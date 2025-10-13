import os
import re
from typing import Tuple, List
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


# -----------------------------
# Binary label mapping
# -----------------------------
CLASS_NAMES = {0: "BI-RADS 1–2", 1: "BI-RADS 3–5"}

# -----------------------------
# Label parsing
# -----------------------------
def _parse_birads_to_binary(s) -> int:
    """
    Accepts 'BI-RADS 3', 'BIRADS 2', '3', 3, etc.
    Returns binary: 0 for BI-RADS 1-2, 1 for BI-RADS 3-5.
    """
    if isinstance(s, (int, np.integer, float, np.floating)):
        k = int(s)
        if 0 <= k <= 4:   # if already 0..4 (sometimes stored), interpret as 1..5
            k = k + 1
        if 1 <= k <= 5:
            return 0 if k <= 2 else 1

    s = str(s).strip().upper().replace("_", "-")
    m = re.search(r"(?:BI-?RADS\s*)?([1-5])", s)
    if not m:
        raise ValueError(f"Unrecognized BI-RADS label: {s}")
    k = int(m.group(1))
    return 0 if k <= 2 else 1


# -----------------------------
# Image I/O
# Reads PNG as grayscale float32 in [0,1]. Handles 8-bit or 16-bit images correctly.
# -----------------------------
def _imread_float01_gray(path: str) -> np.ndarray:
    """Load image as float32 in [0,1], grayscale HxW. Supports 8/16-bit."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype == np.uint16:
        x = im.astype(np.float32) / 65535.0
    else:
        x = im.astype(np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

# These 2 functions provide human (natural) sorting
def _natural_key(s: str):
    """Sort helper: (human order).""" #'warped_image_10.png' after 'warped_image_2.png' 
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]

def _sorted_pngs(folder: str) -> List[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort(key=_natural_key)
    return files


# -----------------------------
# Dataset: fully sequential by folder order
# -----------------------------
class MultiSourceMammoDataset(Dataset):
    """
    Emits a 3-channel tensor for each sample [orig, warped, heat] and a binary label.

    Sequential policy:
      - Read the CSV rows in their existing order.
      - Enumerate PNGs in the warped/heat directories in natural sorted order.
      - Pair row i with warped_pngs[i] and heat_pngs[i].
    """
    def __init__(
        self,
        csv_file: str,
        base_root: str,
        scale: float,
        fwhm: float,
        image_size: Tuple[int, int] = (512, 512),
        mean_3ch: Tuple[float, float, float] = (0.3385, 0.3385, 0.3385),
        std_3ch: Tuple[float, float, float] = (0.3344, 0.3344, 0.3344),
    ):
        self.base_root = base_root
        self.scale = self._fmt(scale)
        self.fwhm = self._fmt(fwhm)
        self.size = image_size
        self.mean = np.array(mean_3ch, dtype=np.float32)
        self.std = np.array(std_3ch, dtype=np.float32)

        # 1) CSV in given order
        df = pd.read_csv(csv_file)
        df = self._harmonize_cols(df)
        paths_src = df["path"].astype(str).tolist()
        labels = df["label"].tolist()

        # 2) Folders → natural-sorted PNG lists
        warp_dir = os.path.join(
            self.base_root,
            f"scale{self.scale}_fwhm{self.fwhm}",
            f"warped_images_scale{self.scale}_fwhm{self.fwhm}",
        )
        heat_dir = os.path.join(
            self.base_root,
            f"scale{self.scale}_fwhm{self.fwhm}",
            f"heatmaps_scale{self.scale}_fwhm{self.fwhm}",
        )
        warped_pngs = _sorted_pngs(warp_dir)
        heat_pngs   = _sorted_pngs(heat_dir)

        # 3) Match by position: i-th row ↔ i-th png
        n = len(paths_src)
        if not (len(warped_pngs) >= n and len(heat_pngs) >= n):
            raise RuntimeError(
                f"Not enough warped/heat PNGs for CSV rows. "
                f"rows={n}, warped={len(warped_pngs)}, heat={len(heat_pngs)}"
            )

        self.samples = []
        for i in range(n):
            src = paths_src[i]
            y_bin = _parse_birads_to_binary(labels[i])
            self.samples.append({
                "src": src,
                "warp": warped_pngs[i],
                "heat": heat_pngs[i],
                "y": int(y_bin),
            })

        # Optional peek
        head = self.samples[:3]
        print("[Dataset:folder-sequential] examples:", [
            (os.path.basename(s['warp']), os.path.basename(s['heat'])) for s in head
        ])

    @staticmethod
    def _harmonize_cols(df: pd.DataFrame) -> pd.DataFrame:
        # renames columns to the expected names
        cols = {c.lower(): c for c in df.columns}
        need = {"path", "label"}
        if not need.issubset(set(cols.keys())):
            raise KeyError(f"CSV must contain columns {need}, got {df.columns.tolist()}")
        return df.rename(columns={cols["path"]: "path", cols["label"]: "label"})[["path", "label"]].copy()

    @staticmethod
    def _fmt(v: float) -> str:
        # formats numbers (e.g., 20.0 → "20") to match folder names
        s = f"{v}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    def __len__(self) -> int:
        # number of samples
        return len(self.samples)

    def _resize_hw(self, x: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        H, W = size
        # Resizes to (H, W) with cv2.INTER_AREA
        return cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA)

    def __getitem__(self, i: int):
        # Reads: original image (src), warped (warp), heatmap (heat) → all as gray float [0,1].
        rec = self.samples[i]
        img = _imread_float01_gray(rec["src"])
        wrp = _imread_float01_gray(rec["warp"])
        hmp = _imread_float01_gray(rec["heat"])

        H, W = self.size
        img = self._resize_hw(img, (H, W))
        wrp = self._resize_hw(wrp, (H, W))
        hmp = self._resize_hw(hmp, (H, W))

        # Stacks to 3×H×W and applies per-channel normalization using provided means/stds
        x = np.stack([img, wrp, hmp], axis=0).astype(np.float32)  # 3xHxW
        x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        y = int(rec["y"])  # 0/1
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        # returns (tensor, label)


# -----------------------------
# Dataloaders (train/test CSVs; val from train)
# -----------------------------
def _read_csv_min(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    cols = {c.lower(): c for c in df.columns}
    if "path" not in cols or "label" not in cols:
        raise KeyError(f"{csv_file} must have 'path' and 'label' columns.")
    return df.rename(columns={cols["path"]: "path", cols["label"]: "label"})[["path", "label"]].copy()

def _stratified_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df["label"].apply(_parse_birads_to_binary).values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, y))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

def build_dataloaders(
    base_root: str,
    train_csv: str,
    test_csv: str,
    scale: float,
    fwhm: float,
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 16,
    num_workers: int = 4,
    means: Tuple[float, float, float] = (0.3385, 0.3385, 0.3385),
    stds: Tuple[float, float, float] = (0.3344, 0.3344, 0.3344),
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Returns {'train','val','test'} dataloaders.
    Fully sequential: pairs CSV rows with warped/heat PNGs by folder order.
    """
    tr_df = _read_csv_min(train_csv)
    te_df = _read_csv_min(test_csv)
    tr_df, va_df = _stratified_split(tr_df, val_ratio=val_ratio, seed=seed)

    # Persist small temp CSVs for Dataset constructor
    tmp_dir = os.path.join(base_root, "_temp_splits_sequential")
    os.makedirs(tmp_dir, exist_ok=True)
    tr_csv = os.path.join(tmp_dir, "train.csv"); tr_df.to_csv(tr_csv, index=False)
    va_csv = os.path.join(tmp_dir, "val.csv");   va_df.to_csv(va_csv, index=False)
    te_csv = os.path.join(tmp_dir, "test.csv");  te_df.to_csv(te_csv, index=False)

    ds_train = MultiSourceMammoDataset(
        csv_file=tr_csv, base_root=base_root, scale=scale, fwhm=fwhm,
        image_size=image_size, mean_3ch=means, std_3ch=stds
    )
    ds_val = MultiSourceMammoDataset(
        csv_file=va_csv, base_root=base_root, scale=scale, fwhm=fwhm,
        image_size=image_size, mean_3ch=means, std_3ch=stds
    )
    ds_test = MultiSourceMammoDataset(
        csv_file=te_csv, base_root=base_root, scale=scale, fwhm=fwhm,
        image_size=image_size, mean_3ch=means, std_3ch=stds
    )

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {"train": dl_train, "val": dl_val, "test": dl_test}
