import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from imageio import imread
from patches_utils_3 import sample_patches

# --------------------
# Config 
# --------------------
home_data = "/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0"
manifest_unified = os.path.join(home_data, "resized_images_manifest.csv")
patches_root = os.path.join(home_data, "patches")
os.makedirs(patches_root, exist_ok=True)

# --------------------
# Helpers
# --------------------
def _valid_bbox(row):
    for k in ("xmin", "ymin", "xmax", "ymax"):
        if k not in row or pd.isna(row[k]):
            return False
    xmin, ymin, xmax, ymax = float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])
    return (xmax > xmin) and (ymax > ymin)

def _parse_index_from_patch_path(p):
    """Extract numeric index from filenames like roi_123.png, roi_123_4.png, bkg_123.png, bkg_123_4.png"""
    m = re.search(r'_(\d+)(?:[_\.])', Path(p).name)
    return int(m.group(1)) if m else None

def _enrich_and_save(df_s, df_s10, idx2imgid, split_name, out_dir):
    """Add image_id, split, set columns and write CSVs."""
    if len(df_s):
        df_s["image_idx"] = df_s["path"].apply(_parse_index_from_patch_path)
        df_s["image_id"]  = df_s["image_idx"].map(idx2imgid).fillna("unknown").astype(str)
        df_s["split"]     = split_name
        df_s["set"]       = "s"
        df_s.drop(columns=["image_idx"], inplace=True)
        df_s.to_csv(os.path.join(out_dir, "s.csv"), index=False)
        print(f"[{split_name}] Wrote S CSV: {os.path.join(out_dir, 's.csv')} ({len(df_s)} rows)")
    else:
        print(f"[{split_name}] No S patches created.")

    if len(df_s10):
        df_s10["image_idx"] = df_s10["path"].apply(_parse_index_from_patch_path)
        df_s10["image_id"]  = df_s10["image_idx"].map(idx2imgid).fillna("unknown").astype(str)
        df_s10["split"]     = split_name
        df_s10["set"]       = "s10"
        df_s10.drop(columns=["image_idx"], inplace=True)
        df_s10.to_csv(os.path.join(out_dir, "s10.csv"), index=False)
        print(f"[{split_name}] Wrote S10 CSV: {os.path.join(out_dir, 's10.csv')} ({len(df_s10)} rows)")
    else:
        print(f"[{split_name}] No S10 patches created.")

# --------------------
# Load unified manifest
# --------------------
if not os.path.isfile(manifest_unified):
    raise FileNotFoundError(f"Manifest not found: {manifest_unified}")

df_all = pd.read_csv(manifest_unified)
df_all.columns = [c.strip().lower() for c in df_all.columns]

required = {"path", "image_id", "xmin", "ymin", "xmax", "ymax", "split"}
missing = required - set(df_all.columns)
if missing:
    raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

# Normalize split values (support exactly 'training' and 'test')
df_all["split"] = df_all["split"].astype(str).str.strip().str.lower()
ok_splits = {"training", "test"}
if not set(df_all["split"]).issubset(ok_splits):
    bad = df_all[~df_all["split"].isin(ok_splits)]
    raise ValueError(
        "Found rows with invalid split values (expected 'training' or 'test'). "
        f"Examples:\n{bad[['path','split']].head()}"
    )

# --------------------
# Generate per-split
# --------------------
for split_name in ["training", "test"]:
    df = df_all[df_all["split"] == split_name].reset_index(drop=True)
    if df.empty:
        print(f"[{split_name}] No rows — skipping.")
        continue

    print(f"\n=== Generating patches for {split_name} (N images: {len(df)}) ===")

    # Output dirs
    split_out_dir    = os.path.join(patches_root, split_name)
    s_patch_folder   = os.path.join(split_out_dir, "s")
    s10_patch_folder = os.path.join(split_out_dir, "s10")
    os.makedirs(s_patch_folder, exist_ok=True)
    os.makedirs(s10_patch_folder, exist_ok=True)

    s_list, s10_list = [], []
    idx2imgid = df["image_id"].to_dict()  # map row index -> image_id

    for index, row in df.iterrows():
        img_path = row["path"]
        if not os.path.isfile(img_path):
            print(f"[{split_name}] Missing file, skipping: {img_path}")
            continue
        if not _valid_bbox(row):
            print(f"[{split_name}] Invalid bbox, skipping: {img_path}")
            continue

        try:
            mam = imread(img_path)
        except Exception as e:
            print(f"[{split_name}] Read failed for {img_path}: {e}")
            continue

        xmin, ymin = int(float(row["xmin"])), int(float(row["ymin"]))
        xmax, ymax = int(float(row["xmax"])), int(float(row["ymax"]))

        # Build rectangular ROI mask from bbox
        mask = np.zeros_like(mam, dtype=np.uint8)
        h, w = mask.shape[:2]
        xmin_c = max(0, min(xmin, w))
        xmax_c = max(0, min(xmax, w))
        ymin_c = max(0, min(ymin, h))
        ymax_c = max(0, min(ymax, h))
        if xmax_c <= xmin_c or ymax_c <= ymin_c:
            print(f"[{split_name}] Clipped bbox invalid, skipping: {img_path}")
            continue
        mask[ymin_c:ymax_c, xmin_c:xmax_c] = 255

        # Ignore string BI-RADS labels; use 1 for ROI-centered/overlap patches, 0 for backgrounds (handled inside sampler)
        label_for_roi = 1

        print(f"[{split_name}] {index+1}/{len(df)}  path={img_path}  image_id={row['image_id']}  bbox=({xmin},{ymin},{xmax},{ymax})")

        try:
            sample_patches(
                mam, index, label_for_roi, mask,
                folder_s=s_patch_folder,
                folder_s10=s10_patch_folder,
                list_s=s_list, list_s10=s10_list,
                patch_size=224,
                pos_cutoff=0.90,
                neg_cutoff=0.10,   
                nb_bkg=11,         # 1 bkg -> S, remaining -> S10
                nb_abn=10,
                start_sample_nb=0,
                verbose=True
            )
        except Exception as e:
            print(f"[{split_name}] Sampling failed for {img_path}: {e}")
            continue

    # Save CSVs with enrichment
    df_s   = pd.DataFrame(s_list)
    df_s10 = pd.DataFrame(s10_list)
    print(f"[{split_name}] S count   : {len(df_s)}")
    print(f"[{split_name}] S10 count : {len(df_s10)}")

    _enrich_and_save(df_s, df_s10, idx2imgid, split_name, split_out_dir)

print("\nDone.")
