import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


# -----------------------------
# 1) Configuration
# -----------------------------
IMG_DIR    = "/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0/images"
META_CSV   = "metadata.csv"
BREAST_CSV = "breast-level_annotations.csv"
FIND_CSV   = "finding_annotations.csv"

TEST_SIZE    = 0.2
RANDOM_STATE = 42

BATCH_SIZE             = 4
IMAGE_DISPLAY_SIZE     = (512, 512)   # width, height
NUM_SAMPLES_PER_VISUAL = 4

# -----------------------------
# 2) Load & Align CSVs
# -----------------------------
meta_df   = pd.read_csv(META_CSV)
# Rename the SOP Instance UID column to match annotation files
meta_df = meta_df.rename(columns={"SOP Instance UID": "image_id"})

breast_df = pd.read_csv(BREAST_CSV)
find_df   = pd.read_csv(FIND_CSV)

# Merge metadata with breast-level annotations on image_id
df = meta_df.merge(breast_df, on="image_id", how="left")

# Group finding annotations by image_id
findings_grp = (
    find_df
    .groupby("image_id")
    .apply(lambda x: x.to_dict(orient="records"))
    .to_dict()
)

# Build full file path for each DICOM
df["filepath"] = df["image_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.dicom"))

# -----------------------------
# 3) Dataset Summary
# -----------------------------
total = len(df)
print(f"Total images: {total}")

# Stratified split on BI-RADS (breast_birads or bi_rads column)
label_col = "breast_birads" if "breast_birads" in df.columns else "bi_rads"
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE,
    stratify=df[label_col],
    random_state=RANDOM_STATE
)

print(f"Train images: {len(train_df)}, Test images: {len(test_df)}")
print("\nTrain BI-RADS distribution:")
print(train_df[label_col].value_counts().sort_index())
print("\nTest BI-RADS distribution:")
print(test_df[label_col].value_counts().sort_index())

# -----------------------------
# 4) PyTorch Dataset
# -----------------------------
class VinDrFullDataset(Dataset):
    def __init__(self, df, findings_map, transforms=None):
        self.df = df.reset_index(drop=True)
        self.findings_map = findings_map
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dic = pydicom.dcmread(row.filepath)
        img = dic.pixel_array.astype(np.float32)
        # Normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # Convert to PIL and to RGB
        img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        # Apply transforms or default to ToTensor
        if self.transforms:
            img_t = self.transforms(img)
        else:
            img_t = T.ToTensor()(img)
        # Label and findings
        label = int(row[label_col])
        findings = self.findings_map.get(row.image_id, [])
        return img_t, label, findings

# -----------------------------
# 5) Transforms & DataLoaders
# -----------------------------
train_transform = T.Compose([
    T.Resize(IMAGE_DISPLAY_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
])
test_transform = T.Compose([
    T.Resize(IMAGE_DISPLAY_SIZE),
    T.ToTensor(),
])

train_ds = VinDrFullDataset(train_df, findings_grp, transforms=train_transform)
test_ds  = VinDrFullDataset(test_df,  findings_grp, transforms=test_transform)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

print(f"\nDataLoaders created: {len(train_loader)} train batches, {len(test_loader)} test batches")

# -----------------------------
# 6) Visualization with Annotations
# -----------------------------
def show_samples(dl):
    imgs, labels, findings = next(iter(dl))
    fig, axes = plt.subplots(1, len(imgs), figsize=(15,5))
    for i in range(len(imgs)):
        ax = axes[i]
        img = imgs[i].permute(1,2,0).numpy()
        ax.imshow(img)
        ax.set_title(f"BI-RADS: {labels[i].item()}")
        ax.axis('off')
        # Overlay bounding boxes
        for f in findings[i]:
            rect = patches.Rectangle(
                (f['xmin'], f['ymin']),
                f['xmax'] - f['xmin'],
                f['ymax'] - f['ymin'],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
    plt.show()

print("\nVisualizing training samples:")
show_samples(train_loader)

print("\nVisualizing testing samples:")
show_samples(test_loader)

print("\nDone! VinDr-Mammo dataset is loaded, split, and visualized.")
