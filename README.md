# Non_uniform_sampling

A complete, reproducible pipeline for screening mammography that follows a **patch → heatmap → non-uniform warp → whole-image** strategy.  
Built around **VinDr-Mammo**, it (1) preprocesses images, (2) creates **S** and **S10** patch datasets, (3) trains a **patch classifier**, (4) produces lesion-probability **heatmaps**, and (5) performs **saliency-guided warping** controlled by **`scale`** and **`FWHM`** to emphasize suspicious regions for the **whole-image classifier**.

---

## Highlights

- End-to-end scripts: preprocess → patches → patch training → heatmaps → warps → whole-image training/eval  
- **Non-uniform sampling** (deformation) with tunable **`scale`** (saliency sharpening) and **`FWHM`** (displacement smoothness)  
- 16-bit PNG pipeline, breast segmentation, consistent bbox remapping  
- Deterministic options and clean separation of data, models, and artifacts

---

## Repo map (code)

**Data & preprocessing**
- `vindr_mammo_full_preprocess.py` – from raw VinDr-Mammo to `resized_images_manifest.csv`
- `My_Resize.py`, `resize_utils.py` – resizing (1152×896), segmentation, bbox remap, 16-bit I/O
- `CSV_generator.py` – splits manifest into `training_resized_images.csv` / `test_resized_images.csv`

**Patch datasets (S/S10)**
- `My_patch_generator3.py` – builds **S** (1 ROI + 1 background) and **S10** (10 ROI-near + backgrounds)
- `patches_utils_3.py` – sampling rules, overlap cutoffs, padding, PNG writing

**Patch classifier**
- `My_patch_classifier_main_3.py` – 3-stage fine-tuning, metrics, checkpointing
- `Patch_classifier_utils_3.py` – dataloaders, transforms, mean/std, weighted sampler

**Heatmaps & non-uniform warping**
- `DeformationUtils.py` – Gaussian builders, pixel/structure-driven grids, blended sampler (`grid_sample`)
- `My_WARP5.py`, `My_deformation2.py` – single-image demos & ablations
- `SingleDeformation_2.py` – one-off: heatmap + quivers + warped output (for figures)
- `WARP_all_images_AUTO.py` – batch heatmaps/overlays/warps (parameterized filenames)

**Whole-image classifier**
- `whole_classifier_model.py` – image-level model (patch backbone + ResNet blocks + classifier head)

---

## Dataset

### VinDr-Mammo Dataset (https://physionet.org/content/vindr-mammo/1.0.0/)
- **Modality:** FFDM screening mammograms with lesion-level annotations.  
- **Preprocess (this repo):**
  1. **Resize to 1152×896** (padding as needed) and save **16-bit PNG**
  2. **Breast segmentation** (contour-based) to remove background
  3. **BBox remap** to the resized space; write `resized_images_manifest.csv` with:
     ```
     image_id, path, split, label, xmin, ymin, xmax, ymax, (scales/pads)
     ```
> You must obtain VinDr-Mammo under its terms. This repo provides **code only**.

### Derived patch datasets: **S** and **S10**
- **S (sparse):** Per image, one ROI-centered abnormal patch (224×224) + one background patch  
- **S10 (dense):** Ten abnormal patches near ROI (20% overlap) + multiple background patches  
- **CSV schema:** `path,label,image_id,split,set` where `set ∈ {S, S10}`  
- Saved under: `patches/{training|test}/{S|S10}/` with mirrored CSVs

### Derived heatmap & warp datasets (new)
- **Heatmaps:** slide 224×224 with stride (default **32**), per-patch lesion probability → coarse grid → resized to full image  
  - Stored as: `heatmaps/{split}/{image_id}_scale{S}_fwhm{F}.npy`  
  - Overlays: `overlays/{split}/{image_id}_overlay_scale{S}_fwhm{F}.png`
- **Warps:** saliency-guided deformations (see below)  
  - Stored as: `warps/{split}/{image_id}_warp_scale{S}_fwhm{F}_lam{L}.png`  
  - Optional sampling grids: `grids/{split}/{image_id}_grid_scale{S}_fwhm{F}_lam{L}.npy` (H×W×2, normalized to [-1,1])

---

## Non-uniform sampling (deformation)

Implemented in `DeformationUtils.py`:

1. **Pixel-driven grid (`warped_imgs`)**  
   - Softmax over the heatmap scaled by **`scale`** → probability mass → smoothed by a 2-D Gaussian with **`FWHM`**  
   - Produces a dense displacement field that prioritizes salient regions

2. **Structure-driven grid (`warped_str`)**  
   - Axial max-reductions + 1-D Gaussian smoothing → global structure; controlled by **`FWHM`**

3. **Blended grid**  
   - `src_grid = (1−λ)·structure + λ·pixel` with **`λ ∈ [0,1]`** for locality control

All grids are sampled via `torch.nn.functional.grid_sample`.  
**Intuition:** larger **`scale`** → sharper, more localized focus; larger **`FWHM`** → smoother, more global deformations.

---

## Quickstart

> Replace paths with your dataset locations. Avoid hard-coding drive letters; pass them as args.

### Requirements
```bash
conda create -n nonuniform python=3.10 -y
conda activate nonuniform
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26 pandas scikit-learn opencv-python pillow matplotlib pyyaml tqdm tensorboard albumentations torchmetrics rich pydicom
