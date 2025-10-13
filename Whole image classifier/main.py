import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm 
from tools import build_dataloaders, CLASS_NAMES
from whole_classifier_model import WholeImageClassifier


# =========================
#        CONFIG
# =========================
CONFIG = {
    # Required CSVs
    #"train_csv":  "/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0/training_resized_images.csv",
    "train_csv":  "E:/VinDr_mammo/training_resized_images.csv",

    #"test_csv":   "/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0/test_resized_images.csv",
    "test_csv":   "E:/VinDr_mammo/test_resized_images.csv",

    # Manifest used during warp/heat generation (for consistent idx mapping)
    #"manifest_csv": "/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0/resized_images_manifestold.csv",
    "manifest_csv": "E:/VinDr_mammo/resized_images_manifest.csv",

    # Root containing the warped/heat folders
    #"base_root": "/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0/warped_datasets",
    "base_root": "E:/VinDr_mammo",


    # Warped/heat parameter (Important: must match generated datasets)
    "scale": 20.0,
    "fwhm":  15.0,

    # Validation split from training CSV
    "val_ratio": 0.15,
    "seed": 42,

    # Image & batching
    "height": 512,
    "width":  512,
    "batch":  16,
    "workers": 4,

    # Normalization (per-channel for [orig, warped, heat])
    "means": (0.3522, 0.3522, 0.3522),
    "stds":  (0.3316, 0.3316, 0.3316),

    # Training schedule
    "epochs_stage1": 15,   # train layer4 + fc
    "epochs_stage2": 25,   # unfreeze all
    "lr1": 1e-4, "wd1": 1e-4,
    "lr2": 1e-5, "wd2": 1e-4,

    # Model init
    "use_imagenet_pretrain": True,
}


# =========================
#     UTIL FUNCTIONS
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_stage(model, loaders, device, epochs=20, lr=1e-4, wd=1e-4, unfreeze_all=False):
    """
    One training stage. If unfreeze_all=False, train only layer4 + fc.
    """
    if not unfreeze_all:
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if name.startswith("backbone.layer4.") or name.startswith("backbone.fc"):
                p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optim = Adam(params, lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_acc = 0.0

    for ep in range(epochs):
        for phase in ("train", "val"):
            model.train(mode=(phase == "train"))
            running_loss, y_true, y_pred = 0.0, [], []

            loader = loaders[phase]
            # Progress bar over batches for this phase/epoch
            for xb, yb in tqdm(loader, desc=f"[{phase}] epoch {ep+1}/{epochs}", dynamic_ncols=True, leave=False):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == "train"):
                    logits = model(xb)                # Bx2
                    loss = criterion(logits, yb)      # CE on logits
                    if phase == "train":
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                running_loss += loss.item() * xb.size(0)
                y_true.extend(yb.detach().cpu().tolist())
                y_pred.extend(logits.detach().cpu().argmax(dim=1).tolist())

            epoch_loss = running_loss / len(loader.dataset)
            acc = accuracy_score(y_true, y_pred)
            print(f"[{phase:5}] epoch {ep+1:02d} | loss={epoch_loss:.4f} | acc={acc:.4f}")

            if phase == "val" and acc > best_acc:
                best_acc = acc
                best_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_wts, strict=True)
    return best_acc


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Binary evaluation: returns accuracy and ROC AUC for the positive class (BI-RADS 3–5).
    """
    model.eval()
    y_true, y_prob1 = [], []  # prob of class 1 (BI-RADS 3–5)

    # Progress bar for evaluation
    for xb, yb in tqdm(loader, desc="[test] inference", dynamic_ncols=True):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # p(class=1)
        y_true.extend(yb.numpy().tolist())
        y_prob1.extend(probs.tolist())

    y_true = np.array(y_true)
    y_prob1 = np.array(y_prob1)
    y_pred = (y_prob1 >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob1)
    except Exception:
        auc = float("nan")
    return acc, auc


# =========================
#          MAIN
# =========================
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    loaders = build_dataloaders(
        base_root=cfg["base_root"],
        train_csv=cfg["train_csv"],
        test_csv=cfg["test_csv"],
        scale=cfg["scale"],
        fwhm=cfg["fwhm"],
        image_size=(cfg["height"], cfg["width"]),
        batch_size=cfg["batch"],
        num_workers=cfg["workers"],
        means=cfg["means"],
        stds=cfg["stds"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
    )

    model = WholeImageClassifier(
        num_classes=2, pretrained=bool(cfg["use_imagenet_pretrain"])
    ).to(device)

    print("\n[Stage 1] Train head & last stage")
    best_val_acc_1 = train_one_stage(
        model, loaders, device,
        epochs=cfg["epochs_stage1"], lr=cfg["lr1"], wd=cfg["wd1"], unfreeze_all=False
    )

    print("\n[Stage 2] Unfreeze all")
    best_val_acc_2 = train_one_stage(
        model, loaders, device,
        epochs=cfg["epochs_stage2"], lr=cfg["lr2"], wd=cfg["wd2"], unfreeze_all=True
    )

    print(f"\n[VAL] best_acc_stage1={best_val_acc_1:.4f} | best_acc_stage2={best_val_acc_2:.4f}")

    print("\n[TEST] evaluating best weights on test split…")
    test_acc, test_auc = evaluate(model, loaders["test"], device)
    print(f"[TEST] acc={test_acc:.4f} | ROC-AUC={test_auc:.4f}")
    print("\n[CLASSES]", CLASS_NAMES)

if __name__ == "__main__":
    main()
