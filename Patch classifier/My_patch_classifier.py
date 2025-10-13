import os
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms as T  
from Patch_classifier_utils import initialize_dataloaders, set_seed, initialize_model, train_model, test_func
import json


# ────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


# ────────────────────────────────────────────────────────────
# 1) Seed for reproducibility
# ────────────────────────────────────────────────────────────
set_seed()
logging.info("Random seed set")


# ────────────────────────────────────────────────────────────
# 2) Paths & parameters
# ────────────────────────────────────────────────────────────
training_patches_path = '/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0/patches/training'
testing_patches_path  = '/home/AD/ses235/physionet.org/files/vindr-mammo/1.0.0/patches/test'

patch_set   = 's'     # 's' or 's10'
batch_size  = 32
workers     = 12     
val_ratio   = 0.2
early_stop_patience = 10

logging.info(f"Training patches path: {training_patches_path}")
logging.info(f"Testing  patches path: {testing_patches_path}")
logging.info(f"Patch set: {patch_set}  |  Batch size: {batch_size}  |  Workers: {workers}")


# ────────────────────────────────────────────────────────────
# 3) Initialize dataloaders (grouped by image_id to prevent leakage)
# ────────────────────────────────────────────────────────────
logging.info("Initializing data loaders …")
train_loader, val_loader, test_loader, mean_pixel, std_pixel = initialize_dataloaders(
    training_patches_path,
    testing_patches_path,
    patch_set=patch_set,
    batch_size=batch_size,
    workers=workers,
    val_ratio=val_ratio,
    group_col='image_id',
)
logging.info(f"Loaded datasets → train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

# SAVE THE NORMALIZATION STATS
os.makedirs("saved_models", exist_ok=True)
norm_path = os.path.join("saved_models", "train_norm.json")
with open(norm_path, "w") as f:
    json.dump({"mean": float(mean_pixel), "std": float(std_pixel)}, f)
logging.info(f"Saved training normalization stats to {norm_path}")

dataloaders = {'train': train_loader, 'val': val_loader}


# ────────────────────────────────────────────────────────────
# 4) Model, device, and loss  (BINARY: num_classes=2)
# ────────────────────────────────────────────────────────────
logging.info("Building model …")
model   = initialize_model('resnet', num_classes=2, use_pretrained=True)
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model   = model.to(device)
loss_fn = nn.CrossEntropyLoss()
logging.info(f"Using device: {device}")


# ────────────────────────────────────────────────────────────
# 5) Training (with early stopping in Stage 3 + progress bars)
# ────────────────────────────────────────────────────────────
logging.info("Starting training …")
stages = ['First_Stage', 'Second_Stage', 'Third_Stage']
best_model_wts, train_loss_history, val_loss_history, val_acc_history = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=loss_fn,
    stages=stages,
    device=device,
    patch_set=patch_set,
    early_stop_patience=early_stop_patience,
)
logging.info("Training complete")


# ────────────────────────────────────────────────────────────
# 6) Save checkpoint for testing
# ────────────────────────────────────────────────────────────
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
ckpt_path = os.path.join(save_dir, "best_patch_classifier_2_classes.pth")
torch.save(best_model_wts, ckpt_path)
logging.info(f"Saved best model weights to {ckpt_path}")


# ────────────────────────────────────────────────────────────
# 7) Test the model (original summary)
# ────────────────────────────────────────────────────────────
logging.info("Running final evaluation on test set …")
bar_test_loader = tqdm(test_loader, desc="Testing batches")
class_correct, class_total = test_func(
    weights_dict=ckpt_path,
    test_loader=bar_test_loader,
    device=device,
    loss_fn=loss_fn
)
logging.info(f"Test samples per class: {class_total}")


# ────────────────────────────────────────────────────────────
# 7b) Test metrics (Recall/Sensitivity, Specificity, Precision, F1, Accuracy)
# ────────────────────────────────────────────────────────────
def compute_binary_metrics_on_loader(weights_path, loader, device, loss_fn):
    eval_model = initialize_model('resnet', num_classes=2, use_pretrained=False).to(device)
    eval_model.load_state_dict(torch.load(weights_path, map_location=device))
    eval_model.eval()

    TP = FP = TN = FN = 0
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Testing (metrics)", leave=False):
            inputs  = inputs.to(device=device, dtype=torch.float)
            targets = targets.to(device=device)

            logits = eval_model(inputs)
            loss   = loss_fn(logits, targets)
            preds  = torch.argmax(logits, dim=1)

            total_loss += loss.item() * inputs.size(0)
            n_samples  += inputs.size(0)

            TP += torch.sum((preds == 1) & (targets == 1)).item()
            TN += torch.sum((preds == 0) & (targets == 0)).item()
            FP += torch.sum((preds == 1) & (targets == 0)).item()
            FN += torch.sum((preds == 0) & (targets == 1)).item()

    avg_loss = total_loss / max(n_samples, 1)

    sens = TP / (TP + FN) if (TP + FN) > 0 else float('nan')   # sensitivity / recall
    rec  = sens
    spec = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
    acc  = (TP + TN) / max((TP + TN + FP + FN), 1)
    prec = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else float('nan')

    logging.info("\n[TEST PATCH METRICS]")
    logging.info(f"Confusion Matrix  TP:{TP}  FP:{FP}  TN:{TN}  FN:{FN}")
    logging.info(f"Avg Test Loss : {avg_loss:.6f}")
    logging.info(f"Accuracy      : {acc:.4f}")
    logging.info(f"Sensitivity   : {sens:.4f}  (a.k.a. Recall)")
    logging.info(f"Specificity   : {spec:.4f}")
    logging.info(f"Precision     : {prec:.4f}")
    logging.info(f"F1-Score      : {f1:.4f}\n")

    return {
        "loss": avg_loss,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": acc, "recall": rec, "sensitivity": sens,
        "specificity": spec, "precision": prec, "f1": f1
    }

_ = compute_binary_metrics_on_loader(
    weights_path=ckpt_path,
    loader=test_loader,
    device=device,
    loss_fn=loss_fn
)


# ────────────────────────────────────────────────────────────
# 8) Plot training & validation loss
# ────────────────────────────────────────────────────────────
def plot_loss(train_loss, val_loss, patch_set):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f'Loss Curve for Patch Set: {patch_set}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    out_file = f'loss_curve_{patch_set}.png'
    plt.savefig(out_file)
    logging.info(f"Saved loss curve to {out_file}")
    plt.show()

plot_loss(train_loss_history, val_loss_history, patch_set)
