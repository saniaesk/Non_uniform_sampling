import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import numpy as np 
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
import copy
from imageio import imread
import random
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    print(f"Random seed set as {seed}")


def split_unbalanced_data(dataframe_path,split_ratio=0.2):

    df = pd.read_csv(dataframe_path)
    # Split the data into train and validation set

    train_df, val_df = train_test_split(df, test_size=split_ratio, stratify=df['label'], random_state=42)

    return train_df, val_df


class VinDrPatchDataset(Dataset):
    """
    Expects a pandas.DataFrame with columns ['path','label'], where label is already an int in 0..4.
    Loads each 16-bit PNG, scales to [0,1], converts to 3×H×W, and applies optional transforms.
    """
    def __init__(self, source, transform=None):
        if isinstance(source, pd.DataFrame):
            self.df = source.reset_index(drop=True)
        else:
            # fallback: read from CSV path
            self.df = pd.read_csv(source)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['path'], cv2.IMREAD_UNCHANGED).astype(np.float32)
        img /= 65535.0
        x = torch.from_numpy(img).unsqueeze(0).repeat(3,1,1)
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(int(row['label']), dtype=torch.long)
        return x, y


def initialize_model(model_name, num_classes, use_pretrained=True):
    """
    Initializes and returns a PyTorch model for patch classification.

    The architecture employed for the patch classifier is ResNet50. Since this patch model classifies a custom number of patch categories,
    it is necessary to adjust the default ResNet50 of PyTorch, which classifies 1000 classes.

    Args:
        model_name (str): The name of the model architecture to use (e.g., "resnet").
        num_classes (int): The number of output classes for classification.
        use_pretrained (bool, optional): Whether to use a model pre-trained on ImageNet. Defaults to True.
      model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)
    Returns:
        torch.nn.Module: The initialized model with the specified number of output classes.
    """
    model_ft = None

    if model_name == "resnet":
        """ Resnet50
        """
        if use_pretrained:
            model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else :
            model_ft = models.resnet50(weights=None)
    
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'. Only 'resnet' is currently supported.")

    return model_ft


def initialize_dataloaders(
    training_patches_path,
    testing_patches_path,
    patch_set='s',
    batch_size=64,
    workers=12,
    split_ratio=0.2
):
    """
    Initialize train/val/test DataLoaders for VinDr patch CSVs.
    Expects two folders:
      training_patches_path/  containing s.csv or s10.csv
      testing_patches_path/   containing s.csv or s10.csv

    Returns:
        train_loader, val_loader, test_loader
    """

    # 1) decide which CSV names to use
    if patch_set == 's10':
        train_csv = os.path.join(training_patches_path, 's10.csv')
        test_csv  = os.path.join(testing_patches_path,  's10.csv')
    elif patch_set == 's':
        train_csv = os.path.join(training_patches_path, 's.csv')
        test_csv  = os.path.join(testing_patches_path,  's.csv')
    else:
        raise ValueError(f"Unknown patch_set '{patch_set}'. Use 's' or 's10'.")

    # 2) load train CSV and split off validation
    train_df, val_df = split_unbalanced_data(train_csv, split_ratio)

    # 3) load test CSV
    test_df = pd.read_csv(test_csv)

    # --- DEBUG: Show raw unique labels in each split ---
    #print("TRAIN raw labels:", sorted(train_df['label'].unique()))
    #print("VAL   raw labels:", sorted(val_df  ['label'].unique()))
    #print("TEST  raw labels:", sorted(test_df ['label'].unique()))

    # 4) map "BI-RADS X" → integer X, then zero-base → [0..4],
    #    but leave background '0' as 0
    def normalize_labels(df, num_classes=6):
        # strip the "BI-RADS " prefix and cast to int:
        df['label'] = (
            df['label']
              .str.replace(r'BI-RADS\s*', '', regex=True)
              .fillna('0')        # if some rows really had no BI-RADS tag
              .astype(int)        # yields 0…5
        )
        # now ensure they’re in [0 .. num_classes-1]
        if df['label'].min() < 0 or df['label'].max() > (num_classes-1):
            raise ValueError(f"Found out‐of‐range label after mapping: {df['label'].unique()}")
 
    for df_ in (train_df, val_df, test_df):
        normalize_labels(df_, num_classes=6)

    # --- DEBUG: Show mapped labels ---
    #print("TRAIN mapped labels:", sorted(train_df['label'].unique()))
    #print("VAL   mapped labels:", sorted(val_df  ['label'].unique()))
    #print("TEST  mapped labels:", sorted(test_df ['label'].unique()))

    # 5) compute global pixel mean/std on train only (16-bit→[0,1])
    total_sum, total_sqsum, total_pix = 0.0, 0.0, 0
    for _, row in train_df.iterrows():
        img = (cv2.imread(row['path'], cv2.IMREAD_UNCHANGED)
               .astype(np.float32) / 65535.0)
        total_sum   += img.sum()
        total_sqsum += (img**2).sum()
        total_pix   += img.size

    mean_pixel = total_sum / total_pix
    var_pixel  = total_sqsum / total_pix - mean_pixel**2
    std_pixel  = np.sqrt(var_pixel)
    print(f"Computed dataset mean = {mean_pixel:.4f}, std = {std_pixel:.4f}")

    # 6) define transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.RandomAffine(degrees=0, scale=(0.8,1.2)),
        T.Normalize(mean=[mean_pixel], std=[std_pixel]),
    ])
    val_transform = T.Normalize(mean=[mean_pixel], std=[std_pixel])

    # 7) create PyTorch dataset objects
    train_set = VinDrPatchDataset(train_df, transform=train_transform)
    val_set   = VinDrPatchDataset(val_df,   transform=val_transform)
    test_set  = VinDrPatchDataset(test_df,  transform=val_transform)

    # 8) build a weighted sampler to rebalance train set
    labels         = train_df['label'].to_numpy()
    class_counts   = np.bincount(labels, minlength=6)
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 9) final DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        sampler=sampler, num_workers=workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_model(model, dataloaders, criterion, stages, device, patch_set='s'):
    since = time.time()

    val_acc_history   = []
    train_loss_history = []
    val_loss_history  = []
    model = model.to(device=device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc       = 0.0

    for stage in stages:
        # configure optimizer for this stage
        if stage == "First_Stage":
            num_epochs = 3
            params_to_update = [param for name, param in model.named_parameters() if name.startswith('fc')]
            for p in model.parameters(): p.requires_grad = False
            for p in params_to_update: p.requires_grad = True
            optimizer = optim.Adam(params_to_update, lr=1e-3, weight_decay=1e-4)

        elif stage == "Second_Stage":
            num_epochs = 10
            params_to_update = [param for name, param in model.named_parameters()
                                if name.startswith('layer4.2') or name.startswith('fc')]
            for p in model.parameters(): p.requires_grad = False
            for p in params_to_update: p.requires_grad = True
            optimizer = optim.Adam(params_to_update, lr=1e-4, weight_decay=1e-4)

        else:  # Third_Stage
            if patch_set == 's':   num_epochs = 187
            else:                  num_epochs = 37
            params_to_update = list(model.parameters())
            for p in params_to_update: p.requires_grad = True
            optimizer = optim.Adam(params_to_update, lr=1e-5, weight_decay=1e-4)

        # Epoch loop with progress bar
        for epoch in range(num_epochs):
            print(f"\n=== Stage {stage} Epoch {epoch+1}/{num_epochs} ===")
            for phase in ['train', 'val']:
                is_train = (phase == 'train')
                model.train() if is_train else model.eval()

                running_loss = 0.0
                running_corrects = 0

                # wrap the DataLoader in tqdm
                loader = dataloaders[phase]
                pbar = tqdm(loader, desc=f"{phase.upper()} ", leave=False)
                for inputs, labels in pbar:
                    inputs = inputs.to(device=device, dtype=torch.float)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(is_train):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if is_train:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # update bar
                    pbar.set_postfix(
                        loss=f"{running_loss/((pbar.n+1)*inputs.size(0)):.4f}",
                        acc=f"{running_corrects.double()/((pbar.n+1)*inputs.size(0)):.4f}"
                    )

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc  = running_corrects.double() / len(loader.dataset)

                if is_train:
                    train_loss_history.append(epoch_loss)
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

                # deep copy best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # load best weights
    model.load_state_dict(best_model_wts)
    return best_model_wts, train_loss_history, val_loss_history, val_acc_history


def test_func(weights_dict, test_loader, device, loss_fn):
    """
    Run final evaluation on the test_loader.
    Returns per-class correct counts and totals.
    """
    # 1) Load model
    model = initialize_model("resnet", num_classes=6, use_pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_dict, map_location=device))
    model.eval()

    # 2) Capture dataset size before wrapping in tqdm
    dataset_size = len(test_loader)

    test_loss      = 0.0
    n_classes      = model.fc.out_features
    class_correct  = [0 for _ in range(n_classes)]
    class_total    = [0 for _ in range(n_classes)]

    # 3) Wrap the loader in a separate pbar variable
    pbar = tqdm(test_loader, desc="Testing batches")
    with torch.no_grad():
        for data, target in pbar:
            data   = data.to(device=device, dtype=torch.float)
            target = target.to(device=device)

            outputs = model(data)
            loss    = loss_fn(outputs, target)

            # accumulate total loss
            test_loss += loss.item() * data.size(0)

            # get predictions and compare
            _, preds = torch.max(outputs, 1)
            correct = preds.eq(target)

            # accumulate per-sample stats
            for i in range(data.size(0)):
                lbl = target[i].item()
                class_correct[lbl] += int(correct[i].item())
                class_total[lbl]   += 1

    # 4) Finalize average loss
    avg_test_loss = test_loss / dataset_size
    print(f"\nTest Loss: {avg_test_loss:.6f}\n")

    # 5) Print per-class accuracies
    for i in range(n_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"Class {i:2d} Accuracy: {acc:5.2f}% ({class_correct[i]}/{class_total[i]})")

    overall_acc = 100.0 * sum(class_correct) / sum(class_total)
    print(f"\nOverall Test Accuracy: {overall_acc:5.2f}% "
          f"({sum(class_correct)}/{sum(class_total)})")

    return class_correct, class_total
    


#############################################################
#   Fully-convolutional patch classifier utilities 
#############################################################

import torch.nn.functional as F
from collections import OrderedDict

class ResNet50Backbone(nn.Module):
    """ResNet-50 trunk up to layer4 (no avgpool/fc)."""
    def __init__(self, use_pretrained=True):
        super().__init__()
        base = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None
        )
        # expose layers
        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu  = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.out_channels = 2048

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x


class PatchClassifierFCN(nn.Module):
    """
    Fully-convolutional version of your patch classifier.
    Output: logits [B, C, U, V], where C = num_classes.
    """
    def __init__(self, num_classes=6, use_pretrained=True):
        super().__init__()
        self.backbone = ResNet50Backbone(use_pretrained=use_pretrained)
        self.head     = nn.Conv2d(self.backbone.out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)          # [B, 2048, U, V]
        logits = self.head(feats)         # [B, C, U, V]
        return logits


def _strip_prefix(k: str) -> str:
    for pref in ("module.", "model.", "backbone."):
        if k.startswith(pref):
            return k[len(pref):]
    return k


def load_patch_checkpoint_into_fcn(fcn_model: nn.Module, ckpt_path: str):
    """
    Load your patch classifier checkpoint into the FCN:
    - copy every matching conv weight
    - convert 'fc.weight' (C,2048) → head.weight (C,2048,1,1)
      and 'fc.bias' → head.bias
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    sd  = raw.get("state_dict", raw)
    sd  = { _strip_prefix(k): v for k, v in sd.items() }

    # 1) copy trunk
    bb_sd = fcn_model.backbone.state_dict()
    bb_new = OrderedDict()
    for k, v in sd.items():
        if k.startswith("fc."):  # skip here; handle in step 2
            continue
        if k in bb_sd and bb_sd[k].shape == v.shape:
            bb_new[k] = v
    fcn_model.backbone.load_state_dict(bb_new, strict=False)

    # 2) map fc → 1x1 head if present
    if "fc.weight" in sd and "fc.bias" in sd:
        with torch.no_grad():
            W = sd["fc.weight"]            # [C, 2048]
            b = sd["fc.bias"]              # [C]
            fcn_model.head.weight.copy_(W.view(W.size(0), W.size(1), 1, 1))
            fcn_model.head.bias.copy_(b)
    # done
    return fcn_model


# builds FCN (ResNet50 trunk + 1x1 conv head), can warm-start from best_patch_classifier.pth
def initialize_model_fcn(num_classes=6, use_pretrained=True, ckpt_path=None):
    """
    Build an FCN patch model that outputs [B,C,U,V].
    Optionally warm-start from your best_patch_classifier.pth.
    """
    model = PatchClassifierFCN(num_classes=num_classes, use_pretrained=use_pretrained)
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print(f"[FCN] Loading conv trunk + converting fc from: {ckpt_path}")
        model = load_patch_checkpoint_into_fcn(model, ckpt_path)
    return model


@torch.no_grad()
# runs FCN on a full image and returns logits [1,C,U,V], probs [1,C,U,V]
def infer_heatmap_whole_image(model_fcn: nn.Module, img_path: str, mean: float, std: float, device: torch.device):
    """
    Run FCN on a whole image and return:
      - logits [1,C,U,V]
      - softmax heatmap [1,C,U,V] if you want probabilities
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    x = torch.from_numpy(img).unsqueeze(0).repeat(3,1,1)  # [3,H,W]
    tfm = T.Normalize(mean=[mean]*3, std=[std]*3)
    x = tfm(x).unsqueeze(0).to(device)                    # [1,3,H,W]
    model_fcn.eval()
    logits = model_fcn(x)                                 # [1,C,U,V]
    prob   = torch.softmax(logits, dim=1)
    return logits, prob
