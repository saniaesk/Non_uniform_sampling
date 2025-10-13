import torch
import torch.nn as nn
import torchvision.models as tv

# Builds a torchvision ResNet-50 (ImageNet-pretrained if pretrained=True)
# Replaces the final fc layer with Linear(in_features, num_classes) for binary classification.
def initialize_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    ResNet-50 backbone for 3-channel inputs ([orig, warped, heat]).
    Final FC -> num_classes (binary).
    """
    model = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


class WholeImageClassifier(nn.Module):
  
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        # stores the initialized ResNet-50 in self.backbone
        super().__init__()
        self.backbone = initialize_model(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Bx3xHxW
        # expects B×3×H×W (channels are [orig, warped, heat]) and returns B×2 logits.
        return self.backbone(x)
