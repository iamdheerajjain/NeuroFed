import torch
import torch.nn as nn
import torchvision.models as models


class PreTrainedCNN(nn.Module):
    def __init__(self, num_classes: int = 2, model_name: str = "resnet50"):
        super().__init__()
        
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            # Freeze early layers
            for param in list(self.backbone.parameters())[:-20]:
                param.requires_grad = False
            self.backbone.fc = nn.Linear(2048, num_classes)
        elif model_name == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.classifier = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int) -> nn.Module:
    return PreTrainedCNN(num_classes=num_classes, model_name="resnet50")
