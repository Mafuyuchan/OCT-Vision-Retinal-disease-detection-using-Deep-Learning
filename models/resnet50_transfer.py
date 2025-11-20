import torch
import torch.nn as nn
from torchvision import models


class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super().__init__()

        # Load ResNet50
        # TorchVision >= 0.13 uses "weights" instead of pretrained=True
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)

        # Freeze all layers except layer4 + fc
        for name, param in resnet.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False

        # Replace FC layer
        in_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        self.model = resnet

        # You will use this in GradCAM(model, model.target_layer)
        self.target_layer = self.model.layer4[-1]

    def forward(self, x):
        return self.model(x)
