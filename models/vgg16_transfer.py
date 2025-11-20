import torch
import torch.nn as nn
from torchvision import models


class VGG16Transfer(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super().__init__()

        # TorchVision >= 0.13 uses "weights" instead of pretrained=True
        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
            vgg = models.vgg16(weights=weights)
        else:
            vgg = models.vgg16(weights=None)

        # Freeze VGG backbone
        for param in vgg.features.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = vgg.classifier[-1].in_features
        vgg.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        self.model = vgg

       
        # Grad-CAM target layer for VGG16
        # Last conv layer = features[-1]
        # Required for GradCAM(model, model.target_layer)
        self.target_layer = self.model.features[-1]

    def forward(self, x):
        return self.model(x)

