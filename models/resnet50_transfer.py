import torch
import torch.nn as nn
from torchvision import models


class ResNet50Transfer(nn.Module):
def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
super().__init__()
resnet = models.resnet50(pretrained=pretrained)
# Freeze early layers (optional)
for name, param in resnet.named_parameters():
if "layer4" not in name:
param.requires_grad = False


in_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
nn.Dropout(dropout),
nn.Linear(in_features, 512),
nn.ReLU(inplace=True),
nn.Dropout(dropout),
nn.Linear(512, num_classes)
)
self.model = resnet


def forward(self, x):
return self.model(x)
