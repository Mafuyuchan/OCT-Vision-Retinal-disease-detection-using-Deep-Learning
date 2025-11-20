import torch
import torch.nn as nn
from torchvision import models


class VGG16Transfer(nn.Module):
def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
super().__init__()
vgg = models.vgg16(pretrained=pretrained)
# Freeze backbone
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


def forward(self, x):
return self.model(x)
