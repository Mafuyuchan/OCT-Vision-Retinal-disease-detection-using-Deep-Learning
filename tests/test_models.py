import torch
from src.models import create_vgg16, create_resnet50

def test_vgg16_creation():
    model = create_vgg16(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 4)

def test_resnet50_creation():
    model = create_resnet50(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 4)
