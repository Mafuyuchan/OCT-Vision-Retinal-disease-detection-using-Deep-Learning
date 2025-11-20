import torch
from torchvision import transforms
from src.dataset import OCTDataset

def test_dataset_loading():
    dataset = OCTDataset(
        root_dir="data/train",
        transform=transforms.ToTensor()
    )

    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, int)
