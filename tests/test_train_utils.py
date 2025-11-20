import torch
from src.train_utils import accuracy

def test_accuracy_function():
    preds = torch.tensor([0, 1, 2, 2])
    labels = torch.tensor([0, 2, 2, 2])

    acc = accuracy(preds, labels)
    assert acc == 75.0  # 3/4 correct
