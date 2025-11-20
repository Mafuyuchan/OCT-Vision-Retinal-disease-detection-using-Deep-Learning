import torch
from torch.utils.data import DataLoader, TensorDataset
from src.evaluate import evaluate_model
from src.models import create_resnet50

def test_evaluation():
    model = create_resnet50(num_classes=4)
    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    acc, true_labels, preds = evaluate_model(model, loader, device="cpu")

    assert isinstance(acc, float)
    assert len(true_labels) == 8
    assert len(preds) == 8
