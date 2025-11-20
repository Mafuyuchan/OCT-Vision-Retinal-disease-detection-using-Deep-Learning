import torch
import torch.nn.functional as F

def evaluate_model(model, dataloader, device):
    """
    Evaluate classification accuracy for a trained model.
    
    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for validation/test data.
        device: "cuda" or "cpu".
        
    Returns:
        accuracy (float), all_labels (list), all_preds (list)
    """
    model.eval()
    total, correct = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_labels, all_preds
