import torch
from torchvision import transforms
from PIL import Image

def predict(model, image_path, device, class_names):
    """
    Predict the class of a single retina image.
    
    Args:
        model: Trained PyTorch model.
        image_path: File path of the input image.
        device: "cuda" or "cpu".
        class_names: List of class names.
        
    Returns:
        Predicted class label (string).
    """
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]
