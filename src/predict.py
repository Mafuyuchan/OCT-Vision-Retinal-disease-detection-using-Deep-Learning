import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from gradcam import GradCAM   # import your GradCAM class


def load_transform():
    """Standard ImageNet pre-processing for VGG16/ResNet50."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict(model, image_path, device, class_names, use_gradcam=False):
    """
    Predict the class of a single OCT image and optionally generate Grad-CAM.

    Args:
        model: Trained PyTorch model.
        image_path: Path to the input image.
        device: "cuda" or "cpu".
        class_names: List of class names.
        use_gradcam: Whether to generate a Grad-CAM heatmap.

    Returns:
        dict containing:
            - predicted_class  (str)
            - predicted_idx    (int)
            - probabilities    (list of floats)
            - gradcam          (numpy array or None)
    """
    model.eval()
    transform = load_transform()

    # load image safely
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    img_tensor = transform(image).unsqueeze(0).to(device)

    # ---- Forward pass ----
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_idx]

    # ---- Grad-CAM ----
    gradcam_map = None

    if use_gradcam:
        if hasattr(model, "target_layer"):
            cam = GradCAM(model, model.target_layer)
            heatmap = cam(img_tensor)  # (H, W) float 0–1
            gradcam_map = heatmap
        else:
            print("[WARN] Grad-CAM requested but model.target_layer not found.")

    # ---- Return response ----
    return {
        "predicted_class": predicted_class,
        "predicted_idx": predicted_idx,
        "probabilities": probabilities.tolist(),
        "gradcam": gradcam_map
    }
