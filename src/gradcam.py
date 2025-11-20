# src/gradcam.py
import torch
import numpy as np
import cv2


class GradCAM:
    """
    Minimal + stable Grad-CAM implementation.
    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam(image_tensor, class_idx=None)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Ensure model in eval mode (important for BatchNorm)
        self.model.eval()

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out: (N, C, H, W)
            self.activations = out.detach().cpu()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0]: (N, C, H, W)
            self.gradients = grad_out[0].detach().cpu()

        # Use full backward hook (PyTorch >= 1.12)
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None, upsample_size=None):
        """
        Returns heatmap numpy array scaled 0..1 (H, W)
        """
        # Forward pass
        logits = self.model(input_tensor)

        # Which class to visualize
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass
        self.model.zero_grad()
        loss = logits[0, class_idx]
        loss.backward(retain_graph=True)

        # Verify hooks captured data
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. Check target_layer or model hooks."
            )

        grads = self.gradients  # (1, C, H, W)
        acts = self.activations  # (1, C, H, W)

        # Global average pooling of gradients → weights
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1).squeeze().numpy()  # (H, W)

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize heatmap
        cam = cam / cam.max() if cam.max() > 0 else np.zeros_like(cam)

        # Optional upsample
        if upsample_size is not None:
            cam = cv2.resize(cam, upsample_size, interpolation=cv2.INTER_LINEAR)

        return cam

    @staticmethod
    def apply_colormap(heatmap: np.ndarray, image: np.ndarray):
        """
        Apply Jet colormap and overlay on the image.
        image: (H, W, 3) uint8 original image
        heatmap: (H, W) float32 0..1
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        return overlay

