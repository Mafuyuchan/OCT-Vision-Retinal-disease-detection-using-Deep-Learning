# src/gradcam.py
import torch
import numpy as np
import cv2

class GradCAM:
    """
    Minimal Grad-CAM implementation.
    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam(image_tensor, class_idx=None)  # heatmap numpy HxW float32 0..1
    Notes:
        - image_tensor shape: (1, C, H, W)
        - target_layer: the nn.Module to hook (e.g. model.layer4[-1] for ResNet50)
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out shape: (N, C, H, W)
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] shape: (N, C, H, W)
            self.gradients = grad_out[0].detach()
        # remove existing hooks if any (not tracked here) — user should create new GradCAM instance per run
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Returns heatmap numpy array scaled 0..1 (H, W)
        """
        # forward
        logits = self.model(input_tensor)  # (1, num_classes)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # backward
        self.model.zero_grad()
        loss = logits[0, class_idx]
        loss.backward(retain_graph=True)

        # gradients & activations
        grads = self.gradients  # (1, C, H, W)
        acts = self.activations  # (1, C, H, W)

        if grads is None or acts is None:
            raise RuntimeError("Gradients or activations were not captured. Ensure target_layer is correct.")

        # global-average-pool grads
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted_acts = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        heatmap = weighted_acts.squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap
