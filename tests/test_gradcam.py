import torch
from PIL import Image
import numpy as np
from src.models import create_resnet50
from streamlit_app.app import generate_gradcam

def test_gradcam_output():
    model = create_resnet50(num_classes=4)
    tensor = torch.randn(1, 3, 224, 224)

    target_layer = model.layer4[-1]

    heatmap, class_idx = generate_gradcam(model, tensor, target_layer)

    assert heatmap.shape[0] > 0
    assert isinstance(class_idx, int)
