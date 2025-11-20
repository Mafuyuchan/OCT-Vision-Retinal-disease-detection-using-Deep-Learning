import torch
from PIL import Image
import numpy as np
from src.predict import predict
from src.models import create_vgg16

def test_predict():
    model = create_vgg16(num_classes=4)
    dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

    pred = predict(model, dummy_img, "cpu", class_names)
    assert pred in class_names
