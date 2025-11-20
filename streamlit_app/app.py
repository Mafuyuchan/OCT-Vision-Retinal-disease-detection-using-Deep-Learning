import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# ---------------------------
# Load model utility
# ---------------------------
def load_model(model_path, model_class, num_classes, device):
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ---------------------------
# Image preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="OCT Retinal Disease Classifier", layout="centered")

st.title("🩺 OCT Retinal Disease Classification")
st.write("Upload an OCT scan and the model will predict the retinal disease class.")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model Path", "models/resnet50_best.pth")

class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
selected_model = st.sidebar.selectbox("Choose Model:", ["VGG16", "ResNet50"])

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load model dynamically
# ---------------------------
if st.sidebar.button("Load Model"):
    try:
        if selected_model == "ResNet50":
            from src.models import create_resnet50
            model = load_model(model_path, create_resnet50, len(class_names), device)

        elif selected_model == "VGG16":
            from src.models import create_vgg16
            model = load_model(model_path, create_vgg16, len(class_names), device)

        st.sidebar.success("Model loaded successfully!")

    except Exception as e:
        st.sidebar.error(f"Failed to load model:\n{str(e)}")

# ---------------------------
# Upload image
# ---------------------------
uploaded_file = st.file_uploader("Upload OCT Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        if 'model' not in locals():
            st.error("⚠️ Please load the model first from the sidebar.")
        else:
            tensor = preprocess_image(image).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = class_names[predicted.item()]

            st.success(f"### 🧾 Prediction: **{predicted_class}**")
