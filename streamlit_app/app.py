import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# LOTTIE ANIMATIONS
# ------------------------------------------------------------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_eye = load_lottie("https://assets9.lottiefiles.com/private_files/lf30_jmgekfqg.json")
lottie_loading = load_lottie("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="OCT Retinal Disease Classifier",
    layout="wide",
    page_icon="🩺",
)

st.markdown("<h1 style='text-align:center;'>🩺 OCT Retinal Disease Diagnosis</h1>", unsafe_allow_html=True)
st.write("---")

# ------------------------------------------------------------------------------
# SIDEBAR: Model Settings
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")
model_path = st.sidebar.text_input("Model Path", "models/resnet50_best.pth")
selected_model = st.sidebar.selectbox("Choose Model:", ["VGG16", "ResNet50"])

class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path, selected_model, num_classes):
    if selected_model == "ResNet50":
        from src.models import create_resnet50
        model = create_resnet50(num_classes)
    else:
        from src.models import create_vgg16
        model = create_vgg16(num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if st.sidebar.button("Load Model"):
    try:
        model = load_model(model_path, selected_model, len(class_names))
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Model Load Failed:\n{str(e)}")

# ------------------------------------------------------------------------------
# TRANSFORM
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------------------------
# GRAD-CAM FUNCTION
# ------------------------------------------------------------------------------
def generate_gradcam(model, image_tensor, target_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)
    class_idx = output.argmax().item()
    output[:, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    handle_f.remove()
    handle_b.remove()

    return heatmap, class_idx

# ------------------------------------------------------------------------------
# TABS UI
# ------------------------------------------------------------------------------
tab_home, tab_predict, tab_gradcam, tab_about = st.tabs(
    ["🏠 Home", "🔮 Prediction", "🔥 Grad-CAM", "ℹ️ About"]
)

# ------------------------------------------------------------------------------
# HOME TAB
# ------------------------------------------------------------------------------
with tab_home:
    st.subheader("👁️ AI-Powered OCT Image Analysis")
    st.write("Upload OCT scans and let the model classify retinal diseases such as:")
    st.write("• CNV\n• DME\n• DRUSEN\n• NORMAL")

    if lottie_eye:
        st_lottie = st.components.v1.html(
            f"""
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <lottie-player src="https://assets9.lottiefiles.com/private_files/lf30_jmgekfqg.json"
                background="transparent" speed="1" style="width: 400px; height: 400px;" loop autoplay>
            </lottie-player>
            """,
            height=400
        )

# ------------------------------------------------------------------------------
# PREDICTION TAB
# ------------------------------------------------------------------------------
with tab_predict:
    st.header("🔮 Disease Prediction")

    uploaded_file = st.file_uploader("Upload an OCT Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded OCT Image", use_column_width=True)

        if st.button("Predict"):
            if "model" not in locals():
                st.error("Please load the model first!")
            else:
                with col2:
                    st.write("### Processing...")
                    if lottie_loading:
                        st_lottie = st.components.v1.html(
                            f"""
                            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
                            <lottie-player src="https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json"
                                background="transparent" speed="1" style="width: 180px; height: 180px;" loop autoplay>
                            </lottie-player>
                            """,
                            height=200
                        )

                img_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs, dim=1)
                    pred_idx = probs.argmax().item()
                    pred_class = class_names[pred_idx]

                st.success(f"### 🧾 Prediction: **{pred_class}**")

                # Chart
                st.write("### 📊 Class Probabilities")
                fig, ax = plt.subplots()
                ax.bar(class_names, probs.cpu().numpy()[0])
                st.pyplot(fig)

# ------------------------------------------------------------------------------
# GRAD-CAM TAB
# ------------------------------------------------------------------------------
with tab_gradcam:
    st.header("🔥 Grad-CAM Heatmap Visualization")

    cam_file = st.file_uploader("Upload Image for Grad-CAM", type=["jpg", "png", "jpeg"])

    if cam_file:
        image = Image.open(cam_file)
        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Generate Grad-CAM"):
            if "model" not in locals():
                st.error("Please load the model first!")
            else:
                img_tensor = transform(image).unsqueeze(0).to(device)

                # Target layer for ResNet50 & VGG16
                if selected_model == "ResNet50":
                    target_layer = model.layer4[-1]
                else:
                    target_layer = model.features[-1]

                heatmap, class_idx = generate_gradcam(model, img_tensor, target_layer)

                # Overlay heatmap
                heatmap = np.uint8(255 * heatmap)
                heatmap_img = Image.fromarray(heatmap).resize(image.size)
                heatmap_img = np.array(heatmap_img)

                fig, ax = plt.subplots()
                ax.imshow(image)
                ax.imshow(heatmap_img, cmap="jet", alpha=0.4)
                ax.axis("off")
                st.pyplot(fig)

                st.success(f"Predicted Class: {class_names[class_idx]}")

# ------------------------------------------------------------------------------
# ABOUT TAB
# ------------------------------------------------------------------------------
with tab_about:
    st.header("About the Project")
    st.write("""
    This project classifies retinal diseases using OCT images with deep learning.

    **Models Supported:**  
    • VGG16  
    • ResNet50  

    **Features:**  
    ✔ Real-time prediction  
    ✔ Grad-CAM visualization  
    ✔ Probability charts  
    ✔ Clean UI with animations  

    **Developer:** Mafuyu
    """)


