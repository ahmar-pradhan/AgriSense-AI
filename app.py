import numpy as np
import sklearn
import torch.nn.functional as F
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=1uKCEY_BVZHeG5lXxwUcnAnXJ5MDkJoEO"
MODEL_PATH = "disease_model.pth"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

YIELD_MODEL_URL = "https://drive.google.com/uc?id=130xiQlzRUI7j7-reXI2F5JWa-wMXet0f"
YIELD_MODEL_PATH = "yield_model.pkl"

if not os.path.exists(YIELD_MODEL_PATH):
    gdown.download(YIELD_MODEL_URL, YIELD_MODEL_PATH, quiet=False)
# ------------------ CONFIG ------------------

CONFIDENCE_THRESHOLD = 75.0  # percent

# ------------------ LOAD MODELS ------------------

@st.cache_resource
def load_disease_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 38)

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


@st.cache_resource
def load_yield_model():
    with open(YIELD_MODEL_PATH, "rb") as f:
        return joblib.load(YIELD_MODEL_PATH)


@st.cache_resource
def load_class_names():
    class_file = Path("models/class_names.json")
    if class_file.exists():
        with open(class_file, "r") as f:
            names = json.load(f)
        if not isinstance(names, list) or len(names) != 38:
            raise ValueError("models/class_names.json must be a list of 38 class names.")
        return names

    return [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry___Powdery_mildew",
        "Cherry___healthy",
        "Corn___Cercospora_leaf_spot",
        "Corn___Common_rust",
        "Corn___Northern_Leaf_Blight",
        "Corn___healthy",
        "Grape___Black_rot",
        "Grape___Esca",
        "Grape___Leaf_blight",
        "Grape___healthy",
        "Orange___Haunglongbing",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper___Bacterial_spot",
        "Pepper___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]


def get_crop(label: str) -> str:
    return label.split("___")[0] if "___" in label else label


def generate_gradcam(model, img_tensor, class_idx):
    activations = None
    gradients = None

    target_layer = model.layer4[-1].conv2

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    out = model(img_tensor)
    score = out[0, class_idx]
    score.backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy()

    fh.remove()
    bh.remove()
    return cam_np


# ------------------ IMAGE TRANSFORM ------------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ------------------ UI ------------------

st.title("AgriSense AI")
st.markdown("AI-powered crop disease detection and yield prediction")

tab1, tab2 = st.tabs(["Disease Detection", "Yield Prediction"])

# ================== DISEASE DETECTION ==================
with tab1:
    st.header("Crop Disease Detection")

    supported_crops = [
        "Apple",
        "Blueberry",
        "Cherry",
        "Corn",
        "Grape",
        "Orange",
        "Peach",
        "Pepper",
        "Potato",
        "Raspberry",
        "Soybean",
        "Squash",
        "Strawberry",
        "Tomato",
    ]

    selected_crop = st.selectbox("Select your crop:", ["Select..."] + supported_crops)
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file and selected_crop != "Select...":
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", width=300)

        if st.button("Detect Disease"):
            model = load_disease_model()
            class_names = load_class_names()

            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            global_conf, global_idx = torch.max(probs, dim=0)
            global_label = class_names[global_idx.item()]
            global_crop = get_crop(global_label)

            crop_indices = [
                i for i, name in enumerate(class_names)
                if name.startswith(f"{selected_crop}___")
            ]

            if not crop_indices:
                st.error(f"No disease classes found for selected crop: {selected_crop}")
                st.stop()

            crop_probs = probs[crop_indices]
            denom = crop_probs.sum().item()
            if denom <= 0:
                st.error("Model returned invalid probabilities for selected crop.")
                st.stop()

            crop_probs = crop_probs / crop_probs.sum()

            best_idx_in_crop = torch.argmax(crop_probs).item()
            predicted_idx = crop_indices[best_idx_in_crop]
            disease = class_names[predicted_idx]
            confidence_pct = crop_probs[best_idx_in_crop].item() * 100.0

            # Generate heatmap after predicted_idx is available
            cam_map = generate_gradcam(model, img_tensor, predicted_idx)

            if global_crop != selected_crop and global_conf.item() >= 0.65:
                st.warning(
                    f"Selected crop is {selected_crop}, but the model globally thinks this image "
                    f"looks like {global_crop} ({global_conf.item() * 100:.1f}%). "
                    "Please verify crop selection or upload a clearer leaf image."
                )

            if "healthy" in disease.lower():
                st.success(f"Your {selected_crop} leaf looks healthy.")
                st.balloons()
            else:
                st.error(f"Disease Detected: {disease}")
                st.warning("Please consult an agricultural expert.")

            st.info(f"Confidence (within {selected_crop} classes): {confidence_pct:.2f}%")

            if confidence_pct < CONFIDENCE_THRESHOLD:
                st.warning("Low confidence prediction. Please verify manually with an expert.")

            st.subheader("Model Insights (Top Predictions for Selected Crop)")

            topk = min(5, len(crop_indices))
            top_probs, top_pos = torch.topk(crop_probs, topk)

            top_labels = [class_names[crop_indices[i]] for i in top_pos.tolist()]
            top_values = top_probs.tolist()

            fig, ax = plt.subplots()
            ax.barh(top_labels[::-1], top_values[::-1])
            ax.set_xlabel("Confidence")
            ax.set_title(f"Top {topk} Predictions ({selected_crop})")
            st.pyplot(fig)

            st.subheader("Model Focus Heatmap")
            fig_hm, ax_hm = plt.subplots()
            base_img = image.resize((224, 224))
            ax_hm.imshow(base_img)
            ax_hm.imshow(cam_map, cmap="jet", alpha=0.4)
            ax_hm.set_title("Grad-CAM (red = high attention)")
            ax_hm.axis("off")
            st.pyplot(fig_hm)

# ================== YIELD PREDICTION ==================
with tab2:
    st.header("Crop Yield Prediction")

    crop_options = [
        "Cassava",
        "Maize",
        "Plantains",
        "Potatoes",
        "Rice",
        "Sorghum",
        "Soybeans",
        "Sweet potatoes",
        "Wheat",
        "Yams",
    ]

    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("Select Crop:", crop_options)
        year = st.number_input("Year:", 1990, 2030, 2024)
        rainfall = st.number_input("Rainfall (mm):", value=1000)

    with col2:
        pesticides = st.number_input("Pesticides (tonnes):", value=10000)
        temperature = st.number_input("Temperature (°C):", value=25)

    if st.button("Predict Yield"):
        yield_model = load_yield_model()

        input_data = pd.DataFrame(
            [[0, crop_options.index(crop), year, rainfall, pesticides, temperature]],
            columns=[
                "Area",
                "Item",
                "Year",
                "average_rain_fall_mm_per_year",
                "pesticides_tonnes",
                "avg_temp",
            ],
        )

        prediction = yield_model.predict(input_data)[0]
        yield_kg = prediction / 10

        st.success(f"Predicted Yield: {yield_kg:,.0f} kg/hectare")
        st.info(f"{yield_kg / 1000:.2f} tonnes/hectare")

        comparison = {"Your Crop": yield_kg, "World Avg": 28774, "High Yield": 50000}
        fig2, ax2 = plt.subplots()
        ax2.bar(comparison.keys(), comparison.values())
        ax2.set_title("Yield Comparison")
        st.pyplot(fig2)

        rainfall_range = list(range(200, 3000, 100))
        yield_range = []

        for r in rainfall_range:
            temp_data = pd.DataFrame(
                [[0, crop_options.index(crop), year, r, pesticides, temperature]],
                columns=input_data.columns,
            )
            pred = yield_model.predict(temp_data)[0] / 10
            yield_range.append(pred)

        fig3, ax3 = plt.subplots()
        ax3.plot(rainfall_range, yield_range)
        ax3.axvline(x=rainfall, linestyle="--")
        ax3.set_title("Yield vs Rainfall")
        st.pyplot(fig3)
