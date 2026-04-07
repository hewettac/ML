import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "artifacts/pneumonia_model.keras"
METADATA_PATH = "artifacts/model_metadata.json"


@st.cache_resource
def load_trained_model(model_path: str):
    return load_model(model_path)


@st.cache_data
def load_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_uploaded_image(uploaded_file, image_size: int):
    image = Image.open(uploaded_file).convert("L")
    resized_image = image.resize((image_size, image_size))

    image_np = np.array(resized_image, dtype=np.float32) / 255.0
    reshaped = image_np.reshape(1, image_size, image_size, 1)

    return image, reshaped


def predict_image(model, processed_image, class_names):
    prediction = model.predict(processed_image, verbose=0)
    score = float(prediction[0][0])

    predicted_label = class_names[1] if score > 0.5 else class_names[0]
    normal_prob = 1.0 - score
    pneumonia_prob = score

    return predicted_label, normal_prob, pneumonia_prob


def check_required_files():
    missing = []
    if not Path(MODEL_PATH).exists():
        missing.append(MODEL_PATH)
    if not Path(METADATA_PATH).exists():
        missing.append(METADATA_PATH)
    return missing


st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩻",
    layout="wide"
)

st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload one or more chest X-ray images from the sidebar.")

missing_files = check_required_files()
if missing_files:
    st.error("Missing required files:")
    for file in missing_files:
        st.code(file)
    st.stop()

try:
    model = load_trained_model(MODEL_PATH)
    metadata = load_metadata(METADATA_PATH)
except Exception as e:
    st.error(f"Error loading model or metadata: {e}")
    st.stop()

image_size = metadata.get("img_size", [224, 224])[0]
class_names = metadata.get("class_names", ["NORMAL", "PNEUMONIA"])

st.sidebar.title("Upload X-Ray Images")
uploaded_files = st.sidebar.file_uploader(
    "Choose image files",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload one or more images from the sidebar to begin.")
    st.stop()

for uploaded_file in uploaded_files:
    try:
        original_image, processed_image = prepare_uploaded_image(uploaded_file, image_size)
        predicted_label, normal_prob, pneumonia_prob = predict_image(
            model, processed_image, class_names
        )

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(original_image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            st.subheader(uploaded_file.name)
            st.write(f"**Prediction:** {predicted_label}")
            st.metric("Normal Probability", f"{normal_prob:.2%}")
            st.metric("Pneumonia Probability", f"{pneumonia_prob:.2%}")

        st.markdown("---")

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
