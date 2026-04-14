#%%
import zipfile
from pathlib import Path

import cv2
import keras
import numpy as np
import streamlit as st
from PIL import Image

IMAGE_SIZE = 224
MODEL_CANDIDATES = [Path("CNN.keras"), Path("CNN.h5")]
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


@st.cache_resource
def load_cnn_model(model_path: str):
    return keras.models.load_model(model_path)


def find_model_file() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def describe_model_file(model_path: Path) -> dict:
    info = {
        "name": model_path.name,
        "size_bytes": model_path.stat().st_size,
        "is_zip_based_keras": False,
        "zip_contents": [],
    }

    if model_path.suffix == ".keras":
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                info["is_zip_based_keras"] = True
                info["zip_contents"] = zf.namelist()
        except Exception:
            info["is_zip_based_keras"] = False

    return info


def preprocess_uploaded_image(uploaded_file) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized.astype("float32") / 255.0
    model_input = normalized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return image_np, model_input


def predict_image(model, model_input: np.ndarray) -> dict:
    raw_pred = model.predict(model_input, verbose=0)
    pneumonia_prob = float(raw_pred[0][0])
    normal_prob = 1.0 - pneumonia_prob

    predicted_class = 1 if pneumonia_prob >= 0.5 else 0
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = pneumonia_prob if predicted_class == 1 else normal_prob

    return {
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "pneumonia_prob": pneumonia_prob,
        "normal_prob": normal_prob,
    }


st.set_page_config(page_title="Chest X-Ray Pneumonia Detector", layout="centered")

st.title("Chest X-Ray Pneumonia Detector")
st.write(
    "Upload a JPEG or PNG chest X-ray image to classify it as NORMAL or PNEUMONIA "
    "using your trained CNN model."
)

with st.expander("Important notes"):
    st.markdown(
        """
        - This app looks for `CNN.keras` first, then `CNN.h5`.
        - Put this app file and your model file in the same folder before running the app.
        - The model was trained on **grayscale 224x224** images, so uploads are converted automatically.
        - This is a demo/inference app and should **not** be used for medical diagnosis.
        """
    )

model_path = find_model_file()

if model_path is None:
    st.error(
        "No model file was found. Put `CNN.keras` or `CNN.h5` in the same folder as this app."
    )
    st.stop()
else:
    model_info = describe_model_file(model_path)

    with st.expander("Model file diagnostics"):
        st.write(f"**Detected model file:** `{model_info['name']}`")
        st.write(f"**Size:** {model_info['size_bytes']:,} bytes")
        if model_path.suffix == ".keras":
            st.write(f"**ZIP-based Keras file:** {model_info['is_zip_based_keras']}")
            if model_info["zip_contents"]:
                st.write("**Archive contents:**")
                st.code("\n".join(model_info["zip_contents"]))

    try:
        model = load_cnn_model(str(model_path))
        st.success(f"Loaded model successfully from `{model_path.name}`")
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.info(
            "Your model file appears valid, so this is likely an environment/version mismatch. "
            "Use the updated requirements file and reinstall dependencies."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    threshold = st.slider(
        "Prediction threshold for pneumonia",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="If pneumonia probability is greater than or equal to this threshold, the app predicts PNEUMONIA.",
    )

    if uploaded_file is not None:
        try:
            display_image, model_input = preprocess_uploaded_image(uploaded_file)
            results = predict_image(model, model_input)

            predicted_label = (
                "PNEUMONIA" if results["pneumonia_prob"] >= threshold else "NORMAL"
            )
            confidence = (
                results["pneumonia_prob"]
                if predicted_label == "PNEUMONIA"
                else results["normal_prob"]
            )

            st.subheader("Uploaded Image")
            st.image(display_image, caption="Uploaded chest X-ray", use_container_width=True)

            st.subheader("Prediction Result")
            if predicted_label == "PNEUMONIA":
                st.error(f"Prediction: {predicted_label}")
            else:
                st.success(f"Prediction: {predicted_label}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{confidence:.2%}")
            col2.metric("Pneumonia Probability", f"{results['pneumonia_prob']:.2%}")
            col3.metric("Normal Probability", f"{results['normal_prob']:.2%}")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
