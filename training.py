from pathlib import Path
import textwrap, json, os, zipfile

repo_dir = Path("/mnt/data/pneumonia-github-ready")
repo_dir.mkdir(parents=True, exist_ok=True)

train_py = r'''"""
Train a CNN to classify chest X-rays as NORMAL or PNEUMONIA.

Usage:
    python src/train.py --data-dir data/chest_xray --epochs 10 --batch-size 32

Expected dataset structure:
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomContrast,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.models import load_model


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to chest_xray dataset root")
    parser.add_argument("--image-size", type=int, default=224, help="Square resize dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split if val folder not used")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Where model and outputs are saved")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to disk")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split_dir: Path, image_size: int):
    data = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        class_num = CLASS_NAMES.index(class_name)
        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue
            try:
                image_array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image_array is None:
                    continue
                resized = cv2.resize(image_array, (image_size, image_size))
                data.append((resized, class_num))
            except Exception:
                continue

    if not data:
        raise ValueError(f"No usable images found in {split_dir}")

    np.random.shuffle(data)
    x = np.array([item[0] for item in data], dtype=np.float32).reshape(-1, image_size, image_size, 1) / 255.0
    y = np.array([item[1] for item in data], dtype=np.int32)
    return x, y


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        RandomRotation(0.05),
        RandomZoom(0.10),
        RandomTranslation(0.05, 0.05),
        RandomContrast(0.10),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def save_class_distribution(y, output_dir: Path):
    classes, counts = np.unique(y, return_counts=True)
    plt.figure()
    bars = plt.bar(classes.astype(str), counts)
    plt.title("Class Distribution")
    plt.xlabel("Class (0 = NORMAL, 1 = PNEUMONIA)")
    plt.ylabel("Number of Images")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()


def save_training_curves(history, output_dir: Path):
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(train_dir, args.image_size)

    use_explicit_val = val_dir.exists() and any(val_dir.iterdir())
    if use_explicit_val:
        x_val, y_val = load_split(val_dir, args.image_size)
        validation_data = (x_val, y_val)
        validation_split = 0.0
    else:
        validation_data = None
        validation_split = args.validation_split

    x_test, y_test = load_split(test_dir, args.image_size)

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train), class_weights_arr)
    }

    model = build_model(x_train.shape[1:])

    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=validation_data,
        validation_split=validation_split,
        class_weight=class_weight,
        verbose=1,
    )

    eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    model_path = output_dir / "pneumonia_model.keras"
    model.save(model_path)

    metadata = {
        "img_size": [args.image_size, args.image_size],
        "class_names": CLASS_NAMES,
        "test_loss": float(eval_loss),
        "test_accuracy": float(eval_acc),
        "class_weight": class_weight,
    }

    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.save(output_dir / "confusion_matrix.npy", cm)

    if args.save_plots:
        save_class_distribution(y_train, output_dir)
        save_training_curves(history, output_dir)

    print(f"Saved model to: {model_path}")
    print(f"Test loss: {eval_loss:.4f}")
    print(f"Test accuracy: {eval_acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
'''

predict_py = r'''"""
Run prediction on a single chest X-ray image.

Usage:
    python src/predict.py --image path/to/image.jpeg --model artifacts/pneumonia_model.keras
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="artifacts/pneumonia_model.keras", help="Path to trained model")
    parser.add_argument("--metadata", type=str, default="artifacts/model_metadata.json", help="Path to metadata JSON")
    return parser.parse_args()


def prepare_image(filepath: str, image_size: int):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        raise ValueError(f"Could not read image: {filepath}")

    img_resized = cv2.resize(img_array, (image_size, image_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_reshaped = img_normalized.reshape(-1, image_size, image_size, 1)
    return img_reshaped


def main():
    args = parse_args()

    model = load_model(args.model)

    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_size = metadata["img_size"][0]
    class_names = metadata["class_names"]

    image_batch = prepare_image(args.image, image_size)
    prediction = model.predict(image_batch, verbose=0)
    score = float(prediction[0][0])

    label = class_names[1] if score > 0.5 else class_names[0]
    normal_prob = 1.0 - score
    pneumonia_prob = score

    print(f"Image: {Path(args.image).name}")
    print(f"Prediction: {label}")
    print(f"Normal probability: {normal_prob:.6f}")
    print(f"Pneumonia probability: {pneumonia_prob:.6f}")


if __name__ == "__main__":
    main()
'''

download_script = r'''"""
Download the Kaggle chest X-ray pneumonia dataset.

Usage:
    python src/download_data.py --kaggle-json /path/to/kaggle.json --output-dir data
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-json", type=str, required=True, help="Path to kaggle.json")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory where data will be downloaded")
    return parser.parse_args()


def main():
    args = parse_args()

    kaggle_json = Path(args.kaggle_json)
    if not kaggle_json.exists():
        raise FileNotFoundError(f"kaggle.json not found: {kaggle_json}")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    dest = kaggle_dir / "kaggle.json"
    shutil.copy2(kaggle_json, dest)
    dest.chmod(0o600)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "paultimothymooney/chest-xray-pneumonia",
            "-p",
            str(output_dir),
            "--unzip",
        ],
        check=True,
    )

    print(f"Dataset downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
'''

requirements_txt = """tensorflow>=2.14
numpy>=1.24
matplotlib>=3.7
opencv-python>=4.8
scikit-learn>=1.3
kaggle>=1.6
"""

readme_md = """# Chest X-Ray Pneumonia Classifier

This repo trains a CNN that classifies chest X-rays as **NORMAL** or **PNEUMONIA**.

## 1. Clone the repo

```bash
git clone <your-repo-url>
cd pneumonia-github-ready
