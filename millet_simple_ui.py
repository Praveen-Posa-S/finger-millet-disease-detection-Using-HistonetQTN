import argparse
import io
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from finger_millet_mffhistonet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MFFHistoNet,
    DenseNet121Base,
    QTN,
)


CLASS_NAMES: List[str] = ["downy", "healthy", "mottle", "seedling", "smut", "wilt"]
IMAGE_SIZE = 224


def build_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = DenseNet121Base(num_classes=len(CLASS_NAMES))
    qtn_model = QTN(input_dim=3 * IMAGE_SIZE * IMAGE_SIZE, hidden_dim=512, output_dim=128)
    model = MFFHistoNet(cnn_model, qtn_model, num_classes=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=True)
def get_model(weights_path: str):
    return build_model(weights_path)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def format_probabilities(probs: np.ndarray):
    sorted_idx = np.argsort(probs)[::-1]
    rows = []
    for idx in sorted_idx:
        rows.append(
            {
                "Class": CLASS_NAMES[idx],
                "Probability": f"{probs[idx] * 100:.2f}%",
            }
        )
    return rows


def main():
    st.set_page_config(page_title="Finger Millet Disease Demo", layout="centered")
    st.title("Finger Millet Disease Detector")
    st.caption("Upload an image to see model predictions.")

    default_weights = "mff_histonet_millet.pth"
    weights_path = st.text_input("Model weights path", value=default_weights)

    if not weights_path or not Path(weights_path).exists():
        st.warning("Provide a valid path to the trained model weights (.pth).")
        st.stop()

    model, device = get_model(weights_path)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded_file is None:
        st.info("Awaiting image upload...")
        st.stop()

    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Analyzing..."):
            tensor = preprocess_image(image).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
            top_idx = int(probabilities.argmax())
            st.success(f"Predicted class: **{CLASS_NAMES[top_idx]}**")
            st.subheader("Class probabilities")
            st.table(format_probabilities(probabilities))


if __name__ == "__main__":
    main()

