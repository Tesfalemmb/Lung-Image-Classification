import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🫁 Lung Image Classification")

# Path to model in repo root
MODEL_PATH = "lung_model.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

CLASS_NAMES = ["healthy", "neoplastic", "inflammation", "undetermined"]

uploaded = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)

    # Predict
    preds = model.predict(arr, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = int(np.argmax(probs))
    st.success(f"Prediction: {CLASS_NAMES[top_idx]}")
    st.write({CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)})
