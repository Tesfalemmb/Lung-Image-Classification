import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0  # register EfficientNet

st.set_page_config(page_title="Lung Image Classification", layout="centered")
st.title("🫁 Lung Image Classification")

# Path to your model in the repo
MODEL_PATH = "lung_model.keras"

# Load model with caching
@st.cache_resource(show_spinner=True)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("Failed to load model.")
    st.exception(e)
    st.stop()

CLASS_NAMES = ["healthy", "neoplastic", "inflammation", "undetermined"]

# Upload image
uploaded = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))  # adjust if your model expects different input size
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    preds = model.predict(arr, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = int(np.argmax(probs))
    st.success(f"Prediction: {CLASS_NAMES[top_idx]}")
    st.subheader("Class probabilities")
    for i, c in enumerate(CLASS_NAMES):
        st.write(f"{c}: {probs[i]:.4f}")
