import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0  # register EfficientNet

# --- Page configuration ---
st.set_page_config(page_title="Lung Image Classification", layout="centered")
st.title("🫁 Lung Image Classification")

# --- Model path ---
MODEL_PATH = "lung_model.keras"

# --- Load model with caching ---
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

# --- Define class names ---
CLASS_NAMES = ["healthy", "neoplastic", "inflammation", "undetermined"]

# --- File uploader ---
uploaded = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])
if uploaded:
    # Open image and ensure 3 channels
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # --- Preprocess image ---
    # Resize to match model input size
    input_size = model.input_shape[1:3]  # automatically get HxW
    img_resized = img.resize((input_size[1], input_size[0]))  # PIL uses (W,H)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

    # --- Predict ---
    preds = model.predict(img_array, verbose=0)
    
    # Apply softmax if output is not already probabilities
    if preds.shape[-1] == len(CLASS_NAMES):
        probs = tf.nn.softmax(preds[0]).numpy()
        top_idx = int(np.argmax(probs))
        st.success(f"Prediction: {CLASS_NAMES[top_idx]}")
        st.subheader("Class probabilities")
        for i, c in enumerate(CLASS_NAMES):
            st.write(f"{c}: {probs[i]:.4f}")
    else:
        # fallback for binary or unusual outputs
        st.warning("Unexpected model output shape.")
        st.write("Raw output:", preds)
