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
    # Load .keras model trained on RGB images
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Ensure input is RGB (3 channels)
    if model.input_shape[-1] != 3:
        from tensorflow.keras import layers, Model
        inputs = layers.Input(shape=(model.input_shape[1], model.input_shape[2], 3))
        outputs = model(inputs)
        model = Model(inputs, outputs)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("Failed to load model.")
    st.exception(e)
    st.stop()

# --- Class names ---
CLASS_NAMES = ["healthy", "neoplastic", "inflammation", "undetermined"]

# --- File uploader ---
uploaded = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])
if uploaded:
    # Open image and ensure RGB
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # --- Preprocess image ---
    # Resize to model input size automatically
    input_size = model.input_shape[1:3]  # (H, W)
    img_resized = img.resize((input_size[1], input_size[0]))  # PIL expects (W,H)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

    # --- Predict ---
    preds = model.predict(img_array, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = int(np.argmax(probs))
    
    # Display results
    st.success(f"Prediction: {CLASS_NAMES[top_idx]}")
    st.subheader("Class probabilities")
    for i, c in enumerate(CLASS_NAMES):
        st.write(f"{c}: {probs[i]:.4f}")
