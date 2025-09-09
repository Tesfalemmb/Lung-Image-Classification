import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0

st.set_page_config(page_title="Lung Image Classification", layout="centered")
st.title("🫁 Lung Image Classification")

# Path to your .h5 model in the repo
MODEL_PATH = "lung_classification_model_efficientnetb0.h5"

# Load model with caching
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Rebuild EfficientNetB0 with correct input shape (RGB, 224x224)
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        base_model = EfficientNetB0(weights=None, include_top=True, classes=4, input_tensor=inputs)

        # Load weights from the .h5 file
        base_model.load_weights(MODEL_PATH)

        return base_model
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.exception(e)
        st.stop()

model = load_model()
st.success("✅ Model loaded successfully!")

# Define class labels
CLASS_NAMES = ["healthy", "neoplastic", "inflammation", "undetermined"]

# Upload image
uploaded = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)  # (1, 224, 224, 3)

    # Predict
    preds = model.predict(arr, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = int(np.argmax(probs))

    # Show results
    st.success(f"Prediction: {CLASS_NAMES[top_idx]}")
    st.subheader("Class probabilities")
    for i, c in enumerate(CLASS_NAMES):
        st.write(f"{c}: {probs[i]:.4f}")
