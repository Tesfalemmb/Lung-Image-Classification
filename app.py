import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Lung Image Classification", layout="centered")

# ---- Config ----
MODEL_PATH = "lung_model.keras"
CLASS_NAMES = ["healthy", "neoplastic", "inflammation", "undetermined"]  # adjust if needed

# ---- Helpers ----
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Make sure 'lung_model.keras' is committed to the repo."
        )
    # Using compile=False speeds loading and avoids missing custom objects
    model = tf.keras.models.load_model(path, compile=False)
    return model

def infer_input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):  # handle multi-input models
        shape = shape[0]
    # Expect (None, H, W, C)
    H = shape[1] if len(shape) > 1 and shape[1] else 224
    W = shape[2] if len(shape) > 2 and shape[2] else 224
    C = shape[3] if len(shape) > 3 and shape[3] else 3
    return int(H), int(W), int(C)

def preprocess_image(img: Image.Image, target_hw_c):
    H, W, C = target_hw_c
    img = img.convert("RGB")
    img = img.resize((W, H))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(model, batch):
    preds = model.predict(batch, verbose=0)
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = tf.nn.softmax(preds[0]).numpy()
        top_idx = int(np.argmax(probs))
        return probs, top_idx
    # Fallback: binary or other shapes
    if preds.ndim == 1:
        prob = float(preds[0])
        return np.array([1 - prob, prob]), int(round(prob))
    raise ValueError(f"Unexpected model output shape: {preds.shape}")

# ---- UI ----
st.title("🫁 Lung Image Classification")
st.caption("Upload a lung image to get the predicted class and probabilities.")

# Load model
try:
    with st.spinner("Loading model…"):
        model = load_model(MODEL_PATH)
        H, W, C = infer_input_size(model)
except Exception as e:
    st.error(
        "Failed to load the model. "
        "If you previously used a `.h5` file, convert it to `.keras` and commit it to `lung_model.keras`."
    )
    st.exception(e)
    st.stop()

st.info(f"Model input expected: **{H}×{W}×{C}** (H×W×C)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Uploaded image", use_container_width=True)

        batch = preprocess_image(image, (H, W, C))
        probs, top_idx = predict(model, batch)

        if len(CLASS_NAMES) == len(probs):
            top_label = CLASS_NAMES[top_idx]
            st.success(f"**Prediction:** {top_label}")
            # Display probabilities as a simple table
            st.subheader("Class probabilities")
            for label, p in zip(CLASS_NAMES, probs):
                st.write(f"- {label}: {p:.4f}")
        else:
            st.success(f"**Predicted class index:** {top_idx}")
            st.write("Raw probabilities:", probs)

    except Exception as e:
        st.error("Error while processing or predicting. See details below.")
        st.exception(e)
else:
    st.caption("Tip: If your model expects different classes, edit `CLASS_NAMES` in `app.py`.")
