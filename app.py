import streamlit as st
import numpy as np
from PIL import Image
import os

# ===============================
# 🫁 Lung Classification Web App
# ===============================

# Set page configuration
st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("❌ TensorFlow not available. Please check requirements.txt.")

# Path to your trained model
MODEL_PATH = "lung_classification_model_efficientnetb0.h5"

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    if not TENSORFLOW_AVAILABLE:
        return None

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        return None

    try:
        # Build base EfficientNetB0 with correct input shape
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=(224, 224, 3),  # ✅ 3-channel input
            weights=None
        )

        # Add custom layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(4, activation="softmax")(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # Load weights
        model.load_weights(MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model once
model = load_model()

# Define class names
class_names = ["Healthy", "Inflammation", "Neoplastic", "Undetermined"]

def preprocess_image(img):
    """Preprocess image for model prediction (RGB, 224x224)"""
    try:
        img = img.convert("RGB")  # ✅ Ensure 3 channels
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # EfficientNet preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it.")

    uploaded_file = st.file_uploader(
        "Choose a lung image",
        type=["jpg", "jpeg", "png"],
        help="Upload a lung X-ray or CT scan image for analysis"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)

        if model is None:
            st.error("⚠️ Model failed to load. Please check the model file.")
        else:
            with st.spinner("🔄 Processing image..."):
                img_array = preprocess_image(img)
                if img_array is None:
                    return

                try:
                    preds = model.predict(img_array, verbose=0)[0]

                    with col2:
                        st.subheader("📊 Prediction Results")
                        for class_name, prob in zip(class_names, preds):
                            st.write(f"**{class_name}: {prob*100:.2f}%**")
                            st.progress(float(prob))

                        pred_class_index = np.argmax(preds)
                        pred_class = class_names[pred_class_index]
                        confidence = np.max(preds) * 100
                        st.success(f"✅ Final Prediction: {pred_class} ({confidence:.2f}%)")

                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")

    else:
        st.info("👆 Please upload a lung image to get started.")

    st.markdown("---")
    st.caption("🔬 For educational and research purposes only. Not for medical use.")

if __name__ == "__main__":
    main()
