import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# Path to your model file
MODEL_PATH = 'lung_classification_model_efficientnetb0.h5'

@st.cache_resource
def load_model():
    """Load the model safely: full model or weights-only"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None

    # Check file size (optional)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    if size_mb < 1:
        st.error(f"❌ Model file seems too small ({size_mb:.2f} MB). Check file integrity.")
        return None

    try:
        # Try loading as a full saved model first
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success(f"✅ Loaded full model successfully ({size_mb:.2f} MB).")
        return model
    except Exception as e_full:
        st.warning("⚠️ Could not load as full model, trying weights-only...")
        try:
            # Rebuild architecture and load weights (weights-only scenario)
            from tensorflow.keras.applications import EfficientNetB0
            from tensorflow.keras import layers, models

            base_model = EfficientNetB0(include_top=False, input_shape=(224,224,3), weights=None)
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            predictions = layers.Dense(4, activation='softmax')(x)
            model = models.Model(inputs=base_model.input, outputs=predictions)
            model.load_weights(MODEL_PATH)
            st.success("✅ Loaded weights-only model successfully.")
            return model
        except Exception as e_weights:
            st.error(f"❌ Failed to load model:\nFull model error: {str(e_full)}\nWeights-only error: {str(e_weights)}")
            return None

# Load model
model = load_model()

# Class names
class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']

def preprocess_image(img):
    img = img.convert('RGB')  # Ensure 3 channels
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it.")

    uploaded_file = st.file_uploader("Choose a lung image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("👆 Please upload a lung image to get started.")
        return

    img = Image.open(uploaded_file)

    if model is None:
        st.error("Model failed to load. Please check the model file.")
        return

    # Preprocess and predict
    img_array = preprocess_image(img)
    preds = model.predict(img_array, verbose=0)[0]
    pred_class_index = np.argmax(preds)
    pred_class = class_names[pred_class_index]
    confidence = np.max(preds) * 100

    # Display original image
    st.image(img, caption="Original Image", use_column_width=True)

    # Prediction chart
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["green", "red", "blue", "orange"]
    ax.barh(class_names, preds * 100, color=colors)
    ax.set_xlim([0, 100])
    ax.set_xlabel("Probability (%)")
    ax.set_title("Prediction Confidence")
    for i, v in enumerate(preds * 100):
        ax.text(v + 1, i, f"{v:.2f}%", va="center")
    st.pyplot(fig)

    # Final prediction
    prediction_color = (
        "green" if pred_class == "Healthy" else
        "red" if pred_class == "Inflammation" else
        "blue" if pred_class == "Neoplastic" else
        "orange"
    )
    st.markdown(f"<h2 style='color: {prediction_color}'>✅ Final Prediction: {pred_class}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
