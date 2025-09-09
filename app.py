import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Page configuration
st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# Path to the full saved model (must be the same as Ngrok)
MODEL_PATH = 'lung_classification_model_efficientnetb0.h5'

@st.cache_resource
def load_model():
    """Load the full saved RGB model directly"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
    try:
        # Load the full saved model (do NOT rebuild architecture)
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

# Class names
class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']

def preprocess_image(img):
    """Convert to RGB and preprocess like Ngrok"""
    img = img.convert('RGB')  # ensure 3 channels
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

    # Prediction bar chart
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
