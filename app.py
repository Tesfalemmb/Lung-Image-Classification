import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

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
    st.error("TensorFlow not available. Please check requirements.txt.")

# Path to your full saved model (same as Ngrok)
MODEL_PATH = 'lung_classification_model_efficientnetb0.h5'

@st.cache_resource
def load_model():
    """Load full saved model to match Ngrok predictions"""
    if not TENSORFLOW_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
    try:
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
    """Preprocess image the same way as Ngrok app"""
    img = img.convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def get_gradcam(img_array, model, class_index):
    """Generate Grad-CAM heatmap"""
    if model is None:
        return None

    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap
    except:
        return None

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify and visualize important regions.")

    uploaded_file = st.file_uploader(
        "Choose a lung image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        if model is None:
            st.error("Model failed to load. Please check the model file.")
            return

        img_array = preprocess_image(img)
        preds = model.predict(img_array, verbose=0)[0]
        pred_class_index = np.argmax(preds)
        pred_class = class_names[pred_class_index]
        confidence = np.max(preds) * 100

        # Grad-CAM
        heatmap = get_gradcam(img_array, model, pred_class_index)
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            img_np = np.array(img.convert('RGB'))
            superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        else:
            superimposed_img = np.array(img.convert('RGB'))

        # Display original and Grad-CAM
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_column_width=True)
        with col2:
            st.image(superimposed_img, caption=f"Grad-CAM Overlay for {pred_class}", use_column_width=True)

        # Display prediction chart and final result side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Prediction Confidence")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ["green", "red", "blue", "orange"]
            ax.barh(class_names, preds * 100, color=colors)
            ax.set_xlim([0, 100])
            ax.set_xlabel("Probability (%)", fontsize=10)
            ax.set_title("Prediction Confidence", fontsize=12)
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            for i, v in enumerate(preds * 100):
                ax.text(v + 1, i, f"{v:.2f}%", va="center", fontsize=10)
            st.pyplot(fig)

        with col2:
            prediction_color = (
                "green" if pred_class == "Healthy" else
                "red" if pred_class == "Inflammation" else
                "blue" if pred_class == "Neoplastic" else
                "orange"
            )
            st.markdown(f"<h2 style='color: {prediction_color}'>✅ Final Prediction: {pred_class}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

    else:
        st.info("👆 Please upload a lung image to get started.")

    st.markdown("---")
    st.caption("🔬 For educational and research purposes. Consult healthcare professionals for medical diagnoses.")

if __name__ == "__main__":
    main()
