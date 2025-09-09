import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "lung_classification_model_efficientnetb0.h5"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model not found at {MODEL_PATH}")
            return None
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

# -------------------------------
# Classes
# -------------------------------
class_names = ["Healthy", "Inflammation", "Neoplastic", "Undetermined"]

# -------------------------------
# Preprocess
# -------------------------------
def preprocess_image(img):
    try:
        img = img.convert("RGB")  # ensure 3 channels
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# -------------------------------
# Grad-CAM
# -------------------------------
def get_gradcam(img_array, model, class_index, layer_name=None):
    if layer_name is None:
        # Pick the last conv layer automatically
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_array = np.array(img.convert("RGB"))
    overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    return overlayed

# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("🫁 Lung Image Classification with Grad-CAM")
    st.write("Upload a lung image to classify it and visualize **where the AI is looking**.")

    uploaded_file = st.file_uploader(
        "Choose a lung image",
        type=["jpg", "jpeg", "png"],
        help="Upload a lung CT/X-ray image"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)

        if model is None:
            st.error("❌ Model not loaded.")
            return

        with st.spinner("🔄 Processing..."):
            img_array = preprocess_image(img)
            if img_array is None:
                return
            
            preds = model.predict(img_array, verbose=0)[0]
            pred_class_index = np.argmax(preds)
            pred_class = class_names[pred_class_index]
            confidence = np.max(preds) * 100

            with col2:
                st.subheader("📊 Prediction Results")
                for cname, prob in zip(class_names, preds):
                    color = "green" if cname == "Healthy" else "red" if cname == "Inflammation" else "blue" if cname == "Neoplastic" else "orange"
                    st.markdown(
                        f"**<span style='color: {color}'>{cname}:</span> {prob*100:.2f}%**",
                        unsafe_allow_html=True
                    )
                    st.progress(float(prob))
                
                st.markdown(
                    f"<h3 style='color: red'>✅ Final Prediction: {pred_class} ({confidence:.2f}%)</h3>",
                    unsafe_allow_html=True
                )

            # ---------------- Grad-CAM ----------------
            st.subheader("🔥 Grad-CAM Visualization")
            heatmap = get_gradcam(img_array, model, pred_class_index)
            overlayed = overlay_gradcam(img, heatmap)

            col3, col4 = st.columns(2)
            with col3:
                st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True, clamp=True)
            with col4:
                st.image(overlayed, caption="Overlayed Heatmap", use_column_width=True)

            st.info("**Color Meaning:** 🔴/🟡 = Strong evidence, 🟢/🔵 = Less important, ⚫ = Ignored")

    else:
        st.info("👆 Upload a lung image to get started.")

    st.markdown("---")
    st.caption("🔬 Educational & research use only. Not for medical diagnosis.")

if __name__ == "__main__":
    main()
