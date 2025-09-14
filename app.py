import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_classification_model_efficientnetb0.h5")
    return model

model = load_model()

# Define your classes
CLASS_NAMES = ["Healthy", "Neoplastic", "Inflammation", "Undetermined"]

# ---------------------------
# Grad-CAM Function
# ---------------------------
def get_gradcam(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Lung Image Classification", layout="wide")

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>🫁 Lung Disease Classification</h1>
    <p style='text-align: center;'>Upload a lung image to classify it and visualize important regions with <b>EfficientNet + Grad-CAM</b>.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📌 Navigation")
st.sidebar.info("Select an option below:")
st.sidebar.markdown("- Upload & Classify")
st.sidebar.markdown("- Model Info")
st.sidebar.markdown("- About")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload a lung image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess Image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))  # Match EfficientNetB0 input size
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Grad-CAM
    heatmap = get_gradcam(img_array, model)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    # ---------------------------
    # Display Results
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="🖼️ Uploaded Lung Image", use_container_width=True)

    with col2:
        st.image(superimposed_img, caption="🔥 Grad-CAM Heatmap", use_container_width=True)

    st.markdown("---")

    st.metric(label="Prediction", value=predicted_class, delta=f"{confidence:.2f}% confidence")

else:
    st.warning("👆 Upload a lung image to get started.")

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center;">Developed by <b>[Your Name]</b> | 
    <a href="https://github.com/your-repo" target="_blank">GitHub</a></p>
    """,
    unsafe_allow_html=True
)
