import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# TensorFlow check
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not available. Please check requirements.txt.")

MODEL_PATH = 'lung_classification_model_efficientnetb0.h5'

@st.cache_resource
def load_model():
    if not TENSORFLOW_AVAILABLE:
        return None
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            return None
        try:
            # Try loading as full model
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception:
            # Fallback: load base architecture + weights
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                input_shape=(224, 224, 3),
                weights=None
            )
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            model.load_weights(MODEL_PATH)
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()
class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']
class_colors = ['green', 'red', 'blue', 'orange']

# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(img: Image.Image):
    img_resized = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# -------------------------
# Grad-CAM generation
# -------------------------
def get_gradcam(img_array, model, class_index):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and 'block7a' in layer.name:
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer.name
                break
    if last_conv_layer is None:
        return None
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

def heatmap_explanation():
    return [
        ("blue", "Low activation: minimal contribution to prediction."),
        ("green", "Moderate activation: moderate contribution."),
        ("yellow", "Strong activation: significant contribution."),
        ("red", "Very high activation: strongest influence on prediction."),
    ]

# -------------------------
# Main App
# -------------------------
def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it and visualize important regions.")

    uploaded_file = st.file_uploader("Choose a lung image", type=["jpg","jpeg","png"])
    if uploaded_file is None:
        st.info("👆 Upload a lung image to get started.")
        return

    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Cannot open image: {str(e)}")
        return

    if model is None:
        st.error("Model failed to load. Check the model file.")
        return

    img_array = preprocess_image(img)
    preds = model.predict(img_array, verbose=0)[0]
    pred_class_index = np.argmax(preds)
    pred_class = class_names[pred_class_index]
    confidence = np.max(preds) * 100
    prediction_color = class_colors[pred_class_index]

    # -------------------------
    # Top row: Uploaded Image + Prediction
    # -------------------------
    col_img, col_pred = st.columns([1.3, 1])
    with col_img:
        st.subheader("🖼️ Uploaded Image")
        st.image(np.array(img), caption="Uploaded Image", use_column_width=True)

    with col_pred:
        st.subheader("📊 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(5,4))
        ax.barh(class_names, preds*100, color=class_colors)
        ax.set_xlim([0,100])
        ax.set_xlabel("Probability (%)")
        for i, v in enumerate(preds*100):
            ax.text(v+1, i, f"{v:.2f}%", va='center', fontsize=12)
        st.pyplot(fig)
        st.markdown(f"<h2 style='color:{prediction_color}; font-size:32px'>✅ Final Prediction: {pred_class}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size:22px'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

    # -------------------------
    # Bottom row: Grad-CAM + Interpretation
    # -------------------------
    col_heatmap, col_interpret = st.columns([1.3, 1])
    with col_heatmap:
        st.subheader("🔥 Grad-CAM Overlay")
        heatmap = get_gradcam(img_array, model, pred_class_index)
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            img_np = np.array(img)
            superimposed = cv2.addWeighted(img_np, 0.6, heatmap_resized, 0.4, 0)
            st.image(np.array(superimposed), caption=f"Grad-CAM Overlay ({pred_class})", use_column_width=True)
        else:
            st.warning("Grad-CAM could not be generated")

    with col_interpret:
        st.subheader("📝 Heatmap Interpretation")
        st.markdown("<p style='font-size:20px'>Colors indicate influence on the model's decision:</p>", unsafe_allow_html=True)
        for color, text in heatmap_explanation():
            st.markdown(f"<p style='font-size:20px'>- <span style='color:{color}'>●</span> {text}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔬 For educational and research purposes. Consult healthcare professionals for medical diagnoses.")

if __name__ == "__main__":
    main()
