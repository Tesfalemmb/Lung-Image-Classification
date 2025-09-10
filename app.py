import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
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

# Model path
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
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception:
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

def preprocess_image(img):
    try:
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def get_gradcam(img_array, model, class_index):
    if model is None:
        return None
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
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap
    except Exception:
        return None

def explanation_for_class(pred_class):
    if pred_class == "Healthy":
        return [
            ("green", "Minimal abnormal activations observed."),
            ("green", "Lung tissue looks consistent with healthy patterns."),
        ]
    elif pred_class == "Inflammation":
        return [
            ("red", "Highlighted regions suggest inflamed tissue."),
            ("red", "Possible infection or irritation detected."),
        ]
    elif pred_class == "Neoplastic":
        return [
            ("blue", "Model detected irregular growth patterns."),
            ("blue", "May indicate tumor or neoplastic changes."),
        ]
    else:
        return [
            ("orange", "Model uncertain about the affected regions."),
            ("orange", "Further examination is recommended."),
        ]

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it and visualize important regions.")

    uploaded_file = st.file_uploader(
        "Choose a lung image",
        type=["jpg", "jpeg", "png"],
        help="Select a lung X-ray or CT scan image for analysis"
    )

    if uploaded_file is not None:
        # Cloud-safe read
        uploaded_bytes = uploaded_file.read()
        img = Image.open(BytesIO(uploaded_bytes)).convert("RGB")

        if model is None:
            st.error("Model failed to load. Please check the model file.")
            return

        with st.spinner("🔄 Processing image..."):
            img_array = preprocess_image(img)
            if img_array is None:
                st.error("Failed to process image")
                return

            preds = model.predict(img_array, verbose=0)[0]
            pred_class_index = np.argmax(preds)
            pred_class = class_names[pred_class_index]
            confidence = np.max(preds) * 100
            prediction_color = class_colors[pred_class_index]

            # Top row: Image | Prediction Confidence | Final Prediction
            col1, col2, col3 = st.columns([1.2, 1.2, 1])

            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True)

            with col2:
                st.subheader("📊 Prediction Confidence")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(class_names, preds * 100, color=class_colors)
                ax.set_xlim([0, 100])
                ax.set_xlabel("Probability (%)")
                ax.set_title("Prediction Confidence")
                for i, v in enumerate(preds * 100):
                    ax.text(v + 1, i, f"{v:.2f}%", va="center", fontsize=10)
                st.pyplot(fig)

            with col3:
                st.markdown(
                    f"<h2 style='color:{prediction_color}; font-size:34px'>✅ Final Prediction: <b>{pred_class}</b></h2>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<h4 style='font-size:24px'>Confidence: {confidence:.2f}%</h4>",
                    unsafe_allow_html=True
                )

        # Bottom row: Grad-CAM Overlay | Interpretation
        st.subheader("🔥 Model Explanation")
        st.write("Grad-CAM overlay highlights the regions influencing the prediction:")

        col4, col5 = st.columns([1.2, 1])

        with col4:
            heatmap = get_gradcam(img_array, model, pred_class_index)
            if heatmap is not None:
                heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                img_np = np.array(img)
                superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                superimposed_img_pil = Image.fromarray(superimposed_img)
                st.image(superimposed_img_pil, caption=f"Grad-CAM Overlay for {pred_class}", use_column_width=True)
            else:
                st.warning("Could not generate Grad-CAM visualization")

        with col5:
            st.markdown("<h3 style='font-size:28px'>📝 Heatmap Interpretation</h3>", unsafe_allow_html=True)
            for color, text in explanation_for_class(pred_class):
                st.markdown(
                    f"<p style='font-size:22px'>- <b><span style='color:{color}'>{color.capitalize()}</span></b> → {text}</p>",
                    unsafe_allow_html=True
                )
    else:
        st.info("👆 Please upload a lung image to get started.")

    st.markdown("---")
    st.caption("🔬 For educational and research purposes. Consult healthcare professionals for medical diagnoses.")

if __name__ == "__main__":
    main()
