import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Streamlit page config
st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

MODEL_PATH = "lung_classification_model_efficientnetb0.h5"

@st.cache_resource
def load_model():
    """Load full model or weights-only safely"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Loaded full model successfully.")
        return model
    except Exception as e_full:
        st.warning("⚠️ Could not load as full model, trying weights-only...")
        try:
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

model = load_model()

class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']
colors = ["green","red","blue","orange"]  # For dynamic bars

def preprocess_image(img):
    img = img.convert('RGB')
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def get_gradcam(img_array, model, class_index):
    """Generate Grad-CAM heatmap"""
    if model is None:
        return None
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
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap,0)/(np.max(heatmap)+1e-8)
        return heatmap
    except:
        return None

def explanation_for_class(pred_class):
    """Dynamic Grad-CAM explanation depending on predicted class"""
    explanations = {
        "Healthy": [
            ("🟢 Green areas", "Minimal model focus – normal lung background."),
            ("🔵 Blue areas", "Regions with little diagnostic relevance."),
            ("🟡 Yellow areas", "Slight model attention, but not strongly indicative."),
            ("🔴 Red areas", "No abnormal hotspot detected in healthy prediction.")
        ],
        "Inflammation": [
            ("🔴 Red areas", "Strong focus on inflamed tissue regions."),
            ("🟡 Yellow areas", "Moderate influence, may indicate mild infection."),
            ("🟢 Green areas", "Healthy lung tissue with minimal relevance."),
            ("🔵 Blue areas", "Background or noise, low importance.")
        ],
        "Neoplastic": [
            ("🔴 Red areas", "Strong attention to potential tumor-like regions."),
            ("🟡 Yellow areas", "Possible secondary abnormal zones."),
            ("🟢 Green areas", "Surrounding healthy tissue."),
            ("🔵 Blue areas", "Least relevant parts of the scan.")
        ],
        "Undetermined": [
            ("🔴 Red areas", "Model found uncertain abnormal regions."),
            ("🟡 Yellow areas", "Possible mild influence."),
            ("🟢 Green areas", "Healthy-like regions."),
            ("🔵 Blue areas", "Ignored by the model.")
        ]
    }
    return explanations.get(pred_class, [])

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify and visualize important regions.")

    uploaded_file = st.file_uploader("Choose a lung image", type=["jpg","jpeg","png"])
    if uploaded_file is None:
        st.info("👆 Please upload a lung image to get started.")
        return

    img = Image.open(uploaded_file)

    if model is None:
        st.error("Model failed to load. Please check the model file.")
        return

    img_array = preprocess_image(img)
    preds = model.predict(img_array, verbose=0)[0]
    pred_class_index = np.argmax(preds)
    pred_class = class_names[pred_class_index]
    confidence = np.max(preds)*100

    # Grad-CAM overlay
    heatmap = get_gradcam(img_array, model, pred_class_index)
    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_np = np.array(img.convert('RGB'))
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    else:
        superimposed_img = np.array(img.convert('RGB'))

    # === Top Row ===
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(img, caption="🩻 Uploaded Lung Image", use_column_width=True)

    with col2:
        prediction_color = (
    "green" if pred_class=="Healthy" else
    "red" if pred_class=="Inflammation" else
    "blue" if pred_class=="Neoplastic" else
    "orange"
)
st.markdown(
    f"<h2 style='color:{prediction_color}; font-size:32px'>✅ Final Prediction: {pred_class}</h2>",
    unsafe_allow_html=True
)
st.markdown(
    f"<h4 style='font-size:22px'>Confidence: {confidence:.2f}%</h4>",
    unsafe_allow_html=True
)

# --- inside col4 (Heatmap Interpretation) ---
st.markdown("<h3 style='font-size:26px'>📝 Interpretation of Heatmap</h3>", unsafe_allow_html=True)
for color, text in explanation_for_class(pred_class):
    st.markdown(
        f"<p style='font-size:20px'>- <b><span style='color:{color}'>{color}</span></b> → {text}</p>",
        unsafe_allow_html=True
    )

    # === Bottom Row ===
    st.markdown("---")
    st.subheader("🔥 Model Explanation (Grad-CAM)")

    col3, col4 = st.columns([1,1])
    with col3:
        st.image(superimposed_img, caption=f"Grad-CAM Overlay for {pred_class}", use_column_width=True)

    with col4:
        st.markdown("### 📝 Interpretation of Heatmap")
        for color, text in explanation_for_class(pred_class):
            st.markdown(f"- **<span style='color:{color}'>{color}</span>** → {text}", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔬 For educational/research purposes. Consult healthcare professionals for medical diagnoses.")

if __name__=="__main__":
    main()
