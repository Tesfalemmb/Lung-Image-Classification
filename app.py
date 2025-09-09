import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# Load your trained model
MODEL_PATH = 'lung_classification_model_efficientnetb0.h5'

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None
        
        # Try different loading methods for compatibility
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except:
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                return model
            except:
                model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
                return model
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# Define class names
class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']

def get_gradcam(img_array, model, class_index):
    """Generate Grad-CAM heatmap"""
    if model is None:
        return None
        
    # Find last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'Conv2D' in str(layer.__class__):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        st.warning("⚠️ No Conv2D layer found in the model.")
        return None

    try:
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap
        
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None

def preprocess_image(img):
    """Preprocess image for model prediction - CONVERT TO 3-CHANNEL RGB"""
    try:
        # Convert to RGB (3 channels) to match model expectations
        img = img.convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Use EfficientNet preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it and visualize important regions using Grad-CAM.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a lung image", 
        type=["jpg", "jpeg", "png"],
        help="Select a lung X-ray or CT scan image for analysis"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)
        
        if model is None:
            st.error("Model failed to load. Please check the model file.")
        else:
            # Preprocess and predict
            with st.spinner("🔄 Processing image..."):
                img_array = preprocess_image(img)
                
                if img_array is None:
                    st.error("Failed to process image")
                    return
                
                # Make prediction
                try:
                    preds = model.predict(img_array, verbose=0)[0]
                    
                    with col2:
                        st.subheader("📊 Prediction Results")
                        
                        # Show confidence scores
                        for i, (class_name, prob) in enumerate(zip(class_names, preds)):
                            color = "green" if class_name == "Healthy" else "red" if class_name == "Inflammation" else "blue" if class_name == "Neoplastic" else "orange"
                            st.markdown(f"**<span style='color: {color}'>{class_name}:</span> {prob*100:.2f}%**", unsafe_allow_html=True)
                            st.progress(float(prob))
                        
                        # Final prediction
                        pred_class_index = np.argmax(preds)
                        pred_class = class_names[pred_class_index]
                        confidence = np.max(preds) * 100
                        
                        prediction_color = "green" if pred_class == "Healthy" else "red" if pred_class == "Inflammation" else "blue" if pred_class == "Neoplastic" else "orange"
                        st.markdown(f"<h3 style='color: {prediction_color}'>✅ Final Prediction: {pred_class}</h3>", unsafe_allow_html=True)
                        st.info(f"**Confidence: {confidence:.2f}%**")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    return
            
            # Grad-CAM visualization
            st.subheader("🔥 Grad-CAM Visualization")
            
            with st.spinner("🔄 Generating explanation..."):
                heatmap = get_gradcam(img_array, model, pred_class_index)
                
                if heatmap is not None:
                    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    # Convert PIL Image to numpy array for OpenCV
                    img_np = np.array(img.convert('RGB'))
                    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                    
                    # Display results
                    cam_col1, cam_col2 = st.columns(2)
                    
                    with cam_col1:
                        st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
                    
                    with cam_col2:
                        st.image(superimposed_img, caption="Overlay on Image", use_column_width=True)
                else:
                    st.warning("Could not generate Grad-CAM visualization")

    else:
        st.info("👆 Please upload a lung image to get started.")

    # Footer
    st.markdown("---")
    st.caption("🔬 For educational and research purposes. Consult healthcare professionals for medical diagnoses.")

# Run the app
if __name__ == "__main__":
    main()
