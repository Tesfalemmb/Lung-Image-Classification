import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2

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

# Load your trained model
MODEL_PATH = 'lung_classification_model_efficientnetb0.h5'

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    if not TENSORFLOW_AVAILABLE:
        return None
        
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            return None
        
        # Build model with correct 3-channel input architecture
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=(224, 224, 3),  # 3-channel input
            weights=None
        )
        
        # Add custom layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Load the weights
        model.load_weights(MODEL_PATH)
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# Define class names
class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']

def get_gradcam(img_array, model, class_index):
    """Generate Grad-CAM heatmap"""
    if model is None:
        return None
        
    # Find the last convolutional layer
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
        return None

def preprocess_image(img):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB (3 channels)
        img = img.convert('RGB')
        img_resized = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Use EfficientNet preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it and visualize important regions.")

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
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
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
            
            # Grad-CAM visualization - ONLY OVERLAY
            st.subheader("🔥 Model Explanation")
            st.write("Heatmap overlay showing the areas that influenced the prediction:")
            
            with st.spinner("🔄 Generating heatmap overlay..."):
                heatmap = get_gradcam(img_array, model, pred_class_index)
                
                if heatmap is not None:
                    # Resize heatmap to match original image
                    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    # Convert PIL Image to numpy array for OpenCV
                    img_np = np.array(img.convert('RGB'))
                    
                    # Superimpose heatmap on original image
                    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                    
                    # Display ONLY the overlay image
                    st.image(superimposed_img, caption=f"Grad-CAM Overlay for {pred_class}", use_column_width=True)
                else:
                    st.warning("Could not generate Grad-CAM visualization")

    else:
        st.info("👆 Please upload a lung image to get started.")

    st.markdown("---")
    st.caption("🔬 For educational and research purposes. Consult healthcare professionals for medical diagnoses.")

if __name__ == "__main__":
    main()
