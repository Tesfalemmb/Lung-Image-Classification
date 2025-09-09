import subprocess
import sys
import os

# Install missing packages before anything else
def install_missing_packages():
    required_packages = [
        'matplotlib==3.7.5',
        'opencv-python-headless==4.8.1.78',
        'numpy==1.24.3',
        'pillow==10.1.0',
        'tensorflow==2.15.0',
        'protobuf==3.20.3',
        'h5py==3.9.0'
    ]
    
    for package in required_packages:
        try:
            __import__(package.split('==')[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Run installation check
install_missing_packages()

# Now import the rest of your packages
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ... rest of your app code
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import subprocess
import sys

# Install matplotlib if not present
try:
    import matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.7.5"])
    import matplotlib

# Rest of your imports
import streamlit as st
import numpy as np

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
        
        # Try multiple loading methods for compatibility
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
        
    except:
        return None

def preprocess_image(img):
    """Preprocess image for model prediction"""
    try:
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Simple normalization
        return img_array
    except:
        return None

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image for analysis with Grad-CAM explanations.")

    uploaded_file = st.file_uploader("Choose a lung image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if model is None:
            st.error("Model failed to load. Please check the model file.")
        else:
            with st.spinner("Processing image..."):
                img_array = preprocess_image(img)
                
                if img_array is None:
                    st.error("Failed to process image")
                    return
                
                try:
                    preds = model.predict(img_array, verbose=0)[0]
                    
                    with col2:
                        st.subheader("Prediction Results")
                        
                        for i, (class_name, prob) in enumerate(zip(class_names, preds)):
                            color = "green" if class_name == "Healthy" else "red" 
                            st.markdown(f"**{class_name}:** {prob*100:.2f}%")
                            st.progress(float(prob))
                        
                        pred_class_index = np.argmax(preds)
                        pred_class = class_names[pred_class_index]
                        confidence = np.max(preds) * 100
                        
                        st.success(f"**Final Prediction: {pred_class}**")
                        st.info(f"**Confidence: {confidence:.2f}%**")
                
                except:
                    st.error("Error making prediction")
                    return
            
            # Grad-CAM visualization
            heatmap = get_gradcam(img_array, model, pred_class_index)
            
            if heatmap is not None:
                st.subheader("Grad-CAM Visualization")
                heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
                
                cam_col1, cam_col2 = st.columns(2)
                with cam_col1:
                    st.image(heatmap, caption="Heatmap", use_column_width=True)
                with cam_col2:
                    st.image(superimposed_img, caption="Overlay", use_column_width=True)

    else:
        st.info("Please upload a lung image to get started.")

    st.markdown("---")
    st.caption("For educational and research purposes only.")

if __name__ == "__main__":
    main()
