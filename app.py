import streamlit as st
import numpy as np
from PIL import Image
import os
import streamlit as st
import numpy as np
from PIL import Image
import os
import h5py  # Add this import

# Set page configuration
st.set_page_config(
    page_title="Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# Check if model file exists and is valid
MODEL_PATH = 'lung_classification_model_efficientnetb0.h5'

def check_model_file():
    """Check if model file exists and is valid"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.info("Please make sure the file 'lung_classification_model_efficientnetb0.h5' is in the same folder as this app.")
        return False
    
    # Check if file is a valid HDF5 file
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            st.success("✅ Model file is a valid HDF5 file")
            # List all keys in the file
            st.write("File structure:", list(f.keys()))
        return True
    except Exception as e:
        st.error(f"❌ Model file is corrupted or invalid: {str(e)}")
        st.info("This usually means:")
        st.info("1. The file didn't upload properly to GitHub")
        st.info("2. The file got corrupted during download")
        st.info("3. It's not actually a .h5 model file")
        return False

# Check model file first
if check_model_file():
    # Try to import TensorFlow
    try:
        import tensorflow as tf
        TENSORFLOW_AVAILABLE = True
        
        @st.cache_resource
        def load_model():
            """Load and cache the trained model"""
            try:
                # Try different loading methods
                try:
                    model = tf.keras.models.load_model(MODEL_PATH)
                    st.success("✅ Model loaded successfully!")
                    return model
                except Exception as e1:
                    try:
                        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                        st.success("✅ Model loaded successfully with compile=False!")
                        return model
                    except Exception as e2:
                        try:
                            model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
                            st.success("✅ Model loaded successfully with safe_mode=False!")
                            return model
                        except Exception as e3:
                            st.error(f"All loading methods failed:")
                            st.error(f"Method 1: {str(e1)}")
                            st.error(f"Method 2: {str(e2)}")
                            st.error(f"Method 3: {str(e3)}")
                            return None
                            
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None

        # Load model
        model = load_model()

    except ImportError:
        st.error("TensorFlow not installed. Please check requirements.txt")
        model = None
else:
    model = None

# Define class names
class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']

def preprocess_image(img):
    """Preprocess image for model prediction - CONVERT TO GRAYSCALE"""
    try:
        # Convert to grayscale (1 channel) to match model expectations
        img = img.convert('L')  # 'L' mode for grayscale
        img_resized = img.resize((224, 224))
        
        # Convert to numpy array and add channel dimension
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a lung image", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None and model is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)
        
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

    elif uploaded_file is None:
        st.info("👆 Please upload a lung image to get started.")

    st.markdown("---")
    st.caption("🔬 For educational and research purposes. Consult healthcare professionals for medical diagnoses.")

if __name__ == "__main__":
    main()

# Try to import TensorFlow with graceful fallback
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not installed. Please check requirements.txt")

# Try to import matplotlib with graceful fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.error("Matplotlib not installed. Please check requirements.txt")

# Try to import OpenCV with graceful fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.error("OpenCV not installed. Please check requirements.txt")

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
    if not TENSORFLOW_AVAILABLE:
        return None
        
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None
        
        # Try different loading methods
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

def preprocess_image(img):
    """Preprocess image for model prediction - CONVERT TO GRAYSCALE"""
    try:
        # Convert to grayscale (1 channel) to match model expectations
        img = img.convert('L')  # 'L' mode for grayscale
        img_resized = img.resize((224, 224))
        
        # Convert to numpy array and add channel dimension
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension: (224, 224) -> (224, 224, 1)
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension: (224, 224, 1) -> (1, 224, 224, 1)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    st.title("🫁 Lung Image Classification App")
    st.write("Upload a lung image to classify it and visualize important regions using Grad-CAM.")

    # Check if all required packages are available
    if not all([TENSORFLOW_AVAILABLE, MATPLOTLIB_AVAILABLE, OPENCV_AVAILABLE]):
        st.error("Some required packages are missing. Please check your requirements.txt file.")
        return

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

    else:
        st.info("👆 Please upload a lung image to get started.")

    st.markdown("---")
    st.caption("🔬 For educational and research purposes. Consult healthcare professionals for medical diagnoses.")

if __name__ == "__main__":
    main()
