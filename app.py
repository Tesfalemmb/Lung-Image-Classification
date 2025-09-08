# @title 🫁 Lung Classification Web App (With Probability Bar Chart + Grad-CAM)
# ===============================
# 1. Install dependencies
# ===============================
!pip install streamlit pyngrok tensorflow pillow matplotlib opencv-python-headless --quiet

# ===============================
# 2. Authenticate ngrok (replace with your token if different)
# ===============================
!pkill -f ngrok || echo "No old ngrok processes running"
!ngrok config add-authtoken 30dOVNGZkPDo7jaZvHxnNLJeEg6_jnGobTNd3XNYoXurk2wv

# ===============================
# 3. Write Streamlit app to file
# ===============================
app_code = """
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Load your trained model
MODEL_PATH = '/content/drive/MyDrive/lung_classification_model_efficientnetb0.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = ['Healthy', 'Inflammation', 'Neoplastic', 'Undetermined']

# -----------------------------
# Grad-CAM function
# -----------------------------
def get_gradcam(img_array, model, class_index):
    # Find last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("⚠️ No Conv2D layer found in the model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # ✅ grads shape = (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🫁 Lung Image Classification App (with Grad-CAM)")
st.write("Upload a lung image to classify it and visualize important regions.")

uploaded_file = st.file_uploader("Upload a lung image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ✅ Proper preprocessing

    # Make prediction
    preds = model.predict(img_array)[0]
    
    st.write("### Prediction Probabilities:")

    # Display class probabilities
    for i, prob in enumerate(preds):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")

    # Final prediction
    pred_class_index = np.argmax(preds)
    pred_class = class_names[pred_class_index]
    confidence = np.max(preds) * 100
    st.subheader(f"✅ Final Prediction: **{pred_class}** ({confidence:.2f}%)")

    # Plot probability bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(class_names, preds * 100, color="skyblue")
    ax.set_xlim([0, 100])
    ax.set_xlabel("Probability (%)")
    ax.set_title("Prediction Confidence")
    for i, v in enumerate(preds * 100):
        ax.text(v + 1, i, f"{v:.2f}%", va="center")
    st.pyplot(fig)

    # -----------------------------
    # Generate and show Grad-CAM
    # -----------------------------
    st.write("### 🔥 Grad-CAM Visualization")

    heatmap = get_gradcam(img_array, model, pred_class_index)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    st.image(superimposed_img, caption=f"Grad-CAM for {pred_class}", use_column_width=True)
"""

with open("app.py", "w") as f:
    f.write(app_code)

# ===============================
# 4. Run Streamlit app in background
# ===============================
!streamlit run app.py --server.port 8501 &>/dev/null&

# ===============================
# 5. Expose with ngrok
# ===============================
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🌍 Your Web App is live at:", public_url)
