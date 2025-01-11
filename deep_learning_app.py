import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Define focal loss with proper Keras serialization
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=1, gamma=0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        ce_loss = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_factor = tf.math.pow(1 - p_t, gamma)
        focal_loss_value = alpha * focal_factor * ce_loss
        return tf.reduce_mean(focal_loss_value)
    return focal_loss_fixed

import tempfile

# Define the model URL
model_url = "https://drive.google.com/uc?id=1JB6KQIAnyTS_aks7e7CxTbnx8z0zL8Ru"

# Use a temporary directory for the model file
temp_dir = tempfile.gettempdir()
model_path = os.path.join(temp_dir, "best_model_transfer_A_to_D.keras")

# Download the model if not already present
if not os.path.exists(model_path):
    print("Downloading the model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)
    print("Model downloaded and saved to:", model_path)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'focal_loss_fixed': focal_loss(alpha=1, gamma=0)},
    )

model = load_model()


# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the uploaded image to match the input shape of the model.
    - Resizes to (256, 256).
    - Ensures 3 channels (RGB).
    - Normalizes pixel values to [0, 1].
    """
    image = image.resize((256, 256))
    image = image.convert("RGB")  # Ensure RGB format
    image_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Streamlit app
st.title("Crack Detection Dashboard")
st.write("Upload an image of a bridge deck or pavement to predict if it is cracked or uncracked.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    preprocessed_image = preprocess_image(image)
    
    # Make prediction
    st.write("Making prediction...")
    prediction = model.predict(preprocessed_image)
    predicted_class = "Cracked" if prediction[0][0] >= 0.5 else "Uncracked"
    confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
    
    # Display results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
