import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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

MODEL_PATH = "C:/Users/ericr/Downloads/best_model_transfer_A_to_D.keras"  # Adjust as needed
# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,  # Update to your model path
        custom_objects={'focal_loss_fixed': focal_loss(alpha=1, gamma=0)}
    )
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the uploaded image to match the input shape of the model.
    - Resizes to (256, 256).
    - Ensures 3 channels (RGB).
    - Normalizes pixel values to [0, 1].
    """
    # Resize to (256, 256)
    image = image.resize((256, 256))
    
    # Convert to RGB if image has alpha channel
    image = image.convert("RGB")
    
    # Convert to NumPy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Add batch dimension (1, 256, 256, 3)
    return np.expand_dims(image_array, axis=0)

# Streamlit app
st.title("Crack Detection Dashboard")
st.write("Upload an image of a bridge deck or pavement to predict if it is cracked or uncracked.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction
    st.write("Making prediction...")
    prediction = model.predict(preprocessed_image)
    predicted_class = "Cracked" if prediction[0][0] >= 0.5 else "Uncracked"
    confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
    
    # Display the results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")




