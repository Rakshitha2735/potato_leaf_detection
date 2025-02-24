import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# Load the trained model
file_id = "1xMYZbuR4mv81pPdg-qS7ud4gQ3uFJVJe"
url = 'https://drive.google.com/file/d/1xMYZbuR4mv81pPdg-qS7ud4gQ3uFJVJe/view?usp=sharing'
model_path = "trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
model = tf.keras.models.load_model(model_path)

# Define class labels for potato leaf diseases
class_labels = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']

# Custom CSS for styling with background image (without dog)
st.markdown(
    """
    <style>
        .stApp {
            background-image: url('https://img.freepik.com/premium-vector/modern-abstract-cute-landscape-background-vector-with-framework_1021635-233.jpg?w=2000');
            background-size: cover;
            background-position: center;
            color: #8B4513; /* SaddleBrown for better visibility */
        }
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .uploadedFile {
            max-width: 400px;
        }
        img {
            max-width: 300px;
            border-radius: 10px;
        }
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
            color: #8B4513 !important; /* Brown for all text elements */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown("<h1>🌱 Detect & Protect: Potato Leaf Disease Classifier 🥔</h1>", unsafe_allow_html=True)
st.write("Upload an image of a potato leaf to detect diseases and take action! 🚜")

# File uploader
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 Uploaded Image', use_container_width=False)
    
    # Ensure image is in RGB mode
    image = image.convert("RGB")
    
    # Preprocess the image
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    # Display prediction results
    st.subheader("🔍 Prediction")
    st.write(f"**📝 Predicted Class:** {class_labels[predicted_class]}")
    st.write(f"**🎯 Confidence:** {confidence:.2f}")
    
    # Display additional message based on prediction
    if class_labels[predicted_class] == 'Potato_Early_blight':
        st.warning("⚠ This leaf has **Early Blight**. Consider using fungicides and improving field management.")
    elif class_labels[predicted_class] == 'Potato_Late_blight':
        st.error("🚨 This leaf has **Late Blight**. Immediate action is needed to prevent crop loss!")
    else:
        st.success("✅ This potato leaf is **healthy**! Keep up the good farming practices! 🌿")
