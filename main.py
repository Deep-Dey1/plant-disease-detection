import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Get the absolute directory of the script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Corrected paths
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "trained_model", "class_indices.json")  # Assuming it's inside trained_model

# Load the pre-trained model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error(f"Model file not found: {model_path}")

# Load class indices
if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
else:
    st.error(f"Class indices file not found: {class_indices_path}")

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize to [0,1]
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Ensure that the predicted class index exists in class_indices
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Class")
    return predicted_class_name

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Check if the model and class indices are loaded
            if 'model' in globals() and 'class_indices' in globals():
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {prediction}')
            else:
                st.error("Model or class indices not loaded properly.")
