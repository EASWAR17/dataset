import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import google.generativeai as genai
import base64
from dotenv import load_dotenv
import pyfiglet

# Load API Key
load_dotenv()
GOOGLE_API_KEY = ""
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# Create directory to save Grad-CAM images
GEN_IMAGES_DIR = "Gen_Images"
os.makedirs(GEN_IMAGES_DIR, exist_ok=True)

# Load pre-trained model
MODEL_PATH = "lung_cancer_model_vgg.h5"
model = load_model(MODEL_PATH)

# Standard image size for consistency
STANDARD_SIZE = (400, 400)

# Function to predict cancer
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Cancerous" if prediction < 0.5 else "Non-Cancerous"

def generate_image_description(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        model = genai.GenerativeModel('gemini-2.0-flash')
        user_prompt = (f"The AI model has predicted this scan as {prediction}. As an expert medical XAI model, analyze the provided Grad-CAM image to identify key regions influencing the model’s decision. Explain how these highlighted areas contribute to the prediction, assess whether the Grad-CAM visualization aligns with the diagnosis, and provide a justification for the AI’s decision based on medical imaging principles. Keep the explanation concise yet informative , ensuring clarity for medical professionals.")
        
        response = model.generate_content([
            user_prompt,
            {"mime_type": "image/png", "data": encoded_image}
        ])
        return response.text
    except Exception as e:
        return f"An error occurred while generating explanation: {e}"

# Grad-CAM Implementation
def generate_gradcam(model, img_path):
    IMG_SIZE = (224, 224)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer('block5_conv3').output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save the Grad-CAM image
    gradcam_path = os.path.join(GEN_IMAGES_DIR, os.path.basename(img_path).replace('.jpg', '_gradcam.jpg'))
    cv2.imwrite(gradcam_path, superimposed_img)

    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)).resize(STANDARD_SIZE), gradcam_path

# Function to resize images
def resize_image(image_path):
    img = Image.open(image_path)
    return img.resize(STANDARD_SIZE)


# Streamlit Page Configuration
st.set_page_config(page_title="XAI Project", layout="wide")

# Move Title Higher
st.markdown("<h1 style='text-align: center; margin-top: -40px;'> XAI Project</h1>", unsafe_allow_html=True)

# Add Top Padding
st.write("\n\n")

# Sample images (Replace with actual image paths)
sample_images = [
    "web-images/grad_cam_enet.png",
    "web-images/grad_cam_mvnet.png",
    "web-images/grad_cam_vgg.png"
]

# Display sample images
st.markdown("<h3 style='text-align: left; margin-bottom: 20px;'> Grad-Cam Results</h3>", unsafe_allow_html=True)
st.write("\n\n")

col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 3, 1, 3, 1, 3, 1])
with col2:
    st.image(resize_image(sample_images[0]), caption="EfficientNet")
with col4:
    st.image(resize_image(sample_images[1]), caption="MobileNet")
with col6:
    st.image(resize_image(sample_images[2]), caption="VGG16")


# Add Bottom Padding
st.write("\n\n")

st.write("### Model Performance Overview📊")
st.write("\n\n")
st.write("\n\n")
metric_images = [
    "web-images/Confusion_matrix.png",
    "web-images/ROC_Curve.png",
    "web-images/Precision_Recall_curve.png",
    "web-images/Accuracy.png",
    "web-images/loss.png"
]
col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 3, 1, 3, 1, 3, 1])
with col2:
    st.image(metric_images[0], caption="Confusion Matrix")
with col4:
    st.image(metric_images[1], caption="ROC Curve")
with col6:
    st.image(metric_images[2], caption="Precision Recall Curve")

st.write("\n\n")

st.write("### Training & Learning Dynamics📈")
col1, col2, col3, col4,col5 = st.columns([1,1,1,1,1])
with col2:
    st.image(metric_images[3], caption="Training Accuracy")
with col4:
    st.image(metric_images[4], caption="Training loss")

st.write("\n\n")

# File uploader (Drag & Drop)
st.write("### Upload Your Scan Image👇")
uploaded_file = st.file_uploader("", type=["png"])

if uploaded_file:
    # Convert uploaded image to a temporary file
    temp_path = "temp_uploaded_image.jpg"
    image_bytes = uploaded_file.read()
    with open(temp_path, "wb") as f:
        f.write(image_bytes)
    
    # Perform Prediction
    prediction = predict_image(temp_path, model)
    # st.write("### Prediction Result")
    st.markdown(f"<h3 style='text-align: center;'>Prediction: {prediction}</h3>", unsafe_allow_html=True)

    # Display uploaded image
    #st.write("### Uploaded Image")
    uploaded_img = Image.open(BytesIO(image_bytes)).resize(STANDARD_SIZE)
    # Generate Grad-CAM
    gradcam_image, gradcam_path = generate_gradcam(model, temp_path)
    #st.image(uploaded_img, caption="Uploaded Image", use_container_width=False)

    col1, col2, col3= st.columns([1, 0.5, 1])
    with col1:
        st.write("### Uploaded Image")
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=False)
    with col3:
        st.write("### Grad-CAM Heatmap")
        st.image(gradcam_image, caption="Grad-CAM Output", use_container_width=False)
    # AI-based explanation using Gemini API


    # Display Results
    # st.write("### Grad-CAM Heatmap")
    #st.image(gradcam_image, caption="Grad-CAM Output", use_container_width=False)
    
    # Generated Explanation
    st.write("#### Explanation:")
    explanation = generate_image_description(gradcam_path)
    st.write(explanation)

    # Cleanup temporary file
    os.remove(temp_path)

# Add Footer
st.write("---")

st.markdown(
    """
    <style>
    @keyframes glowing {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    @keyframes typing {
        0%   { content: "B"; }
        10%  { content: "BY"; }
        20%  { content: "BY "; }
        30%  { content: "BY E"; }
        40%  { content: "BY EA"; }
        50%  { content: "BY EAS"; }
        60%  { content: "BY EASW"; }
        70%  { content: "BY EASWA"; }
        80%  { content: "BY EASWAR"; }
        100% { content: "BY EASWAR"; }
    }

    .animated-text::after {
        content: "";
        font-size: 15px;
        font-weight: thin;
        position: absolute;
        right: 20px; /* Adjust for fine-tuning */
        top: 10px;
        background: linear-gradient(270deg, #ffffff, #bfbfbf, #808080, #ffffff); /* White to grey shades */
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: typing 4s steps(8, end) infinite, glowing 3s ease infinite;
    }
    </style>

    <p class="animated-text"></p>
    """,
    unsafe_allow_html=True
)
