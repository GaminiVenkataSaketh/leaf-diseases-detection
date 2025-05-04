import streamlit as st
import cv2 as cv
import numpy as np
import keras
import base64
import os

# Labels for prediction
label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

# Page configuration
st.set_page_config(page_title="Leaf Disease Recognition", layout="wide")

# Sidebar for theme toggle
theme = st.sidebar.radio("Select Theme", ["Light Mode", "Dark Mode"])
dark_mode = theme == "Dark Mode"

# Set background image
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_image_path = r"C:\Users\venka\Documents\Projects\leaf-diseases-detect-main\Media\Background.jpg"

if os.path.exists(background_image_path):
    img_base64 = get_base64_of_image(background_image_path)
    bg_color = "rgba(0, 0, 0, 0.85)" if dark_mode else "rgba(255, 255, 255, 0.85)"
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: {bg_color};
        }}
        </style>
    """, unsafe_allow_html=True)
else:
    st.error("⚠️ Background image not found! Please check the file path.")

# App title and instructions
header_color = "white" if dark_mode else "black"
st.markdown(f"<h1 style='text-align: center; color: white;'>🌿 Leaf Disease Diagnosis 🌿</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center; color: {header_color};'>Upload an image of a leaf to detect disease</h3>", unsafe_allow_html=True)

# Load model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# File uploader
uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    resized_img = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150))
    normalized_image = np.expand_dims(resized_img, axis=0)
    display_img = cv.resize(resized_img, (300, 300))

    st.image(image_bytes, caption="Uploaded Leaf Image", use_container_width=False)

    with st.spinner("🔍 Analyzing leaf image..."):
        predictions = model.predict(normalized_image)
        confidence = predictions[0][np.argmax(predictions)] * 100
        predicted_label = label_name[np.argmax(predictions)]

    if confidence >= 80:
        st.markdown(f"""
            <style>
            .result-text {{
                font-size: 30px;
                font-weight: bold;
                color: {'white' if dark_mode else 'gold'};
                background-color: rgba(0, 0, 0, 0.85); 
                padding: 10px;
                border-radius: 10px;
                display: inline-block;
            }}
            </style>
            <p class='result-text'>✅ <strong>Result:</strong> {predicted_label} ({confidence:.2f}% confidence)</p>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Try another image for better accuracy!")

    if "healthy" in predicted_label.lower():
        st.markdown(f"""
            <style>
            .healthy-text {{
                font-size: 18px;
                font-weight: bold;
                color: {'white' if dark_mode else 'gold'};
                background-color: rgba(0, 128, 0, 0.85); 
                padding: 10px;
                border-radius: 10px;
                display: inline-block;
            }}
            </style>
            <p class='healthy-text'>🌱 The plant appears healthy! No action required.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <style>
            .infected-text {{
                font-size: 18px;
                font-weight: bold;
                color: {'white' if dark_mode else 'gold'};
                background-color: rgba(128, 0, 0, 0.85); 
                padding: 10px;
                border-radius: 10px;
                display: inline-block;
            }}
            </style>
            <p class='infected-text'>⚠️ This plant might be affected. Consider taking precautions!</p>
        """, unsafe_allow_html=True)

        # Precautions section
        precautions = [
            "🧪 Remove and destroy infected leaves to prevent spreading.",
            "🌿 Use recommended fungicides or bactericides for the detected disease.",
            "💧 Avoid overhead irrigation; water at the base of the plant.",
            "🧼 Sterilize tools before and after use.",
            "🚜 Rotate crops and avoid planting the same crop repeatedly in the same soil."
        ]

        st.markdown("### 🛡️ Precautions to be Taken:")
        for precaution in precautions:
            st.markdown(f"- {precaution}")
