import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import streamlit.components.v1 as html_components
from .utils import set_css
from .components import title
from predictor import get_model
from mask import crop_img

def load_model():
    model, metrics = get_model(0)
    return model, metrics

def main():
    set_css("pages/css/streamlit.css")
    html_components.html(title())
    st.write("These are pre-cropped MRI scan samples. They were used to validate the model.\n")
    
    samples = os.listdir("pages/samples")
    option = st.selectbox("Select an image for analysis", range(1, len(samples) + 1))
    
    collection = []
    for i in range(0, len(samples), 3):
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            if i + idx < len(samples):
                col.image(Image.open(f"pages/samples/{samples[i + idx]}"))
                col.subheader(i + idx + 1)
        collection.append(cols)

    if st.button("Analyze"):
        with st.spinner(text="Analyzing..."):
            model, metrics = load_model()
            image_path = f"pages/samples/{samples[option-1]}"
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[-1]
            img = np.array([cv2.resize(image, (50, 50))])  # Ensure the size matches training size
            prediction = model.predict(img)

            st.write("#### Mask Threshold")
            st.image(thresh, caption=f"Threshold of sample {option}")

            st.write("#### Prediction")
            st.image(image, caption=f"Sample {option}")
            confidence = prediction[0][0]
            if confidence >= 0.5:
                st.write(f"Sample {option} has a tumor with confidence: {confidence * 100:.2f}%")
            else:
                st.write(f"Sample {option} has no tumor with confidence: {(1 - confidence) * 100:.2f}%")

            st.write("#### Model Metrics")
            st.write(f"Accuracy: {metrics['accuracy']:.2%}")
            st.write(f"Loss: {metrics['loss']:.4f}")
            st.write(f"Precision: {metrics['precision']:.2%}")
            st.write(f"Recall: {metrics['recall']:.2%}")
            st.write(f"AUC: {metrics['auc']:.2%}")
