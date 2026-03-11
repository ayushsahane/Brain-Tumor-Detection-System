import streamlit as st
import os
import numpy as np
import cv2
import streamlit.components.v1 as html_components
from .utils import set_css
from .components import title
from predictor import get_model
from mask import crop_img

@st.cache_resource  # Updated cache decorator for resources like models
def load_model():
    model, metrics = get_model(0)
    return model, metrics

def main():
    html_components.html(title())
    set_css("pages/css/streamlit.css")
    st.write(
        """Here, you can upload your MRI image of choice and see the analysis results.
    The program will automatically crop the image to the brain area and then analyzes the 
    image. The results will be displayed in the browser."""
    )
    image_bytes = st.file_uploader(
        "Upload a brain MRI scan image", type=["png", "jpeg", "jpg"]
    )

    if image_bytes:
        array = np.frombuffer(image_bytes.read(), np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))
        st.write(
            """
                #### Brain MRI scan image
                """
        )
        st.image(image)

    if st.button("Analyze"):
        with st.spinner(text="Analyzing..."):
            model, metrics = load_model()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = crop_img(gray, image, None)
            cv2.imwrite("temp.png", img)
            img_mask = crop_img(gray, image, None)
            gray_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_OTSU)[-1]
            img_resized = cv2.resize(img, (50, 50))  # Ensure size matches training size
            img_resized = np.array([img_resized])
            prediction = model.predict(img_resized)

            st.write(
                """
                #### Mask Threshold
                """
            )
            st.image(cv2.resize(thresh, (128, 128)), caption="Threshold Image")

            st.write("#### Prediction")
            st.image(cv2.resize(img_mask, (128, 128)), caption="Cropped Image")
            confidence = prediction[0][0]
            if confidence >= 0.5:
                st.write(f"The sample has a tumor with confidence: {confidence * 100:.2f}%")
            else:
                st.write(f"The sample has no tumor with confidence: {(1 - confidence) * 100:.2f}%")

            st.write("#### Model Metrics")
            st.write(f"Accuracy: {metrics['accuracy']:.2%}")
            st.write(f"Loss: {metrics['loss']:.4f}")
            st.write(f"Precision: {metrics['precision']:.2%}")
            st.write(f"Recall: {metrics['recall']:.2%}")
            st.write(f"AUC: {metrics['auc']:.2%}")

