import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os

model = YOLO("modelzoo/safety.pt")

st.title("YOLOv8 Demo")


media_type = st.radio("Choose media type", ("Image", "Video"))

if media_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        np_image = np.array(image)
        results = model(np_image)
        result_img = results[0].plot()
        st.image(result_img, caption='Detection Result', use_column_width=True)

else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
        cap.release()
        os.remove(video_path)