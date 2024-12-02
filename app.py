import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile

st.title("Enhanced Object Detection App")

# Display detected objects on video frames
def _display_detected_frames(conf, model, st_frame, image):
    """
    Display detected objects on a video frame using YOLOv8.
    """
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Objects', channels="BGR", use_column_width=True)

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    """
    Load YOLO model from the specified path.
    """
    return YOLO(model_path)

# Inference on uploaded images
def infer_uploaded_image(conf, model):
    """
    Perform object detection on uploaded images.
    """
    source_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    col1, col2 = st.columns(2)
    if source_img:
        uploaded_image = Image.open(source_img)
        with col1:
            st.image(source_img, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Processing..."):
                res = model.predict(uploaded_image, conf=conf)
                res_plotted = res[0].plot()[:, :, ::-1]
                with col2:
                    st.image(res_plotted, caption="Detected Objects", use_column_width=True)
                st.sidebar.write("Detected Classes:")
                for box in res[0].boxes:
                    st.sidebar.write(box.label)

# Inference on uploaded videos
def infer_uploaded_video(conf, model):
    """
    Perform object detection on uploaded videos.
    """
    source_video = st.sidebar.file_uploader("Upload a video")
    if source_video:
        st.video(source_video)
        if st.button("Detect Objects in Video"):
            with st.spinner("Processing..."):
                try:
                    temp_video = tempfile.NamedTemporaryFile(delete=False)
                    temp_video.write(source_video.read())
                    vid_cap = cv2.VideoCapture(temp_video.name)
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf, model, st_frame, image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error processing video: {e}")

# Inference using webcam
def infer_uploaded_webcam(conf, model):
    """
    Perform object detection using a webcam.
    """
    st.warning("Ensure webcam permissions are granted!")
    if st.button("Start Webcam Detection"):
        try:
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            while True:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")

# Sidebar controls
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Path", "path/to/yolo/model.pt")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
task = st.sidebar.selectbox("Choose Task", ["Image", "Video", "Webcam"])

# Load the model
model = load_model("yolo11x-seg.pt")

# Task selection
if task == "Image":
    infer_uploaded_image(conf_threshold, model)
elif task == "Video":
    infer_uploaded_video(conf_threshold, model)
elif task == "Webcam":
    infer_uploaded_webcam(conf_threshold, model)

# Additional analytics and visualizations
st.sidebar.header("Statistics")
st.sidebar.write("Confidence Threshold:", conf_threshold)
st.sidebar.write("Detected Classes (Real-time analysis will show here):")
st.markdown("## Additional Features")
st.write("Explore object detection with various inputs and real-time analysis!")

# Placeholder for additional plots or data insights
st.markdown("### Insights")
st.write("Visualizations of detected classes and their counts will appear here.")
