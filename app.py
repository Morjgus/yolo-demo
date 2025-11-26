import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

# Set page config
st.set_page_config(page_title="YOLO 11n Object Detection", layout="wide")

def main():
    st.title("YOLO 11n Object Detection")

    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Model selection
    use_ncnn = st.sidebar.checkbox("Use Optimized NCNN Model (for Pi)", value=False)
    
    if use_ncnn:
        model_path = 'yolo11n_ncnn_model'
    else:
        model_path = 'yolo11n.pt'
    
    # Optimization button
    if st.sidebar.button("Optimize for Pi (Export to NCNN)"):
        with st.spinner("Exporting model to NCNN format... this may take a while"):
            try:
                # Load the pt model first to export
                pt_model = YOLO('yolo11n.pt')
                pt_model.export(format="ncnn")
                st.success("Model exported successfully to 'yolo11n_ncnn_model'!")
            except Exception as e:
                st.error(f"Export failed: {e}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        if use_ncnn:
            st.error(f"Error loading NCNN model: {e}. Did you run 'Optimize for Pi'?")
        else:
            st.error(f"Error loading model: {e}")
        return

    # Video Source Selection
    source_type = st.sidebar.selectbox("Select Video Source", ("Webcam", "Network Stream", "Pi Camera"))

    source = None
    if source_type == "Webcam":
        webcam_index = st.sidebar.number_input("Webcam Index", min_value=0, value=0, step=1)
        source = webcam_index
    elif source_type == "Network Stream":
        source = st.sidebar.text_input("Enter Video Stream URL (e.g., RTSP, HTTP)")
    elif source_type == "Pi Camera":
        # GStreamer pipeline for Pi Camera (libcamera)
        # Adjust width/height/framerate as needed
        source = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! videoscale ! video/x-raw, format=BGR ! appsink"


    # Session state for running
    if 'run' not in st.session_state:
        st.session_state['run'] = False

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start Detection"):
            st.session_state['run'] = True
    with col2:
        if st.button("Stop Detection"):
            st.session_state['run'] = False

    # Placeholder for the video frame
    stframe = st.empty()
    
    if st.session_state['run']:
        if source is None or (isinstance(source, str) and source.strip() == ""):
            st.error("Please provide a valid video source.")
            st.session_state['run'] = False
        else:
            use_gstreamer = (source_type == "Pi Camera")
            run_detection(model, source, stframe, use_gstreamer)

def run_detection(model, source, stframe, use_gstreamer=False):
    if use_gstreamer:
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error(f"Error opening video source: {source}")
        st.session_state['run'] = False
        return

    while st.session_state['run'] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame from source. Stream may have ended.")
            st.session_state['run'] = False
            break

        # Run YOLO inference
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Streamlit
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        stframe.image(rgb_frame, channels="RGB", width="stretch")
        
        # Small delay to prevent UI freezing (optional, but good for responsiveness)
        # time.sleep(0.01) 

    cap.release()
    # Clear the frame when stopped
    stframe.empty()

if __name__ == "__main__":
    main()
