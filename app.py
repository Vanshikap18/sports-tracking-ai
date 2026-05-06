import streamlit as st
import cv2
import os
import tempfile
import sys
import time

# Maintain system paths for module discovery
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tracker import SportsTracker
from utils import get_video_properties, create_video_writer

# Professional Dark & Purple Theme Injection
st.set_page_config(page_title="AI Sports Tracker", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 2px solid #7D4CDB;
    }
    label, p, .stMarkdown, [data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
    }
    .stButton>button p {
        color: #FFFFFF !important;
    }
    .stButton>button {
        background-color: #7D4CDB;
        color: #FFFFFF !important;
        border-radius: 8px;
        width: 100%;
        border: none;
        height: 3em;
    }
    .stAlert p, .stText p {
        color: #FFFFFF !important;
    }
    h1, h2, h3 {
        color: #9B6DFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title(" AI Sports Tracker: Pro Dashboard")
st.markdown("Advanced Multi-Object Tracking with **YOLOv8** and **BoT-SORT**.")

# Sidebar Configuration
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
uploaded_file = st.sidebar.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Use a fixed temp location for the input to ensure read permissions
    input_path = os.path.join(tempfile.gettempdir(), "input_video.mp4")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Stream")
        st.video(input_path)
    
    if st.button("Run AI Analysis"):
        with col2:
            st.subheader("Tracking Result")
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            cap = cv2.VideoCapture(input_path)
            width, height, fps = get_video_properties(cap)
            
            # Use a fixed temp location for the output
            output_path = os.path.join(tempfile.gettempdir(), "output_processed.mp4")
            
            # Pre-emptive cleanup of old results
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
                
            writer = create_video_writer(output_path, fps, (width, height))
            
            tracker = SportsTracker()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                processed_frame = tracker.process_frame(frame)
                writer.write(processed_frame)
                
                frame_idx += 1
                progress_bar.progress(frame_idx / total_frames)
                status_text.text(f"Analyzing: Frame {frame_idx}/{total_frames}")
            
            # CRITICAL: Close handles immediately to unlock the file for reading
            cap.release()
            writer.release()
            
            # Small delay to ensure OS file buffers are flushed
            time.sleep(1)
            
            status_text.success("Analysis Complete!")
            
            # Read and display as binary stream with explicit format to fix "not playing" error
            if os.path.exists(output_path):
                with open(output_path, 'rb') as v_file:
                    video_bytes = v_file.read()
                st.video(video_bytes, format="video/mp4")
            else:
                st.error("Error: Could not find the processed video file.")