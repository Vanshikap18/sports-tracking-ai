import streamlit as st
import cv2
import os
import tempfile
import sys
import time
import uuid
import subprocess

# Maintain system paths for module discovery
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tracker import SportsTracker
from utils import get_video_properties

# Professional Dark & Purple Theme Injection
st.set_page_config(page_title="AI Sports Tracker", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 2px solid #7D4CDB; }
    label, p, .stMarkdown, [data-testid="stWidgetLabel"] { color: #FFFFFF !important; }
    .stButton>button { background-color: #7D4CDB; color: #FFFFFF !important; border-radius: 8px; width: 100%; border: none; height: 3em; }
    h1, h2, h3 { color: #9B6DFF !important; }
    </style>
    """, unsafe_allow_html=True)

st.title(" AI Sports Tracker: Pro Dashboard")
st.markdown("Advanced Multi-Object Tracking with **YOLOv8** and **BoT-SORT**.")

# Sidebar Configuration
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
uploaded_file = st.sidebar.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    unique_id = uuid.uuid4().hex[:8]
    input_path = os.path.join(tempfile.gettempdir(), f"in_{unique_id}.mp4")
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
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
            frame_size = (int(width), int(height))
            
            # Temporary paths
            temp_output = os.path.join(tempfile.gettempdir(), f"raw_{unique_id}.mp4")
            final_output = os.path.join(tempfile.gettempdir(), f"final_{unique_id}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)
            
            tracker = SportsTracker()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_idx = 0
            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success: break
                    
                    processed_frame = tracker.process_frame(frame)
                    
                    if (processed_frame.shape[1], processed_frame.shape[0]) != frame_size:
                        processed_frame = cv2.resize(processed_frame, frame_size)
                    
                    writer.write(processed_frame)
                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)
                    status_text.text(f"Analyzing: Frame {frame_idx}/{total_frames}")
            finally:
                if 'writer' in locals(): writer.release()
                if 'cap' in locals(): cap.release()
            
            # --- WEB COMPATIBILITY FIX (FFMPEG) ---
            status_text.text("Optimizing video for web playback...")
            # This converts the 'mp4v' file to 'h264' so the browser can play it
            conversion_cmd = f"ffmpeg -i {temp_output} -vcodec libx264 -crf 25 -preset fast {final_output} -y"
            subprocess.run(conversion_cmd, shell=True, capture_output=True)
            
            if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
                status_text.success("Analysis Complete!")
                with open(final_output, 'rb') as v_file:
                    st.video(v_file.read())
                
                # Cleanup
                os.remove(input_path)
                os.remove(temp_output)
                os.remove(final_output)
            else:
                st.error("Error: Video conversion failed. Please try again.")