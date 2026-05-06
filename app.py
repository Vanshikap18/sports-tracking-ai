import streamlit as st
import cv2
import os
import tempfile
import sys
import time
import uuid

# Maintain system paths for module discovery
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tracker import SportsTracker
from utils import get_video_properties

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
    # Use a unique ID for this session
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
            
            # 1. FIX: Ensure dimensions are integers and consistent
            frame_size = (int(width), int(height))
            
            # 2. FIX: Use 'mp4v' codec for maximum compatibility on Linux Cloud servers
            output_path = os.path.join(tempfile.gettempdir(), f"out_{unique_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            
            tracker = SportsTracker()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_idx = 0
            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    processed_frame = tracker.process_frame(frame)
                    
                    # 3. FIX: Force resize processed frame to match writer dimensions
                    # This prevents 0-byte files if the tracker alters dimensions
                    if (processed_frame.shape[1], processed_frame.shape[0]) != frame_size:
                        processed_frame = cv2.resize(processed_frame, frame_size)
                    
                    writer.write(processed_frame)
                    
                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)
                    status_text.text(f"Analyzing: Frame {frame_idx}/{total_frames}")
            finally:
                # Always release handles to unlock the file
                if 'writer' in locals():
                    writer.release()
                if 'cap' in locals():
                    cap.release()
            
            # Give the OS time to finalize the file write
            time.sleep(2)
            
            # Verify file existence AND size before proceeding
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                status_text.success("Analysis Complete!")
                with open(output_path, 'rb') as v_file:
                    video_bytes = v_file.read()
                st.video(video_bytes, format="video/mp4")
                
                # Cleanup
                try:
                    os.remove(input_path)
                    os.remove(output_path)
                except:
                    pass
            else:
                st.error(f"Technical Error: Encoder failed. Result at {output_path} is empty. Check codec support.")