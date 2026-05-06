import streamlit as st
import cv2
import os
import tempfile
import sys

# Ensure the 'src' directory is in the system path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tracker import SportsTracker
from utils import get_video_properties, create_video_writer

# Professional Dark & Purple Theme Injection
st.set_page_config(page_title="AI Sports Tracker", layout="wide")

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 2px solid #7D4CDB;
    }

    /* Force visibility for ALL labels and text descriptions */
    label, p, .stMarkdown, [data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
    }

    /* SPECIFIC FIX: Button text visibility */
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

    /* Success and Status messages visibility */
    .stAlert p, .stText p {
        color: #FFFFFF !important;
    }

    /* Titles */
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
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Stream")
        st.video(tfile.name)
    
    if st.button("Run AI Analysis"):
        with col2:
            st.subheader("Tracking Result")
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            cap = cv2.VideoCapture(tfile.name)
            width, height, fps = get_video_properties(cap)
            
            # Using a consistent output name
            output_path = "data/output/web_result.mp4"
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
            
            cap.release()
            writer.release()
            
            # Final visualization without download button
            status_text.success("Analysis Complete!")
            st.video(output_path)