import streamlit as st
import cv2
import os
import tempfile
import sys
import time
import uuid
import av # Ensure 'av' is in requirements.txt

# Maintain system paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tracker import SportsTracker
from utils import get_video_properties

st.set_page_config(page_title="AI Sports Tracker", layout="wide")

# Theme styling
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 2px solid #7D4CDB; }
    .stButton>button { background-color: #7D4CDB; color: #FFFFFF !important; border-radius: 8px; width: 100%; height: 3em; }
    h1, h2, h3 { color: #9B6DFF !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("AI Sports Tracker: Pro Dashboard")

# Configuration
st.sidebar.header("Configuration")
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
            
            output_path = os.path.join(tempfile.gettempdir(), f"final_{unique_id}.mp4")
            
            cap = cv2.VideoCapture(input_path)
            width, height, fps = get_video_properties(cap)
            
            # Setup PyAV container for H.264 (Web-friendly)
            container = av.open(output_path, mode='w')
            stream = container.add_stream('libx264', rate=int(fps))
            stream.width = int(width)
            stream.height = int(height)
            stream.pix_fmt = 'yuv420p'
            
            tracker = SportsTracker()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_idx = 0
            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success: break
                    
                    processed_frame = tracker.process_frame(frame)
                    
                    # Convert for PyAV
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    av_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
                    
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                    
                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)
                    status_text.text(f"Analyzing: Frame {frame_idx}/{total_frames}")
                
                # Close encoder
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                cap.release()
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    status_text.success("Analysis Complete!")
                    with open(output_path, 'rb') as v_file:
                        st.video(v_file.read())
                else:
                    st.error("Error: Video generation failed.")
            except Exception as e:
                st.error(f"Technical Error: {e}")
            finally:
                if 'cap' in locals(): cap.release()