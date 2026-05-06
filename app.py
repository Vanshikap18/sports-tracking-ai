import streamlit as st
import cv2
import os
import tempfile
import sys
import time
import uuid
import av  # Final fix for web-ready encoding

# Maintain system paths for module discovery
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tracker import SportsTracker
from utils import get_video_properties

# ... (Keep your existing Theme styling and Page Config) ...

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
            
            # Final output path for web-ready video
            output_path = os.path.join(tempfile.gettempdir(), f"final_{unique_id}.mp4")
            
            # Setup PyAV container for H.264 encoding
            container = av.open(output_path, mode='w')
            stream = container.add_stream('libx264', rate=int(fps))
            stream.width = int(width)
            stream.height = int(height)
            stream.pix_fmt = 'yuv420p' # Standard format for web players
            
            tracker = SportsTracker()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_idx = 0
            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    processed_frame = tracker.process_frame(frame)
                    
                    # Convert BGR (OpenCV) to RGB (PyAV/Web)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    av_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
                    
                    # Encode and write frame
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                    
                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)
                    status_text.text(f"Analyzing: Frame {frame_idx}/{total_frames}")
                
                # Flush encoder
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                cap.release()
                
                # Final check and display
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