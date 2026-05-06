import cv2
import os
import sys

# Ensure the 'src' directory is in the system path for module discovery
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tracker import SportsTracker
from utils import get_video_properties, create_video_writer

def main():
    """
    Primary entry point for the Multi-Object Tracking pipeline.
    Handles video I/O initialization and execution of the tracking loop.
    """
    # Define input and output file paths
    input_video = "data/input/sample_video.mp4" 
    output_video = "data/output/annotated_result.mp4"
    
    # Validate the existence of the source video file
    if not os.path.exists(input_video):
        print(f"CRITICAL ERROR: Input video not found at: {input_video}")
        print("Please ensure the source file is correctly placed in the data/input directory.")
        return

    # Initialize video capture and retrieve stream properties
    cap = cv2.VideoCapture(input_video)
    width, height, fps = get_video_properties(cap)
    
    # Initialize the video writer for output generation
    writer = create_video_writer(output_video, fps, (width, height))
    
    # Initialize the Computer Vision tracking engine
    tracker = SportsTracker()

    print(f"Starting inference pipeline for: {input_video}")
    frame_index = 0

    # Main processing loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # End of video stream reached
            break

        # Apply detection and tracking logic to the current frame
        processed_frame = tracker.process_frame(frame)
        
        # Write the annotated frame to the output file
        writer.write(processed_frame)
        
        frame_index += 1
        # Log progress every 30 frames for monitoring
        if frame_index % 30 == 0:
            print(f"Progress: {frame_index} frames processed...")

    # Release resources and finalize output file
    cap.release()
    writer.release()
    print(f"Inference complete. Annotated video saved to: {output_video}")

if __name__ == "__main__":
    main()