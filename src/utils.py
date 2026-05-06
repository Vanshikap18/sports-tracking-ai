import cv2

def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return width, height, fps

def create_video_writer(output_path, fps, size):
    """
    Initializes a video writer with H.264 encoding for web compatibility.
    """
    # 'avc1' or 'mp4v' are standard, but 'avc1' is better for web playback
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    return cv2.VideoWriter(output_path, fourcc, fps, size)   
    