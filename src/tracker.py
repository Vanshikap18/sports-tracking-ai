import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class SportsTracker:
    """
    A class to handle object detection and temporal tracking for sports footage.
    Uses YOLOv8 for detection and BoT-SORT for persistent ID assignment.
    """
    
    def __init__(self, model_name='yolov8m.pt'):
        """
        Initializes the tracker with a pre-trained YOLOv8 model.
        """
        self.model = YOLO(model_name)
        # Store historical center points for trajectory visualization
        self.track_history = defaultdict(lambda: [])

    def process_frame(self, frame):
        """
        Processes a single video frame to detect subjects, update tracks,
        and visualize movement trajectories.
        """
        # Perform multi-object tracking with BoT-SORT (handles camera motion compensation)
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml", 
            verbose=False
        )[0]
        
        # Check if active tracks exist in the current frame
        if results.boxes.id is not None:
            boxes = results.boxes.xywh.cpu()
            track_ids = results.boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                # Track the bottom-center point (base of the subject) for paths
                self.track_history[track_id].append((float(x), float(y + h/2)))
                
                # Maintain a rolling window of the last 30 frames to limit path length
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)

                # Draw movement trajectories on the frame (Enhancement)
                points = np.hstack(self.track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

        # Generate the standard YOLO annotations (bounding boxes and IDs)
        annotated_frame = results.plot()
        return annotated_frame