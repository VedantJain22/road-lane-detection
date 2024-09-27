import cv2
import numpy as np
from utils import region_of_interest, hough_lines, preprocess_frame
from cnn_model import LaneCNN

class LaneDetector:
    def __init__(self):
        self.model = LaneCNN().load_model('models/lane_cnn_model.h5')

    def detect_lanes(self, frame):
        preprocessed = preprocess_frame(frame)
        edges = cv2.Canny(preprocessed, 50, 150)
        
        # Define region of interest
        roi = region_of_interest(edges)
        
        # Hough Transform to detect lines
        lines = hough_lines(roi)
        
        # Use CNN to filter lane lines
        lanes = self.model.predict(lines)
        
        return lanes

if __name__ == "__main__":
    cap = cv2.VideoCapture('data/video.mp4')
    lane_detector = LaneDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lanes = lane_detector.detect_lanes(frame)
        cv2.imshow('Lane Detection', lanes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
