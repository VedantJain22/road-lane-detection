import cv2
from lane_detector.py import LaneDetector

def process_video_feed(video_source=0):
    cap = cv2.VideoCapture(video_source)
    lane_detector = LaneDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lanes = lane_detector.detect_lanes(frame)
        cv2.imshow('Lane Detection', lanes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_feed()
