import cv2
import time
from src.pipeline import run_pipeline_frame


class RTSPProcessor:
    def __init__(self, url, frame_skip=5):
        self.url = url
        self.frame_skip = frame_skip
        self.running = False
        self.cap = None

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            print("RTSP not opened")
            return

        frame_id = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                time.sleep(1)
                continue

            if frame_id % self.frame_skip != 0:
                frame_id += 1
                continue

            frame = cv2.resize(frame, (640, 640))

            detections = run_pipeline_frame(frame)

            if detections:
                print("RTSP detections:", detections)

            frame_id += 1

        self.cap.release()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()