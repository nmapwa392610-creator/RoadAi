class RTSPProcessor:
    def __init__(self, url, frame_skip=5):
        self.url = url
        self.frame_skip = frame_skip
        self.running = False

    def start(self):
        import cv2
        import time
        from src.pipeline import run_pipeline_frame  # 👈 ВНУТРИ метода

        self.running = True
        cap = cv2.VideoCapture(self.url)

        frame_id = 0

        while self.running:
            ret, frame = cap.read()

            if not ret:
                time.sleep(1)
                continue

            if frame_id % self.frame_skip != 0:
                frame_id += 1
                continue

            frame = cv2.resize(frame, (640, 640))

            detections = run_pipeline_frame(frame)

            if detections:
                print(detections)

            frame_id += 1

        cap.release()