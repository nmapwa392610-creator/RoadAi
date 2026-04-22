from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)

    def predict(self, source):
        return self.model(source)