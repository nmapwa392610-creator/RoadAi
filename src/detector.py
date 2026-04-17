from pathlib import Path
from ultralytics import YOLO

class Detector:
    def __init__(self):
        base = Path(__file__).resolve().parent.parent
        model_path = base / "models" / "best.pt"
        self.model = YOLO(str(model_path))

    def predict(self, frame):
        return self.model(frame)