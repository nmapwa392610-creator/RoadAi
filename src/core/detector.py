import torch
from ultralytics import YOLO

# FIX CPU backend crash
torch.backends.mkldnn.enabled = False

try:
    torch.backends.nnpack.enabled = False
except Exception:
    pass


class Detector:
    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)

    def predict(self, image):
        if image is None:
            raise ValueError("Empty image input")

        return self.model.predict(image, verbose=False, device="cpu")

    def track(self, image, persist=True):
        if image is None:
            raise ValueError("Empty image input")

        return self.model.track(image, persist=persist, verbose=False, device="cpu")