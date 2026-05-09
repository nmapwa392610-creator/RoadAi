from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)

    def predict(self, source):
        """Обычное предсказание для одного кадра"""
        return self.model(source)

    def track(self, source, persist=True):
        """Tracking — каждая яма получает свой ID"""
        return self.model.track(source, persist=persist)