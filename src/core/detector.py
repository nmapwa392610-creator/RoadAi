from ultralytics import YOLO


class Detector:

    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)

    def predict(self, source):
        # Обычное предсказание — без отслеживания объектов между кадрами
        return self.model(source)

    def track(self, source, persist=True):
        # Предсказание с трекингом — каждая яма получает уникальный track_id
        # persist=True сохраняет ID между кадрами одного видео/потока
        return self.model.track(source, persist=persist)