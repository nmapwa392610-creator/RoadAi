import threading
from src.core.detector import Detector
from src.utils.nms import filter_boxes

detector = None
lock = threading.Lock()


def get_detector():
    global detector
    with lock:
        if detector is None:
            detector = Detector()
    return detector


# Минимальный порог уверенности — детекции ниже этого значения игнорируются
FILTER_THRESHOLD = 0.4


def get_severity(conf: float) -> str:
    # Определяем серьёзность дефекта по уверенности модели
    if conf > 0.8:
        return "high"
    elif conf > 0.6:
        return "medium"
    return "low"


def run_pipeline_image(image):
    try:
        # 1. Получаем инстанс детектора ОДИН раз
        detector_instance = get_detector()

        # 2. Запускаем на GPU, чтобы избежать бага primitives на CPU сервера
        results = detector_instance.predict(image)
        r = results[0]

        # Кэшируем имена классов модели
        names = detector_instance.model.names

        detections = []
        for box in r.boxes:
            conf = float(box.conf)
            # пропуск слабых дефекций
            if conf < FILTER_THRESHOLD:
                continue

            cls_id = int(box.cls)
            # 3. ИСПРАВЛЕНО: берем имя класса из локальной переменной names
            cls_name = names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
                "area": area,
                "severity": get_severity(conf)
            })

        # Финальная фильтрация через NMS — убираем перекрывающиеся боксы
        cleaned = filter_boxes(detections)
        return {
            "detections": cleaned,
            "count": len(cleaned),
            "status": "ok"
        }
    except Exception as e:
        return {
            "error": str(e),
            "where": "pipeline_image"
        }
