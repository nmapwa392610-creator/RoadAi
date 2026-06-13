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


FILTER_THRESHOLD = 0.4


def get_severity(conf: float) -> str:
    if conf > 0.8:
        return "high"
    elif conf > 0.6:
        return "medium"
    return "low"


def run_pipeline_frame(frame, use_tracking: bool = False):
    # 1. ИСПРАВЛЕНО: Изменили имя на detector_instance, чтобы не затирать глобальный detector
    detector_instance = get_detector()
    try:
        # Кэшируем имена классов сразу, чтобы не лезть в модель в цикле
        names = detector_instance.model.names

        # ---------------------------
        # INFERENCE (Явно передаем device="cuda")
        # ---------------------------
        if use_tracking:
            results = detector_instance.track(frame, persist=True)
        else:
            results = detector_instance.predict(frame)

        r = results[0]

        # если нет боксов — сразу выход
        if r.boxes is None or len(r.boxes) == 0:
            return []

        # ---------------------------
        # OPTIMIZED EXTRACTION
        # ---------------------------
        xyxy_np = r.boxes.xyxy.cpu().numpy()
        conf_np = r.boxes.conf.cpu().numpy()
        cls_np = r.boxes.cls.cpu().numpy()

        has_ids = use_tracking and hasattr(r.boxes, "id") and r.boxes.id is not None
        id_np = r.boxes.id.cpu().numpy() if has_ids else None

        detections = []
        for i in range(len(xyxy_np)):
            score = float(conf_np[i])
            if score < FILTER_THRESHOLD:
                continue

            cls_id = int(cls_np[i])
            # 2. ИСПРАВЛЕНО: берем имена из локального кэша names
            cls_name = names[cls_id]

            x1, y1, x2, y2 = xyxy_np[i]
            area = float((x2 - x1) * (y2 - y1))

            track_id = int(id_np[i]) if has_ids else None

            detections.append({
                "track_id": track_id if track_id is not None else "untracked",
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": score,
                "class_id": cls_id,
                "class_name": cls_name,
                "area": area,
                "severity": get_severity(score)
            })

        return filter_boxes(detections)

    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")
        return {
            "status": "error",
            "error": str(e),
            "where": "run_pipeline_frame"
        }
