from src.core.detector import Detector
from src.utils.nms import filter_boxes

detector = Detector()

FILTER_THRESHOLD = 0.5


def get_severity(conf):
    if conf > 0.8:
        return "high"
    elif conf > 0.6:
        return "medium"
    return "low"


def run_pipeline_frame(frame, use_tracking=False):
    try:
        # используем tracking если нужно
        if use_tracking:
            results = detector.track(frame, persist=True)
        else:
            results = detector.predict(frame)

        r = results[0]
        detections = []

        for box in r.boxes:
            conf = float(box.conf)

            if conf < FILTER_THRESHOLD:
                continue

            cls_id = int(box.cls)
            cls_name = detector.model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)

            # track_id только если tracking включён
            track_id = None
            if use_tracking and box.id is not None:
                track_id = int(box.id)

            detections.append({
                "track_id": track_id if track_id is not None else "untracked",
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
                "area": area,
                "severity": get_severity(conf)
            })

        return filter_boxes(detections)

    except Exception as e:
        return {
            "error": str(e),
            "where": "pipeline_frame"
        }