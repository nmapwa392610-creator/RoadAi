from src.core.detector import Detector
from src.utils.nms import filter_boxes

FILTER_THRESHOLD = 0.25

detector = Detector()


def get_severity(conf):
    if conf > 0.8:
        return "high"
    elif conf > 0.6:
        return "medium"
    return "low"


def run_pipeline_image(image_path):
    try:
        results = detector.predict(image_path)
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

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
                "area": area,
                "severity": get_severity(conf)
            })

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