import math

def filter_boxes(detections):
    if not detections:
        return []

    safe = []

    for det in detections:
        if not isinstance(det, dict):
            continue

        required = ["bbox", "confidence", "class_id"]
        if not all(k in det for k in required):
            continue

        conf = det.get("confidence", 0)

        # ❌ мусорные значения
        if not isinstance(conf, (int, float)):
            continue

        if math.isnan(conf) or math.isinf(conf):
            continue

        if conf < 0.1:
            continue

        bbox = det.get("bbox")

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        # проверка координат
        if any(not isinstance(x, (int, float)) for x in bbox):
            continue

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            continue

        safe.append(det)

    return safe