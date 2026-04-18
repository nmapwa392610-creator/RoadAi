def filter_boxes(detections):
    if not detections:
        return []

    safe = []

    for det in detections:
        if not isinstance(det, dict):
            continue

        # 🔥 защита от KeyError
        if "class_id" not in det:
            continue

        safe.append(det)

    return safe