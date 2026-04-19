import cv2
from src.detector import Detector
from utils.nms import filter_boxes

detector = Detector()


def get_severity(conf):
    if conf > 0.8:
        return "high"
    elif conf > 0.6:
        return "medium"
    return "low"


# ----------------------------
# IMAGE PIPELINE
# ----------------------------
def run_pipeline_image(image_path):
    try:
        results = detector.predict(image_path)
        r = results[0]

        detections = []

        for box in r.boxes:
            conf = float(box.conf)

            if conf < 0.15:
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

        return filter_boxes(detections)

    except Exception as e:
        return {"error": str(e), "where": "pipeline_image"}


# ----------------------------
# FRAME PIPELINE
# ----------------------------
def run_pipeline_frame(frame):
    try:
        results = detector.predict(frame)
        r = results[0]

        detections = []

        print("BOXES:", len(r.boxes))  # DEBUG

        for box in r.boxes:
            conf = float(box.conf)

            print("CONF:", conf)  # DEBUG

            if conf < 0.15:
                continue

            cls_id = int(box.cls)
            cls_name = detector.model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)

            # 🔥 ВАЖНО: class_id ОБЯЗАТЕЛЬНО
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
                "area": area,
                "severity": get_severity(conf)
            })

        return filter_boxes(detections)

    except Exception as e:
        return {"error": str(e), "where": "pipeline_frame"}


# ----------------------------
# VIDEO PIPELINE
# ----------------------------
def process_video(video_path, frame_skip=20):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return {"error": "video not opened", "path": video_path}

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_id = 0
    results_all = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, (640, 640))

        detections = run_pipeline_frame(frame)

        # 🔥 ВАЖНО: защита от мусора
        if isinstance(detections, list) and detections:
            results_all.append({
                "frame": frame_id,
                "timestamp": frame_id / fps if fps else None,
                "detections": detections
            })

        frame_id += 1

    cap.release()
    return results_all

# -------------------------
# RTSP (SIMPLE VERSION)
# -------------------------
_rtsp_cap = None


def start_rtsp_stream(url):
    global _rtsp_cap
    _rtsp_cap = cv2.VideoCapture(url)
    return {"status": "rtsp started"}


def stop_rtsp_stream():
    global _rtsp_cap
    if _rtsp_cap:
        _rtsp_cap.release()
    return {"status": "rtsp stopped"}


def get_live_frame():
    global _rtsp_cap

    if _rtsp_cap is None:
        return {"error": "rtsp not started"}

    ret, frame = _rtsp_cap.read()

    if not ret:
        return {"error": "frame not received"}

    detections = run_pipeline_frame(frame)

    return {
        "detections": detections
    }