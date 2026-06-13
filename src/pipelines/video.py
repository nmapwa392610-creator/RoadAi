import cv2
import threading
from src.core.detector import Detector
from src.utils.nms import filter_boxes
from src.pipelines.frame import run_pipeline_frame

detector = None
lock = threading.Lock()


def get_detector():
    global detector
    with lock:
        if detector is None:
            detector = Detector()
    return detector


FILTER_THRESHOLD = 0.5


def get_severity(conf: float) -> str:
    if conf > 0.8: return "high"
    if conf > 0.6: return "medium"
    return "low"


def run_pipeline_image(image):
    try:
        # 1. Получаем инстанс ОДИН раз
        detector_instance = get_detector()

        # Кэшируем имена, чтобы не заходить под Lock в цикле
        names = detector_instance.model.names

        # 2. Жестко привязываем к CUDA для сервера Ubuntu
        results = detector_instance.predict(image)
        r = results[0]

        detections = []
        # Выгружаем боксы на CPU/NumPy сразу
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy_array = r.boxes.xyxy.cpu().numpy()
            conf_array = r.boxes.conf.cpu().numpy()
            cls_array = r.boxes.cls.cpu().numpy()

            for i in range(len(xyxy_array)):
                conf = float(conf_array[i])
                if conf < FILTER_THRESHOLD:
                    continue

                cls_id = int(cls_array[i])
                # ИСПРАВЛЕНО: берем из кэшированной переменной, БЕЗ вызова get_detector()
                cls_name = names[cls_id]

                x1, y1, x2, y2 = xyxy_array[i]
                area = (x2 - x1) * (y2 - y1)

                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "area": float(area),
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


def calc_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def process_video(video_path, frame_skip=60):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return {"error": "video not opened", "path": video_path}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_id = 0
    results_all = []
    seen_track_ids = set()
    FORGET_AFTER_FRAMES = int(fps * 10)
    seen_boxes_with_frame = []  # [(box, frame_id), ...]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            # Очищаем старые bbox, которые за пределами 10-секундного окна
            seen_boxes_with_frame = [
                (b, fid) for b, fid in seen_boxes_with_frame if frame_id - fid < FORGET_AFTER_FRAMES
            ]
            seen_boxes = [b for b, _ in seen_boxes_with_frame]

            frame_resized = cv2.resize(frame, (640, 640))

            try:
                # Внутри run_pipeline_frame тоже должен быть device="cuda"
                detections = run_pipeline_frame(frame_resized, use_tracking=True)
            except Exception as e:
                print(f"Warning: skipped corrupted frame {frame_id} due to error: {e}")
                frame_id += 1
                continue

            if not isinstance(detections, list) or len(detections) == 0:
                frame_id += 1
                continue

            new_detections = []
            for det in detections:
                tid = det.get("track_id")
                box = det["bbox"]

                if tid and tid != "untracked":
                    if tid in seen_track_ids:
                        continue
                    is_dup = any(calc_iou(box, b) >= 0.3 for b in seen_boxes)
                    if not is_dup:
                        seen_track_ids.add(tid)
                        seen_boxes_with_frame.append((box, frame_id))
                        seen_boxes.append(box)  # Теперь это безопасно добавляется в локальный список кадра
                        new_detections.append(det)
                else:
                    is_dup = any(calc_iou(box, b) >= 0.3 for b in seen_boxes)
                    if not is_dup:
                        seen_boxes_with_frame.append((box, frame_id))
                        seen_boxes.append(box)
                        new_detections.append(det)

            if new_detections:
                results_all.append({
                    "frame": frame_id,
                    "timestamp": round(frame_id / fps, 2),
                    "detections": new_detections,
                })
            frame_id += 1

        return {
            "status": "ok",
            "frames_processed": frame_id,
            "unique_defects": len(seen_track_ids),
            "results": results_all,
        }
    finally:
        cap.release()
