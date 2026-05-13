import cv2
from src.pipelines.frame import run_pipeline_frame


def calc_iou(box1, box2) -> float:
    # Считаем overlap между двумя bbox — чем больше значение тем сильнее перекрытие
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


def process_video(video_path, frame_skip=60):  # каждые 2 сек при 30fps
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return {"error": "video not opened", "path": video_path}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_id = 0
    results_all = []
    seen_track_ids = set()
    seen_boxes = []

    # Сколько кадров держать bbox в памяти (10 секунд)
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

            # Забываем старые bbox — камера уже уехала
            seen_boxes_with_frame = [
                (b, fid) for b, fid in seen_boxes_with_frame
                if frame_id - fid < FORGET_AFTER_FRAMES
            ]
            seen_boxes = [b for b, _ in seen_boxes_with_frame]

            frame_resized = cv2.resize(frame, (640, 640))
            detections = run_pipeline_frame(frame_resized, use_tracking=True)

            if not detections or not isinstance(detections, list):
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
                        seen_boxes.append(box)
                        new_detections.append(det)
                    else:
                        seen_track_ids.add(tid)
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
            "unique_defects": len(seen_boxes),
            "results": results_all,
        }
    finally:
        cap.release()