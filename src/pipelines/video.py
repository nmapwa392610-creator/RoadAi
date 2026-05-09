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


def process_video(video_path, frame_skip=20):
    # Открываем видео через FFMPEG
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return {"error": "video not opened", "path": video_path}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    frame_id = 0
    results_all = []
    seen_track_ids = set()  # уже виденные track_id
    seen_boxes = []         # уже виденные bbox — для untracked случаев

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Обрабатываем только каждый frame_skip-й кадр — экономим ресурсы
            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            frame = cv2.resize(frame, (640, 640))

            # Запускаем детекцию с tracking — каждая яма получает track_id
            detections = run_pipeline_frame(frame, use_tracking=True)

            if detections and isinstance(detections, list):
                new_detections = []

                for det in detections:
                    tid = det.get("track_id")
                    box = det["bbox"]

                    if tid is not None and tid != "untracked":
                        if tid not in seen_track_ids:
                            # Проверяем по IoU — вдруг эту яму уже видели как untracked
                            is_dup = any(calc_iou(box, seen) >= 0.4 for seen in seen_boxes)
                            if not is_dup:
                                seen_track_ids.add(tid)
                                seen_boxes.append(box)
                                new_detections.append(det)
                            else:
                                # Дубликат untracked — просто запоминаем track_id
                                seen_track_ids.add(tid)

                    else:
                        # track_id нет — проверяем только по IoU
                        is_dup = any(calc_iou(box, seen) >= 0.4 for seen in seen_boxes)
                        if not is_dup:
                            seen_boxes.append(box)
                            new_detections.append(det)

                if new_detections:
                    results_all.append({
                        "frame": frame_id,
                        "timestamp": frame_id / fps if fps else None,
                        "detections": new_detections
                    })

            frame_id += 1

        return {
            "status": "ok",
            "frames_processed": frame_id,
            "unique_defects": len(seen_boxes),  # количество уникальных ям во всём видео
            "results": results_all
        }

    finally:
        # Освобождаем ресурсы даже если произошла ошибка
        cap.release()