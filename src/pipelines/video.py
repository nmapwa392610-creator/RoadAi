import cv2
from src.pipelines.frame import run_pipeline_frame


def process_video(video_path, frame_skip=20):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return {"error": "video not opened", "path": video_path}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    frame_id = 0
    results_all = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            frame = cv2.resize(frame, (640, 640))

            detections = run_pipeline_frame(frame)

            if detections:
                results_all.append({
                    "frame": frame_id,
                    "timestamp": frame_id / fps if fps else None,
                    "detections": detections
                })

            frame_id += 1

        return {
            "status": "ok",
            "frames_processed": frame_id,
            "results": results_all
        }

    finally:
        cap.release()