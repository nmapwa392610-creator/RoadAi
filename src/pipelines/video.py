import subprocess
import numpy as np

from src.pipelines.frame import run_pipeline_frame


def calc_iou(box1: list, box2: list) -> float:
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


def process_video(
    video_path: str,
    fps_target: int = 1,
    memory_lifetime_sec: float = 10.0,
):
    """
    Обработка видео через FFmpeg Pipe.
    Логика полностью совпадает со старой версией,
    но используется более быстрый способ чтения кадров.
    """

    width = 640
    height = 640
    frame_size = width * height * 3

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps_target},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-an",
        "-sn",
        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_size * 2,
    )

    frame_id = 0
    results_all = []

    # Аналог старого кода
    seen_track_ids = set()
    seen_boxes_with_time = []

    # Для корректной статистики
    all_track_ids = set()

    time_step = 1.0 / fps_target

    try:
        while True:
            raw = process.stdout.read(frame_size)

            if not raw or len(raw) < frame_size:
                break

            current_time = round(frame_id * time_step, 2)

            # забываем bbox спустя N секунд
            seen_boxes_with_time = [
                (box, t)
                for box, t in seen_boxes_with_time
                if current_time - t < memory_lifetime_sec
            ]

            seen_boxes = [box for box, _ in seen_boxes_with_time]

            frame = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape((height, width, 3))

            try:
                detections = run_pipeline_frame(
                    frame,
                    use_tracking=True,
                )
            except Exception as e:
                print(
                    f"[VIDEO PIPELINE WARNING] "
                    f"Skipped frame {frame_id}: {e}"
                )
                frame_id += 1
                continue

            if not detections:
                frame_id += 1
                continue

            new_detections = []

            for det in detections:
                tid = det.get("track_id")
                box = det["bbox"]

                if tid and tid != "untracked":

                    # Поведение как в оригинале:
                    # уже найденный track больше не добавляем
                    if tid in seen_track_ids:
                        continue

                    is_dup = any(
                        calc_iou(box, b) >= 0.3
                        for b in seen_boxes
                    )

                    if is_dup:
                        continue

                    seen_track_ids.add(tid)
                    all_track_ids.add(tid)

                    seen_boxes_with_time.append(
                        (box, current_time)
                    )
                    seen_boxes.append(box)

                    new_detections.append(det)

                else:
                    is_dup = any(
                        calc_iou(box, b) >= 0.3
                        for b in seen_boxes
                    )

                    if is_dup:
                        continue

                    seen_boxes_with_time.append(
                        (box, current_time)
                    )
                    seen_boxes.append(box)

                    new_detections.append(det)

            if new_detections:
                results_all.append(
                    {
                        "frame": frame_id,
                        "timestamp": current_time,
                        "detections": new_detections,
                    }
                )

            frame_id += 1

        return {
            "status": "ok",
            "frames_processed": frame_id,
            "unique_defects": len(all_track_ids),
            "results": results_all,
        }

    finally:
        if process.stdout:
            process.stdout.close()

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()