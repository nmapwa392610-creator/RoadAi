import cv2
import threading
from src.pipelines.frame import run_pipeline_frame


def start_rtsp_stream(url, frame_callback, frame_skip=20):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        raise Exception("RTSP stream not opened")

    stop_flag = {"running": True}
    seen_track_ids = set()  # запоминаем ямы которые уже видели
    frame_id = 0

    def loop():
        nonlocal frame_id

        while stop_flag["running"]:
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # пропускаем кадры
            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            frame = cv2.resize(frame, (640, 640))

            # tracking
            detections = run_pipeline_frame(frame, use_tracking=True)

            if detections and isinstance(detections, list):
                # фильтруем дубликаты по track_id
                new_detections = []
                for det in detections:
                    tid = det.get("track_id")
                    if tid is None:
                        new_detections.append(det)
                    elif tid not in seen_track_ids:
                        seen_track_ids.add(tid)
                        new_detections.append(det)

                # отправляем только новые ямы
                if new_detections:
                    frame_callback(new_detections)

            frame_id += 1

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    return {
        "cap": cap,
        "thread": t,
        "stop_flag": stop_flag
    }


def stop_rtsp_stream(stream):
    stream["stop_flag"]["running"] = False

    cap = stream.get("cap")
    if cap:
        cap.release()