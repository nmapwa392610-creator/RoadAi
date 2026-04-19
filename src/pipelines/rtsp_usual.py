import cv2
import threading


def start_rtsp_stream(url, frame_callback):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        raise Exception("RTSP stream not opened")

    stop_flag = {"running": True}

    def loop():
        while stop_flag["running"]:
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            frame_callback(frame)

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