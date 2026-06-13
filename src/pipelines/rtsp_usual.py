import cv2
import threading
import logging
import time
from typing import Callable, Optional, Dict, Tuple, List
from src.pipelines.frame import run_pipeline_frame

logger = logging.getLogger(__name__)


class RTSPStream:
    def __init__(
        self,
        url: str,
        reconnect: bool = True,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
        frame_skip: int = 20,
        memory_lifetime: float = 10.0,
    ):
        self.url = url
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.frame_skip = frame_skip
        self.memory_lifetime = memory_lifetime

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._is_running = False
        self._reconnect_count = 0

        # memory
        self._seen_tracks: Dict[int, float] = {}
        self._seen_boxes: List[Tuple[list, float]] = []

    # ---------------- PUBLIC ----------------

    def start(self, frame_callback: Callable) -> "RTSPStream":
        if self._is_running:
            return self

        self._stop_event.clear()
        self._is_running = True

        self._thread = threading.Thread(
            target=self._loop,
            args=(frame_callback,),
            daemon=True,
        )
        self._thread.start()

        logger.info("RTSP started: %s", self.url)
        return self

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        self._is_running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

        self._release()

    # ---------------- LOOP ----------------

    def _loop(self, frame_callback: Callable):
        frame_id = 0

        while not self._stop_event.is_set():
            if not self._open():
                if not self._reconnect():
                    break
                continue

            self._reconnect_count = 0

            while not self._stop_event.is_set():
                if frame_id % self.frame_skip != 0:
                    self._cap.grab()
                    frame_id += 1
                    continue

                ok, frame = self._cap.read()
                frame_id += 1

                if not ok:
                    self._release()
                    if not self._reconnect():
                        return
                    break

                try:
                    frame = cv2.resize(frame, (640, 640))

                    detections = run_pipeline_frame(frame, use_tracking=True)

                    if detections:
                        filtered = self._dedup(detections)

                        if filtered:
                            frame_callback(filtered)

                except Exception:
                    logger.exception("Frame processing error")

        self._is_running = False

    # ---------------- DEDUP ----------------

    def _iou(self, a, b) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        return inter / (area_a + area_b - inter)

    def _clean_memory(self, now: float):
        self._seen_tracks = {
            k: v for k, v in self._seen_tracks.items()
            if now - v < self.memory_lifetime
        }

        self._seen_boxes = [
            (b, t) for b, t in self._seen_boxes
            if now - t < self.memory_lifetime
        ]

    def _is_duplicate(self, box) -> bool:
        return any(self._iou(box, old[0]) > 0.4 for old in self._seen_boxes)

    def _dedup(self, detections: list) -> list:
        now = time.time()
        self._clean_memory(now)

        out = []

        for d in detections:
            box = d["bbox"]
            tid = d.get("track_id")

            # track-based
            if tid and tid != "untracked":
                if tid in self._seen_tracks:
                    self._seen_tracks[tid] = now
                    out.append(d)
                    continue

                if self._is_duplicate(box):
                    continue

                self._seen_tracks[tid] = now
                self._seen_boxes.append((box, now))
                out.append(d)

            else:
                if self._is_duplicate(box):
                    continue

                self._seen_boxes.append((box, now))
                out.append(d)

        return out

    # ---------------- CAMERA ----------------

    def _open(self) -> bool:
        with self._lock:
            src = int(self.url) if self.url.isdigit() else self.url

            self._cap = cv2.VideoCapture(src)

            if not self._cap.isOpened():
                self._cap = None
                return False

            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True

    def _release(self):
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    def _reconnect(self) -> bool:
        if not self.reconnect:
            return False

        if self._reconnect_count >= self.max_reconnects:
            return False

        self._reconnect_count += 1
        time.sleep(self.reconnect_delay)
        return True


# ---------------- BACKWARD COMPAT (ВАЖНО) ----------------

def start_rtsp_stream(url: str, frame_callback: Callable) -> RTSPStream:
    return RTSPStream(url).start(frame_callback)


def stop_rtsp_stream(stream: RTSPStream) -> None:
    stream.stop()