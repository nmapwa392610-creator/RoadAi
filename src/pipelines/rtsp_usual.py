import subprocess
import threading
import logging
import time
import numpy as np
from typing import Callable, Optional, Dict, List, Tuple
from queue import Queue, Empty

from src.pipelines.frame import run_pipeline_frame

logger = logging.getLogger(__name__)


class RTSPStreamV3:
    def __init__(
        self,
        url: str,
        fps_target: int = 2,
        memory_lifetime: float = 10.0,
        queue_size: int = 30,
        max_reconnects: int = 10,
        reconnect_delay: float = 3.0,
    ):
        self.url = url
        self.fps_target = fps_target
        self.memory_lifetime = memory_lifetime

        self.queue = Queue(maxsize=queue_size)

        self.max_reconnects = max_reconnects
        self.reconnect_delay = reconnect_delay

        self._process: Optional[subprocess.Popen] = None

        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._ffmpeg_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._running = False
        self._reconnect_count = 0

        self.width, self.height = 640, 640
        self.frame_size = self.width * self.height * 3

        # dedup memory
        self._seen_tracks: Dict[int, float] = {}
        self._seen_boxes: List[Tuple[list, float]] = []
        self._dedup_lock = threading.Lock()

    # ---------------- PUBLIC ----------------

    def start(self, callback: Callable[[dict], None]) -> "RTSPStreamV3":
        with self._lock:
            if self._running:
                return self

            self._stop_event.clear()
            self._running = True

            # очистка очереди
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Empty:
                    break

            self._ffmpeg_thread = threading.Thread(target=self._ffmpeg_loop, daemon=True)
            self._worker_thread = threading.Thread(target=self._worker_loop, args=(callback,), daemon=True)

            self._ffmpeg_thread.start()
            self._worker_thread.start()

            logger.info("RTSP V3 started: %s", self.url)
            return self

    def stop(self):
        with self._lock:
            if not self._running:
                return
            self._stop_event.set()
            self._running = False

        self._release()

        if self._ffmpeg_thread:
            self._ffmpeg_thread.join(timeout=3)
        if self._worker_thread:
            self._worker_thread.join(timeout=3)

        logger.info("RTSP V3 stopped: %s", self.url)

    # ---------------- FFmpeg ----------------

    def _ffmpeg_loop(self):
        while not self._stop_event.is_set():

            if not self._open():
                if not self._reconnect():
                    break
                continue

            self._reconnect_count = 0

            while not self._stop_event.is_set():
                if not self._process or not self._process.stdout:
                    break

                raw = self._process.stdout.read(self.frame_size)

                if not raw or len(raw) < self.frame_size:
                    logger.warning("RTSP broken → reconnecting")
                    if not self._reconnect():
                        return
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )

                ts = time.time()

                # backpressure
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass

                self.queue.put((frame, ts))

    # ---------------- WORKER ----------------

    def _worker_loop(self, callback: Callable[[dict], None]):
        frame_id = 0

        while not self._stop_event.is_set():
            try:
                frame, ts = self.queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                detections = run_pipeline_frame(frame, use_tracking=True)

                if detections:
                    filtered = self._dedup(detections, ts)

                    if filtered:
                        callback({
                            "frame": frame_id,
                            "timestamp": round(ts, 2),
                            "detections": filtered,
                        })

            except Exception as e:
                logger.exception(f"RTSP worker error: {e}")

            frame_id += 1

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

    def _clean(self, now: float):
        self._seen_tracks = {
            k: v for k, v in self._seen_tracks.items()
            if now - v < self.memory_lifetime
        }
        self._seen_boxes = [
            (b, t) for b, t in self._seen_boxes
            if now - t < self.memory_lifetime
        ]

    def _is_dup(self, box) -> bool:
        return any(self._iou(box, b[0]) > 0.4 for b in self._seen_boxes)

    def _dedup(self, detections: list, now: float) -> list:
        with self._dedup_lock:
            self._clean(now)

            out = []

            for d in detections:
                box = d["bbox"]
                tid = d.get("track_id")

                if tid and tid != "untracked":
                    if tid in self._seen_tracks:
                        self._seen_tracks[tid] = now
                        out.append(d)
                        continue

                    if self._is_dup(box):
                        continue

                    self._seen_tracks[tid] = now
                    self._seen_boxes.append((box, now))
                    out.append(d)

                else:
                    if self._is_dup(box):
                        continue

                    self._seen_boxes.append((box, now))
                    out.append(d)

            return out

    # ---------------- FFmpeg ----------------

    def _open(self) -> bool:
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", self.url,
            "-vf", f"fps={self.fps_target},scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-sn",
            "pipe:1",
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_size,
            )
            return True
        except Exception as e:
            logger.error(f"FFmpeg open error: {e}")
            return False

    def _release(self):
        if self._process:
            try:
                if self._process.stdout:
                    self._process.stdout.close()
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                pass
            self._process = None

    def _reconnect(self) -> bool:
        self._release()

        if self._reconnect_count >= self.max_reconnects:
            logger.error("RTSP permanently failed after max retries")
            return False

        self._reconnect_count += 1
        time.sleep(self.reconnect_delay)
        return True


def start_rtsp_stream(url: str, callback: Callable):
    return RTSPStreamV3(url).start(callback)


def stop_rtsp_stream(stream: RTSPStreamV3):
    stream.stop()