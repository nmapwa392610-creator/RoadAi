import cv2
import threading
import logging
import time
from typing import Callable, Optional
from src.pipelines.frame import run_pipeline_frame

logger = logging.getLogger(__name__)


class RTSPStream:
    """Управляет RTSP потоком в отдельном потоке с автопереподключением и дедупликацией ям."""

    def __init__(
        self,
        url: str,
        reconnect: bool = True,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
        frame_skip: int = 20,
    ):
        self.url = url
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay  # секунд между попытками переподключения
        self.max_reconnects = max_reconnects    # максимум попыток переподключения
        self.frame_skip = frame_skip            # обрабатываем каждый N-й кадр

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._reconnect_count = 0
        self._is_running = False

        # Хранит уже виденные ямы — чтобы не отправлять дубликаты
        self._seen_track_ids: set = set()
        self._seen_boxes: list = []


    # Public API
    def start(self, frame_callback: Callable) -> "RTSPStream":
        """Запускает захват потока. frame_callback получает список новых уникальных ям."""
        if self._is_running:
            logger.warning("Stream already running: %s", self.url)
            return self

        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(
            target=self._loop,
            args=(frame_callback,),
            daemon=True,
            name=f"rtsp-{self.url[-20:]}",
        )
        self._thread.start()
        logger.info("Stream started: %s", self.url)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Останавливает поток и освобождает ресурсы."""
        self._stop_event.set()
        self._is_running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._release_cap()
        logger.info("Stream stopped: %s", self.url)

    @property
    def is_running(self) -> bool:
        return self._is_running and not self._stop_event.is_set()



    # Internal
    def _calc_iou(self, box1, box2) -> float:
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

    def _is_duplicate(self, box) -> bool:
        # Яма считается дубликатом если IoU с уже виденным bbox >= 0.4
        return any(self._calc_iou(box, seen) >= 0.4 for seen in self._seen_boxes)

    def _filter_detections(self, detections: list) -> list:
        # Фильтруем дубликаты — оставляем только новые ямы
        # Используем track_id если есть, иначе сравниваем по IoU
        new_detections = []

        for det in detections:
            tid = det.get("track_id")
            box = det["bbox"]

            if tid is not None and tid != "untracked":
                if tid not in self._seen_track_ids:
                    if not self._is_duplicate(box):
                        self._seen_track_ids.add(tid)
                        self._seen_boxes.append(box)
                        new_detections.append(det)
                    else:
                        # bbox уже видели через untracked — просто запоминаем track_id
                        self._seen_track_ids.add(tid)
            else:
                # track_id нет — проверяем только по IoU
                if not self._is_duplicate(box):
                    self._seen_boxes.append(box)
                    new_detections.append(det)

        return new_detections

    def _loop(self, frame_callback: Callable) -> None:
        frame_id = 0

        while not self._stop_event.is_set():
            if not self._open_cap():
                if not self._try_reconnect():
                    break
                continue

            # Сбрасываем счётчик после успешного подключения
            self._reconnect_count = 0

            while not self._stop_event.is_set():
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    logger.warning("Frame read failed, reconnecting: %s", self.url)
                    self._release_cap()
                    if not self._try_reconnect():
                        return
                    break

                # Пропускаем кадры — обрабатываем только каждый frame_skip-й
                if frame_id % self.frame_skip != 0:
                    frame_id += 1
                    continue

                try:
                    frame = cv2.resize(frame, (640, 640))
                    detections = run_pipeline_frame(frame, use_tracking=True)

                    if detections and isinstance(detections, list):
                        # Отправляем только новые уникальные ямы
                        new_detections = self._filter_detections(detections)
                        if new_detections:
                            frame_callback(new_detections)

                except Exception as exc:
                    logger.exception("frame_callback raised an exception: %s", exc)

                frame_id += 1

        self._is_running = False

    def _open_cap(self) -> bool:
        with self._lock:
            self._cap = cv2.VideoCapture(self.url)
            if not self._cap.isOpened():
                logger.error("Cannot open stream: %s", self.url)
                self._cap = None
                return False
            # Минимальный буфер для минимальной задержки
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True

    def _release_cap(self) -> None:
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    def _try_reconnect(self) -> bool:
        if not self.reconnect:
            logger.error("Reconnect disabled, stopping: %s", self.url)
            return False

        if self._reconnect_count >= self.max_reconnects:
            logger.error(
                "Max reconnects (%d) reached, stopping: %s",
                self.max_reconnects,
                self.url,
            )
            return False

        self._reconnect_count += 1
        logger.info(
            "Reconnecting (%d/%d) in %.1fs: %s",
            self._reconnect_count,
            self.max_reconnects,
            self.reconnect_delay,
            self.url,
        )
        time.sleep(self.reconnect_delay)
        return True

    def __enter__(self) -> "RTSPStream":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"<RTSPStream url={self.url!r} running={self.is_running}>"



# Обратная совместимость со старым функциональным API
def start_rtsp_stream(url: str, frame_callback: Callable) -> RTSPStream:
    """Устаревший интерфейс. Используйте RTSPStream напрямую."""
    stream = RTSPStream(url)
    stream.start(frame_callback)
    return stream


def stop_rtsp_stream(stream: RTSPStream) -> None:
    """Устаревший интерфейс."""
    stream.stop()