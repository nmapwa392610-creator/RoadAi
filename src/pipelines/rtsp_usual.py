import cv2
import threading
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class RTSPStream:
    """
    Управляет RTSP-потоком в отдельном потоке.

    Пример использования:
        stream = RTSPStream("rtsp://admin:admin@192.168.0.118:554/stream")
        stream.start(callback=my_frame_handler)
        ...
        stream.stop()
    """

    def __init__(
        self,
        url: str,
        reconnect: bool = True,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 10,
    ):
        self.url = url
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._reconnect_count = 0
        self._is_running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, frame_callback: Callable) -> "RTSPStream":
        """Запустить захват потока. frame_callback вызывается для каждого кадра."""
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
        """Остановить поток и освободить ресурсы."""
        self._stop_event.set()
        self._is_running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._release_cap()
        logger.info("Stream stopped: %s", self.url)

    @property
    def is_running(self) -> bool:
        return self._is_running and not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self, frame_callback: Callable) -> None:
        while not self._stop_event.is_set():
            if not self._open_cap():
                # Не смогли открыть — пробуем переподключиться
                if not self._try_reconnect():
                    break
                continue

            self._reconnect_count = 0  # сбрасываем счётчик после успеха

            while not self._stop_event.is_set():
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    logger.warning("Frame read failed, reconnecting: %s", self.url)
                    self._release_cap()
                    if not self._try_reconnect():
                        return
                    break  # выходим во внешний while для переоткрытия

                try:
                    frame_callback(frame)
                except Exception as exc:
                    logger.exception("frame_callback raised an exception: %s", exc)

        self._is_running = False

    def _open_cap(self) -> bool:
        with self._lock:
            self._cap = cv2.VideoCapture(self.url)
            if not self._cap.isOpened():
                logger.error("Cannot open stream: %s", self.url)
                self._cap = None
                return False

            # Уменьшаем буфер для минимальной задержки
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


# ------------------------------------------------------------------
# Обратная совместимость со старым функциональным API
# ------------------------------------------------------------------

def start_rtsp_stream(url: str, frame_callback: Callable) -> RTSPStream:
    """Устаревший интерфейс. Используйте RTSPStream напрямую."""
    stream = RTSPStream(url)
    stream.start(frame_callback)
    return stream


def stop_rtsp_stream(stream: RTSPStream) -> None:
    """Устаревший интерфейс."""
    stream.stop()