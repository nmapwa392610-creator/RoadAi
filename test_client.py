# tests/test_rtsp_v3.py

import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.pipelines.rtsp_usual import RTSPStreamV3


# =========================
# IOU TEST
# =========================

def test_rtsp_stream_v3_iou():
    stream = RTSPStreamV3(url="rtsp://fake")

    a = [100, 100, 200, 200]
    b = [100, 100, 200, 200]
    assert stream._iou(a, b) == 1.0

    c = [300, 300, 400, 400]
    assert stream._iou(a, c) == 0.0

    d = [150, 100, 250, 200]
    val = stream._iou(a, d)

    assert 0 < val < 1
    assert round(val, 2) == 0.33


# =========================
# BACKPRESSURE TEST
# =========================

def test_rtsp_stream_backpressure():
    stream = RTSPStreamV3(url="rtsp://fake", queue_size=2)

    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    stream.queue.put((frame, time.time()))
    stream.queue.put((frame, time.time()))

    assert stream.queue.full()

    # simulate drop logic
    if stream.queue.full():
        stream.queue.get_nowait()

    stream.queue.put((frame, time.time()))

    assert stream.queue.qsize() == 2


# =========================
# START PROTECTION
# =========================

def test_rtsp_stream_duplicate_start():
    stream = RTSPStreamV3(url="rtsp://fake")

    stream._running = True

    result = stream.start(lambda x: None)

    assert result is stream


# =========================
# LIFECYCLE TEST (FIXED)
# =========================

@pytest.fixture
def mock_pipeline():
    with patch("src.pipelines.rtsp_usual.run_pipeline_frame") as m:
        m.return_value = [
            {
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "defect",
                "track_id": 1,
                "area": 10000.0,
                "severity": "high",
            }
        ]
        yield m


def test_rtsp_stream_v3_lifecycle(mock_pipeline):
    results = []

    def callback(data):
        results.append(data)

    stream = RTSPStreamV3(
        url="rtsp://fake",
        fps_target=1
    )

    mock_process = MagicMock()
    fake_frame = bytes(stream.frame_size)

    mock_process.stdout.read.return_value = fake_frame

    with patch("subprocess.Popen", return_value=mock_process):

        stream.start(callback)

        # даём потокам гарантированно стартануть
        time.sleep(0.5)

        stream.stop()

    assert stream._running is False

    assert len(results) > 0

    payload = results[0]

    assert isinstance(payload["frame"], int)
    assert isinstance(payload["timestamp"], float)
    assert isinstance(payload["detections"], list)

    det = payload["detections"][0]
    assert det["track_id"] == 1
    assert det["class_name"] == "defect"
    assert det["severity"] == "high"
    assert len(det["bbox"]) == 4