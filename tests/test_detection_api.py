"""检测 API 端点测试。"""

from __future__ import annotations

import json

import numpy as np
import cv2
import pytest

from fireagent.detection.registry import DetectionModelRegistry
from fireagent.detection.schemas import DetectionBox, DetectionFrameResult
from fireagent.utils.config import (
    DetectionConfig,
    DetectionModelConfig,
    FireAgentConfig,
    clear_config_cache,
    load_config,
)

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class FakeDetector:
    def __init__(self, model_id: str = "yolo_fire_smoke", labels=None):
        self.model_id = model_id
        self.labels = labels or ["fire", "smoke", "others"]

    def warmup(self):
        pass

    def predict_bgr(self, frame_bgr, conf_threshold, iou_threshold):
        h, w = frame_bgr.shape[:2]
        return DetectionFrameResult(
            width=w, height=h, latency_ms=5.0,
            detections=[
                DetectionBox(label="fire", class_id=0, confidence=0.85,
                             box={"x1": 10, "y1": 10, "x2": 100, "y2": 100})
            ],
        )


@pytest.fixture()
def client():
    if not HAS_FASTAPI:
        pytest.skip("fastapi not installed")
    clear_config_cache()
    cfg = load_config()
    from fireagent.api.server import create_app
    app = create_app(config=cfg)
    # 注入 fake detector
    registry = app.state.detection_registry
    fake = FakeDetector()
    registry._instances["yolo_fire_smoke"] = fake
    return TestClient(app)


def test_list_models(client) -> None:
    resp = client.get("/detect/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    ids = [m["model_id"] for m in data["models"]]
    assert "yolo_fire_smoke" in ids


def test_detect_frame(client) -> None:
    # 创建一个测试图片
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    resp = client.post(
        "/detect/frame",
        data={"model_id": "yolo_fire_smoke", "conf_threshold": "0.35"},
        files={"image": ("test.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["width"] == 100
    assert data["height"] == 100
    assert len(data["detections"]) == 1
    assert data["detections"][0]["label"] == "fire"


def test_detect_frame_unknown_model(client) -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    resp = client.post(
        "/detect/frame",
        data={"model_id": "nonexistent"},
        files={"image": ("test.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert resp.status_code == 404
