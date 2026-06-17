"""检测模型注册表测试。"""

from __future__ import annotations

from fireagent.detection.registry import DetectionModelRegistry
from fireagent.detection.schemas import DetectionBox, DetectionFrameResult
from fireagent.utils.config import DetectionConfig, DetectionModelConfig

import numpy as np


class FakeDetector:
    """用于测试的假检测模型。"""

    def __init__(self, model_id: str = "fake", labels: list[str] | None = None) -> None:
        self.model_id = model_id
        self.labels = labels or ["fire", "smoke"]

    def warmup(self) -> None:
        pass

    def predict_bgr(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> DetectionFrameResult:
        h, w = frame_bgr.shape[:2]
        return DetectionFrameResult(
            width=w,
            height=h,
            latency_ms=1.0,
            detections=[
                DetectionBox(
                    label="fire",
                    class_id=0,
                    confidence=0.9,
                    box={"x1": 10, "y1": 10, "x2": 100, "y2": 100},
                )
            ],
        )


def _make_config() -> DetectionConfig:
    return DetectionConfig(
        models={
            "test_model": DetectionModelConfig(
                type="yolo",
                display_name="Test Model",
                weights="nonexistent.pt",
                labels=["fire", "smoke"],
            ),
        }
    )


def test_list_models_empty_registry() -> None:
    registry = DetectionModelRegistry(_make_config())
    models = registry.list_models()
    assert len(models) == 1
    assert models[0].model_id == "test_model"
    assert models[0].ready is False


def test_get_unknown_model_raises() -> None:
    registry = DetectionModelRegistry(_make_config())
    import pytest
    with pytest.raises(KeyError, match="未知模型"):
        registry.get_model("nonexistent")


def test_list_models_after_injection() -> None:
    """手动注入模型实例后，list_models 应返回 ready=True。"""
    registry = DetectionModelRegistry(_make_config())
    # 手动注入
    fake = FakeDetector(model_id="test_model")
    registry._instances["test_model"] = fake

    models = registry.list_models()
    assert models[0].ready is True
