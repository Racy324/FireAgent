"""检测配置解析测试。"""

from __future__ import annotations

from fireagent.utils.config import (
    DetectionConfig,
    DetectionModelConfig,
    FireAgentConfig,
    clear_config_cache,
    load_config,
)


def test_detection_model_config_defaults() -> None:
    mc = DetectionModelConfig()
    assert mc.type == "yolo"
    assert mc.image_size == 640
    assert mc.conf_threshold == 0.35
    assert mc.iou_threshold == 0.7
    assert mc.labels == ["fire", "smoke"]


def test_detection_config_defaults() -> None:
    dc = DetectionConfig()
    assert dc.enabled is True
    assert dc.device == "auto"
    assert dc.default_model == "yolo_fire_smoke"
    assert dc.max_upload_mb == 500
    assert dc.models == {}


def test_fire_agent_config_has_detection() -> None:
    cfg = FireAgentConfig()
    assert isinstance(cfg.detection, DetectionConfig)


def test_load_config_with_detection() -> None:
    clear_config_cache()
    cfg = load_config()
    assert isinstance(cfg.detection, DetectionConfig)
    assert cfg.detection.enabled is True
    assert "yolo_fire_smoke" in cfg.detection.models
    mc = cfg.detection.models["yolo_fire_smoke"]
    assert mc.type == "yolo"
    assert mc.weights == "detect_models/exp110/weights/best.pt"
    assert mc.image_size == 640
