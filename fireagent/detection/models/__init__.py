"""检测模型适配器。"""

from fireagent.detection.models.base import DetectionModel
from fireagent.detection.models.yolo_adapter import YoloDetectionModel

__all__ = ["DetectionModel", "YoloDetectionModel"]
