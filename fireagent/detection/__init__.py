"""FireAgent 视觉检测模块。"""

from fireagent.detection.drawing import draw_detections
from fireagent.detection.registry import DetectionModelRegistry
from fireagent.detection.schemas import (
    BoundingBox,
    DetectionBox,
    DetectionFrameResult,
    DetectionModelInfo,
    VideoJobInfo,
)
from fireagent.detection.video_jobs import VideoJobManager

__all__ = [
    "BoundingBox",
    "DetectionBox",
    "DetectionFrameResult",
    "DetectionModelInfo",
    "DetectionModelRegistry",
    "VideoJobInfo",
    "VideoJobManager",
    "draw_detections",
]
