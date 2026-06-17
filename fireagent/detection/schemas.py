"""视觉检测数据结构。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class _APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class BoundingBox(_APIModel):
    """检测框坐标。"""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionBox(_APIModel):
    """单个检测结果。"""

    label: str
    class_id: int
    confidence: float
    box: BoundingBox


class DetectionFrameResult(_APIModel):
    """单帧检测结果。"""

    width: int
    height: int
    latency_ms: float
    detections: list[DetectionBox] = Field(default_factory=list)


class DetectionModelInfo(_APIModel):
    """模型信息（API 返回用）。"""

    model_id: str
    display_name: str
    type: str
    ready: bool
    labels: list[str] = Field(default_factory=list)
    default_conf_threshold: float = 0.35


class VideoJobInfo(_APIModel):
    """视频任务状态。"""

    job_id: str
    status: str  # queued / running / succeeded / failed
    progress: float = 0.0
    input_filename: str = ""
    output_url: Optional[str] = None
    error: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
