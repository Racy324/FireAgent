"""检测模型统一协议。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from fireagent.detection.schemas import DetectionFrameResult


@runtime_checkable
class DetectionModel(Protocol):
    """所有检测模型必须实现的接口。"""

    model_id: str
    labels: list[str]

    def warmup(self) -> None:
        """预热模型，首次推理前调用。"""
        ...

    def predict_bgr(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> DetectionFrameResult:
        """对单帧 BGR 图像执行检测。"""
        ...
