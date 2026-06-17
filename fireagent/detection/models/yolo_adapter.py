"""YOLO 检测模型适配器。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from fireagent.detection.schemas import (
    BoundingBox,
    DetectionBox,
    DetectionFrameResult,
)

logger = logging.getLogger(__name__)


class YoloDetectionModel:
    """基于 ultralytics YOLO 的检测模型。"""

    def __init__(
        self,
        weights_path: str,
        labels: list[str] | None = None,
        image_size: int = 640,
        device: str = "auto",
    ) -> None:
        self.model_id = "yolo"
        self.labels = labels or ["fire", "smoke"]
        self.weights_path = weights_path
        self.image_size = image_size
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        path = Path(self.weights_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO 权重文件不存在: {path}")
        logger.info("加载 YOLO 模型: %s", path)
        self._model = YOLO(str(path))
        # 从模型中读取真实类别名
        if hasattr(self._model, "names") and self._model.names:
            self.labels = [self._model.names[i] for i in sorted(self._model.names)]
        logger.info("YOLO 模型加载完成，类别: %s", self.labels)

    def warmup(self) -> None:
        """预热：加载模型并对空白图像做一次推理。"""
        self._load()
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.predict_bgr(dummy, conf_threshold=0.5, iou_threshold=0.5)

    def predict_bgr(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> DetectionFrameResult:
        """对单帧 BGR 图像执行 YOLO 检测。"""
        self._load()
        height, width = frame_bgr.shape[:2]
        start = time.perf_counter()

        results = self._model.predict(
            source=frame_bgr,
            imgsz=self.image_size,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections: list[DetectionBox] = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    class_id = int(clss[i])
                    label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
                    detections.append(
                        DetectionBox(
                            label=label,
                            class_id=class_id,
                            confidence=round(float(confs[i]), 4),
                            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        )
                    )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return DetectionFrameResult(
            width=width,
            height=height,
            latency_ms=latency_ms,
            detections=detections,
        )
