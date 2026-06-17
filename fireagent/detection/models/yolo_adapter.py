"""YOLO 检测模型适配器。

支持两种加载方式：
1. ultralytics YOLO（.pt）— 需要训练时的 ultralytics 版本
2. onnxruntime（.onnx）— 通用方案，不依赖 ultralytics
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from fireagent.detection.schemas import (
    BoundingBox,
    DetectionBox,
    DetectionFrameResult,
)

logger = logging.getLogger(__name__)


class YoloDetectionModel:
    """YOLO 检测模型，优先 .pt，失败时降级 .onnx。"""

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
        self._backend = None  # "ultralytics" or "onnx"
        self._model = None
        self._session = None

    def _load(self) -> None:
        if self._backend is not None:
            return

        path = Path(self.weights_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO 权重文件不存在: {path}")

        # 尝试 1: ultralytics .pt
        if path.suffix == ".pt":
            try:
                from ultralytics import YOLO

                logger.info("尝试 ultralytics 加载: %s", path)
                self._model = YOLO(str(path))
                if hasattr(self._model, "names") and self._model.names:
                    self.labels = [self._model.names[i] for i in sorted(self._model.names)]
                self._backend = "ultralytics"
                logger.info("ultralytics 加载成功，类别: %s", self.labels)
                return
            except Exception as exc:
                logger.warning("ultralytics 加载失败: %s，尝试 ONNX", exc)

        # 尝试 2: onnxruntime .onnx
        onnx_path = path if path.suffix == ".onnx" else path.with_suffix(".onnx")
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                logger.info("尝试 ONNX 加载: %s", onnx_path)
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._session = ort.InferenceSession(str(onnx_path), providers=providers)
                self._backend = "onnx"
                logger.info("ONNX 加载成功，providers: %s", self._session.get_providers())
                return
            except Exception as exc:
                logger.error("ONNX 加载失败: %s", exc)
                raise RuntimeError(f"YOLO ONNX 加载失败: {exc}") from exc

        raise RuntimeError(
            f"YOLO 加载失败: ultralytics 不支持自定义模块，且未找到 {onnx_path}"
        )

    def warmup(self) -> None:
        self._load()
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.predict_bgr(dummy, conf_threshold=0.5, iou_threshold=0.5)

    def predict_bgr(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> DetectionFrameResult:
        self._load()
        height, width = frame_bgr.shape[:2]
        start = time.perf_counter()

        if self._backend == "ultralytics":
            detections = self._predict_ultralytics(frame_bgr, conf_threshold, iou_threshold)
        else:
            detections = self._predict_onnx(frame_bgr, conf_threshold, iou_threshold)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return DetectionFrameResult(
            width=width,
            height=height,
            latency_ms=latency_ms,
            detections=detections,
        )

    def _predict_ultralytics(
        self, frame_bgr: np.ndarray, conf_threshold: float, iou_threshold: float
    ) -> list[DetectionBox]:
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
                            label=label, class_id=class_id,
                            confidence=round(float(confs[i]), 4),
                            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        )
                    )
        return detections

    def _predict_onnx(
        self, frame_bgr: np.ndarray, conf_threshold: float, iou_threshold: float
    ) -> list[DetectionBox]:
        orig_h, orig_w = frame_bgr.shape[:2]
        img = self._preprocess_onnx(frame_bgr)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: img})

        # ultralytics ONNX 输出: [1, num_classes+4, num_anchors]
        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]  # [num_classes+4, num_anchors]

        num_classes = len(self.labels)
        # 转置: [num_anchors, num_classes+4]
        if preds.shape[0] == num_classes + 4:
            preds = preds.T

        # 前 4 列是 xywh，后面是类别分数
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        # 每个 anchor 取最大类别分数
        max_scores = class_scores.max(axis=1)
        max_class_ids = class_scores.argmax(axis=1)

        # 置信度过滤
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        max_class_ids = max_class_ids[mask]

        if len(max_scores) == 0:
            return []

        # xywh -> xyxy（像素坐标，已缩放到 image_size）
        xyxy = np.zeros_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # 缩放回原图坐标
        scale_x = orig_w / self.image_size
        scale_y = orig_h / self.image_size
        xyxy[:, [0, 2]] *= scale_x
        xyxy[:, [1, 3]] *= scale_y

        # NMS
        keep = self._nms(xyxy, max_scores, iou_threshold)
        xyxy = xyxy[keep]
        max_scores = max_scores[keep]
        max_class_ids = max_class_ids[keep]

        detections: list[DetectionBox] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            class_id = int(max_class_ids[i])
            label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
            detections.append(
                DetectionBox(
                    label=label, class_id=class_id,
                    confidence=round(float(max_scores[i]), 4),
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections

    def _preprocess_onnx(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB, resize, normalize, NCHW."""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # CHW -> NCHW
        return np.ascontiguousarray(img)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """简单 NMS 实现。"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
