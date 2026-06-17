"""后端检测框绘制。"""

from __future__ import annotations

import numpy as np

from fireagent.detection.schemas import DetectionBox

# 类别 → BGR 颜色
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "fire": (0, 100, 255),   # 橙红色
    "smoke": (180, 160, 120),  # 灰蓝色
}
_DEFAULT_COLOR = (0, 255, 0)  # 绿色兜底


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[DetectionBox],
) -> np.ndarray:
    """在 BGR 帧上绘制检测框和标签。"""
    import cv2

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    # 线宽和字体大小根据画面尺寸自适应
    thickness = max(1, min(h, w) // 400 + 1)
    font_scale = max(0.4, min(h, w) / 1600)

    for det in detections:
        color = _LABEL_COLORS.get(det.label.lower(), _DEFAULT_COLOR)
        x1, y1 = int(det.box.x1), int(det.box.y1)
        x2, y2 = int(det.box.x2), int(det.box.y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # 标签背景
        cv2.rectangle(
            annotated,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label_text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated
