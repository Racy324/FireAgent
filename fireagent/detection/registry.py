"""检测模型注册表。"""

from __future__ import annotations

import logging
from typing import Any

from fireagent.detection.models.yolo_adapter import YoloDetectionModel
from fireagent.detection.schemas import DetectionModelInfo
from fireagent.utils.config import DetectionConfig

logger = logging.getLogger(__name__)


class DetectionModelRegistry:
    """管理检测模型的懒加载、缓存和查询。"""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._instances: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

    def list_models(self) -> list[DetectionModelInfo]:
        """返回所有配置模型的信息。"""
        models: list[DetectionModelInfo] = []
        for model_id, mc in self._config.models.items():
            ready = model_id in self._instances
            error = self._errors.get(model_id)
            models.append(
                DetectionModelInfo(
                    model_id=model_id,
                    display_name=mc.display_name or model_id,
                    type=mc.type,
                    ready=ready and error is None,
                    labels=list(mc.labels),
                    default_conf_threshold=mc.conf_threshold,
                )
            )
        return models

    def get_model(self, model_id: str) -> Any:
        """获取模型实例，懒加载。"""
        if model_id in self._instances:
            return self._instances[model_id]
        if model_id in self._errors:
            raise RuntimeError(f"模型 {model_id} 加载失败: {self._errors[model_id]}")

        mc = self._config.models.get(model_id)
        if mc is None:
            raise KeyError(f"未知模型: {model_id}")

        try:
            model = self._create_model(model_id, mc)
            self._instances[model_id] = model
            return model
        except Exception as exc:
            self._errors[model_id] = str(exc)
            logger.error("模型 %s 加载失败: %s", model_id, exc)
            raise RuntimeError(f"模型 {model_id} 加载失败: {exc}") from exc

    def _create_model(self, model_id: str, mc: Any) -> Any:
        """根据配置创建模型实例。"""
        device = self._config.device
        if mc.type == "yolo":
            model = YoloDetectionModel(
                weights_path=mc.weights,
                labels=list(mc.labels),
                image_size=mc.image_size,
                device=device,
            )
            model.model_id = model_id
            return model
        raise ValueError(f"不支持的模型类型: {mc.type}")

    def warmup(self, model_id: str) -> None:
        """预热指定模型。"""
        model = self.get_model(model_id)
        model.warmup()
