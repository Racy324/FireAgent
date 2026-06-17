"""视觉检测 API 端点。"""

import io
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def register_detection_endpoints(app: Any) -> None:
    """在 FastAPI app 上注册检测相关端点。"""
    from fastapi import HTTPException, UploadFile, File, Form
    from fastapi.responses import FileResponse, JSONResponse

    from fireagent.detection import (
        DetectionModelRegistry,
        VideoJobManager,
        draw_detections,
    )

    cfg = app.state.config
    if not cfg.detection.enabled:
        logger.info("检测模块已禁用，跳过注册端点")
        return

    # 单例：registry 和 job manager
    registry = DetectionModelRegistry(cfg.detection)
    job_manager = VideoJobManager(cfg.detection, registry)
    app.state.detection_registry = registry
    app.state.detection_job_manager = job_manager

    @app.get("/detect/models")
    def list_models() -> dict[str, Any]:
        """返回可用检测模型列表。"""
        return {"models": [m.model_dump() for m in registry.list_models()]}

    @app.post("/detect/frame")
    async def detect_frame(
        model_id: str = Form(default=""),
        conf_threshold: float = Form(default=0.35),
        iou_threshold: float = Form(default=0.7),
        image: UploadFile = File(...),
    ) -> dict[str, Any]:
        """单帧检测。"""
        if not model_id:
            model_id = cfg.detection.default_model

        # 读取图片
        data = await image.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="无法解码图片")

        try:
            model = registry.get_model(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        result = model.predict_bgr(frame, conf_threshold, iou_threshold)
        return result.model_dump()

    @app.post("/detect/videos")
    async def create_video_job(
        model_id: str = Form(default=""),
        conf_threshold: float = Form(default=0.35),
        iou_threshold: float = Form(default=0.7),
        video: UploadFile = File(...),
    ) -> dict[str, Any]:
        """上传视频创建检测任务。"""
        if not model_id:
            model_id = cfg.detection.default_model

        # 验证模型存在
        if model_id not in cfg.detection.models:
            raise HTTPException(status_code=404, detail=f"未知模型: {model_id}")

        # 验证模型可用
        try:
            registry.get_model(model_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        # 读取视频
        video_bytes = await video.read()
        max_bytes = cfg.detection.max_upload_mb * 1024 * 1024
        if len(video_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"视频文件超过 {cfg.detection.max_upload_mb} MB 限制",
            )

        filename = video.filename or "upload.mp4"
        job_id = job_manager.create_job(
            model_id=model_id,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            video_bytes=video_bytes,
            filename=filename,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/detect/videos/{job_id}")
    def get_video_job(job_id: str) -> dict[str, Any]:
        """查询视频任务状态。"""
        try:
            info = job_manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return info

    @app.get("/detect/videos/{job_id}/output")
    def download_video_output(job_id: str) -> Any:
        """下载/播放检测后视频。"""
        try:
            path = job_manager.get_output_path(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{job_id}_detected.mp4",
        )
