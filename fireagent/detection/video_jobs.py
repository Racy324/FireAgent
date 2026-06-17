"""视频上传检测任务管理。"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fireagent.detection.drawing import draw_detections
from fireagent.detection.registry import DetectionModelRegistry
from fireagent.utils.config import DetectionConfig

logger = logging.getLogger(__name__)


class VideoJobManager:
    """管理视频检测任务的创建、执行和状态查询。"""

    def __init__(self, config: DetectionConfig, registry: DetectionModelRegistry) -> None:
        self._config = config
        self._registry = registry
        self._output_dir = Path(config.job_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create_job(
        self,
        model_id: str,
        conf_threshold: float,
        iou_threshold: float,
        video_bytes: bytes,
        filename: str,
    ) -> str:
        """创建视频检测任务，返回 job_id。"""
        job_id = f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        job_dir = self._output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # 保存上传视频
        ext = Path(filename).suffix or ".mp4"
        input_path = job_dir / f"input{ext}"
        input_path.write_bytes(video_bytes)

        # 初始化 job.json
        job_info = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "input_filename": filename,
            "output_url": None,
            "error": None,
            "metrics": {},
            "model_id": model_id,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
        }
        self._write_job_json(job_dir, job_info)

        # 后台线程执行
        thread = threading.Thread(
            target=self._process_video,
            args=(job_id, job_dir, model_id, conf_threshold, iou_threshold),
            daemon=True,
        )
        thread.start()

        return job_id

    def get_job(self, job_id: str) -> dict[str, Any]:
        """查询任务状态。"""
        job_dir = self._output_dir / job_id
        job_json = job_dir / "job.json"
        if not job_json.exists():
            raise KeyError(f"任务不存在: {job_id}")
        return json.loads(job_json.read_text(encoding="utf-8"))

    def get_output_path(self, job_id: str) -> Path:
        """获取输出视频路径。"""
        job_dir = self._output_dir / job_id
        output = job_dir / "output.mp4"
        if not output.exists():
            raise FileNotFoundError(f"输出视频不存在: {job_id}")
        return output

    def _write_job_json(self, job_dir: Path, info: dict[str, Any]) -> None:
        path = job_dir / "job.json"
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _update_job(self, job_dir: Path, **kwargs: Any) -> None:
        job_json = job_dir / "job.json"
        info = json.loads(job_json.read_text(encoding="utf-8"))
        info.update(kwargs)
        self._write_job_json(job_dir, info)

    def _process_video(
        self,
        job_id: str,
        job_dir: Path,
        model_id: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> None:
        """后台执行视频逐帧检测。"""
        try:
            self._update_job(job_dir, status="running")

            # 获取模型
            model = self._registry.get_model(model_id)

            # 读取输入视频
            input_path = next(job_dir.glob("input.*"))
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            # 准备输出
            output_path = job_dir / "output_raw.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            frame_idx = 0
            total_latency = 0.0
            last_update = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = model.predict_bgr(frame, conf_threshold, iou_threshold)
                annotated = draw_detections(frame, result.detections)
                writer.write(annotated)

                frame_idx += 1
                total_latency += result.latency_ms

                # 每 0.5 秒更新一次进度
                now = time.perf_counter()
                if now - last_update > 0.5:
                    progress = frame_idx / total_frames if total_frames > 0 else 0.0
                    self._update_job(
                        job_dir,
                        progress=round(progress, 4),
                        metrics={
                            "frames_total": total_frames,
                            "frames_done": frame_idx,
                            "avg_latency_ms": round(total_latency / frame_idx, 2),
                        },
                    )
                    last_update = now

            cap.release()
            writer.release()

            # 用 ffmpeg 转码为 H.264（浏览器兼容性最好）
            final_output = job_dir / "output.mp4"
            import subprocess
            import shutil

            ffmpeg_ok = False
            if shutil.which("ffmpeg"):
                try:
                    result = subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(output_path),
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-preset", "fast",
                            "-movflags", "+faststart",
                            str(final_output),
                        ],
                        capture_output=True,
                        timeout=300,
                    )
                    if result.returncode == 0 and final_output.exists() and final_output.stat().st_size > 0:
                        ffmpeg_ok = True
                        # 删除原始 mp4v 文件
                        if output_path != final_output and output_path.exists():
                            output_path.unlink()
                    else:
                        logger.error("ffmpeg 转码失败: %s", result.stderr.decode(errors="replace")[:500])
                except Exception as exc:
                    logger.error("ffmpeg 执行异常: %s", exc)

            if not ffmpeg_ok:
                # ffmpeg 不可用或转码失败，直接用 OpenCV 输出
                if final_output != output_path and output_path.exists():
                    output_path.rename(final_output)
                if not final_output.exists() or final_output.stat().st_size == 0:
                    raise RuntimeError("视频输出失败：ffmpeg 不可用且 OpenCV 输出无效")

            avg_latency = round(total_latency / max(frame_idx, 1), 2)
            self._update_job(
                job_dir,
                status="succeeded",
                progress=1.0,
                output_url=f"/detect/videos/{job_id}/output",
                metrics={
                    "frames_total": total_frames,
                    "frames_done": frame_idx,
                    "avg_latency_ms": avg_latency,
                },
            )
            logger.info("视频任务 %s 完成: %d 帧, 平均 %.1f ms", job_id, frame_idx, avg_latency)

        except Exception as exc:
            logger.error("视频任务 %s 失败: %s", job_id, exc)
            self._update_job(job_dir, status="failed", error=str(exc))
