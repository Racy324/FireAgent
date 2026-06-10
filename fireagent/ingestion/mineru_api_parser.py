"""MinerU 云端 API 解析器。

通过 mineru.net 的开放 API 提交 PDF 解析任务，获取结构化 JSON 输出，
再映射为 FireAgent 统一的 ParsedDocument。

使用方式：
    1. 在 .env 中配置 MINERU_API_KEY
    2. 设置 PDF_PARSER=mineru_api
    3. 正常运行入库脚本
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from fireagent.ingestion.mineru_parser import MinerUPDFParser
from fireagent.ingestion.schema import ParsedDocument

logger = logging.getLogger(__name__)


class MinerUAPIError(RuntimeError):
    """MinerU API 调用失败时抛出的异常。"""


class MinerUAPIParser(MinerUPDFParser):
    """通过 MinerU 云端 API 解析 PDF。

    继承 MinerUPDFParser 的输出解析逻辑（_parse_content_list_v2 等），
    只替换 _run_mineru 方法，改为调用云端 API。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://mineru.net",
        timeout: float = 300.0,
        poll_interval: float = 3.0,
        max_pages: Optional[int] = None,
        output_dir: str | Path = "data/parsed/mineru",
        keep_output: bool = True,
        reuse_output: bool = True,
        enable_ocr: bool = True,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "ch",
    ) -> None:
        # 调用父类 __init__，但 cli_path 不会用到
        super().__init__(
            output_dir=output_dir,
            backend="pipeline",
            model_source="",
            timeout=timeout,
            keep_output=keep_output,
            cli_path="mineru",
            max_pages=max_pages,
            reuse_output=reuse_output,
        )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.enable_ocr = enable_ocr
        self.enable_formula = enable_formula
        self.enable_table = enable_table
        self.language = language

    def _run_mineru(self, pdf_path: Path, output_dir: Path) -> None:
        """调用 MinerU 云端 API 解析 PDF，将结果保存到 output_dir。

        覆盖父类的 CLI 调用逻辑，改为：
        1. 上传 PDF 获取文件 URL（或直接用本地文件的 base64）
        2. 提交解析任务
        3. 轮询任务状态直到完成
        4. 下载结果 JSON 到 output_dir
        """
        if not self.api_key:
            raise MinerUAPIError(
                "未配置 MINERU_API_KEY。请在 .env 中设置 MinerU 云端 API 密钥。"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # ── 1. 上传文件并提交解析任务 ──
        logger.info("正在上传 PDF 到 MinerU API: %s", pdf_path.name)
        task_id = self._submit_task(pdf_path, headers)
        logger.info("任务已提交，task_id: %s", task_id)

        # ── 2. 轮询任务状态 ──
        result = self._poll_task(task_id, headers)
        logger.info("任务完成，正在保存结果...")

        # ── 3. 保存结果到 output_dir ──
        self._save_api_result(result, output_dir, pdf_path)

    def _submit_task(self, pdf_path: Path, headers: dict) -> str:
        """上传 PDF 并提交解析任务，返回 task_id。"""
        url = f"{self.base_url}/api/v4/extract/task"

        # 读取 PDF 文件为 base64
        import base64

        pdf_bytes = pdf_path.read_bytes()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        payload = {
            "file_name": pdf_path.name,
            "file_base64": pdf_base64,
            "enable_ocr": self.enable_ocr,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
        }

        # 如果有 max_pages 限制
        if self.max_pages is not None:
            payload["max_pages"] = self.max_pages

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise MinerUAPIError(
                f"MinerU API 提交任务失败 ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise MinerUAPIError(f"MinerU API 网络错误: {exc}") from exc

        # 解析响应获取 task_id
        if isinstance(data, dict):
            task_id = data.get("data", {}).get("task_id") or data.get("task_id")
            if task_id:
                return str(task_id)
            # 兼容直接返回 task_id 的情况
            if "task_id" in data:
                return str(data["task_id"])

        raise MinerUAPIError(f"MinerU API 响应格式异常: {json.dumps(data, ensure_ascii=False)[:500]}")

    def _poll_task(self, task_id: str, headers: dict) -> dict:
        """轮询任务状态，直到完成或超时。"""
        url = f"{self.base_url}/api/v4/extract/task/{task_id}"
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > self.timeout:
                raise MinerUAPIError(f"MinerU API 任务超时 ({self.timeout}s): {task_id}")

            try:
                with httpx.Client(timeout=30) as client:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
            except Exception as exc:
                logger.warning("轮询 MinerU API 失败，重试中: %s", exc)
                time.sleep(self.poll_interval)
                continue

            # 解析状态
            status = self._extract_status(data)
            logger.debug("任务 %s 状态: %s (%.0fs)", task_id, status, elapsed)

            if status in ("completed", "done", "success", "finished"):
                return data
            elif status in ("failed", "error"):
                error_msg = data.get("data", {}).get("error") or data.get("error", "未知错误")
                raise MinerUAPIError(f"MinerU API 任务失败: {error_msg}")
            # 其他状态（processing, pending, running）继续轮询

            time.sleep(self.poll_interval)

    def _extract_status(self, data: dict) -> str:
        """从 API 响应中提取任务状态。"""
        if not isinstance(data, dict):
            return "unknown"
        # 兼容多种响应格式
        d = data.get("data", data)
        return str(d.get("status", d.get("state", "unknown"))).lower()

    def _save_api_result(self, result: dict, output_dir: Path, pdf_path: Path) -> None:
        """将 API 返回的结果保存为 MinerU 格式的 JSON 文件。

        MinerU API 响应结构：
            data.result.content_list → 结构化块列表
            data.result.markdown    → Markdown 全文

        保存为 content_list_v2.json 以复用父类解析逻辑。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._safe_stem(pdf_path)

        # 提取实际内容数据（兼容 data.result 和 data 两种层级）
        data = result.get("data", result)
        result_data = data.get("result", data)

        # ── 1. 优先提取 content_list ──
        content_list = result_data.get("content_list") or data.get("content_list")
        if isinstance(content_list, list) and content_list:
            out_file = output_dir / f"{stem}_content_list_v2.json"
            out_file.write_text(json.dumps(content_list, ensure_ascii=False, indent=2))
            logger.info("已保存 content_list (%d 个块) 到 %s", len(content_list), out_file)
            return

        # ── 2. 尝试 content_list_v2 ──
        content_list_v2 = result_data.get("content_list_v2") or data.get("content_list_v2")
        if isinstance(content_list_v2, list) and content_list_v2:
            out_file = output_dir / f"{stem}_content_list_v2.json"
            out_file.write_text(json.dumps(content_list_v2, ensure_ascii=False, indent=2))
            logger.info("已保存 content_list_v2 (%d 个块) 到 %s", len(content_list_v2), out_file)
            return

        # ── 3. 如果只有 markdown，保存并尝试转换 ──
        markdown = result_data.get("markdown") or data.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            # 保存 markdown
            md_file = output_dir / f"{stem}.md"
            md_file.write_text(markdown, encoding="utf-8")
            logger.info("已保存 Markdown 到 %s", md_file)

            # 保存完整响应供调试
            out_file = output_dir / f"{stem}_api_result.json"
            out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

            # 尝试从 pages/blocks 结构转换
            converted = self._convert_api_result_to_content_list(data)
            if converted:
                cl_file = output_dir / f"{stem}_content_list_v2.json"
                cl_file.write_text(json.dumps(converted, ensure_ascii=False, indent=2))
                logger.info("已从 pages/blocks 转换并保存 content_list_v2 (%d 个块)", len(converted))
            return

        # ── 4. 兜底：保存原始响应 ──
        out_file = output_dir / f"{stem}_raw_response.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        logger.warning("MinerU API 响应格式未识别，已保存原始响应到 %s", out_file)

    def _convert_api_result_to_content_list(self, data: dict) -> list[dict] | None:
        """尝试将 MinerU API 的结果转换为 content_list_v2 格式。"""
        # 如果有 pages 或 blocks 结构
        pages = data.get("pages") or data.get("result", {}).get("pages")
        if isinstance(pages, list):
            content_list = []
            for page in pages:
                page_idx = page.get("page_index", page.get("page", 0))
                for block in page.get("blocks", page.get("content", [])):
                    if isinstance(block, dict):
                        item = {
                            "type": block.get("type", "text"),
                            "text": block.get("text", block.get("content", "")),
                            "page_idx": page_idx,
                        }
                        if "bbox" in block:
                            item["bbox"] = block["bbox"]
                        if "level" in block:
                            item["text_level"] = block["level"]
                        content_list.append(item)
            return content_list if content_list else None
        return None
