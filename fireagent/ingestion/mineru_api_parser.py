"""MinerU cloud API parser.

This parser uses MinerU's precise parsing API for local PDFs:

1. Request a batch of pre-signed upload URLs.
2. Upload the local PDF bytes to the returned URL.
3. Poll the batch extraction result.
4. Download and unpack the result zip.
5. Reuse ``MinerUPDFParser`` to map MinerU JSON output to ``ParsedDocument``.
"""

from __future__ import annotations

import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import httpx

from fireagent.ingestion.mineru_parser import MinerUPDFParser

logger = logging.getLogger(__name__)


class MinerUAPIError(RuntimeError):
    """Raised when a MinerU cloud API call or result download fails."""


class MinerUAPIParser(MinerUPDFParser):
    """Parse PDFs through MinerU's cloud precise parsing API."""

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
        """Call MinerU cloud precise parsing and save its zip contents locally."""
        if not self.api_key:
            raise MinerUAPIError("未配置 MINERU_API_KEY。请在 .env 中设置 MinerU 云端 API 密钥。")

        headers = self._auth_headers()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Requesting MinerU upload URL for %s", pdf_path.name)
        batch_id, upload_url = self._request_upload_url(pdf_path, headers)

        logger.info("Uploading %s to MinerU batch %s", pdf_path.name, batch_id)
        self._upload_pdf(pdf_path, upload_url)

        logger.info("Polling MinerU batch result: %s", batch_id)
        result_item = self._poll_batch_result(batch_id, pdf_path.name, headers)

        self._save_batch_result(result_item, output_dir, pdf_path)
        full_zip_url = self._extract_full_zip_url(result_item)
        logger.info("Downloading MinerU result zip for %s", pdf_path.name)
        self._download_and_extract_zip(full_zip_url, output_dir)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_upload_url(self, pdf_path: Path, headers: dict[str, str]) -> tuple[str, str]:
        url = f"{self.base_url}/api/v4/file-urls/batch"
        payload: dict[str, Any] = {
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
            "files": [
                {
                    "name": pdf_path.name,
                    "is_ocr": self.enable_ocr,
                    "data_id": self._safe_stem(pdf_path),
                }
            ],
        }
        if self.max_pages is not None:
            payload["max_pages"] = self.max_pages

        data = self._post_json(url, headers=headers, payload=payload, context="申请上传链接")
        body = self._response_data(data)
        batch_id = body.get("batch_id") or body.get("id")
        upload_url = self._first_upload_url(body)

        if not batch_id or not upload_url:
            raise MinerUAPIError(
                f"MinerU API 上传链接响应缺少 batch_id 或 upload_url: "
                f"{json.dumps(data, ensure_ascii=False)[:500]}"
            )
        return str(batch_id), upload_url

    def _upload_pdf(self, pdf_path: Path, upload_url: str) -> None:
        headers = {"Content-Type": "application/pdf"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.put(upload_url, headers=headers, content=pdf_path.read_bytes())
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - normalize HTTP/client failures.
            raise MinerUAPIError(f"MinerU API 上传 PDF 失败: {exc}") from exc

    def _poll_batch_result(
        self,
        batch_id: str,
        file_name: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        start = time.time()

        while True:
            if time.time() - start > self.timeout:
                raise MinerUAPIError(f"MinerU API 任务超时 ({self.timeout}s): {batch_id}")

            data = self._get_json(url, headers=headers, context="轮询解析结果")
            item = self._select_result_item(data, file_name)
            state = self._extract_state(item or data)

            if state in {"done", "completed", "complete", "success", "finished"}:
                if not item:
                    raise MinerUAPIError(
                        f"MinerU API 结果缺少文件条目: {json.dumps(data, ensure_ascii=False)[:500]}"
                    )
                return item

            if state in {"failed", "fail", "error"}:
                error_msg = self._extract_error(item or data)
                raise MinerUAPIError(f"MinerU API 任务失败: {error_msg}")

            time.sleep(self.poll_interval)

    def _save_batch_result(self, result_item: dict[str, Any], output_dir: Path, pdf_path: Path) -> None:
        raw_file = output_dir / f"{self._safe_stem(pdf_path)}_api_result.json"
        raw_file.write_text(json.dumps(result_item, ensure_ascii=False, indent=2), encoding="utf-8")

    def _download_and_extract_zip(self, full_zip_url: str, output_dir: Path) -> None:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(full_zip_url)
                response.raise_for_status()
                zip_bytes = response.content
        except Exception as exc:  # noqa: BLE001 - normalize HTTP/client failures.
            raise MinerUAPIError(f"MinerU API 下载解析结果失败: {exc}") from exc

        zip_file = output_dir / "mineru_result.zip"
        zip_file.write_bytes(zip_bytes)
        try:
            self._extract_zip_safely(zip_file, output_dir)
        except zipfile.BadZipFile as exc:
            raise MinerUAPIError("MinerU API 返回的解析结果不是有效 zip 文件。") from exc

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as exc:  # noqa: BLE001 - normalize HTTP/client failures.
            raise MinerUAPIError(f"MinerU API {context}失败: {exc}") from exc

    def _get_json(self, url: str, headers: dict[str, str], context: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as exc:  # noqa: BLE001 - normalize HTTP/client failures.
            raise MinerUAPIError(f"MinerU API {context}失败: {exc}") from exc

    @staticmethod
    def _response_data(data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        if data.get("code") not in (None, 0, 200, "0", "200"):
            raise MinerUAPIError(
                f"MinerU API 返回错误: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        body = data.get("data", data)
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _first_upload_url(body: dict[str, Any]) -> str:
        file_urls = body.get("file_urls") or body.get("files") or []
        if isinstance(file_urls, dict):
            file_urls = list(file_urls.values())
        if not isinstance(file_urls, list) or not file_urls:
            return ""

        first = file_urls[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("upload_url", "url", "file_url"):
                value = first.get(key)
                if value:
                    return str(value)
        return ""

    def _select_result_item(self, data: dict[str, Any], file_name: str) -> dict[str, Any] | None:
        body = self._response_data(data)
        results = body.get("extract_result") or body.get("results") or body.get("files")
        if isinstance(results, dict):
            results = list(results.values())
        if not isinstance(results, list):
            return body if self._extract_full_zip_url(body, required=False) else None

        fallback: dict[str, Any] | None = None
        for item in results:
            if not isinstance(item, dict):
                continue
            fallback = fallback or item
            if item.get("file_name") == file_name or item.get("name") == file_name:
                return item
        return fallback

    @staticmethod
    def _extract_state(data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return "unknown"
        body = data.get("data", data)
        if not isinstance(body, dict):
            return "unknown"
        return str(body.get("state", body.get("status", "unknown"))).lower()

    @staticmethod
    def _extract_error(data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return "unknown error"
        body = data.get("data", data)
        if not isinstance(body, dict):
            body = data
        return str(body.get("err_msg") or body.get("error") or body.get("message") or "unknown error")

    @staticmethod
    def _extract_full_zip_url(data: dict[str, Any], required: bool = True) -> str:
        if not isinstance(data, dict):
            if required:
                raise MinerUAPIError("MinerU API 结果缺少 full_zip_url。")
            return ""
        for key in ("full_zip_url", "zip_url", "result_url"):
            value = data.get(key)
            if value:
                return str(value)
        if required:
            raise MinerUAPIError(
                f"MinerU API 结果缺少 full_zip_url: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        return ""

    @staticmethod
    def _extract_zip_safely(zip_file: Path, output_dir: Path) -> None:
        root = output_dir.resolve()
        with zipfile.ZipFile(zip_file) as archive:
            for member in archive.infolist():
                target = (output_dir / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise MinerUAPIError(f"MinerU API zip 包含非法路径: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as destination:
                        destination.write(source.read())

