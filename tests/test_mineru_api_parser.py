"""Tests for the MinerU cloud API parser."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from fireagent.ingestion.mineru_api_parser import MinerUAPIParser


class _FakeResponse:
    def __init__(
        self,
        json_data: dict[str, Any] | None = None,
        content: bytes = b"",
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._json_data = json_data or {}
        self.content = content
        self.status_code = status_code
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.text or f"HTTP {self.status_code}")


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_mineru_api_parser_uses_official_batch_upload_flow(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "fire-paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfake")
    output_dir = tmp_path / "mineru"

    content_list = [
        [
            {"type": "title", "page_idx": 0, "text_level": 1, "text": "摘要"},
            {"type": "text", "page_idx": 0, "text": "火灾烟气控制研究内容。"},
        ]
    ]
    result_zip = _zip_bytes(
        {
            "fire-paper/fire-paper_content_list_v2.json": json.dumps(
                content_list,
                ensure_ascii=False,
            )
        }
    )

    calls: list[tuple[str, str, dict[str, Any]]] = []

    class FakeClient:
        def __init__(self, timeout: float | None = None) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            calls.append(("POST", url, kwargs))
            assert url == "https://mineru.net/api/v4/file-urls/batch"
            return _FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": [
                            {
                                "file_name": "fire-paper.pdf",
                                "upload_url": "https://upload.example/fire-paper.pdf",
                            }
                        ],
                    },
                }
            )

        def put(self, url: str, **kwargs: Any) -> _FakeResponse:
            calls.append(("PUT", url, kwargs))
            assert url == "https://upload.example/fire-paper.pdf"
            assert kwargs["content"] == pdf_path.read_bytes()
            return _FakeResponse()

        def get(self, url: str, **kwargs: Any) -> _FakeResponse:
            calls.append(("GET", url, kwargs))
            if url == "https://mineru.net/api/v4/extract-results/batch/batch-1":
                return _FakeResponse(
                    {
                        "code": 0,
                        "data": {
                            "batch_id": "batch-1",
                            "extract_result": [
                                {
                                    "file_name": "fire-paper.pdf",
                                    "state": "done",
                                    "full_zip_url": "https://download.example/result.zip",
                                }
                            ],
                        },
                    }
                )
            if url == "https://download.example/result.zip":
                return _FakeResponse(content=result_zip)
            raise AssertionError(f"Unexpected GET URL: {url}")

    monkeypatch.setattr("fireagent.ingestion.mineru_api_parser.httpx.Client", FakeClient)

    parser = MinerUAPIParser(
        api_key="token",
        output_dir=output_dir,
        poll_interval=0,
        timeout=5,
        reuse_output=False,
    )
    document = parser.parse(pdf_path)

    assert document.metadata["parser"] == "mineru"
    assert document.metadata["parser_variant"] == "content_list_v2"
    assert "火灾烟气控制研究内容" in document.pages[0].text

    post_call = calls[0]
    assert post_call[0] == "POST"
    assert post_call[2]["headers"]["Authorization"] == "Bearer token"
    assert post_call[2]["json"]["files"] == [
        {"name": "fire-paper.pdf", "is_ocr": True, "data_id": "fire-paper"}
    ]
    assert post_call[2]["json"]["enable_formula"] is True
    assert post_call[2]["json"]["enable_table"] is True
    assert post_call[2]["json"]["language"] == "ch"

