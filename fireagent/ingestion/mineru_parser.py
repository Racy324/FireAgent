"""MinerU 增强版 PDF 结构解析适配器。

该模块通过 MinerU CLI 生成结构化输出，再把 ``content_list_v2.json``、
``content_list.json`` 或 ``middle.json`` 映射为 FireAgent 统一的
``ParsedDocument``。这样后续清洗、章节切分、切块和入库流程不需要关心
MinerU 的运行细节。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

from fireagent.ingestion.pdf_parser import BasePDFParser
from fireagent.ingestion.schema import FigureCaption, ParsedDocument, ParsedTable, PDFPage


TEXT_TYPES = {
    "title",
    "text",
    "list",
    "list_item",
    "equation",
    "interline_equation",
    "inline_equation",
    "code",
    "algorithm",
    "reference",
    "page_footnote",
}
TABLE_TYPES = {"table"}
FIGURE_TYPES = {"image", "chart"}
AUXILIARY_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "page_aside",
    "page_aside_text",
}


class MinerUParserError(RuntimeError):
    """MinerU 解析失败时抛出的领域异常。"""


class MinerUPDFParser(BasePDFParser):
    """通过 MinerU CLI 提取版面结构、表格、图注和页码。"""

    def __init__(
        self,
        output_dir: str | Path = "data/parsed/mineru",
        backend: str = "pipeline",
        model_source: str = "",
        timeout: float = 1800.0,
        keep_output: bool = True,
        cli_path: str = "mineru",
        max_pages: Optional[int] = None,
        reuse_output: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.backend = backend
        self.model_source = model_source
        self.timeout = timeout
        self.keep_output = keep_output
        self.cli_path = cli_path
        self.max_pages = max_pages
        self.reuse_output = reuse_output

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """调用 MinerU 并读取结构化 JSON 输出。"""
        path = self.validate_pdf_path(pdf_path)
        doc_id = self.make_doc_id(path)
        job_output_dir = self.output_dir / self._safe_stem(path)
        job_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.reuse_output or not self._find_existing_output(job_output_dir):
            self._run_mineru(path, job_output_dir)

        document = self._load_mineru_output(path=path, doc_id=doc_id, output_dir=job_output_dir)
        if not self.keep_output:
            shutil.rmtree(job_output_dir, ignore_errors=True)
        return document

    def _run_mineru(self, pdf_path: Path, output_dir: Path) -> None:
        """执行 MinerU CLI，生成中间结构文件。"""
        command = [self.cli_path, "-p", str(pdf_path), "-o", str(output_dir)]
        if self.backend:
            command.extend(["-b", self.backend])

        env = os.environ.copy()
        if self.model_source:
            env["MINERU_MODEL_SOURCE"] = self.model_source

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=env,
            )
        except FileNotFoundError as exc:
            raise MinerUParserError(
                "未找到 mineru CLI。请先安装 MinerU，例如：uv pip install -U \"mineru[all]\"。"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MinerUParserError(f"MinerU 解析超时：{pdf_path}") from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise MinerUParserError(f"MinerU 解析失败：{stderr[:1200]}")

    def _load_mineru_output(self, path: Path, doc_id: str, output_dir: Path) -> ParsedDocument:
        """按优先级读取 MinerU 输出文件并转成 ParsedDocument。"""
        output_file = self._select_output_file(output_dir, "_content_list_v2.json")
        if output_file:
            return self._parse_content_list_v2(path, doc_id, output_file)

        output_file = self._select_output_file(output_dir, "_content_list.json")
        if output_file:
            return self._parse_content_list(path, doc_id, output_file)

        output_file = self._select_output_file(output_dir, "_middle.json")
        if output_file:
            return self._parse_middle_json(path, doc_id, output_file)

        raise MinerUParserError(f"MinerU 输出目录中未找到可解析 JSON：{output_dir}")

    def _parse_content_list_v2(self, path: Path, doc_id: str, output_file: Path) -> ParsedDocument:
        """解析新版 content_list_v2.json。"""
        raw = self._read_json(output_file)
        pages_raw = self._normalize_v2_pages(raw)
        page_data = _PageCollector(doc_id=doc_id, max_pages=self.max_pages)

        for outer_index, blocks in pages_raw:
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                page_number = self._page_number(block, default=outer_index)
                if self._skip_page(page_number):
                    continue
                self._collect_block(page_data, block, page_number)

        return self._build_document(path, doc_id, output_file, page_data, parser_variant="content_list_v2")

    def _parse_content_list(self, path: Path, doc_id: str, output_file: Path) -> ParsedDocument:
        """解析旧版 content_list.json。"""
        raw = self._read_json(output_file)
        blocks = raw.get("content") if isinstance(raw, dict) else raw
        if not isinstance(blocks, list):
            raise MinerUParserError(f"content_list 结构不是列表：{output_file}")

        page_data = _PageCollector(doc_id=doc_id, max_pages=self.max_pages)
        for block in blocks:
            if not isinstance(block, dict):
                continue
            page_number = self._page_number(block, default=1)
            if self._skip_page(page_number):
                continue
            self._collect_block(page_data, block, page_number)

        return self._build_document(path, doc_id, output_file, page_data, parser_variant="content_list")

    def _parse_middle_json(self, path: Path, doc_id: str, output_file: Path) -> ParsedDocument:
        """从 middle.json 中兜底抽取页面文本块。"""
        raw = self._read_json(output_file)
        pdf_info = raw.get("pdf_info", []) if isinstance(raw, dict) else []
        if not isinstance(pdf_info, list):
            raise MinerUParserError(f"middle.json 缺少 pdf_info 列表：{output_file}")

        page_data = _PageCollector(doc_id=doc_id, max_pages=self.max_pages)
        for outer_index, page in enumerate(pdf_info, start=1):
            if not isinstance(page, dict):
                continue
            page_number = int(page.get("page_idx", outer_index - 1)) + 1
            if self._skip_page(page_number):
                continue
            blocks = page.get("para_blocks") or page.get("preproc_blocks") or []
            for block in blocks if isinstance(blocks, list) else []:
                text = self._middle_block_text(block)
                block_type = str(block.get("type", "text")) if isinstance(block, dict) else "text"
                if text:
                    page_data.add_text_block(
                        page_number=page_number,
                        block_type="title" if block_type == "title" else "text",
                        text=text,
                        text_level=_as_int(block.get("level") if isinstance(block, dict) else None, default=0),
                        metadata={"mineru_raw_type": block_type},
                    )

        return self._build_document(path, doc_id, output_file, page_data, parser_variant="middle_json")

    def _collect_block(self, page_data: "_PageCollector", block: dict[str, Any], page_number: int) -> None:
        """把 MinerU block 分流到正文、表格或图注集合。"""
        block_type = str(block.get("type", block.get("block_type", "text"))).lower()
        if block_type in AUXILIARY_TYPES:
            return

        if block_type in TABLE_TYPES:
            page_data.add_table(page_number, block)
            return

        if block_type in FIGURE_TYPES:
            page_data.add_caption(page_number, block)
            return

        if block_type in TEXT_TYPES or block_type:
            text = self._block_text(block)
            if text:
                page_data.add_text_block(
                    page_number=page_number,
                    block_type=block_type,
                    text=text,
                    text_level=_as_int(block.get("text_level", block.get("level")), default=0),
                    metadata={
                        "bbox": block.get("bbox"),
                        "mineru_raw_type": block_type,
                    },
                )

    def _block_text(self, block: dict[str, Any]) -> str:
        """从 MinerU block 中抽取可读文本。"""
        keys = (
            "title_content",
            "paragraph_content",
            "text",
            "content",
            "list_items",
            "math_content",
            "equation",
            "code_content",
            "html",
        )
        for key in keys:
            text = _content_to_text(block.get(key))
            if text:
                return text
        return ""

    def _middle_block_text(self, block: Any) -> str:
        """从 middle.json 的嵌套 line/span 结构中抽取文本。"""
        if not isinstance(block, dict):
            return ""
        if block.get("text"):
            return _normalize_text(str(block["text"]))
        texts: list[str] = []
        for line in block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            line_parts: list[str] = []
            for span in line.get("spans", []) or []:
                if isinstance(span, dict):
                    line_parts.append(_content_to_text(span.get("content") or span.get("text")))
            line_text = _normalize_text("".join(line_parts))
            if line_text:
                texts.append(line_text)
        return "\n".join(texts).strip()

    def _build_document(
        self,
        path: Path,
        doc_id: str,
        output_file: Path,
        page_data: "_PageCollector",
        parser_variant: str,
    ) -> ParsedDocument:
        """组装 FireAgent 标准 ParsedDocument。"""
        pages = page_data.build_pages()
        if not pages:
            raise MinerUParserError(f"MinerU 输出未包含可用正文：{output_file}")

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(path),
            file_name=path.name,
            pages=pages,
            tables=page_data.tables,
            figure_captions=page_data.captions,
            metadata={
                "parser": "mineru",
                "parser_variant": parser_variant,
                "mineru_output_file": str(output_file),
                "mineru_output_dir": str(output_file.parent),
                "mineru_backend": self.backend,
                "mineru_model_source": self.model_source,
            },
        )

    def _find_existing_output(self, output_dir: Path) -> bool:
        """判断输出目录里是否已经有 MinerU 可复用结果。"""
        return bool(
            self._select_output_file(output_dir, "_content_list_v2.json")
            or self._select_output_file(output_dir, "_content_list.json")
            or self._select_output_file(output_dir, "_middle.json")
        )

    @staticmethod
    def _read_json(path: Path) -> Any:
        """读取 UTF-8 JSON 文件。"""
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _select_output_file(output_dir: Path, suffix: str) -> Path | None:
        """在 MinerU 输出目录中查找指定后缀文件，优先返回最新文件。"""
        normalized_suffix = suffix.lstrip("_")
        candidates = [
            path
            for path in output_dir.rglob("*.json")
            if path.is_file()
            and (path.name.endswith(suffix) or path.name.endswith(normalized_suffix))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.stat().st_mtime)

    @staticmethod
    def _normalize_v2_pages(raw: Any) -> list[tuple[int, list[Any]]]:
        """兼容 content_list_v2 的列表或字典包装格式。"""
        if isinstance(raw, dict):
            raw = raw.get("pages") or raw.get("content") or raw.get("pdf_info") or []
        if not isinstance(raw, list):
            raise MinerUParserError("content_list_v2 结构不是列表。")

        pages: list[tuple[int, list[Any]]] = []
        for index, page in enumerate(raw, start=1):
            if isinstance(page, list):
                pages.append((index, page))
            elif isinstance(page, dict):
                blocks = page.get("items") or page.get("blocks") or page.get("content") or []
                page_number = int(page.get("page_idx", index - 1)) + 1
                pages.append((page_number, blocks if isinstance(blocks, list) else []))
        return pages

    @staticmethod
    def _page_number(block: dict[str, Any], default: int) -> int:
        """把 MinerU 的 0-based page_idx 转为 1-based 页码。"""
        if "page_idx" in block:
            return max(_as_int(block.get("page_idx"), default=default - 1) + 1, 1)
        if "page_number" in block:
            return max(_as_int(block.get("page_number"), default=default), 1)
        return max(default, 1)

    def _skip_page(self, page_number: int) -> bool:
        """根据 max_pages 判断是否跳过页面。"""
        return bool(self.max_pages and page_number > self.max_pages)

    @staticmethod
    def _safe_stem(path: Path) -> str:
        """生成适合作为输出目录名的 PDF stem。"""
        stem = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", path.stem, flags=re.UNICODE)
        return stem.strip("._") or "pdf"


class _PageCollector:
    """临时收集 MinerU 页面块并生成 ParsedDocument 组件。"""

    def __init__(self, doc_id: str, max_pages: Optional[int]) -> None:
        self.doc_id = doc_id
        self.max_pages = max_pages
        self.page_lines: dict[int, list[str]] = {}
        self.page_blocks: dict[int, list[dict[str, Any]]] = {}
        self.tables: list[ParsedTable] = []
        self.captions: list[FigureCaption] = []

    def add_text_block(
        self,
        page_number: int,
        block_type: str,
        text: str,
        text_level: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """追加正文块，同时保留 MinerU 原始结构元数据。"""
        text = _normalize_text(text)
        if not text:
            return
        self.page_lines.setdefault(page_number, []).append(text)
        block_metadata = metadata or {}
        self.page_blocks.setdefault(page_number, []).append(
            {
                "type": block_type,
                "text": text,
                "text_level": text_level,
                "page_number": page_number,
                **block_metadata,
            }
        )

    def add_table(self, page_number: int, block: dict[str, Any]) -> None:
        """把 MinerU 表格块转成 ParsedTable。"""
        caption = _content_to_text(block.get("table_caption") or block.get("caption"))
        raw_body = block.get("table_body") or block.get("html") or block.get("text")
        table_body = _content_to_text(raw_body)
        footnote = _content_to_text(block.get("table_footnote") or block.get("footnote"))
        rows = _html_table_to_rows(str(raw_body or ""))
        table_text_parts = [part for part in (caption, _rows_to_text(rows) or table_body, footnote) if part]
        table_index = len(self.tables) + 1
        if not table_text_parts:
            return
        self.tables.append(
            ParsedTable(
                table_id=f"{self.doc_id}-p{page_number}-mineru-t{table_index}",
                page_number=page_number,
                rows=rows,
                text="\n".join(table_text_parts).strip(),
                caption=caption or None,
                metadata={
                    "parser": "mineru",
                    "bbox": block.get("bbox"),
                    "raw_type": block.get("type"),
                },
            )
        )

    def add_caption(self, page_number: int, block: dict[str, Any]) -> None:
        """提取图片或图表说明。"""
        caption = _content_to_text(
            block.get("image_caption")
            or block.get("chart_caption")
            or block.get("caption")
            or block.get("text")
        )
        if not caption:
            return
        caption_index = len(self.captions) + 1
        self.captions.append(
            FigureCaption(
                caption_id=f"{self.doc_id}-p{page_number}-mineru-f{caption_index}",
                page_number=page_number,
                text=caption,
                metadata={
                    "parser": "mineru",
                    "bbox": block.get("bbox"),
                    "raw_type": block.get("type"),
                },
            )
        )

    def build_pages(self) -> list[PDFPage]:
        """按页码顺序生成 PDFPage。"""
        pages: list[PDFPage] = []
        for page_number in sorted(self.page_lines):
            if self.max_pages and page_number > self.max_pages:
                continue
            pages.append(
                PDFPage(
                    page_number=page_number,
                    text="\n".join(self.page_lines[page_number]).strip(),
                    metadata={"mineru_blocks": self.page_blocks.get(page_number, [])},
                )
            )
        return pages


class _TableHTMLParser(HTMLParser):
    """极简 HTML 表格解析器，用于把 MinerU table_body 转成 rows。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(_normalize_text("".join(self._current_cell)))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def _content_to_text(value: Any) -> str:
    """递归提取 MinerU rich content 中的文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_text(_strip_html(value))
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return _normalize_text("\n".join(filter(None, (_content_to_text(item) for item in value))))
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "title_content",
            "paragraph_content",
            "caption",
            "table_body",
            "image_caption",
            "chart_caption",
            "math_content",
        ):
            text = _content_to_text(value.get(key))
            if text:
                return text
    return ""


def _html_table_to_rows(html_text: str) -> list[list[str]]:
    """从 HTML 表格中提取行列；不是 HTML 时返回空列表。"""
    if "<tr" not in html_text.lower():
        return []
    parser = _TableHTMLParser()
    parser.feed(html_text)
    return parser.rows


def _rows_to_text(rows: list[list[str]]) -> str:
    """把表格行列转成制表符分隔文本。"""
    return "\n".join("\t".join(cell for cell in row) for row in rows if any(row)).strip()


def _strip_html(text: str) -> str:
    """去掉 HTML 标签，保留单元格中的可读文本。"""
    if "<" not in text or ">" not in text:
        return text
    text = re.sub(r"</(?:p|div|tr|li|h\d)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:td|th)>", "\t", text, flags=re.IGNORECASE)
    return re.sub(r"<[^>]+>", "", text)


def _normalize_text(text: str) -> str:
    """压缩空白并清理多余空行。"""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _as_int(value: Any, default: int = 0) -> int:
    """把未知值安全转成 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
