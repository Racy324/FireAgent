"""Structure-aware section splitting for Chinese fire-domain papers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from fireagent.ingestion.schema import PaperMetadata, ParsedDocument, Section


CHINESE_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百\d]+[章节篇]\s*.{0,70}$")
NUMERIC_HEADING_RE = re.compile(
    r"^(?P<num>\d+(?:[.．]\d+){0,4})(?:[.．、])?\s*(?P<title>[\u4e00-\u9fffA-Za-z][^\n]{0,70})$"
)
COMMON_HEADING_RE = re.compile(
    r"^(摘要|Abstract|ABSTRACT|引言|绪论|结论|总结|Conclusion|Conclusions|CONCLUSION|"
    r"参考文献|References|REFERENCES|致谢|Acknowledgements?|附录|Appendix)$"
)
BAD_HEADING_ENDINGS = ("。", "；", "，", ",", ";", "：")


@dataclass(frozen=True)
class Heading:
    """A detected section heading."""

    title: str
    level: int
    chunk_type: str = "section"


class SectionSplitter:
    """Split cleaned page text into paper sections while retaining page spans."""

    def split(self, document: ParsedDocument, metadata: PaperMetadata | None = None) -> list[Section]:
        """Split a parsed document into sections."""
        mineru_sections = self._split_mineru_blocks(document, metadata=metadata)
        if mineru_sections:
            return mineru_sections

        sections: list[Section] = []
        current_title = "正文"
        current_path = [current_title]
        current_level = 1
        current_type = "section"
        current_lines: list[str] = []
        page_start = document.pages[0].page_number if document.pages else 1
        page_end = page_start
        heading_stack: list[str] = [current_title]

        def flush() -> None:
            nonlocal current_lines, page_start, page_end
            text = "\n".join(current_lines).strip()
            if not text:
                current_lines = []
                return
            index = len(sections)
            section_id = self._make_section_id(document.doc_id, current_title, page_start, index)
            sections.append(
                Section(
                    section_id=section_id,
                    doc_id=document.doc_id,
                    title=current_title,
                    path=current_path,
                    level=current_level,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    chunk_type=current_type,
                    metadata={"paper_title": metadata.paper_title if metadata else ""},
                )
            )
            current_lines = []

        for page in document.pages:
            page_number = page.page_number
            for raw_line in page.text.splitlines():
                line = raw_line.strip()
                if not line:
                    if current_lines and current_lines[-1] != "":
                        current_lines.append("")
                    continue

                heading = self._detect_heading(line)
                if heading:
                    flush()
                    current_title = heading.title
                    current_level = heading.level
                    current_type = heading.chunk_type
                    heading_stack = self._update_heading_stack(heading_stack, current_title, current_level)
                    current_path = heading_stack.copy()
                    page_start = page_number
                    page_end = page_number
                    continue

                current_lines.append(line)
                page_end = page_number

        flush()

        if sections:
            return sections

        fallback_text = document.text.strip()
        if not fallback_text:
            return []
        return [
            Section(
                section_id=self._make_section_id(document.doc_id, "正文", page_start, 0),
                doc_id=document.doc_id,
                title="正文",
                path=["正文"],
                level=1,
                page_start=page_start,
                page_end=document.pages[-1].page_number if document.pages else page_start,
                text=fallback_text,
                chunk_type="section",
                metadata={"paper_title": metadata.paper_title if metadata else ""},
            )
        ]

    def _detect_heading(self, line: str) -> Heading | None:
        """Detect whether a text line is likely a section heading."""
        normalized = re.sub(r"\s+", " ", line).strip()
        if not self._is_heading_candidate(normalized):
            return None

        common = COMMON_HEADING_RE.match(normalized)
        if common:
            title = common.group(1)
            return Heading(title=title, level=1, chunk_type=self._chunk_type_for_title(title))

        if CHINESE_CHAPTER_RE.match(normalized):
            return Heading(title=normalized, level=1, chunk_type=self._chunk_type_for_title(normalized))

        numeric = NUMERIC_HEADING_RE.match(normalized)
        if numeric:
            num = numeric.group("num")
            level = min(num.count(".") + num.count("．") + 1, 6)
            return Heading(title=normalized, level=level, chunk_type=self._chunk_type_for_title(normalized))

        return None

    def _split_mineru_blocks(
        self,
        document: ParsedDocument,
        metadata: PaperMetadata | None = None,
    ) -> list[Section]:
        """优先使用 MinerU 的 title/text_level 结构切分章节。"""
        structured_blocks: list[dict[str, object]] = []
        for page in document.pages:
            blocks = page.metadata.get("mineru_blocks", [])
            if isinstance(blocks, list):
                structured_blocks.extend(block for block in blocks if isinstance(block, dict))
        if not structured_blocks:
            return []

        sections: list[Section] = []
        current_title = "正文"
        current_path = [current_title]
        current_level = 1
        current_type = "section"
        current_lines: list[str] = []
        page_start = int(structured_blocks[0].get("page_number", 1))
        page_end = page_start
        heading_stack: list[str] = [current_title]

        def flush() -> None:
            nonlocal current_lines, page_start, page_end
            text = "\n".join(current_lines).strip()
            if not text:
                current_lines = []
                return
            index = len(sections)
            sections.append(
                Section(
                    section_id=self._make_section_id(document.doc_id, current_title, page_start, index),
                    doc_id=document.doc_id,
                    title=current_title,
                    path=current_path,
                    level=current_level,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    chunk_type=current_type,
                    metadata={
                        "paper_title": metadata.paper_title if metadata else "",
                        "parser": "mineru",
                    },
                )
            )
            current_lines = []

        for block in structured_blocks:
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            page_number = int(block.get("page_number", page_end))
            block_type = str(block.get("type", "")).lower()
            text_level = self._safe_level(block.get("text_level"))
            is_heading = block_type == "title" or text_level > 0

            if is_heading:
                flush()
                current_title = text
                current_level = text_level or 1
                current_type = self._chunk_type_for_title(current_title)
                heading_stack = self._update_heading_stack(heading_stack, current_title, current_level)
                current_path = heading_stack.copy()
                page_start = page_number
                page_end = page_number
                continue

            current_lines.append(text)
            page_end = page_number

        flush()
        if sections:
            return sections

        fallback_text = document.text.strip()
        if not fallback_text:
            return []
        return [
            Section(
                section_id=self._make_section_id(document.doc_id, "正文", page_start, 0),
                doc_id=document.doc_id,
                title="正文",
                path=["正文"],
                level=1,
                page_start=page_start,
                page_end=document.pages[-1].page_number if document.pages else page_start,
                text=fallback_text,
                chunk_type="section",
                metadata={
                    "paper_title": metadata.paper_title if metadata else "",
                    "parser": "mineru",
                },
            )
        ]

    @staticmethod
    def _is_heading_candidate(line: str) -> bool:
        """Apply conservative filters before heading regex checks."""
        if len(line) > 90:
            return False
        if line.endswith(BAD_HEADING_ENDINGS):
            return False
        if re.search(r"[。；;]\s*$", line):
            return False
        if len(line) <= 1:
            return False
        return True

    @staticmethod
    def _chunk_type_for_title(title: str) -> str:
        """Map a section title to a chunk type."""
        lowered = title.lower()
        if "摘要" in title or lowered == "abstract":
            return "abstract"
        if "参考文献" in title or lowered == "references":
            return "references"
        if "关键词" in title:
            return "keywords"
        return "section"

    @staticmethod
    def _update_heading_stack(stack: list[str], title: str, level: int) -> list[str]:
        """Update hierarchical section path for a new heading."""
        if level <= 1:
            return [title]
        new_stack = stack[: level - 1]
        while len(new_stack) < level - 1:
            new_stack.append(new_stack[-1] if new_stack else "正文")
        new_stack.append(title)
        return new_stack

    @staticmethod
    def _safe_level(value: object) -> int:
        """把 MinerU 的 text_level 转成 0-6 的章节层级。"""
        try:
            level = int(value)
        except (TypeError, ValueError):
            return 0
        return min(max(level, 0), 6)

    @staticmethod
    def _make_section_id(doc_id: str, title: str, page_start: int, index: int) -> str:
        """Create a stable section id."""
        raw = f"{doc_id}|{title}|{page_start}|{index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
