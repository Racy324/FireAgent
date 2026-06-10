"""Paper metadata extraction from parsed PDF text and filenames."""

from __future__ import annotations

import re
from pathlib import Path

from fireagent.ingestion.schema import PaperMetadata, ParsedDocument


ABSTRACT_RE = re.compile(
    r"(?:^|\n)\s*摘\s*要\s*[:：]?\s*(?P<abstract>.*?)(?=\n\s*(?:关\s*键\s*词|关键字|Abstract|目\s*录|第[一二三四五六七八九十百\d]+[章节]|1[.．、\s]|引言|绪论))",
    re.DOTALL,
)
KEYWORDS_RE = re.compile(r"(?:关\s*键\s*词|关键字)\s*[:：]?\s*(?P<keywords>[^\n]{2,200})")
AUTHOR_RE = re.compile(r"(?:作者|研究生|姓名)\s*[:：]\s*(?P<authors>[^\n]{2,80})")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
FILENAME_COPY_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")


class PaperMetadataExtractor:
    """Extract title, abstract, keywords, authors, and year from a parsed paper."""

    def extract(self, document: ParsedDocument) -> PaperMetadata:
        """Extract best-effort paper metadata."""
        full_text = document.text
        title_from_file, authors_from_file = self._extract_from_filename(document.source_path)
        raw_title = self._clean_metadata_value(document.metadata.get("Title", ""))
        title_from_text = self._extract_title_from_text(full_text)

        paper_title = self._choose_title(title_from_file, title_from_text, raw_title)
        authors = self._extract_authors(full_text) or authors_from_file
        year = self._extract_year(document, full_text)
        abstract = self._extract_abstract(full_text)
        keywords = self._extract_keywords(full_text)

        return PaperMetadata(
            doc_id=document.doc_id,
            paper_title=paper_title,
            authors=authors,
            year=year,
            abstract=abstract,
            keywords=keywords,
            source_path=document.source_path,
            raw_metadata=document.metadata,
        )

    def _extract_from_filename(self, source_path: str) -> tuple[str, list[str]]:
        """Infer title and authors from common CNKI filename patterns."""
        stem = FILENAME_COPY_SUFFIX_RE.sub("", Path(source_path).stem).strip()
        if "_" not in stem:
            return stem, []
        title, author_part = stem.rsplit("_", 1)
        authors = self._split_people(author_part)
        return title.strip(), authors

    @staticmethod
    def _clean_metadata_value(value: object) -> str:
        """Normalize a raw PDF metadata field."""
        if not isinstance(value, str):
            return ""
        value = re.sub(r"\s+", " ", value).strip()
        if value.lower() in {"untitled", "none"}:
            return ""
        return value

    def _choose_title(self, title_from_file: str, title_from_text: str, raw_title: str) -> str:
        """Choose the most reliable title candidate for CNKI-style PDFs."""
        for candidate in (title_from_file, title_from_text, raw_title):
            candidate = candidate.strip()
            if candidate and not self._is_bad_title(candidate):
                return candidate
        return title_from_file or title_from_text or raw_title

    @staticmethod
    def _is_bad_title(title: str) -> bool:
        """Filter generic PDF metadata titles that are not paper titles."""
        normalized = title.lower()
        bad_markers = (
            "university",
            "cnki",
            "cajviewer",
            "microsoft word",
            "untitled",
            "硕士学位论文",
            "博士学位论文",
            "电子科技大学",
            "毕业论文",
        )
        return any(marker in normalized or marker in title for marker in bad_markers)

    def _extract_title_from_text(self, text: str) -> str:
        """Choose a likely title from the first page text."""
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()[:40]]
        lines = [line for line in lines if self._is_title_candidate(line)]
        if not lines:
            return ""
        return max(lines[:8], key=len)

    @staticmethod
    def _is_title_candidate(line: str) -> bool:
        """Return whether a line can be a paper title."""
        if not (6 <= len(line) <= 80):
            return False
        noise_words = ("摘要", "关键词", "目录", "作者", "导师", "学院", "大学", "硕士", "博士")
        if any(word == line or line.startswith(f"{word}:") or line.startswith(f"{word}：") for word in noise_words):
            return False
        return not YEAR_RE.fullmatch(line)

    def _extract_authors(self, text: str) -> list[str]:
        """Extract author names from common labeled lines."""
        first_text = "\n".join(text.splitlines()[:80])
        match = AUTHOR_RE.search(first_text)
        if not match:
            return []
        return self._split_people(match.group("authors"))

    @staticmethod
    def _split_people(raw: str) -> list[str]:
        """Split a compact author string."""
        raw = re.sub(r"\s+", "", raw.strip())
        raw = re.sub(r"(等|著)$", "", raw)
        parts = re.split(r"[、,，;/；]+", raw)
        return [part for part in parts if 1 < len(part) <= 12]

    @staticmethod
    def _extract_year(document: ParsedDocument, text: str) -> int | None:
        """Extract a likely publication or thesis year."""
        candidates: list[int] = []
        for raw in document.metadata.values():
            if isinstance(raw, str):
                candidates.extend(int(match.group()) for match in YEAR_RE.finditer(raw))
        candidates.extend(int(match.group()) for match in YEAR_RE.finditer(Path(document.source_path).stem))
        candidates.extend(int(match.group()) for match in YEAR_RE.finditer("\n".join(text.splitlines()[:120])))
        candidates = [year for year in candidates if 1900 <= year <= 2100]
        if not candidates:
            return None
        return max(set(candidates), key=candidates.count)

    @staticmethod
    def _extract_abstract(text: str) -> str:
        """Extract a Chinese abstract block."""
        match = ABSTRACT_RE.search(text)
        if not match:
            return ""
        abstract = re.sub(r"\s+", " ", match.group("abstract")).strip()
        return abstract[:3000]

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract Chinese keywords."""
        match = KEYWORDS_RE.search(text)
        if not match:
            return []
        raw = match.group("keywords")
        raw = re.sub(r"\s+", "", raw)
        parts = re.split(r"[;；,，、\s]+", raw)
        return [part for part in parts if 1 < len(part) <= 40]
