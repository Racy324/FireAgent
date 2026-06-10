"""Cleaning utilities for parsed PDF text."""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, Field

from fireagent.ingestion.schema import ParsedDocument, PDFPage


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
REPEATED_SYMBOL_RE = re.compile(r"([=_\-~*·•])\1{5,}")
PAGE_NUMBER_RE = re.compile(r"^\s*(?:第\s*)?\d{1,4}\s*(?:页)?\s*$")
REFERENCE_LINE_RE = re.compile(r"^\s*(?:\[\d+\]|\d+\s*[.．、])\s*.{8,}$")


class CleanerConfig(BaseModel):
    """Options controlling conservative PDF text cleaning."""

    remove_repeated_headers: bool = True
    remove_page_numbers: bool = True
    remove_reference_noise_lines: bool = True
    min_repeated_boundary_count: int = Field(default=3, ge=2)


class PDFTextCleaner:
    """Clean PDF page text while preserving page boundaries."""

    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()

    def clean_document(self, document: ParsedDocument) -> ParsedDocument:
        """Return a copy of ``document`` with cleaned page text."""
        boundary_noise = self._detect_repeated_boundary_lines(document.pages)
        cleaned_pages: list[PDFPage] = []

        for page in document.pages:
            text = self.clean_text(page.text, boundary_noise=boundary_noise)
            cleaned_pages.append(page.model_copy(update={"text": text}))

        return document.model_copy(update={"pages": cleaned_pages})

    def clean_text(self, text: str, boundary_noise: set[str] | None = None) -> str:
        """Clean a page or section text string."""
        boundary_noise = boundary_noise or set()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = CONTROL_CHAR_RE.sub("", text)
        text = REPEATED_SYMBOL_RE.sub(lambda match: match.group(1) * 3, text)

        cleaned_lines: list[str] = []
        previous_line = ""
        for raw_line in text.splitlines():
            line = self._normalize_line(raw_line)
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            if line in boundary_noise:
                continue
            if self.config.remove_page_numbers and PAGE_NUMBER_RE.match(line):
                continue
            if self._is_reference_noise_line(line):
                continue
            if line == previous_line:
                continue
            cleaned_lines.append(line)
            previous_line = line

        return self._collapse_blank_lines("\n".join(cleaned_lines)).strip()

    def _detect_repeated_boundary_lines(self, pages: list[PDFPage]) -> set[str]:
        """Find repeated first/last lines that likely represent headers or footers."""
        if not self.config.remove_repeated_headers or len(pages) < self.config.min_repeated_boundary_count:
            return set()

        candidates: list[str] = []
        for page in pages:
            lines = [self._normalize_line(line) for line in page.text.splitlines()]
            lines = [line for line in lines if self._is_boundary_candidate(line)]
            if not lines:
                continue
            candidates.extend(lines[:2])
            candidates.extend(lines[-2:])

        counter = Counter(candidates)
        threshold = max(self.config.min_repeated_boundary_count, int(len(pages) * 0.45))
        return {line for line, count in counter.items() if count >= threshold}

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Normalize whitespace within a single line."""
        return re.sub(r"\s+", " ", line).strip()

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """Collapse runs of blank lines."""
        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def _is_boundary_candidate(line: str) -> bool:
        """Return whether a line is plausible header/footer noise."""
        if not line:
            return False
        if PAGE_NUMBER_RE.match(line):
            return True
        return 3 <= len(line) <= 80

    def _is_reference_noise_line(self, line: str) -> bool:
        """Drop citation-style reference lines that rarely help QA evidence."""
        if not self.config.remove_reference_noise_lines:
            return False
        if "中国知网" in line or "CNKI" in line.upper():
            return True
        return bool(REFERENCE_LINE_RE.match(line) and len(line) > 60)

