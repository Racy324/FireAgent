"""Base interfaces for PDF parsers."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from fireagent.ingestion.schema import ParsedDocument


class BasePDFParser(ABC):
    """Abstract PDF parser interface used by the ingestion pipeline."""

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF file into a ``ParsedDocument``."""

    def validate_pdf_path(self, pdf_path: str | Path) -> Path:
        """Validate and normalize a PDF path."""
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path}")
        return path

    def make_doc_id(self, pdf_path: str | Path) -> str:
        """Create a stable document id from file name, size, and absolute path."""
        path = Path(pdf_path).expanduser().resolve()
        stat = path.stat() if path.exists() else None
        size = stat.st_size if stat else 0
        raw = f"{path.name}|{size}|{path.as_posix()}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

