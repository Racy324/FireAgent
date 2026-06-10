"""Shared data structures for FireAgent PDF ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class IngestionModel(BaseModel):
    """Base model for ingestion data structures."""

    model_config = ConfigDict(extra="ignore")


class PDFPage(IngestionModel):
    """Text extracted from a single PDF page."""

    page_number: int = Field(gt=0)
    text: str = ""
    width: Optional[float] = None
    height: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedTable(IngestionModel):
    """A table extracted from a PDF page."""

    table_id: str
    page_number: int = Field(gt=0)
    rows: list[list[str]] = Field(default_factory=list)
    text: str = ""
    caption: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FigureCaption(IngestionModel):
    """A figure caption detected in page text."""

    caption_id: str
    page_number: int = Field(gt=0)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(IngestionModel):
    """A parsed PDF document before metadata extraction and chunking."""

    doc_id: str
    source_path: str
    file_name: str
    pages: list[PDFPage] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    figure_captions: list[FigureCaption] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return full document text joined in page order."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def page_count(self) -> int:
        """Return the number of extracted pages."""
        return len(self.pages)

    @property
    def path(self) -> Path:
        """Return the source path as a ``Path`` object."""
        return Path(self.source_path)


class PaperMetadata(IngestionModel):
    """Metadata inferred from PDF text, filename, and PDF metadata."""

    doc_id: str
    paper_title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    source_path: str = ""
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class Section(IngestionModel):
    """A structure-aware section from a parsed paper."""

    section_id: str
    doc_id: str
    title: str
    path: list[str] = Field(default_factory=list)
    level: int = Field(default=1, ge=1)
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    text: str
    chunk_type: str = "section"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkDraft(IngestionModel):
    """An intermediate chunk before paper-level metadata is attached."""

    parent_id: str
    doc_id: str
    section_id: str
    section_title: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    text: str
    chunk_type: str = "section"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(IngestionModel):
    """Standard local evidence chunk written to vector stores and retrievers."""

    chunk_id: str
    parent_id: str
    doc_id: str
    paper_title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    section_title: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    text: str
    chunk_type: str
    source_type: str = "local_pdf"
    metadata: dict[str, Any] = Field(default_factory=dict)
