"""PDF ingestion and chunking package for FireAgent."""

from fireagent.ingestion.index_builder import (
    PDFIndexBuilder,
    build_chunks_from_document,
    build_chunks_from_pdf,
    create_pdf_parser_from_config,
)
from fireagent.ingestion.mineru_parser import MinerUPDFParser
from fireagent.ingestion.pdf_parser import BasePDFParser
from fireagent.ingestion.pdfplumber_parser import PdfPlumberPDFParser
from fireagent.ingestion.schema import (
    ChunkDraft,
    DocumentChunk,
    FigureCaption,
    PaperMetadata,
    ParsedDocument,
    ParsedTable,
    PDFPage,
    Section,
)

__all__ = [
    "BasePDFParser",
    "ChunkDraft",
    "DocumentChunk",
    "FigureCaption",
    "PDFIndexBuilder",
    "PDFPage",
    "PaperMetadata",
    "ParsedDocument",
    "ParsedTable",
    "MinerUPDFParser",
    "PdfPlumberPDFParser",
    "Section",
    "build_chunks_from_document",
    "build_chunks_from_pdf",
    "create_pdf_parser_from_config",
]
