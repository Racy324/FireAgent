"""Build standard DocumentChunk objects from parsed PDF documents."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fireagent.ingestion.cleaner import PDFTextCleaner
from fireagent.ingestion.metadata_extractor import PaperMetadataExtractor
from fireagent.ingestion.mineru_parser import MinerUPDFParser
from fireagent.ingestion.pdf_parser import BasePDFParser
from fireagent.ingestion.pdfplumber_parser import PdfPlumberPDFParser
from fireagent.ingestion.schema import (
    ChunkDraft,
    DocumentChunk,
    PaperMetadata,
    ParsedDocument,
)
from fireagent.ingestion.section_splitter import SectionSplitter
from fireagent.ingestion.semantic_chunker import RecursiveSemanticChunker
from fireagent.utils.config import FireAgentConfig, get_config


class PDFIndexBuilder:
    """End-to-end builder from parsed PDFs to standard document chunks."""

    def __init__(
        self,
        parser: BasePDFParser | None = None,
        cleaner: PDFTextCleaner | None = None,
        metadata_extractor: PaperMetadataExtractor | None = None,
        section_splitter: SectionSplitter | None = None,
        chunker: RecursiveSemanticChunker | None = None,
    ) -> None:
        self.parser = parser or PdfPlumberPDFParser()
        self.cleaner = cleaner or PDFTextCleaner()
        self.metadata_extractor = metadata_extractor or PaperMetadataExtractor()
        self.section_splitter = section_splitter or SectionSplitter()
        self.chunker = chunker or RecursiveSemanticChunker()

    @classmethod
    def from_config(
        cls,
        config: FireAgentConfig | None = None,
        parser_name: str | None = None,
        max_pages: int | None = None,
    ) -> "PDFIndexBuilder":
        """Create a builder using configured chunk sizes."""
        cfg = config or get_config()
        return cls(
            parser=create_pdf_parser_from_config(cfg, parser_name=parser_name, max_pages=max_pages),
            chunker=RecursiveSemanticChunker(
                chunk_size=cfg.rag.chunk_size,
                chunk_overlap=cfg.rag.chunk_overlap,
                parent_chunk_size=cfg.rag.parent_chunk_size,
            )
        )

    def build_from_pdf(self, pdf_path: str | Path) -> list[DocumentChunk]:
        """Parse, clean, split, and chunk a PDF file."""
        document = self.parser.parse(pdf_path)
        return self.build_from_document(document)

    def build_from_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Convert a parsed document into standard chunks."""
        cleaned_document = self.cleaner.clean_document(document)
        metadata = self.metadata_extractor.extract(cleaned_document)
        sections = self.section_splitter.split(cleaned_document, metadata=metadata)
        drafts = self.chunker.split_sections(sections)
        chunks = self._attach_metadata(drafts, metadata, cleaned_document)
        chunks.extend(self._table_chunks(cleaned_document, metadata))
        chunks.extend(self._figure_caption_chunks(cleaned_document, metadata))
        return chunks

    def _attach_metadata(
        self,
        drafts: list[ChunkDraft],
        metadata: PaperMetadata,
        document: ParsedDocument,
    ) -> list[DocumentChunk]:
        """Attach paper-level metadata to chunk drafts."""
        chunks: list[DocumentChunk] = []
        for index, draft in enumerate(drafts):
            chunk_id = self._make_chunk_id(draft.doc_id, draft.parent_id, index, draft.text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    parent_id=draft.parent_id,
                    doc_id=draft.doc_id,
                    paper_title=metadata.paper_title,
                    authors=metadata.authors,
                    year=metadata.year,
                    section_title=draft.section_title,
                    section_path=draft.section_path,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    text=draft.text,
                    chunk_type=draft.chunk_type,
                    source_type="local_pdf",
                    metadata={
                        **draft.metadata,
                        "source_path": document.source_path,
                        "file_name": document.file_name,
                        "abstract": metadata.abstract,
                        "keywords": metadata.keywords,
                    },
                )
            )
        return chunks

    def _table_chunks(
        self,
        document: ParsedDocument,
        metadata: PaperMetadata,
    ) -> list[DocumentChunk]:
        """Convert extracted tables into chunks."""
        chunks: list[DocumentChunk] = []
        for index, table in enumerate(document.tables):
            if not table.text.strip():
                continue
            parent_id = self._make_aux_parent_id(document.doc_id, table.table_id)
            section_title = table.caption or f"Table on page {table.page_number}"
            chunks.append(
                DocumentChunk(
                    chunk_id=self._make_chunk_id(document.doc_id, parent_id, index, table.text),
                    parent_id=parent_id,
                    doc_id=document.doc_id,
                    paper_title=metadata.paper_title,
                    authors=metadata.authors,
                    year=metadata.year,
                    section_title=section_title,
                    section_path=[section_title],
                    page_start=table.page_number,
                    page_end=table.page_number,
                    text=table.text,
                    chunk_type="table",
                    source_type="local_pdf",
                    metadata={
                        "source_path": document.source_path,
                        "file_name": document.file_name,
                        "table_id": table.table_id,
                        **table.metadata,
                    },
                )
            )
        return chunks

    def _figure_caption_chunks(
        self,
        document: ParsedDocument,
        metadata: PaperMetadata,
    ) -> list[DocumentChunk]:
        """Convert detected figure captions into chunks."""
        chunks: list[DocumentChunk] = []
        for index, caption in enumerate(document.figure_captions):
            parent_id = self._make_aux_parent_id(document.doc_id, caption.caption_id)
            chunks.append(
                DocumentChunk(
                    chunk_id=self._make_chunk_id(document.doc_id, parent_id, index, caption.text),
                    parent_id=parent_id,
                    doc_id=document.doc_id,
                    paper_title=metadata.paper_title,
                    authors=metadata.authors,
                    year=metadata.year,
                    section_title=caption.text[:80],
                    section_path=["图注"],
                    page_start=caption.page_number,
                    page_end=caption.page_number,
                    text=caption.text,
                    chunk_type="figure_caption",
                    source_type="local_pdf",
                    metadata={
                        "source_path": document.source_path,
                        "file_name": document.file_name,
                        "caption_id": caption.caption_id,
                        **caption.metadata,
                    },
                )
            )
        return chunks

    @staticmethod
    def _make_chunk_id(doc_id: str, parent_id: str, index: int, text: str) -> str:
        """Create a stable chunk id."""
        raw = f"{doc_id}|{parent_id}|{index}|{text[:120]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _make_aux_parent_id(doc_id: str, aux_id: str) -> str:
        """Create a parent id for tables and figure captions."""
        raw = f"{doc_id}|aux|{aux_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_chunks_from_document(document: ParsedDocument) -> list[DocumentChunk]:
    """Convenience function for building chunks from a parsed document."""
    return PDFIndexBuilder.from_config().build_from_document(document)


def build_chunks_from_pdf(pdf_path: str | Path) -> list[DocumentChunk]:
    """Convenience function for building chunks directly from a PDF path."""
    return PDFIndexBuilder.from_config().build_from_pdf(pdf_path)


def create_pdf_parser_from_config(
    config: FireAgentConfig | None = None,
    parser_name: str | None = None,
    max_pages: int | None = None,
) -> BasePDFParser:
    """根据配置创建 PDF 解析器。"""
    cfg = config or get_config()
    selected = (parser_name or cfg.ingestion.pdf_parser).lower()
    if selected == "mineru":
        return MinerUPDFParser(
            output_dir=cfg.ingestion.mineru_output_dir,
            backend=cfg.ingestion.mineru_backend,
            model_source=cfg.ingestion.mineru_model_source,
            timeout=cfg.ingestion.mineru_timeout,
            keep_output=cfg.ingestion.mineru_keep_output,
            cli_path=cfg.ingestion.mineru_cli_path,
            max_pages=max_pages,
        )
    if selected == "mineru_api":
        from fireagent.ingestion.mineru_api_parser import MinerUAPIParser

        return MinerUAPIParser(
            api_key=cfg.ingestion.mineru_api_key,
            base_url=cfg.ingestion.mineru_api_base_url,
            timeout=cfg.ingestion.mineru_timeout,
            output_dir=cfg.ingestion.mineru_output_dir,
            keep_output=cfg.ingestion.mineru_keep_output,
            max_pages=max_pages,
            enable_ocr=cfg.ingestion.mineru_api_enable_ocr,
            enable_formula=cfg.ingestion.mineru_api_enable_formula,
            enable_table=cfg.ingestion.mineru_api_enable_table,
            language=cfg.ingestion.mineru_api_language,
        )
    if selected == "pdfplumber":
        return PdfPlumberPDFParser(max_pages=max_pages)
    raise ValueError(f"Unsupported PDF parser: {selected}")
