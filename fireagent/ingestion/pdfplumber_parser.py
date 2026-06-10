"""PDF parser backed by pdfplumber."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from fireagent.ingestion.pdf_parser import BasePDFParser
from fireagent.ingestion.schema import FigureCaption, ParsedDocument, ParsedTable, PDFPage


FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:图|Fig\.?|Figure)\s*[\d一二三四五六七八九十IVXivx\-_.．]+[：:、.\s].{2,120}$",
    re.IGNORECASE,
)


class PdfPlumberPDFParser(BasePDFParser):
    """Extract page text, simple tables, and figure captions with pdfplumber."""

    def __init__(
        self,
        x_tolerance: float = 1.5,
        y_tolerance: float = 3.0,
        max_pages: Optional[int] = None,
    ) -> None:
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        self.max_pages = max_pages

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF into page-level text and lightweight layout artifacts."""
        path = self.validate_pdf_path(pdf_path)

        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PdfPlumberPDFParser. "
                "Install project dependencies or run: pip install pdfplumber"
            ) from exc

        doc_id = self.make_doc_id(path)
        pages: list[PDFPage] = []
        tables: list[ParsedTable] = []
        captions: list[FigureCaption] = []

        with pdfplumber.open(path) as pdf:
            raw_metadata: dict[str, Any] = dict(pdf.metadata or {})

            pages_to_parse = pdf.pages[: self.max_pages] if self.max_pages else pdf.pages
            for page_index, page in enumerate(pages_to_parse, start=1):
                text = page.extract_text(
                    x_tolerance=self.x_tolerance,
                    y_tolerance=self.y_tolerance,
                ) or ""

                pages.append(
                    PDFPage(
                        page_number=page_index,
                        text=text,
                        width=float(page.width) if page.width else None,
                        height=float(page.height) if page.height else None,
                    )
                )

                for table_index, rows in enumerate(page.extract_tables() or [], start=1):
                    cleaned_rows = self._clean_table_rows(rows)
                    table_text = self._table_to_text(cleaned_rows)
                    tables.append(
                        ParsedTable(
                            table_id=f"{doc_id}-p{page_index}-t{table_index}",
                            page_number=page_index,
                            rows=cleaned_rows,
                            text=table_text,
                        )
                    )

                for caption_index, caption in enumerate(self._extract_figure_captions(text), start=1):
                    captions.append(
                        FigureCaption(
                            caption_id=f"{doc_id}-p{page_index}-f{caption_index}",
                            page_number=page_index,
                            text=caption,
                        )
                    )

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(path),
            file_name=path.name,
            pages=pages,
            tables=tables,
            figure_captions=captions,
            metadata=raw_metadata,
        )

    @staticmethod
    def _clean_table_rows(rows: list[list[Any]]) -> list[list[str]]:
        """Normalize pdfplumber table rows into strings."""
        cleaned: list[list[str]] = []
        for row in rows:
            cleaned.append(["" if cell is None else str(cell).strip() for cell in row])
        return cleaned

    @staticmethod
    def _table_to_text(rows: list[list[str]]) -> str:
        """Convert table rows to a tab-separated plain-text representation."""
        return "\n".join("\t".join(cell for cell in row) for row in rows if any(row)).strip()

    @staticmethod
    def _extract_figure_captions(text: str) -> list[str]:
        """Detect figure captions from extracted page text."""
        captions: list[str] = []
        for line in text.splitlines():
            normalized = " ".join(line.strip().split())
            if normalized and FIGURE_CAPTION_RE.match(normalized):
                captions.append(normalized)
        return captions
