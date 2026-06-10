"""Structure-first recursive semantic chunking."""

from __future__ import annotations

import hashlib
import re

from fireagent.ingestion.schema import ChunkDraft, Section


DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "，", ",", " ", ""]


class RecursiveSemanticChunker:
    """Split sections into parent-aware child chunks using recursive separators."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        parent_chunk_size: int = 1800,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if parent_chunk_size < chunk_size:
            raise ValueError("parent_chunk_size must be greater than or equal to chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.separators = separators or DEFAULT_SEPARATORS

    def split_sections(self, sections: list[Section]) -> list[ChunkDraft]:
        """Split all sections into chunk drafts."""
        drafts: list[ChunkDraft] = []
        for section in sections:
            drafts.extend(self.split_section(section))
        return drafts

    def split_section(self, section: Section) -> list[ChunkDraft]:
        """Split one section into parent-aware child chunks."""
        text = self._normalize_text(section.text)
        if not text:
            return []

        parent_texts = self._split_without_overlap(text, self.parent_chunk_size)
        drafts: list[ChunkDraft] = []

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = self._make_parent_id(section, parent_index)
            child_texts = self._split_with_overlap(parent_text)
            for child_index, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                drafts.append(
                    ChunkDraft(
                        parent_id=parent_id,
                        doc_id=section.doc_id,
                        section_id=section.section_id,
                        section_title=section.title,
                        section_path=section.path,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        text=child_text.strip(),
                        chunk_type=section.chunk_type,
                        metadata={
                            **section.metadata,
                            "parent_index": parent_index,
                            "child_index": child_index,
                        },
                    )
                )

        return drafts

    def _split_with_overlap(self, text: str) -> list[str]:
        """Split text into child chunks, adding overlap from the previous chunk."""
        if len(text) <= self.chunk_size:
            return [text]

        base_size = self.chunk_size - self.chunk_overlap if self.chunk_overlap else self.chunk_size
        base_chunks = self._split_without_overlap(text, base_size)
        if not self.chunk_overlap:
            return base_chunks

        chunks: list[str] = []
        previous_base = ""
        for index, chunk in enumerate(base_chunks):
            if index == 0:
                chunks.append(chunk)
            else:
                prefix = previous_base[-self.chunk_overlap :]
                chunks.append((prefix + chunk)[-self.chunk_size :])
            previous_base = chunk
        return chunks

    def _split_without_overlap(self, text: str, max_size: int) -> list[str]:
        """Recursively split text into chunks no longer than ``max_size``."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_size:
            return [text]

        pieces = self._recursive_split(text, max_size, self.separators)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            candidate = current + piece if current else piece
            if len(candidate) <= max_size:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
            current = piece

        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if chunk]

    def _recursive_split(self, text: str, max_size: int, separators: list[str]) -> list[str]:
        """Split text by the first useful separator, recursing when pieces are too long."""
        if len(text) <= max_size:
            return [text]
        if not separators:
            return [text[index : index + max_size] for index in range(0, len(text), max_size)]

        separator = separators[0]
        if separator == "":
            return [text[index : index + max_size] for index in range(0, len(text), max_size)]

        if separator not in text:
            return self._recursive_split(text, max_size, separators[1:])

        raw_parts = text.split(separator)
        parts: list[str] = []
        for index, part in enumerate(raw_parts):
            if index < len(raw_parts) - 1:
                part = part + separator
            if len(part) > max_size:
                parts.extend(self._recursive_split(part, max_size, separators[1:]))
            else:
                parts.append(part)
        return parts

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize excessive whitespace without flattening paragraph boundaries."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _make_parent_id(section: Section, parent_index: int) -> str:
        """Create a stable parent chunk id."""
        raw = f"{section.doc_id}|{section.section_id}|parent|{parent_index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

