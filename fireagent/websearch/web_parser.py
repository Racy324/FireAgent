"""将联网搜索结果解析为临时 evidence chunk。"""

from __future__ import annotations

import re
from typing import Optional

from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.websearch.schema import (
    TavilySearchResponse,
    TavilySearchResult,
    WebEvidenceChunk,
    make_web_chunk_id,
    make_web_doc_id,
)


class WebSearchResultParser:
    """把 Tavily 搜索结果转换成 WebEvidenceChunk。"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunks_per_source: Optional[int] = None,
        config: Optional[FireAgentConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self.chunk_size = chunk_size or self.config.rag.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else self.config.rag.chunk_overlap
        self.chunks_per_source = chunks_per_source or self.config.tavily.chunks_per_source

    def parse_response(self, response: TavilySearchResponse) -> list[WebEvidenceChunk]:
        """解析整份 Tavily 搜索响应。"""
        chunks: list[WebEvidenceChunk] = []
        for result_index, result in enumerate(response.results):
            chunks.extend(self.parse_result(result, result_index=result_index, query=response.query))

        # Tavily answer 是搜索引擎综合摘要，不等同网页原文；保留为弱证据，排序时会带较低分。
        if response.answer.strip():
            answer_text = self._clean_text(response.answer)
            chunk_id = make_web_chunk_id("tavily://answer", answer_text, 0)
            chunks.append(
                WebEvidenceChunk(
                    chunk_id=chunk_id,
                    parent_id=chunk_id,
                    doc_id=make_web_doc_id("tavily://answer"),
                    title="Tavily 搜索摘要",
                    url="",
                    text=answer_text,
                    snippet=answer_text[:300],
                    score=0.2,
                    metadata={"query": response.query, "evidence_kind": "search_answer"},
                )
            )
        return chunks

    def parse_result(
        self,
        result: TavilySearchResult,
        result_index: int = 0,
        query: str = "",
    ) -> list[WebEvidenceChunk]:
        """解析单条 Tavily 搜索结果。"""
        source_text = result.raw_content or result.content
        source_text = self._clean_text(source_text)
        if not source_text:
            return []

        doc_id = make_web_doc_id(result.url or f"tavily-result-{result_index}")
        parent_id = doc_id
        text_chunks = self._split_text(source_text)
        evidence_chunks: list[WebEvidenceChunk] = []

        for index, text in enumerate(text_chunks[: self.chunks_per_source]):
            chunk_id = make_web_chunk_id(result.url, text, index)
            evidence_chunks.append(
                WebEvidenceChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    doc_id=doc_id,
                    title=result.title,
                    url=result.url,
                    text=text,
                    snippet=result.content[:500],
                    score=result.score,
                    published_date=result.published_date,
                    metadata={
                        **result.metadata,
                        "query": query,
                        "result_index": result_index,
                        "chunk_index": index,
                        "evidence_kind": "search_result",
                    },
                )
            )
        return evidence_chunks

    def _split_text(self, text: str) -> list[str]:
        """按网页段落递归切分文本。"""
        if len(text) <= self.chunk_size:
            return [text]

        pieces = self._recursive_split(text, self.chunk_size, ["\n\n", "\n", "。", "；", "，", " ", ""])
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = current + piece if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
            if self.chunk_overlap and chunks:
                current = chunks[-1][-self.chunk_overlap :] + piece
                if len(current) > self.chunk_size:
                    current = current[-self.chunk_size :]
            else:
                current = piece
        if current.strip():
            chunks.append(current.strip())
        return [chunk for chunk in chunks if chunk.strip()]

    def _recursive_split(self, text: str, max_size: int, separators: list[str]) -> list[str]:
        """使用递归分隔符切分长文本。"""
        if len(text) <= max_size:
            return [text]
        if not separators:
            return [text[index : index + max_size] for index in range(0, len(text), max_size)]

        separator = separators[0]
        if separator == "":
            return [text[index : index + max_size] for index in range(0, len(text), max_size)]
        if separator not in text:
            return self._recursive_split(text, max_size, separators[1:])

        parts: list[str] = []
        raw_parts = text.split(separator)
        for index, part in enumerate(raw_parts):
            if index < len(raw_parts) - 1:
                part = part + separator
            if len(part) > max_size:
                parts.extend(self._recursive_split(part, max_size, separators[1:]))
            else:
                parts.append(part)
        return parts

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗网页摘要或正文文本。"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

