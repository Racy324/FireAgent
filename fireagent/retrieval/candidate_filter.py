"""常规 RAG 候选过滤：按 chunk_type 过滤 references 和极短 figure_caption。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from fireagent.utils.config import FireAgentConfig, get_config


class RetrievalCandidate(Protocol):
    """过滤器接受的候选结果接口。"""

    text: str
    payload: dict[str, Any]


T = TypeVar("T", bound=RetrievalCandidate)


@dataclass
class CandidateFilterStats:
    """过滤统计信息。"""

    input_count: int = 0
    output_count: int = 0
    filtered_references: int = 0
    filtered_short_figure_captions: int = 0
    bypassed: bool = False
    mode: str = "off"


class RegularRAGCandidateFilter:
    """按 chunk_type 过滤常规 RAG 候选。"""

    def __init__(self, config: FireAgentConfig | None = None) -> None:
        self.config = config or get_config()

    def filter_many(
        self, query: str, candidates: list[T]
    ) -> tuple[list[T], CandidateFilterStats]:
        """过滤候选列表，返回 (保留的候选, 统计信息)。"""
        mode = self.config.retrieval.chunk_type_filter_mode
        stats = CandidateFilterStats(input_count=len(candidates), mode=mode)

        if mode == "off":
            stats.output_count = len(candidates)
            return candidates, stats

        if self.should_bypass(query):
            stats.output_count = len(candidates)
            stats.bypassed = True
            return candidates, stats

        kept: list[T] = []
        for candidate in candidates:
            keep, reason = self.keep(candidate)
            if keep:
                kept.append(candidate)
            elif reason == "references":
                stats.filtered_references += 1
            elif reason == "short_figure_caption":
                stats.filtered_short_figure_captions += 1

        stats.output_count = len(kept)
        return kept, stats

    def should_bypass(self, query: str) -> bool:
        """判断当前 query 是否应绕过 chunk_type 过滤。"""
        normalized = query.lower()
        keywords = self.config.retrieval.chunk_type_filter_bypass_keywords
        return any(keyword.lower() in normalized for keyword in keywords)

    def keep(self, candidate: RetrievalCandidate) -> tuple[bool, str]:
        """判断单个候选是否保留。返回 (是否保留, 过滤原因)。"""
        payload = candidate.payload or {}
        chunk_type = str(payload.get("chunk_type", "") or "")
        text = candidate.text or str(payload.get("text", "") or "")

        if self.config.retrieval.filter_references_for_regular_rag and chunk_type == "references":
            return False, "references"

        min_caption_chars = self.config.retrieval.min_figure_caption_chars_for_regular_rag
        if chunk_type == "figure_caption" and min_caption_chars > 0 and len(text.strip()) < min_caption_chars:
            return False, "short_figure_caption"

        return True, ""
