"""检索阶段使用的标准数据结构。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievalModel(BaseModel):
    """检索模块的 Pydantic 基类。"""

    model_config = ConfigDict(extra="ignore")


class QueryRewriteResult(RetrievalModel):
    """查询改写结果，必须保留用户原始问题。"""

    original_query: str
    main_query: str
    expanded_queries: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_queries(self) -> list[str]:
        """返回去重后的查询列表，顺序为原始问题、主查询、扩展查询。"""
        queries: list[str] = []
        for query in [self.original_query, self.main_query, *self.expanded_queries]:
            normalized = " ".join(query.strip().split())
            if normalized and normalized not in queries:
                queries.append(normalized)
        return queries


class FusedRetrievalResult(RetrievalModel):
    """dense 与 sparse 融合后的候选结果。"""

    chunk_id: str
    text: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    fusion_score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    source_scores: dict[str, float] = Field(default_factory=dict)
    source_ranks: dict[str, int] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    rank: Optional[int] = None


class RerankedRetrievalResult(RetrievalModel):
    """cross-encoder 或 fallback 重排后的候选结果。"""

    chunk_id: str
    text: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    source_scores: dict[str, float] = Field(default_factory=dict)
    source_ranks: dict[str, int] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    rank: int = 0
    reranker_name: str = ""
    fallback_reason: Optional[str] = None


class EvidenceItem(RetrievalModel):
    """进入最终上下文的证据片段。"""

    evidence_id: str
    chunk_id: str
    parent_id: str = ""
    doc_id: str = ""
    paper_title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    section_title: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    source_type: str = "local_pdf"
    score: float = 0.0
    citation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredCitation(RetrievalModel):
    """结构化引用，对应一条实际被模型使用过的证据。"""

    citation_id: str
    marker: str
    source_type: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    doc_id: str = ""
    chunk_id: str = ""
    parent_id: str = ""
    section_title: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    url: str = ""
    score: float = 0.0
    text_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBuildResult(RetrievalModel):
    """上下文构建结果。"""

    final_context: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    candidate_citations: list[StructuredCitation] = Field(default_factory=list)
    used_citations: list[StructuredCitation] = Field(default_factory=list)
    used_citation_markers: list[str] = Field(default_factory=list)
    total_chars: int = 0


class SufficiencyResult(RetrievalModel):
    """本地证据充分性判断结果。"""

    sufficient: bool
    reason: str
    needs_web: bool = False
    top_score: float = 0.0
    evidence_count: int = 0
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

