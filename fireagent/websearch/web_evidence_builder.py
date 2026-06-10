"""本地证据与联网证据的统一处理。"""

from __future__ import annotations

import hashlib
from typing import Optional

from fireagent.retrieval.reranker import BaseReranker, LexicalReranker
from fireagent.retrieval.schema import FusedRetrievalResult, RerankedRetrievalResult, SufficiencyResult
from fireagent.retrieval.sufficiency_checker import TEMPORAL_KEYWORDS
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.websearch.schema import WebEvidenceChunk


class WebEvidenceBuilder:
    """将 web evidence 与 local evidence 统一成可重排候选。"""

    def __init__(
        self,
        reranker: Optional[BaseReranker] = None,
        config: Optional[FireAgentConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self.reranker = reranker or LexicalReranker()

    def should_trigger_web_search(
        self,
        query: str,
        sufficiency: Optional[SufficiencyResult] = None,
        local_results: Optional[list[RerankedRetrievalResult]] = None,
    ) -> bool:
        """根据配置、时效词和本地证据状态判断是否需要联网兜底。"""
        if not self.config.rag.enable_web_fallback:
            return False
        if sufficiency is not None and (not sufficiency.sufficient or sufficiency.needs_web):
            return True
        if any(keyword in query for keyword in self.config.web.temporal_keywords or TEMPORAL_KEYWORDS):
            return True
        if local_results is None:
            return False
        if len(local_results) < self.config.retrieval.sufficiency_min_evidence:
            return True
        top_score = max((item.final_score for item in local_results), default=0.0)
        return top_score < self.config.retrieval.sufficiency_min_score

    def web_chunks_to_fused_candidates(
        self,
        web_chunks: list[WebEvidenceChunk],
    ) -> list[FusedRetrievalResult]:
        """将 WebEvidenceChunk 转换为统一重排候选。"""
        candidates: list[FusedRetrievalResult] = []
        for rank, chunk in enumerate(web_chunks, start=1):
            payload = self._web_chunk_payload(chunk)
            candidates.append(
                FusedRetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    payload=payload,
                    fusion_score=max(chunk.score, 0.0) / (100.0 + rank),
                    source_scores={"web": chunk.score},
                    source_ranks={"web": rank},
                    sources=["web"],
                    rank=rank,
                )
            )
        return candidates

    def rerank_web_evidence(
        self,
        query: str,
        web_chunks: list[WebEvidenceChunk],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """对 web evidence 单独重排。"""
        candidates = self.web_chunks_to_fused_candidates(web_chunks)
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def combine_and_rerank(
        self,
        query: str,
        local_results: list[RerankedRetrievalResult],
        web_chunks: list[WebEvidenceChunk],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """将本地重排结果和联网 evidence 合并后统一重排、去重。"""
        candidates: list[FusedRetrievalResult] = []
        for local in local_results:
            candidates.append(self._local_reranked_to_fused(local))
        candidates.extend(self.web_chunks_to_fused_candidates(web_chunks))

        deduped = self._dedupe_candidates(candidates)
        limit = top_k or self.config.retrieval.rerank_top_k
        return self.reranker.rerank(query, deduped, top_k=limit)

    def _local_reranked_to_fused(self, local: RerankedRetrievalResult) -> FusedRetrievalResult:
        """将本地 reranked 结果转回统一候选，便于和 web evidence 共同重排。"""
        fusion_score = local.fusion_score or min(local.final_score, 1.0) / 100.0
        payload = dict(local.payload)
        payload.setdefault("source_type", payload.get("source_type", "local_pdf"))
        return FusedRetrievalResult(
            chunk_id=local.chunk_id,
            text=local.text,
            payload=payload,
            fusion_score=fusion_score,
            dense_score=local.dense_score,
            sparse_score=local.sparse_score,
            dense_rank=local.dense_rank,
            sparse_rank=local.sparse_rank,
            source_scores={**local.source_scores, "local_rerank": local.final_score},
            source_ranks=local.source_ranks,
            sources=local.sources or ["local"],
            rank=local.rank,
        )

    @staticmethod
    def _web_chunk_payload(chunk: WebEvidenceChunk) -> dict[str, object]:
        """将 WebEvidenceChunk 转成 ContextBuilder 可识别的 payload。"""
        return {
            "chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id,
            "doc_id": chunk.doc_id,
            "paper_title": chunk.title,
            "authors": [],
            "year": None,
            "section_title": "联网资料",
            "section_path": ["联网资料"],
            "page_start": None,
            "page_end": None,
            "text": chunk.text,
            "chunk_type": "web",
            "source_type": chunk.source_type,
            "metadata": {
                **chunk.metadata,
                "title": chunk.title,
                "url": chunk.url,
                "published_date": chunk.published_date,
                "snippet": chunk.snippet,
            },
        }

    def _dedupe_candidates(self, candidates: list[FusedRetrievalResult]) -> list[FusedRetrievalResult]:
        """按 URL、chunk_id 和文本指纹去重。"""
        best_by_key: dict[str, FusedRetrievalResult] = {}
        for candidate in candidates:
            payload = candidate.payload or {}
            metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
            url = str(metadata.get("url", ""))
            normalized_text = "".join((candidate.text or str(payload.get("text", ""))).split())
            text_hash = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
            keys = [candidate.chunk_id, f"text:{text_hash}"]
            if url:
                keys.append(f"url:{url}")
            for key in keys:
                if not key:
                    continue
                current = best_by_key.get(key)
                if current is None or candidate.fusion_score > current.fusion_score:
                    best_by_key[key] = candidate

        unique: dict[str, FusedRetrievalResult] = {}
        for candidate in best_by_key.values():
            unique[candidate.chunk_id] = candidate
        return sorted(unique.values(), key=lambda item: item.fusion_score, reverse=True)


def should_trigger_web_search(
    query: str,
    sufficiency: Optional[SufficiencyResult] = None,
    local_results: Optional[list[RerankedRetrievalResult]] = None,
) -> bool:
    """便捷函数：判断是否需要联网搜索。"""
    return WebEvidenceBuilder().should_trigger_web_search(
        query=query,
        sufficiency=sufficiency,
        local_results=local_results,
    )

