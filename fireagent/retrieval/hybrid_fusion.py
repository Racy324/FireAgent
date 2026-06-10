"""Dense 与 sparse 结果融合。"""

from __future__ import annotations

from fireagent.retrieval.schema import FusedRetrievalResult
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.schema import VectorSearchResult


class WeightedRRFFusion:
    """加权 Reciprocal Rank Fusion。

    公式：score += weight / (rrf_k + rank)。同一个 chunk_id 只保留一条候选，
    同时记录 dense/sparse 的原始分数、rank 和来源。
    """

    def __init__(
        self,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        rrf_k: int | None = None,
        top_k: int | None = None,
        config: FireAgentConfig | None = None,
    ) -> None:
        self.config = config or get_config()
        self.dense_weight = self.config.retrieval.dense_weight if dense_weight is None else dense_weight
        self.sparse_weight = self.config.retrieval.sparse_weight if sparse_weight is None else sparse_weight
        self.rrf_k = rrf_k or self.config.retrieval.rrf_k
        self.top_k = top_k or self.config.retrieval.fusion_top_k

    def fuse(
        self,
        dense_results: list[VectorSearchResult],
        sparse_results: list[VectorSearchResult],
        top_k: int | None = None,
    ) -> list[FusedRetrievalResult]:
        """融合 dense 与 sparse 候选。"""
        candidates: dict[str, FusedRetrievalResult] = {}
        self._accumulate(candidates, dense_results, source_name="dense", weight=self.dense_weight)
        self._accumulate(candidates, sparse_results, source_name="sparse", weight=self.sparse_weight)

        fused = sorted(candidates.values(), key=lambda item: item.fusion_score, reverse=True)
        limit = top_k or self.top_k
        for rank, candidate in enumerate(fused[:limit], start=1):
            candidate.rank = rank
        return fused[:limit]

    def _accumulate(
        self,
        candidates: dict[str, FusedRetrievalResult],
        results: list[VectorSearchResult],
        source_name: str,
        weight: float,
    ) -> None:
        """累加单路检索结果的 RRF 贡献。"""
        if weight <= 0:
            return

        for index, result in enumerate(results, start=1):
            rank = result.rank or index
            contribution = weight / (self.rrf_k + rank)
            candidate = candidates.get(result.chunk_id)
            if candidate is None:
                candidate = FusedRetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    payload=result.payload,
                    fusion_score=0.0,
                )
                candidates[result.chunk_id] = candidate

            candidate.fusion_score += contribution
            candidate.source_scores[source_name] = float(result.score)
            candidate.source_ranks[source_name] = int(rank)
            if source_name not in candidate.sources:
                candidate.sources.append(source_name)

            if source_name == "dense":
                candidate.dense_score = float(result.score)
                candidate.dense_rank = int(rank)
            elif source_name == "sparse":
                candidate.sparse_score = float(result.score)
                candidate.sparse_rank = int(rank)

            if not candidate.text and result.text:
                candidate.text = result.text
            if not candidate.payload and result.payload:
                candidate.payload = result.payload


def weighted_rrf(
    dense_results: list[VectorSearchResult],
    sparse_results: list[VectorSearchResult],
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
    rrf_k: int = 60,
    top_k: int = 50,
) -> list[FusedRetrievalResult]:
    """便捷函数：执行加权 RRF 融合。"""
    return WeightedRRFFusion(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
        top_k=top_k,
    ).fuse(dense_results, sparse_results, top_k=top_k)

