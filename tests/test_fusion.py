"""Weighted RRF 融合测试。"""

from __future__ import annotations

import pytest

from fireagent.retrieval.hybrid_fusion import WeightedRRFFusion
from fireagent.vectorstore.schema import VectorSearchResult


def result(chunk_id: str, score: float, rank: int, source: str) -> VectorSearchResult:
    """构造检索结果。"""
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        text=f"text-{chunk_id}",
        payload={"chunk_id": chunk_id, "source": source},
    )


def test_weighted_rrf_ranks_shared_candidate_first() -> None:
    """同时被 dense 和 sparse 命中的候选应获得两路 RRF 加分。"""
    dense = [
        result("a", 0.90, 1, "dense"),
        result("b", 0.80, 2, "dense"),
    ]
    sparse = [
        result("b", 5.00, 1, "sparse"),
        result("c", 4.00, 2, "sparse"),
    ]

    fused = WeightedRRFFusion(
        dense_weight=0.65,
        sparse_weight=0.35,
        rrf_k=60,
        top_k=10,
    ).fuse(dense, sparse)

    assert fused[0].chunk_id == "b"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    assert fused[0].dense_score == pytest.approx(0.80)
    assert fused[0].sparse_score == pytest.approx(5.00)
    assert set(fused[0].sources) == {"dense", "sparse"}


def test_weighted_rrf_dedupes_by_chunk_id_and_limits_top_k() -> None:
    """融合应按 chunk_id 去重，并遵守 top_k。"""
    dense = [
        result("a", 0.90, 1, "dense"),
        result("a", 0.70, 2, "dense"),
        result("b", 0.60, 3, "dense"),
    ]
    sparse = [result("c", 3.00, 1, "sparse")]

    fused = WeightedRRFFusion(
        dense_weight=1.0,
        sparse_weight=1.0,
        rrf_k=10,
        top_k=2,
    ).fuse(dense, sparse)

    assert len(fused) == 2
    assert len({item.chunk_id for item in fused}) == 2
    assert all(item.rank in {1, 2} for item in fused)

