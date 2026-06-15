"""多改写 query 并发检索单元测试。"""

from __future__ import annotations

import pytest

from fireagent.retrieval.query_parallel import (
    ParallelQueryStats,
    merge_query_results,
    retrieve_queries_parallel,
)
from fireagent.retrieval.schema import QueryRewriteResult
from fireagent.vectorstore.schema import VectorSearchResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, score: float) -> VectorSearchResult:
    return VectorSearchResult(chunk_id=chunk_id, score=score, text=f"text_{chunk_id}")


def _make_rewrite(*queries: str) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=queries[0],
        main_query=queries[0],
        expanded_queries=list(queries[1:]) if len(queries) > 1 else [],
    )


class FakeVectorStore:
    """可编程的假向量库，支持 dense/sparse 检索。"""

    def __init__(self, mapping: dict[str, list[VectorSearchResult] | Exception]) -> None:
        self.mapping = mapping

    def dense_search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        value = self.mapping[query]
        if isinstance(value, Exception):
            raise value
        return value[:top_k]

    def sparse_search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        value = self.mapping[query]
        if isinstance(value, Exception):
            raise value
        return value[:top_k]


# ---------------------------------------------------------------------------
# merge_query_results
# ---------------------------------------------------------------------------


class TestMergeQueryResults:
    def test_dedup_keeps_highest_score(self) -> None:
        q1_results = [_make_result("c1", 0.9), _make_result("c2", 0.7)]
        q2_results = [_make_result("c2", 0.8), _make_result("c3", 0.6)]
        merged = merge_query_results(
            [("q1", q1_results), ("q2", q2_results)],
            limit=10,
        )
        ids = [r.chunk_id for r in merged]
        assert ids == ["c1", "c2", "c3"]
        c2 = next(r for r in merged if r.chunk_id == "c2")
        assert c2.score == 0.8

    def test_sorted_by_score_desc(self) -> None:
        results = [
            _make_result("a", 0.5),
            _make_result("b", 0.9),
            _make_result("c", 0.7),
        ]
        merged = merge_query_results([("q", results)], limit=10)
        assert [r.chunk_id for r in merged] == ["b", "c", "a"]

    def test_limit_truncates(self) -> None:
        results = [_make_result(f"c{i}", float(i)) for i in range(10)]
        merged = merge_query_results([("q", results)], limit=3)
        assert len(merged) == 3

    def test_rank_assigned(self) -> None:
        results = [_make_result("a", 0.9), _make_result("b", 0.7)]
        merged = merge_query_results([("q", results)], limit=10)
        assert merged[0].rank == 1
        assert merged[1].rank == 2

    def test_matched_queries_tracked(self) -> None:
        r1 = _make_result("c1", 0.9)
        r2 = _make_result("c1", 0.8)
        merged = merge_query_results([("q1", [r1]), ("q2", [r2])], limit=10)
        assert len(merged) == 1
        assert "q1" in merged[0].payload["matched_queries"]
        assert "q2" in merged[0].payload["matched_queries"]


# ---------------------------------------------------------------------------
# retrieve_queries_parallel
# ---------------------------------------------------------------------------


class TestRetrieveQueriesParallel:
    def test_basic_parallel(self) -> None:
        def retrieve_one(query: str) -> list[VectorSearchResult]:
            return [_make_result(f"{query}_c1", 0.9)]

        results, stats = retrieve_queries_parallel(
            queries=["q1", "q2", "q3"],
            retrieve_one=retrieve_one,
            limit=10,
            max_workers=2,
        )
        assert stats.mode == "parallel"
        assert stats.query_count == 3
        assert stats.max_workers == 2
        assert len(stats.failed_queries) == 0
        assert len(results) == 3

    def test_single_query_still_parallel(self) -> None:
        def retrieve_one(query: str) -> list[VectorSearchResult]:
            return [_make_result(f"{query}_c1", 0.9)]

        results, stats = retrieve_queries_parallel(
            queries=["q1"],
            retrieve_one=retrieve_one,
            limit=10,
            max_workers=4,
        )
        assert stats.max_workers == 1
        assert len(results) == 1

    def test_partial_failure_preserved(self) -> None:
        call_count = 0

        def retrieve_one(query: str) -> list[VectorSearchResult]:
            nonlocal call_count
            call_count += 1
            if query == "q2":
                raise RuntimeError("q2 failed")
            return [_make_result(f"{query}_c1", 0.9)]

        results, stats = retrieve_queries_parallel(
            queries=["q1", "q2", "q3"],
            retrieve_one=retrieve_one,
            limit=10,
            max_workers=3,
        )
        assert len(results) == 2
        assert stats.failed_queries == ["q2"]

    def test_all_failure_empty_results(self) -> None:
        def retrieve_one(query: str) -> list[VectorSearchResult]:
            raise RuntimeError(f"{query} failed")

        results, stats = retrieve_queries_parallel(
            queries=["q1", "q2"],
            retrieve_one=retrieve_one,
            limit=10,
            max_workers=2,
        )
        assert len(results) == 0
        assert len(stats.failed_queries) == 2

    def test_elapsed_ms_recorded(self) -> None:
        def retrieve_one(query: str) -> list[VectorSearchResult]:
            return [_make_result(f"{query}_c1", 0.9)]

        _, stats = retrieve_queries_parallel(
            queries=["q1", "q2"],
            retrieve_one=retrieve_one,
            limit=10,
            max_workers=2,
        )
        assert stats.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# DenseRetriever / SparseRetriever 集成
# ---------------------------------------------------------------------------


class TestDenseRetrieverParallel:
    def _make_retriever(self, parallel: bool, max_workers: int = 4):
        from fireagent.retrieval.dense_retriever import DenseRetriever
        from fireagent.utils.config import RetrievalConfig

        config = type(
            "FakeConfig",
            (),
            {"retrieval": RetrievalConfig(
                parallel_rewrite_queries=parallel,
                rewrite_query_max_workers=max_workers,
            )},
        )()
        vs = FakeVectorStore({
            "q1": [_make_result("c1", 0.9), _make_result("c2", 0.7)],
            "q2": [_make_result("c2", 0.8), _make_result("c3", 0.6)],
            "q3": [_make_result("c4", 0.85)],
        })
        return DenseRetriever(vs, config=config)

    def test_serial_mode(self) -> None:
        retriever = self._make_retriever(parallel=False)
        rewrite = _make_rewrite("q1", "q2", "q3")
        results = retriever.retrieve_many(rewrite)
        assert retriever.last_query_stats is not None
        assert retriever.last_query_stats.mode == "serial"
        assert len(results) > 0

    def test_parallel_mode(self) -> None:
        retriever = self._make_retriever(parallel=True, max_workers=2)
        rewrite = _make_rewrite("q1", "q2", "q3")
        results = retriever.retrieve_many(rewrite)
        assert retriever.last_query_stats is not None
        assert retriever.last_query_stats.mode == "parallel"
        assert retriever.last_query_stats.max_workers == 2
        assert len(results) > 0

    def test_serial_and_parallel_same_results(self) -> None:
        serial = self._make_retriever(parallel=False)
        parallel = self._make_retriever(parallel=True, max_workers=4)
        rewrite = _make_rewrite("q1", "q2", "q3")

        serial_results = serial.retrieve_many(rewrite)
        parallel_results = parallel.retrieve_many(rewrite)

        assert len(serial_results) == len(parallel_results)
        for s, p in zip(serial_results, parallel_results):
            assert s.chunk_id == p.chunk_id
            assert abs(s.score - p.score) < 1e-9

    def test_single_query_uses_serial(self) -> None:
        retriever = self._make_retriever(parallel=True, max_workers=4)
        rewrite = _make_rewrite("q1")
        retriever.retrieve_many(rewrite)
        assert retriever.last_query_stats is not None
        assert retriever.last_query_stats.mode == "serial"

    def test_all_queries_failed_raises(self) -> None:
        from fireagent.retrieval.dense_retriever import DenseRetriever
        from fireagent.utils.config import RetrievalConfig

        config = type(
            "FakeConfig",
            (),
            {"retrieval": RetrievalConfig(
                parallel_rewrite_queries=True,
                rewrite_query_max_workers=4,
            )},
        )()
        vs = FakeVectorStore({
            "q1": RuntimeError("fail"),
            "q2": RuntimeError("fail"),
        })
        retriever = DenseRetriever(vs, config=config)
        rewrite = _make_rewrite("q1", "q2")
        with pytest.raises(RuntimeError, match="all dense rewrite queries failed"):
            retriever.retrieve_many(rewrite)


class TestSparseRetrieverParallel:
    def _make_retriever(self, parallel: bool, max_workers: int = 4):
        from fireagent.retrieval.sparse_retriever import SparseRetriever
        from fireagent.utils.config import RetrievalConfig

        config = type(
            "FakeConfig",
            (),
            {"retrieval": RetrievalConfig(
                parallel_rewrite_queries=parallel,
                rewrite_query_max_workers=max_workers,
            )},
        )()
        vs = FakeVectorStore({
            "q1": [_make_result("c1", 0.9), _make_result("c2", 0.7)],
            "q2": [_make_result("c2", 0.8), _make_result("c3", 0.6)],
            "q3": [_make_result("c4", 0.85)],
        })
        return SparseRetriever(vs, config=config)

    def test_serial_mode(self) -> None:
        retriever = self._make_retriever(parallel=False)
        rewrite = _make_rewrite("q1", "q2", "q3")
        results = retriever.retrieve_many(rewrite)
        assert retriever.last_query_stats is not None
        assert retriever.last_query_stats.mode == "serial"
        assert len(results) > 0

    def test_parallel_mode(self) -> None:
        retriever = self._make_retriever(parallel=True, max_workers=2)
        rewrite = _make_rewrite("q1", "q2", "q3")
        results = retriever.retrieve_many(rewrite)
        assert retriever.last_query_stats is not None
        assert retriever.last_query_stats.mode == "parallel"
        assert retriever.last_query_stats.max_workers == 2
        assert len(results) > 0

    def test_serial_and_parallel_same_results(self) -> None:
        serial = self._make_retriever(parallel=False)
        parallel = self._make_retriever(parallel=True, max_workers=4)
        rewrite = _make_rewrite("q1", "q2", "q3")

        serial_results = serial.retrieve_many(rewrite)
        parallel_results = parallel.retrieve_many(rewrite)

        assert len(serial_results) == len(parallel_results)
        for s, p in zip(serial_results, parallel_results):
            assert s.chunk_id == p.chunk_id
            assert abs(s.score - p.score) < 1e-9

    def test_all_queries_failed_raises(self) -> None:
        from fireagent.retrieval.sparse_retriever import SparseRetriever
        from fireagent.utils.config import RetrievalConfig

        config = type(
            "FakeConfig",
            (),
            {"retrieval": RetrievalConfig(
                parallel_rewrite_queries=True,
                rewrite_query_max_workers=4,
            )},
        )()
        vs = FakeVectorStore({
            "q1": RuntimeError("fail"),
            "q2": RuntimeError("fail"),
        })
        retriever = SparseRetriever(vs, config=config)
        rewrite = _make_rewrite("q1", "q2")
        with pytest.raises(RuntimeError, match="all sparse rewrite queries failed"):
            retriever.retrieve_many(rewrite)
