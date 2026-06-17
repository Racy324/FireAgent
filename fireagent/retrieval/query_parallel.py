"""多改写 query 并发检索工具。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter

from fireagent.vectorstore.schema import VectorSearchResult


@dataclass
class ParallelQueryStats:
    """并发检索统计信息。"""

    mode: str
    query_count: int
    max_workers: int
    failed_queries: list[str] = field(default_factory=list)
    elapsed_ms: int = 0


def merge_query_results(
    query_results: Sequence[tuple[str, list[VectorSearchResult]]],
    limit: int,
) -> list[VectorSearchResult]:
    """合并多个 query 的检索结果，按 chunk_id 去重并保留最高分。"""
    best_by_chunk: OrderedDict[str, VectorSearchResult] = OrderedDict()

    for query, results in query_results:
        for result in results:
            result.payload.setdefault("matched_queries", [])
            result.payload["matched_queries"].append(query)
            current = best_by_chunk.get(result.chunk_id)
            if current is None:
                best_by_chunk[result.chunk_id] = result
            elif result.score > current.score:
                # 合并已有的 matched_queries 到更高分的结果。
                existing_queries = current.payload.get("matched_queries", [])
                result.payload["matched_queries"] = list(set(existing_queries) | set(result.payload["matched_queries"]))
                best_by_chunk[result.chunk_id] = result
            else:
                # 当前结果分数更高，把新 query 追加进去。
                current.payload["matched_queries"] = list(set(current.payload["matched_queries"]) | {query})

    merged = sorted(
        best_by_chunk.values(),
        key=lambda item: (-item.score, item.chunk_id),
    )
    for rank, result in enumerate(merged[:limit], start=1):
        result.rank = rank
    return merged[:limit]


def retrieve_queries_parallel(
    queries: Sequence[str],
    retrieve_one: Callable[[str], list[VectorSearchResult]],
    limit: int,
    max_workers: int,
) -> tuple[list[VectorSearchResult], ParallelQueryStats]:
    """并发执行多个 query 的检索并合并结果。"""
    started = perf_counter()
    worker_count = max(1, min(max_workers, len(queries)))
    stats = ParallelQueryStats(
        mode="parallel",
        query_count=len(queries),
        max_workers=worker_count,
    )

    indexed_results: dict[int, tuple[str, list[VectorSearchResult]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(retrieve_one, query): index
            for index, query in enumerate(queries)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            query = queries[index]
            try:
                indexed_results[index] = (query, future.result())
            except Exception:  # noqa: BLE001 - 单个 query 失败不阻断其他结果。
                stats.failed_queries.append(query)

    ordered_results = [
        indexed_results[index]
        for index in sorted(indexed_results)
    ]
    stats.elapsed_ms = int((perf_counter() - started) * 1000)
    return merge_query_results(ordered_results, limit=limit), stats
