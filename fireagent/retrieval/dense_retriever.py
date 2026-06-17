"""Dense 检索器封装。"""

from __future__ import annotations

from collections import OrderedDict

from fireagent.retrieval.query_parallel import (
    ParallelQueryStats,
    merge_query_results,
    retrieve_queries_parallel,
)
from fireagent.retrieval.schema import QueryRewriteResult
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.qdrant_client import FireAgentQdrantClient
from fireagent.vectorstore.schema import VectorSearchResult


class DenseRetriever:
    """调用向量库 dense_search 的检索器。"""

    last_query_stats: ParallelQueryStats | None = None

    def __init__(
        self,
        vectorstore: FireAgentQdrantClient,
        top_k: int | None = None,
        config: FireAgentConfig | None = None,
    ) -> None:
        self.vectorstore = vectorstore
        self.config = config or get_config()
        self.top_k = top_k or self.config.retrieval.dense_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        """对单个查询执行 dense 检索。"""
        return self.vectorstore.dense_search(query, top_k=top_k or self.top_k)

    def retrieve_many(
        self,
        rewrite_result: QueryRewriteResult,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """对原始查询、主查询和扩展查询执行检索，并按 chunk_id 去重。"""
        per_query_k = top_k or self.top_k
        queries = list(rewrite_result.all_queries)

        if self.config.retrieval.parallel_rewrite_queries and len(queries) > 1:
            results, stats = retrieve_queries_parallel(
                queries=queries,
                retrieve_one=lambda query: self.retrieve(query, top_k=per_query_k),
                limit=self.top_k,
                max_workers=self.config.retrieval.rewrite_query_max_workers,
            )
            self.last_query_stats = stats
            if len(stats.failed_queries) == len(queries):
                raise RuntimeError("all dense rewrite queries failed")
            return results

        query_results = [
            (query, self.retrieve(query, top_k=per_query_k))
            for query in queries
        ]
        self.last_query_stats = ParallelQueryStats(
            mode="serial",
            query_count=len(queries),
            max_workers=1,
        )
        return merge_query_results(query_results, limit=self.top_k)

