"""Dense 检索器封装。"""

from __future__ import annotations

from collections import OrderedDict

from fireagent.retrieval.schema import QueryRewriteResult
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.qdrant_client import FireAgentQdrantClient
from fireagent.vectorstore.schema import VectorSearchResult


class DenseRetriever:
    """调用向量库 dense_search 的检索器。"""

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
        best_by_chunk: OrderedDict[str, VectorSearchResult] = OrderedDict()

        for query in rewrite_result.all_queries:
            for result in self.retrieve(query, top_k=per_query_k):
                result.payload.setdefault("matched_queries", [])
                result.payload["matched_queries"].append(query)
                current = best_by_chunk.get(result.chunk_id)
                if current is None or result.score > current.score:
                    best_by_chunk[result.chunk_id] = result

        results = sorted(best_by_chunk.values(), key=lambda item: item.score, reverse=True)
        for rank, result in enumerate(results[: self.top_k], start=1):
            result.rank = rank
        return results[: self.top_k]

