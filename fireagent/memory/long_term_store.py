"""Qdrant-backed 长期记忆存储。"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fireagent.memory.decay import age_days_from_now, memory_final_score
from fireagent.memory.long_term_schema import (
    LongTermMemoryRecord,
    LongTermMemorySearchResult,
)
from fireagent.utils.config import FireAgentConfig, LongTermMemoryConfig, get_config
from fireagent.vectorstore.dense_embedding import BaseDenseEmbedder, create_dense_embedder

logger = logging.getLogger(__name__)


class LongTermMemoryStore:
    """Qdrant 长期记忆 collection 管理。"""

    def __init__(
        self,
        config: Optional[FireAgentConfig] = None,
        client: Any = None,
        dense_embedder: Optional[BaseDenseEmbedder] = None,
    ) -> None:
        self.config = config or get_config()
        self.lt_config: LongTermMemoryConfig = self.config.memory.long_term
        self.collection_name = self.lt_config.collection
        self._client = client
        self._dense_embedder = dense_embedder

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.config.qdrant.url,
                api_key=self.config.qdrant.api_key or None,
            )
        return self._client

    @property
    def dense_embedder(self) -> BaseDenseEmbedder:
        if self._dense_embedder is None:
            self._dense_embedder = create_dense_embedder(self.config)
        return self._dense_embedder

    def create_collection(self, recreate: bool = False) -> None:
        """创建长期记忆 collection。"""
        from qdrant_client import models

        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:  # noqa: BLE001
                pass

        try:
            self.client.get_collection(self.collection_name)
            return  # 已存在
        except Exception:  # noqa: BLE001
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.config.qdrant.dense_vector_name: models.VectorParams(
                    size=self.config.qdrant.dense_vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
        )
        # 创建 payload 索引
        for field_name in ("user_id", "status", "memory_type"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert_memory(self, memory: LongTermMemoryRecord) -> None:
        """写入一条长期记忆。"""
        from qdrant_client import models

        vector = self.dense_embedder.embed_documents([memory.content])[0]
        point = models.PointStruct(
            id=memory.memory_id,
            vector={self.config.qdrant.dense_vector_name: vector},
            payload=memory.model_dump(),
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def search(
        self,
        query: str,
        user_id: str = "local",
        top_k: int = 5,
    ) -> list[LongTermMemorySearchResult]:
        """按 query 检索相关长期记忆，经过时间衰退和重要性重排。"""
        from qdrant_client import models

        query_vector = self.dense_embedder.embed_query(query)
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value="active"),
                ),
            ]
        )

        try:
            raw = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=self.config.qdrant.dense_vector_name,
                query_filter=search_filter,
                limit=top_k * 3,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆检索失败: %s", exc)
            return []

        results: list[LongTermMemorySearchResult] = []
        for point in raw.points:
            payload = point.payload or {}
            memory = LongTermMemoryRecord(**payload)
            semantic_score = float(point.score or 0.0)
            age = age_days_from_now(memory.created_at)
            half_life = self._half_life_for_type(memory.memory_type)
            decay = memory_final_score(semantic_score, memory.importance, age, half_life) / max(semantic_score * (0.75 + 0.5 * memory.importance), 0.001)
            final = memory_final_score(semantic_score, memory.importance, age, half_life)

            if final >= self.lt_config.min_relevance_score:
                results.append(LongTermMemorySearchResult(
                    memory=memory,
                    semantic_score=semantic_score,
                    decay_factor=decay,
                    final_score=final,
                ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    def find_duplicate(self, content: str, user_id: str = "local") -> LongTermMemoryRecord | None:
        """查找高度相似的已有记忆用于去重。"""
        from qdrant_client import models

        query_vector = self.dense_embedder.embed_query(content)
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value="active"),
                ),
            ]
        )
        try:
            raw = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=self.config.qdrant.dense_vector_name,
                query_filter=search_filter,
                limit=1,
                with_payload=True,
            )
        except Exception:  # noqa: BLE001
            return None

        if not raw.points:
            return None

        point = raw.points[0]
        score = float(point.score or 0.0)
        if score >= self.lt_config.dedup_similarity_threshold:
            payload = point.payload or {}
            return LongTermMemoryRecord(**payload)
        return None

    def update_memory_fields(self, memory_id: str, **fields: Any) -> None:
        """更新已有记忆的部分字段。"""
        from qdrant_client import models

        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )
        except Exception:  # noqa: BLE001
            return

        if not result:
            return

        point = result[0]
        payload = dict(point.payload or {})
        payload.update(fields)
        payload["updated_at"] = _now()

        # 合并 tags
        if "tags" in fields and "tags" in (point.payload or {}):
            old_tags = set(point.payload.get("tags", []))
            new_tags = set(fields["tags"])
            payload["tags"] = list(old_tags | new_tags)

        vector = self.dense_embedder.embed_documents([payload.get("content", "")])[0]
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(
                id=memory_id,
                vector={self.config.qdrant.dense_vector_name: vector},
                payload=payload,
            )],
        )

    def _half_life_for_type(self, memory_type: str) -> float:
        mapping = {
            "semantic": self.lt_config.semantic_half_life_days,
            "episodic": self.lt_config.episodic_half_life_days,
            "procedural": self.lt_config.procedural_half_life_days,
        }
        return mapping.get(memory_type, self.lt_config.default_half_life_days)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_memory_id() -> str:
    """生成新的记忆 ID。"""
    return str(uuid.uuid4())
