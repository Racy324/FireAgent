"""FireAgent 的 Qdrant 客户端封装。"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional

from fireagent.ingestion.schema import DocumentChunk
from fireagent.utils.config import FireAgentConfig, PROJECT_ROOT, get_config
from fireagent.vectorstore.dense_embedding import BaseDenseEmbedder, create_dense_embedder
from fireagent.vectorstore.payload_index import create_default_payload_indexes
from fireagent.vectorstore.schema import (
    HybridSearchResultSet,
    QdrantCollectionSchema,
    VectorSearchResult,
    VectorStoreError,
    build_dense_vector_params,
    build_not_deleted_filter,
    build_sparse_vector_params,
    chunk_to_payload,
    payload_to_text,
    sparse_data_to_qdrant,
)
from fireagent.vectorstore.sparse_embedding import BM25SparseEncoder, BaseSparseEncoder, LocalBM25SparseIndex


logger = logging.getLogger(__name__)

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")
FAILED_CHUNK_PREVIEW_CHARS = 300
SANITIZED_EMBEDDING_MAX_CHARS = 1000
EMBEDDING_DIAGNOSTIC_LOG = PROJECT_ROOT / "data" / "logs" / "failed_embedding_chunks.jsonl"


class FireAgentQdrantClient:
    """封装 Qdrant collection 创建、写入与 dense/sparse 检索。"""

    def __init__(
        self,
        config: Optional[FireAgentConfig] = None,
        client: Any = None,
        dense_embedder: Optional[BaseDenseEmbedder] = None,
        sparse_encoder: Optional[BaseSparseEncoder] = None,
        enable_local_sparse_fallback: bool = True,
        allow_hash_dense_fallback: bool = False,
    ) -> None:
        self.config = config or get_config()
        self.collection_name = self.config.qdrant.collection
        self.schema = QdrantCollectionSchema(
            collection_name=self.collection_name,
            dense_vector_name=self.config.qdrant.dense_vector_name,
            sparse_vector_name=self.config.qdrant.sparse_vector_name,
            dense_vector_size=self.config.qdrant.dense_vector_size,
            distance=self.config.qdrant.distance,
            hnsw_m=self.config.qdrant.hnsw.m,
            hnsw_ef_construct=self.config.qdrant.hnsw.ef_construct,
        )
        self.client = client or self._create_client()
        self.dense_embedder = dense_embedder or create_dense_embedder(
            self.config,
            allow_hash_fallback=allow_hash_dense_fallback,
        )
        self.sparse_encoder = sparse_encoder or BM25SparseEncoder()
        self.local_sparse_index = LocalBM25SparseIndex() if enable_local_sparse_fallback else None

    def create_collection(self, recreate: bool = False, create_payload_indexes: bool = True) -> None:
        """创建 Qdrant collection，支持 dense named vector 与 bm25 sparse named vector。"""
        if self._collection_exists():
            if not recreate:
                if create_payload_indexes:
                    create_default_payload_indexes(self.client, self.collection_name)
                return
            self.client.delete_collection(collection_name=self.collection_name)

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={self.schema.dense_vector_name: build_dense_vector_params(self.schema)},
                sparse_vectors_config={
                    self.schema.sparse_vector_name: build_sparse_vector_params(),
                },
            )
        except TypeError:
            # 兼容较老版本 qdrant-client：若 sparse_vectors_config 不被支持，则先建 dense collection。
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={self.schema.dense_vector_name: build_dense_vector_params(self.schema)},
            )

        if create_payload_indexes:
            create_default_payload_indexes(self.client, self.collection_name)

    def upsert_chunks(self, chunks: Sequence[DocumentChunk], batch_size: int = 64) -> None:
        """将 DocumentChunk 写入 Qdrant。

        dense vector 始终写入；sparse vector 会优先写入 Qdrant named sparse vector。
        若当前 Qdrant 版本或 collection 不支持 sparse vector，后续 sparse_search 会退回本地 BM25。
        """
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        self.sparse_encoder.fit(texts)
        if self.local_sparse_index is not None:
            self.local_sparse_index.fit(list(chunks))

        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            try:
                dense_vectors = self.dense_embedder.embed_documents([chunk.text for chunk in batch])
            except Exception as exc:  # noqa: BLE001 - fall back to per-chunk diagnostics.
                logger.warning(
                    "Batch dense embedding failed; locating bad chunks one by one. "
                    "batch_start=%s batch_size=%s error=%s",
                    start,
                    len(batch),
                    exc,
                )
                self._upsert_chunks_one_by_one(batch)
                continue
            sparse_vectors = self.sparse_encoder.encode_documents([chunk.text for chunk in batch])
            points = self._make_points(batch, dense_vectors, sparse_vectors)

            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError("写入 Qdrant chunks 失败，请确认 collection schema 与向量维度匹配。") from exc

    def _upsert_chunks_one_by_one(self, chunks: Sequence[DocumentChunk]) -> None:
        """Embed and write chunks one by one, logging bad chunks and skipping unrecoverable ones."""
        skipped = 0
        recovered = 0
        points = []
        for chunk in chunks:
            recovered_chunk, dense_vector = self._embed_chunk_with_recovery(chunk)
            if recovered_chunk is None or dense_vector is None:
                skipped += 1
                continue

            if recovered_chunk is not chunk:
                recovered += 1
            sparse_vector = self.sparse_encoder.encode_documents([recovered_chunk.text])[0]
            vectors: dict[str, Any] = {self.schema.dense_vector_name: dense_vector}
            if not sparse_vector.is_empty():
                vectors[self.schema.sparse_vector_name] = sparse_data_to_qdrant(sparse_vector)
            points.append(self._make_point(recovered_chunk, vectors))

        if points:
            self._upsert_points(points)
        if skipped or recovered:
            logger.warning(
                "Per-chunk upsert completed: recovered=%s skipped=%s attempted=%s",
                recovered,
                skipped,
                len(chunks),
            )

    def _embed_chunk_with_recovery(
        self,
        chunk: DocumentChunk,
    ) -> tuple[DocumentChunk | None, list[float] | None]:
        """Embed one chunk; on failure, log details, retry sanitized text, then skip."""
        try:
            return chunk, self.dense_embedder.embed_documents([chunk.text])[0]
        except Exception as exc:  # noqa: BLE001 - diagnostic fallback.
            sanitized = self._sanitize_chunk_for_embedding(chunk)
            self._log_failed_chunk(
                "Dense embedding failed; retrying with sanitized/truncated text",
                chunk,
                exc,
                retry_chunk=sanitized,
                event="embedding_failed",
            )

            if not sanitized.text:
                logger.error("Sanitized text is empty; skipping chunk. chunk_id=%s", chunk.chunk_id)
                self._write_embedding_diagnostic(
                    "embedding_skipped_empty_after_sanitize",
                    chunk,
                    "Sanitized text is empty",
                    retry_chunk=sanitized,
                )
                return None, None
            if sanitized.text == chunk.text:
                logger.error("Sanitized text did not change; skipping chunk. chunk_id=%s", chunk.chunk_id)
                self._write_embedding_diagnostic(
                    "embedding_skipped_unchanged_after_sanitize",
                    chunk,
                    "Sanitized text did not change",
                    retry_chunk=sanitized,
                )
                return None, None

        try:
            dense_vector = self.dense_embedder.embed_documents([sanitized.text])[0]
            logger.warning(
                "Recovered bad chunk with sanitized/truncated text. "
                "chunk_id=%s original_chars=%s sanitized_chars=%s",
                chunk.chunk_id,
                len(chunk.text),
                len(sanitized.text),
            )
            self._write_embedding_diagnostic(
                "embedding_recovered_with_sanitized_text",
                chunk,
                "Recovered with sanitized/truncated text",
                retry_chunk=sanitized,
            )
            return sanitized, dense_vector
        except Exception as retry_exc:  # noqa: BLE001 - skip one bad chunk instead of failing the PDF.
            self._log_failed_chunk(
                "Dense embedding still failed after sanitizing/truncating; skipping chunk",
                sanitized,
                retry_exc,
                event="embedding_skipped_after_retry_failed",
            )
            return None, None

    def _sanitize_chunk_for_embedding(self, chunk: DocumentChunk) -> DocumentChunk:
        """Remove suspicious characters and cap text length for Ollama embedding retries."""
        text = CONTROL_CHAR_RE.sub(" ", chunk.text).replace("\ufffd", " ")
        text = WHITESPACE_RE.sub(" ", text).strip()
        truncated = len(text) > SANITIZED_EMBEDDING_MAX_CHARS
        if truncated:
            text = text[:SANITIZED_EMBEDDING_MAX_CHARS].rstrip()

        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "embedding_text_sanitized": True,
                "embedding_original_chars": len(chunk.text),
                "embedding_sanitized_chars": len(text),
                "embedding_text_truncated": truncated,
            }
        )
        return chunk.model_copy(update={"text": text, "metadata": metadata})

    def _log_failed_chunk(
        self,
        message: str,
        chunk: DocumentChunk,
        exc: Exception,
        retry_chunk: DocumentChunk | None = None,
        event: str = "embedding_failed",
    ) -> None:
        """Log enough context to locate a chunk that Ollama cannot embed."""
        logger.error(
            "%s. %s error=%s\nPreview: %s",
            message,
            self._chunk_debug_summary(chunk),
            exc,
            self._chunk_preview(chunk.text),
        )
        self._write_embedding_diagnostic(
            event,
            chunk,
            str(exc),
            retry_chunk=retry_chunk,
            message=message,
        )
        if retry_chunk is not None:
            logger.error(
                "Sanitized retry text: chars=%s changed=%s preview=%s",
                len(retry_chunk.text),
                retry_chunk.text != chunk.text,
                self._chunk_preview(retry_chunk.text),
            )

    def _chunk_debug_summary(self, chunk: DocumentChunk) -> str:
        """Return compact chunk metadata for diagnostic logs."""
        return (
            f"chunk_id={chunk.chunk_id} doc_id={chunk.doc_id} "
            f"pages={chunk.page_start}-{chunk.page_end} type={chunk.chunk_type} "
            f"chars={len(chunk.text)} title={chunk.paper_title!r} section={chunk.section_title!r}"
        )

    @staticmethod
    def _chunk_preview(text: str) -> str:
        """Return a one-line preview safe for logs."""
        preview = WHITESPACE_RE.sub(" ", text).strip()
        if len(preview) > FAILED_CHUNK_PREVIEW_CHARS:
            return f"{preview[:FAILED_CHUNK_PREVIEW_CHARS].rstrip()}..."
        return preview

    def _make_points(
        self,
        chunks: Sequence[DocumentChunk],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[Any],
    ) -> list[Any]:
        """Build Qdrant points from chunks and their dense/sparse vectors."""
        points = []
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
            vectors: dict[str, Any] = {self.schema.dense_vector_name: dense_vector}
            if not sparse_vector.is_empty():
                vectors[self.schema.sparse_vector_name] = sparse_data_to_qdrant(sparse_vector)
            points.append(self._make_point(chunk, vectors))
        return points

    def _upsert_points(self, points: Sequence[Any]) -> None:
        """Write points to Qdrant with a consistent error message."""
        if not points:
            return
        try:
            self.client.upsert(collection_name=self.collection_name, points=list(points))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("写入 Qdrant chunks 失败，请确认 collection schema 与向量维度匹配。") from exc

    def _write_embedding_diagnostic(
        self,
        event: str,
        chunk: DocumentChunk,
        detail: str,
        retry_chunk: DocumentChunk | None = None,
        message: str | None = None,
    ) -> None:
        """Append a structured JSONL diagnostic record for later inspection."""
        try:
            EMBEDDING_DIAGNOSTIC_LOG.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "detail": detail,
                "message": message,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_type": chunk.chunk_type,
                "chars": len(chunk.text),
                "paper_title": chunk.paper_title,
                "section_title": chunk.section_title,
                "preview": self._chunk_preview(chunk.text),
                "metadata": chunk.metadata,
            }
            if retry_chunk is not None:
                record["retry_chars"] = len(retry_chunk.text)
                record["retry_preview"] = self._chunk_preview(retry_chunk.text)
                record["retry_metadata"] = retry_chunk.metadata
            with EMBEDDING_DIAGNOSTIC_LOG.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 - diagnostics must never break ingestion.
            logger.exception("Failed to write embedding diagnostic log: %s", EMBEDDING_DIAGNOSTIC_LOG)

    def dense_search(self, query: str, top_k: int = 10, query_filter: Any = None) -> list[VectorSearchResult]:
        """执行 dense named vector 检索。自动排除已删除的 chunks。"""
        query_vector = self.dense_embedder.embed_query(query)
        effective_filter = build_not_deleted_filter(query_filter)
        raw_results = self._query_points(
            query=query_vector,
            using=self.schema.dense_vector_name,
            top_k=top_k,
            query_filter=effective_filter,
        )
        return self._to_results(raw_results, source="qdrant_dense", vector_name=self.schema.dense_vector_name)

    def sparse_search(self, query: str, top_k: int = 10, query_filter: Any = None) -> list[VectorSearchResult]:
        """执行 BM25 sparse 检索；Qdrant 不可用时退回本地 BM25。自动排除已删除的 chunks。"""
        sparse_query = self.sparse_encoder.encode_query(query)
        effective_filter = build_not_deleted_filter(query_filter)
        if not sparse_query.is_empty():
            try:
                raw_results = self._query_points(
                    query=sparse_data_to_qdrant(sparse_query),
                    using=self.schema.sparse_vector_name,
                    top_k=top_k,
                    query_filter=effective_filter,
                )
                return self._to_results(
                    raw_results,
                    source="qdrant_sparse",
                    vector_name=self.schema.sparse_vector_name,
                )
            except Exception:
                if self.local_sparse_index is None:
                    raise

        if self.local_sparse_index is None:
            return []
        return self.local_sparse_index.search(query, top_k=top_k)

    def hybrid_search(
        self,
        query: str,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        query_filter: Any = None,
    ) -> HybridSearchResultSet:
        """并行检索接口的同步 MVP：返回 dense 与 sparse 两路结果，融合留给 retrieval 阶段。"""
        dense_results = self.dense_search(query, top_k=dense_top_k, query_filter=query_filter)
        sparse_results = self.sparse_search(query, top_k=sparse_top_k, query_filter=query_filter)
        return HybridSearchResultSet(dense_results=dense_results, sparse_results=sparse_results)

    # ── 增量索引支持 ──

    def soft_delete_by_doc_id(self, doc_id: str) -> int:
        """将指定 doc_id 的所有 chunks 标记为 status=deleted。

        返回受影响的 point 数量。
        """
        from qdrant_client import models as qmodels

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))],
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            return 0

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": "deleted"},
            points=[p.id for p in points],
        )
        return len(points)

    def count_by_doc_id(self, doc_id: str) -> int:
        """统计指定 doc_id 下 active 状态的 chunks 数量。"""
        from qdrant_client import models as qmodels

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
                    qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value="active")),
                ],
            ),
            exact=True,
        )
        return result.count

    def get_content_hash_by_doc_id(self, doc_id: str) -> str | None:
        """获取指定 doc_id 的 content_hash（取第一个 active chunk 的值）。"""
        from qdrant_client import models as qmodels

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
                    qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value="active")),
                ],
            ),
            limit=1,
            with_payload=["content_hash"],
            with_vectors=False,
        )
        if not points:
            return None
        return points[0].payload.get("content_hash") if points[0].payload else None

    def _create_client(self) -> Any:
        """根据配置创建 qdrant-client。"""
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorStoreError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client") from exc

        kwargs = {"url": self.config.qdrant.url}
        if self.config.qdrant.api_key:
            kwargs["api_key"] = self.config.qdrant.api_key
        return QdrantClient(**kwargs)

    def _collection_exists(self) -> bool:
        """兼容不同 qdrant-client 版本的 collection 存在性检查。"""
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection_name=self.collection_name))
        try:
            self.client.get_collection(collection_name=self.collection_name)
            return True
        except Exception:
            return False

    def _make_point(self, chunk: DocumentChunk, vectors: dict[str, Any]) -> Any:
        """创建 Qdrant PointStruct。"""
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise VectorStoreError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client") from exc

        return models.PointStruct(
            id=self._point_id(chunk.chunk_id),
            vector=vectors,
            payload=chunk_to_payload(chunk),
        )

    def _query_points(self, query: Any, using: str, top_k: int, query_filter: Any = None) -> list[Any]:
        """兼容 query_points 与旧版 search API 的查询方法。"""
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query,
                using=using,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            return list(getattr(response, "points", response))

        # 旧版 qdrant-client 常用 search API，named vector 用 (name, vector) 表达。
        return list(
            self.client.search(
                collection_name=self.collection_name,
                query_vector=(using, query),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        )

    @staticmethod
    def _to_results(raw_results: list[Any], source: str, vector_name: str) -> list[VectorSearchResult]:
        """将 Qdrant 返回值转换为统一检索结果。"""
        results: list[VectorSearchResult] = []
        for rank, point in enumerate(raw_results, start=1):
            payload = dict(getattr(point, "payload", {}) or {})
            chunk_id = str(payload.get("chunk_id") or getattr(point, "id", ""))
            results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    text=payload_to_text(payload),
                    payload=payload,
                    source=source,
                    vector_name=vector_name,
                    rank=rank,
                )
            )
        return results

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """将任意 chunk_id 转成 Qdrant 接受的 UUID 字符串。"""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fireagent:{chunk_id}"))
