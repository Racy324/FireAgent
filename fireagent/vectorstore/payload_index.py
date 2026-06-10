"""Qdrant payload index 创建工具。"""

from __future__ import annotations

from typing import Any

from fireagent.vectorstore.schema import VectorStoreError


DEFAULT_PAYLOAD_INDEXES: tuple[tuple[str, str], ...] = (
    ("chunk_id", "keyword"),
    ("parent_id", "keyword"),
    ("doc_id", "keyword"),
    ("paper_title", "text"),
    ("authors", "keyword"),
    ("year", "integer"),
    ("section_title", "text"),
    ("chunk_type", "keyword"),
    ("source_type", "keyword"),
    ("page_start", "integer"),
    ("page_end", "integer"),
)


class QdrantPayloadIndexManager:
    """负责创建 FireAgent 常用 payload index。"""

    def __init__(self, client: Any, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    def create_indexes(self, indexes: tuple[tuple[str, str], ...] = DEFAULT_PAYLOAD_INDEXES) -> None:
        """批量创建 payload index；已存在时由 Qdrant 幂等处理或忽略错误。"""
        for field_name, schema_name in indexes:
            self.create_index(field_name, schema_name)

    def create_index(self, field_name: str, schema_name: str) -> None:
        """创建单个 payload index。"""
        field_schema = self._payload_schema(schema_name)
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:  # noqa: BLE001 - Qdrant 版本错误类型不完全一致。
            message = str(exc).lower()
            if "already exists" in message or "duplicate" in message:
                return
            raise VectorStoreError(f"创建 payload index 失败：{field_name} ({schema_name})") from exc

    @staticmethod
    def _payload_schema(schema_name: str) -> Any:
        """将简短 schema 名称映射为 qdrant-client PayloadSchemaType。"""
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise VectorStoreError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client") from exc

        normalized = schema_name.lower()
        mapping = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "integer": models.PayloadSchemaType.INTEGER,
            "float": models.PayloadSchemaType.FLOAT,
            "bool": models.PayloadSchemaType.BOOL,
            "geo": models.PayloadSchemaType.GEO,
            "text": models.PayloadSchemaType.TEXT,
            "datetime": getattr(models.PayloadSchemaType, "DATETIME", models.PayloadSchemaType.KEYWORD),
        }
        if normalized not in mapping:
            raise VectorStoreError(f"不支持的 payload index 类型：{schema_name}")
        return mapping[normalized]


def create_default_payload_indexes(client: Any, collection_name: str) -> None:
    """便捷函数：创建默认 payload index。"""
    QdrantPayloadIndexManager(client=client, collection_name=collection_name).create_indexes()

