"""Qdrant 向量库的数据结构与 schema 工具。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from fireagent.ingestion.schema import DocumentChunk


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


class VectorStoreError(RuntimeError):
    """向量库操作失败时抛出的领域异常。"""


class VectorStoreModel(BaseModel):
    """向量库模块的 Pydantic 基类。"""

    model_config = ConfigDict(extra="ignore")


class QdrantCollectionSchema(VectorStoreModel):
    """FireAgent 在 Qdrant 中使用的 collection schema 配置。"""

    collection_name: str = "fireagent_papers"
    dense_vector_name: str = DENSE_VECTOR_NAME
    sparse_vector_name: str = SPARSE_VECTOR_NAME
    dense_vector_size: int = Field(default=1024, gt=0)
    distance: str = "cosine"
    hnsw_m: int = Field(default=16, gt=0)
    hnsw_ef_construct: int = Field(default=100, gt=0)


class SparseVectorData(VectorStoreModel):
    """与 Qdrant sparse vector 兼容的轻量稀疏向量结构。"""

    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """判断稀疏向量是否为空。"""
        return not self.indices or not self.values


class VectorSearchResult(VectorStoreModel):
    """向量检索返回的标准候选结果。"""

    chunk_id: str
    score: float
    text: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    vector_name: str = ""
    rank: Optional[int] = None


class HybridSearchResultSet(VectorStoreModel):
    """dense 与 sparse 两路检索结果集合。"""

    dense_results: list[VectorSearchResult] = Field(default_factory=list)
    sparse_results: list[VectorSearchResult] = Field(default_factory=list)


def chunk_to_payload(chunk: DocumentChunk) -> dict[str, Any]:
    """将标准 DocumentChunk 转为 Qdrant payload。"""
    payload = {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "doc_id": chunk.doc_id,
        "paper_title": chunk.paper_title,
        "authors": chunk.authors,
        "year": chunk.year,
        "section_title": chunk.section_title,
        "section_path": chunk.section_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text": chunk.text,
        "chunk_type": chunk.chunk_type,
        "source_type": chunk.source_type,
        "metadata": chunk.metadata,
    }
    return payload


def payload_to_text(payload: dict[str, Any]) -> str:
    """从 payload 中读取候选文本。"""
    text = payload.get("text", "")
    return text if isinstance(text, str) else ""


def distance_name_to_qdrant(distance: str) -> Any:
    """将配置中的距离名称转换为 qdrant-client 的 Distance 枚举。

    这里使用懒导入，避免在未安装 qdrant-client 时影响本地测试。
    """
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise VectorStoreError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client") from exc

    normalized = distance.lower()
    if normalized == "cosine":
        return models.Distance.COSINE
    if normalized == "dot":
        return models.Distance.DOT
    if normalized == "euclid":
        return models.Distance.EUCLID
    if normalized == "manhattan" and hasattr(models.Distance, "MANHATTAN"):
        return models.Distance.MANHATTAN
    raise VectorStoreError(f"不支持的 Qdrant distance 配置：{distance}")


def build_dense_vector_params(schema: QdrantCollectionSchema) -> Any:
    """构建 Qdrant dense named vector 参数。"""
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise VectorStoreError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client") from exc

    hnsw_config = models.HnswConfigDiff(
        m=schema.hnsw_m,
        ef_construct=schema.hnsw_ef_construct,
    )
    return models.VectorParams(
        size=schema.dense_vector_size,
        distance=distance_name_to_qdrant(schema.distance),
        hnsw_config=hnsw_config,
    )


def build_sparse_vector_params() -> Any:
    """构建 Qdrant sparse named vector 参数。

    不同 qdrant-client 版本对 sparse vector 的构造参数略有差异，因此这里采用
    逐步降级的方式：优先启用 IDF modifier，若当前版本不支持，则退回基础稀疏向量。
    """
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise VectorStoreError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client") from exc

    sparse_index = None
    if hasattr(models, "SparseIndexParams"):
        try:
            sparse_index = models.SparseIndexParams(on_disk=False)
        except TypeError:
            sparse_index = models.SparseIndexParams()

    modifier = getattr(getattr(models, "Modifier", None), "IDF", None)

    try:
        if modifier is not None and sparse_index is not None:
            return models.SparseVectorParams(index=sparse_index, modifier=modifier)
        if sparse_index is not None:
            return models.SparseVectorParams(index=sparse_index)
        return models.SparseVectorParams()
    except TypeError:
        return models.SparseVectorParams()


def sparse_data_to_qdrant(vector: SparseVectorData) -> Any:
    """将轻量稀疏向量转换成 qdrant-client 的 SparseVector。"""
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise VectorStoreError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client") from exc

    return models.SparseVector(indices=vector.indices, values=vector.values)

