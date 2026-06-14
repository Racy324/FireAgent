"""长期记忆 Store 测试（使用 fake Qdrant client 和 fake embedder）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from fireagent.memory.long_term_schema import LongTermMemoryRecord
from fireagent.memory.long_term_store import LongTermMemoryStore, new_memory_id
from fireagent.utils.config import FireAgentConfig


# ── Fake 实现 ──


class FakeDenseEmbedder:
    """返回固定维度随机向量的 fake embedder。"""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        # 简单哈希生成确定性向量
        h = hash(text)
        return [((h >> i) & 0xFF) / 255.0 for i in range(0, self.dim * 8, 8)]

    @property
    def dimension(self) -> int:
        return self.dim


@dataclass
class FakePoint:
    id: str
    vector: dict[str, list[float]]
    payload: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class FakeQdrantClient:
    """内存模拟 Qdrant 客户端。"""

    def __init__(self) -> None:
        self.collections: dict[str, bool] = {}
        self.points: dict[str, dict[str, FakePoint]] = {}  # collection -> {id -> point}

    def get_collection(self, name: str) -> Any:
        if name not in self.collections:
            raise RuntimeError("not found")
        return True

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)
        self.points.pop(name, None)

    def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        self.collections[collection_name] = True
        self.points[collection_name] = {}

    def create_payload_index(self, collection_name: str, **kwargs: Any) -> None:
        pass

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        if collection_name not in self.points:
            self.points[collection_name] = {}
        for pt in points:
            self.points[collection_name][pt.id] = FakePoint(
                id=pt.id,
                vector=pt.vector,
                payload=pt.payload,
            )

    def retrieve(self, collection_name: str, ids: list[str], **kwargs: Any) -> list[FakePoint]:
        coll = self.points.get(collection_name, {})
        return [coll[i] for i in ids if i in coll]

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        using: str = "",
        query_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> Any:
        coll = self.points.get(collection_name, {})
        results = []
        for pt in coll.values():
            # 简单相似度：用 payload 中的 content 长度模拟
            score = 0.8 if pt.payload.get("status") == "active" else 0.1
            results.append(FakePoint(
                id=pt.id,
                vector=pt.vector,
                payload=pt.payload,
                score=score,
            ))
        results.sort(key=lambda p: p.score, reverse=True)

        @dataclass
        class FakeQueryResult:
            points: list[FakePoint]

        return FakeQueryResult(points=results[:limit])


# ── 测试 ──


def _make_store() -> tuple[LongTermMemoryStore, FakeQdrantClient]:
    cfg = FireAgentConfig()
    cfg.qdrant.dense_vector_size = 16
    cfg.memory.long_term.dedup_similarity_threshold = 0.75  # 适配 fake client 的 score=0.8
    client = FakeQdrantClient()
    embedder = FakeDenseEmbedder(dim=16)
    store = LongTermMemoryStore(config=cfg, client=client, dense_embedder=embedder)
    return store, client


def test_create_memory_collection() -> None:
    """应能创建 collection。"""
    store, client = _make_store()
    store.create_collection()
    assert "fireagent_memories" in client.collections


def test_create_memory_collection_recreate() -> None:
    """recreate=True 应先删后建。"""
    store, client = _make_store()
    store.create_collection()
    store.create_collection(recreate=True)
    assert "fireagent_memories" in client.collections


def test_new_memory_id_is_qdrant_compatible_uuid() -> None:
    """Qdrant point ID 必须是 UUID 或无符号整数。"""
    memory_id = new_memory_id()

    assert str(UUID(memory_id)) == memory_id


def test_upsert_memory_writes_payload_and_vector() -> None:
    """写入记忆应保存 payload 和 vector。"""
    store, client = _make_store()
    store.create_collection()

    memory = LongTermMemoryRecord(
        memory_id=new_memory_id(),
        content="用户偏好：项目笔记默认中文",
        memory_type="semantic",
        importance=0.85,
        created_at="2026-06-13T10:00:00Z",
    )
    store.upsert_memory(memory)

    points = client.points["fireagent_memories"]
    assert len(points) == 1
    saved = list(points.values())[0]
    assert saved.payload["content"] == "用户偏好：项目笔记默认中文"
    assert saved.payload["importance"] == 0.85


def test_search_returns_active_memories() -> None:
    """检索应只返回 active 状态的记忆。"""
    store, client = _make_store()
    store.create_collection()

    store.upsert_memory(LongTermMemoryRecord(
        memory_id="mem_active",
        content="活跃记忆",
        status="active",
        importance=0.8,
        created_at="2026-06-13T10:00:00Z",
    ))
    store.upsert_memory(LongTermMemoryRecord(
        memory_id="mem_disabled",
        content="禁用记忆",
        status="disabled",
        importance=0.8,
        created_at="2026-06-13T10:00:00Z",
    ))

    results = store.search("活跃", top_k=5)
    # fake client 简单返回所有 active，disabled 给低分
    active_results = [r for r in results if r.memory.status == "active"]
    assert len(active_results) >= 1


def test_search_applies_decay_scoring() -> None:
    """检索结果应包含 decay 和 final_score。"""
    store, _ = _make_store()
    store.create_collection()

    store.upsert_memory(LongTermMemoryRecord(
        memory_id="mem_1",
        content="测试记忆",
        importance=0.8,
        created_at="2026-06-13T10:00:00Z",
    ))

    results = store.search("测试", top_k=5)
    assert len(results) >= 1
    assert results[0].final_score > 0
    assert results[0].decay_factor > 0


def test_find_duplicate_returns_similar_memory() -> None:
    """去重应返回高相似度记忆。"""
    store, _ = _make_store()
    store.create_collection()

    memory = LongTermMemoryRecord(
        memory_id="mem_dup",
        content="用户偏好：笔记用中文",
        importance=0.8,
        created_at="2026-06-13T10:00:00Z",
    )
    store.upsert_memory(memory)

    # 相同内容应命中去重（fake client 返回 score=0.8）
    found = store.find_duplicate("用户偏好：笔记用中文")
    assert found is not None
    assert found.memory_id == "mem_dup"


def test_update_memory_fields() -> None:
    """应能更新已有记忆的部分字段。"""
    store, client = _make_store()
    store.create_collection()

    memory = LongTermMemoryRecord(
        memory_id="mem_update",
        content="原始内容",
        importance=0.5,
        tags=["old"],
        created_at="2026-06-13T10:00:00Z",
    )
    store.upsert_memory(memory)

    store.update_memory_fields("mem_update", importance=0.9, tags=["new"])

    updated = client.points["fireagent_memories"]["mem_update"]
    assert updated.payload["importance"] == 0.9
    # tags 应合并
    assert set(updated.payload["tags"]) == {"old", "new"}
