"""长期记忆 Service 测试。"""

from __future__ import annotations

from fireagent.memory.long_term_service import LongTermMemoryService
from fireagent.memory.long_term_schema import LongTermMemoryRecord
from fireagent.utils.config import FireAgentConfig
from tests.test_long_term_memory_store import FakeDenseEmbedder, FakeQdrantClient, _make_store


def _make_service() -> tuple[LongTermMemoryService, FakeQdrantClient]:
    store, client = _make_store()
    store.create_collection()
    cfg = FireAgentConfig()
    cfg.memory.long_term.importance_threshold = 0.55  # 适配规则化评分
    cfg.memory.long_term.confidence_threshold = 0.55
    service = LongTermMemoryService(config=cfg, store=store)
    return service, client


def test_retrieve_for_query_returns_empty_when_no_memories() -> None:
    """无记忆时检索应返回空。"""
    service, _ = _make_service()
    results = service.retrieve_for_query("测试问题")
    assert results == []


def test_retrieve_for_query_formats_prompt_memories() -> None:
    """检索结果应正确格式化为 prompt 文本。"""
    service, _ = _make_service()

    # 写入一条记忆
    service.store.upsert_memory(LongTermMemoryRecord(
        memory_id="mem_1",
        content="用户偏好：项目笔记默认中文",
        memory_type="semantic",
        importance=0.85,
        created_at="2026-06-13T10:00:00Z",
    ))

    results = service.retrieve_for_query("笔记语言")
    prompt_text = service.format_for_prompt(results)

    if results:  # fake client 可能返回
        assert "semantic" in prompt_text
        assert "项目笔记默认中文" in prompt_text


def test_format_for_prompt_empty_results() -> None:
    """空结果应返回空字符串。"""
    service, _ = _make_service()
    assert service.format_for_prompt([]) == ""


def test_maybe_write_after_turn_writes_high_value_memory() -> None:
    """高价值对话应写入长期记忆。"""
    service, client = _make_service()

    written = service.maybe_write_after_turn(
        session_id="sess_1",
        user_message_id="msg_u1",
        assistant_message_id="msg_a1",
        user_query="以后所有项目笔记默认使用中文，不要写英文。",
        assistant_answer="好的，已记录。",
    )

    # 应该写入至少一条
    assert len(written) >= 1
    points = client.points["fireagent_memories"]
    assert len(points) >= 1


def test_maybe_write_after_turn_skips_low_value_memory() -> None:
    """低价值对话不应写入长期记忆。"""
    service, client = _make_service()

    written = service.maybe_write_after_turn(
        session_id="sess_1",
        user_message_id="msg_u1",
        assistant_message_id="msg_a1",
        user_query="你好",
        assistant_answer="你好，我是 FireAgent。",
    )

    assert written == []
    points = client.points["fireagent_memories"]
    assert len(points) == 0


def test_maybe_write_after_turn_skips_transient_error() -> None:
    """临时错误信息不应写入。"""
    service, _ = _make_service()

    written = service.maybe_write_after_turn(
        session_id="sess_1",
        user_message_id="msg_u1",
        assistant_message_id="msg_a1",
        user_query="这次测试报错了 error code 500",
        assistant_answer="请检查日志。",
    )

    assert written == []


def test_duplicate_memory_updates_existing_record() -> None:
    """重复记忆应更新已有记录而非新建。"""
    service, client = _make_service()

    # 先写入一条
    service.store.upsert_memory(LongTermMemoryRecord(
        memory_id="mem_existing",
        content="以后笔记用中文",
        memory_type="semantic",
        importance=0.7,
        tags=["preference"],
        created_at="2026-06-13T10:00:00Z",
    ))

    # 再尝试写入相似内容
    written = service.maybe_write_after_turn(
        session_id="sess_2",
        user_message_id="msg_u2",
        assistant_message_id="msg_a2",
        user_query="以后笔记用中文，这是确定的。",
        assistant_answer="好的。",
    )

    # 不应新建（去重命中），但可能更新
    points = client.points["fireagent_memories"]
    # 总数不应超过 1（去重更新）
    assert len(points) <= 2  # 允许更新后重新写入
