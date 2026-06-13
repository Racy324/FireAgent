"""会话历史 SQLite 存储测试。"""

from __future__ import annotations

from fireagent.memory import SQLiteMemoryStore


def test_sqlite_memory_store_creates_session_and_messages(tmp_path) -> None:
    """SQLite store 应能创建会话、保存消息并按时间读取。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()

    session = store.create_session(title="测试会话")
    user_message = store.save_message(
        session_id=session.session_id,
        role="user",
        content="隧道火灾烟气有什么影响？",
    )
    assistant_message = store.save_message(
        session_id=session.session_id,
        role="assistant",
        content="烟气会影响能见度和疏散。",
        citations=["证据 1"],
        metadata={"intent": "rag"},
    )

    sessions = store.list_sessions()
    messages = store.list_messages(session.session_id)

    assert sessions[0].session_id == session.session_id
    assert sessions[0].title == "测试会话"
    assert [message.message_id for message in messages] == [
        user_message.message_id,
        assistant_message.message_id,
    ]
    assert messages[1].citations == ["证据 1"]
    assert messages[1].metadata["intent"] == "rag"


def test_sqlite_memory_store_builds_recent_context(tmp_path) -> None:
    """最近上下文应只包含最近 N 条消息，并排除当前刚保存的用户消息。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="上下文测试")
    for index in range(5):
        store.save_message(session.session_id, "user", f"问题 {index}")
        store.save_message(session.session_id, "assistant", f"回答 {index}")

    recent = store.list_recent_messages(session.session_id, limit=4)
    context = store.build_recent_context(session.session_id, limit=4)

    assert [item.content for item in recent] == ["问题 3", "回答 3", "问题 4", "回答 4"]
    assert "用户：问题 3" in context
    assert "助手：回答 4" in context
    assert "问题 0" not in context


def test_session_state_create_update_roundtrip(tmp_path) -> None:
    """session_states 可以创建、读取、更新，JSON 字段正确往返。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="状态测试")

    # 初始状态为空
    state = store.get_or_create_session_state(session.session_id)
    assert state.current_goal == ""
    assert state.completed_items == []
    assert state.constraints == []

    # 更新字段
    updated = store.update_session_state(
        session.session_id,
        current_goal="实现 MVP-2",
        current_step="编写 store 测试",
        constraints=["不引入新数据库"],
        pending_items=["ContextManager", "API 接入"],
        completed_items=["schema 定义"],
    )
    assert updated.current_goal == "实现 MVP-2"
    assert updated.constraints == ["不引入新数据库"]
    assert len(updated.pending_items) == 2

    # 重新读取验证持久化
    loaded = store.get_session_state(session.session_id)
    assert loaded is not None
    assert loaded.current_goal == "实现 MVP-2"
    assert loaded.completed_items == ["schema 定义"]


def test_append_intermediate_result_respects_limit(tmp_path) -> None:
    """intermediate_results 超过上限时移除最旧的。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="中间结果测试")

    for i in range(5):
        store.append_intermediate_result(
            session.session_id,
            {"step": i, "status": "ok"},
            limit=3,
        )

    state = store.get_session_state(session.session_id)
    assert state is not None
    assert len(state.intermediate_results) == 3
    assert state.intermediate_results[0]["step"] == 2
    assert state.intermediate_results[2]["step"] == 4


def test_delete_session_cleans_up_session_state(tmp_path) -> None:
    """软删除 session 时应同步清理 session_states。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="清理测试")
    store.update_session_state(session.session_id, current_goal="测试")

    assert store.get_session_state(session.session_id) is not None

    store.delete_session(session.session_id)

    # session 被软删除
    assert store.get_session(session.session_id) is None
    # session_states 被物理删除
    assert store.get_session_state(session.session_id) is None


def test_get_or_create_session_state_rejects_missing_session(tmp_path) -> None:
    """对不存在的 session 创建 state 应抛出 ValueError。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()

    import pytest
    with pytest.raises(ValueError, match="不存在或已删除"):
        store.get_or_create_session_state("sess_nonexistent")
