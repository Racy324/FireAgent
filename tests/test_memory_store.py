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
