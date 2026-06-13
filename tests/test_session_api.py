"""会话历史 API 测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from fireagent.api.server import create_app
from fireagent.graph.state import FireAgentState
from fireagent.utils.config import FireAgentConfig


def test_session_api_creates_lists_and_deletes_sessions(tmp_path) -> None:
    """会话 API 应支持创建、列表、消息查询和删除。"""
    cfg = FireAgentConfig()
    cfg.memory.database_path = str(tmp_path / "fireagent.db")
    app = create_app(config=cfg)
    client = TestClient(app)

    created = client.post("/sessions", json={"title": "第三阶段"}).json()
    session_id = created["session_id"]

    listed = client.get("/sessions").json()
    messages = client.get(f"/sessions/{session_id}/messages").json()
    deleted = client.delete(f"/sessions/{session_id}")

    assert listed["sessions"][0]["session_id"] == session_id
    assert messages["session_id"] == session_id
    assert messages["messages"] == []
    assert deleted.status_code == 204
    assert client.get("/sessions").json()["sessions"] == []


def test_chat_api_persists_user_and_assistant_messages(monkeypatch, tmp_path) -> None:
    """chat 请求应创建/复用 session，并保存 user 与 assistant 消息。"""
    cfg = FireAgentConfig()
    cfg.memory.database_path = str(tmp_path / "fireagent.db")

    def fake_workflow(user_query, **_kwargs):
        return FireAgentState(
            user_query=user_query,
            intent="rag",
            final_answer="烟气会降低能见度。",
            evidence_sufficient=True,
            citations=["证据 1"],
            errors=[],
        )

    monkeypatch.setattr("fireagent.api.server.run_fireagent_workflow", fake_workflow)
    app = create_app(config=cfg)
    client = TestClient(app)

    response = client.post("/chat", json={"query": "烟气有什么影响？"})
    payload = response.json()
    session_id = payload["session_id"]
    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]

    assert response.status_code == 200
    assert payload["answer"] == "烟气会降低能见度。"
    assert payload["session_id"] == session_id
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "烟气有什么影响？"
    assert messages[1]["content"] == "烟气会降低能见度。"
