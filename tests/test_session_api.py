"""会话历史 API 测试。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from fireagent.api.server import create_app
from fireagent.graph.state import FireAgentState
from fireagent.memory.long_term_schema import LongTermMemoryRecord, LongTermMemorySearchResult
from fireagent.retrieval.schema import StructuredCitation
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
            citations=["[L1]: 隧道火灾烟气控制研究，张三，2024，烟气控制，第1页"],
            used_citation_markers=["[L1]"],
            used_citations=[
                StructuredCitation(
                    citation_id="L1",
                    marker="[L1]",
                    source_type="local_pdf",
                    title="隧道火灾烟气控制研究",
                    authors=["张三"],
                    year=2024,
                    section_title="烟气控制",
                    page_start=1,
                    score=0.9,
                ),
            ],
            invalid_citation_markers=[],
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

    # used_citations 字段断言
    assert payload["used_citation_markers"] == ["[L1]"]
    assert payload["invalid_citation_markers"] == []
    assert len(payload["used_citations"]) == 1
    assert payload["used_citations"][0]["marker"] == "[L1]"
    assert payload["used_citations"][0]["title"] == "隧道火灾烟气控制研究"
    assert payload["used_citations"][0]["source_type"] == "local_pdf"

    # citations 字段等同于 used citation strings
    assert len(payload["citations"]) == 1
    assert "[L1]" in payload["citations"][0]

    # 历史消息 metadata 中保存 used_citations
    assistant_meta = messages[1]["metadata"]
    assert assistant_meta["evidence_sufficient"] is True
    assert len(assistant_meta["used_citations"]) == 1
    assert assistant_meta["used_citations"][0]["marker"] == "[L1]"
    assert assistant_meta["used_citation_markers"] == ["[L1]"]
    assert assistant_meta["invalid_citation_markers"] == []


def test_chat_api_returns_empty_citations_for_fallback(monkeypatch, tmp_path) -> None:
    """当 LLM 不可用时，fallback 回答应返回空 used_citations。"""
    cfg = FireAgentConfig()
    cfg.memory.database_path = str(tmp_path / "fireagent.db")
    cfg.llm.enabled = False

    def fake_workflow(user_query, **_kwargs):
        return FireAgentState(
            user_query=user_query,
            intent="rag",
            final_answer="证据不足，无法回答。",
            evidence_sufficient=False,
            citations=[],
            used_citation_markers=[],
            used_citations=[],
            invalid_citation_markers=[],
            errors=[],
        )

    monkeypatch.setattr("fireagent.api.server.run_fireagent_workflow", fake_workflow)
    app = create_app(config=cfg)
    client = TestClient(app)

    response = client.post("/chat", json={"query": "烟气有什么影响？"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["citations"] == []
    assert payload["used_citations"] == []
    assert payload["used_citation_markers"] == []
    assert payload["invalid_citation_markers"] == []


def test_chat_injects_long_term_memories(monkeypatch, tmp_path) -> None:
    """长期记忆应被检索并传入 workflow。"""
    cfg = FireAgentConfig()
    cfg.memory.database_path = str(tmp_path / "fireagent.db")

    captured_kwargs: dict[str, Any] = {}

    def fake_workflow(user_query, **kwargs):
        captured_kwargs.update(kwargs)
        return FireAgentState(
            user_query=user_query,
            intent="rag",
            final_answer="好的，遵循你的偏好。",
            citations=[],
            used_citation_markers=[],
            used_citations=[],
            invalid_citation_markers=[],
            errors=[],
        )

    # mock LongTermMemoryService 类
    mock_lt_service = MagicMock()
    mock_lt_service.retrieve_for_query.return_value = [
        LongTermMemorySearchResult(
            memory=LongTermMemoryRecord(
                memory_id="mem_1",
                content="用户偏好：项目笔记默认中文",
                importance=0.85,
                created_at="2026-06-13T10:00:00Z",
            ),
            semantic_score=0.9,
            final_score=0.8,
        ),
    ]
    mock_lt_service.format_for_prompt.return_value = "- [semantic | 0.80] 用户偏好：项目笔记默认中文"
    mock_lt_service.maybe_write_after_turn.return_value = []

    monkeypatch.setattr("fireagent.api.server.run_fireagent_workflow", fake_workflow)
    monkeypatch.setattr(
        "fireagent.api.server.LongTermMemoryService",
        lambda **kwargs: mock_lt_service,
    )

    app = create_app(config=cfg)
    client = TestClient(app)

    response = client.post("/chat", json={"query": "帮我写个笔记", "include_debug": True})
    payload = response.json()

    assert response.status_code == 200
    # 长期记忆被检索
    mock_lt_service.retrieve_for_query.assert_called_once_with("帮我写个笔记")
    # 长期记忆被传入 workflow
    assert captured_kwargs.get("long_term_memories") == "- [semantic | 0.80] 用户偏好：项目笔记默认中文"
    # debug 中包含长期记忆
    assert payload["debug"]["long_term_memories"] == "- [semantic | 0.80] 用户偏好：项目笔记默认中文"


def test_chat_continues_when_long_term_memory_retrieval_fails(monkeypatch, tmp_path) -> None:
    """长期记忆检索失败不应影响主问答流程。"""
    cfg = FireAgentConfig()
    cfg.memory.database_path = str(tmp_path / "fireagent.db")

    def fake_workflow(user_query, **_kwargs):
        return FireAgentState(
            user_query=user_query,
            intent="rag",
            final_answer="正常回答。",
            citations=[],
            used_citation_markers=[],
            used_citations=[],
            invalid_citation_markers=[],
            errors=[],
        )

    mock_lt_service = MagicMock()
    mock_lt_service.retrieve_for_query.side_effect = RuntimeError("Qdrant 连接失败")

    monkeypatch.setattr("fireagent.api.server.run_fireagent_workflow", fake_workflow)
    monkeypatch.setattr(
        "fireagent.api.server.LongTermMemoryService",
        lambda **kwargs: mock_lt_service,
    )

    app = create_app(config=cfg)
    client = TestClient(app)

    response = client.post("/chat", json={"query": "测试问题", "include_debug": True})
    payload = response.json()

    # 主流程正常完成
    assert response.status_code == 200
    assert payload["answer"] == "正常回答。"
    # 长期记忆为空（检索失败被跳过）
    assert payload["debug"]["long_term_memories"] == ""


def test_chat_continues_when_long_term_memory_write_fails(monkeypatch, tmp_path) -> None:
    """长期记忆写入失败不应影响主问答流程。"""
    cfg = FireAgentConfig()
    cfg.memory.database_path = str(tmp_path / "fireagent.db")

    def fake_workflow(user_query, **_kwargs):
        return FireAgentState(
            user_query=user_query,
            intent="rag",
            final_answer="正常回答。",
            citations=[],
            used_citation_markers=[],
            used_citations=[],
            invalid_citation_markers=[],
            errors=[],
        )

    mock_lt_service = MagicMock()
    mock_lt_service.retrieve_for_query.return_value = []
    mock_lt_service.format_for_prompt.return_value = ""
    mock_lt_service.maybe_write_after_turn.side_effect = RuntimeError("写入失败")

    monkeypatch.setattr("fireagent.api.server.run_fireagent_workflow", fake_workflow)
    monkeypatch.setattr(
        "fireagent.api.server.LongTermMemoryService",
        lambda **kwargs: mock_lt_service,
    )

    app = create_app(config=cfg)
    client = TestClient(app)

    response = client.post("/chat", json={"query": "以后笔记用中文"})
    payload = response.json()

    # 主流程正常完成，不因写入失败而报错
    assert response.status_code == 200
    assert payload["answer"] == "正常回答。"
