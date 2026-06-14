"""Trace 与 /chat、/chat/stream 接口集成测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def mock_workflow_state():
    """模拟 run_fireagent_workflow 返回的状态。"""
    return {
        "final_answer": "测试回答内容",
        "intent": "chat",
        "evidence_sufficient": False,
        "used_citations": [],
        "used_citation_markers": [],
        "invalid_citation_markers": [],
        "citations": [],
        "errors": [],
        "final_context": "",
        "rewritten_queries": [],
        "route_decision": {"sub_intent": "memory_query"},
        "hallucination_warnings": [],
    }


def _mock_streaming_events(*args, **kwargs):
    """模拟 run_streaming_workflow 返回的 SSE 事件序列（含 trace 事件）。"""
    from fireagent.api.streaming import _sse_event
    recorder = kwargs.get("recorder")
    if recorder and recorder.trace:
        yield _sse_event("trace", {"trace_id": recorder.trace.trace_id})
    yield _sse_event("stage", {"stage": "intent", "value": "chat", "route_decision": {}})
    yield _sse_event("token", {"token": "你"})
    yield _sse_event("token", {"token": "好"})
    yield _sse_event("done", {
        "citations": [],
        "used_citations": [],
        "used_citation_markers": [],
        "invalid_citation_markers": [],
        "intent": "chat",
        "route_decision": {},
        "evidence_sufficient": False,
        "elapsed": 0.5,
    })


@pytest.fixture
def app_config(tmp_path):
    """创建测试用配置，trace_dir 指向临时目录，禁用 memory。"""
    from fireagent.utils.config import load_config, ObservabilityConfig

    cfg = load_config()
    cfg.observability = ObservabilityConfig(
        enabled=True,
        trace_dir=str(tmp_path),
        save_trace=True,
        save_index=True,
    )
    # 禁用 memory 避免连接 SQLite/Qdrant
    cfg.memory.enabled = False
    cfg.memory.long_term.enabled = False
    return cfg


@pytest.fixture
def client(app_config):
    """创建 FastAPI 测试客户端。"""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi[testclient] not installed")

    from fireagent.api.server import create_app

    app = create_app(config=app_config)
    return TestClient(app)


# ── /chat 测试 ──


def test_chat_returns_trace_id(client, mock_workflow_state, tmp_path):
    """/chat 响应应包含非空 trace_id。"""
    with patch("fireagent.api.server.run_fireagent_workflow", return_value=mock_workflow_state):
        resp = client.post("/chat", json={"query": "你好"})
    assert resp.status_code == 200
    data = resp.json()
    assert "trace_id" in data
    assert data["trace_id"].startswith("trace_")


def test_chat_creates_trace_file(client, mock_workflow_state, tmp_path):
    """/chat 请求后应生成 trace JSON 文件。"""
    with patch("fireagent.api.server.run_fireagent_workflow", return_value=mock_workflow_state):
        resp = client.post("/chat", json={"query": "测试问题"})
    trace_id = resp.json()["trace_id"]
    json_files = list(tmp_path.glob("**/*.json"))
    assert len(json_files) >= 1
    assert any(trace_id in f.name for f in json_files)


def test_chat_trace_has_api_steps(client, mock_workflow_state, tmp_path):
    """trace 中应包含 api_receive 和 api_response 步骤。"""
    with patch("fireagent.api.server.run_fireagent_workflow", return_value=mock_workflow_state):
        resp = client.post("/chat", json={"query": "测试"})
    trace_id = resp.json()["trace_id"]

    trace_file = [f for f in tmp_path.glob("**/*.json") if trace_id in f.name][0]
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    step_types = [s["step_type"] for s in data["steps"]]
    assert "api_receive" in step_types
    assert "api_response" in step_types


def test_chat_trace_has_intent(client, mock_workflow_state, tmp_path):
    """trace 中应记录 intent。"""
    mock_workflow_state["intent"] = "chat"
    with patch("fireagent.api.server.run_fireagent_workflow", return_value=mock_workflow_state):
        resp = client.post("/chat", json={"query": "你好"})
    trace_id = resp.json()["trace_id"]

    trace_file = [f for f in tmp_path.glob("**/*.json") if trace_id in f.name][0]
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert data["intent"] == "chat"


def test_chat_trace_index_jsonl(client, mock_workflow_state, tmp_path):
    """/chat 请求后应追加 index.jsonl。"""
    with patch("fireagent.api.server.run_fireagent_workflow", return_value=mock_workflow_state):
        client.post("/chat", json={"query": "问题1"})
    index_files = list(tmp_path.glob("**/index.jsonl"))
    assert len(index_files) == 1
    lines = index_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[0])
    assert entry["trace_id"].startswith("trace_")


def test_chat_trace_final_status_success(client, mock_workflow_state, tmp_path):
    """正常请求的 trace final_status 应为 success。"""
    mock_workflow_state["errors"] = []
    with patch("fireagent.api.server.run_fireagent_workflow", return_value=mock_workflow_state):
        resp = client.post("/chat", json={"query": "测试"})
    trace_id = resp.json()["trace_id"]

    trace_file = [f for f in tmp_path.glob("**/*.json") if trace_id in f.name][0]
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert data["final_status"] == "success"


def test_chat_trace_degraded_on_errors(client, mock_workflow_state, tmp_path):
    """工作流有 errors 时 trace final_status 应为 degraded。"""
    mock_workflow_state["errors"] = ["some error"]
    with patch("fireagent.api.server.run_fireagent_workflow", return_value=mock_workflow_state):
        resp = client.post("/chat", json={"query": "测试"})
    trace_id = resp.json()["trace_id"]

    trace_file = [f for f in tmp_path.glob("**/*.json") if trace_id in f.name][0]
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert data["final_status"] == "degraded"


def test_chat_workflow_exception_saves_failed_trace(client, tmp_path):
    """工作流抛异常时 trace final_status 应为 failed。"""
    with patch("fireagent.api.server.run_fireagent_workflow", side_effect=RuntimeError("boom")):
        resp = client.post("/chat", json={"query": "测试"})
    assert resp.status_code == 500
    json_files = list(tmp_path.glob("**/*.json"))
    assert len(json_files) >= 1
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert data["final_status"] == "failed"


# ── /chat/stream 测试 ──


def _parse_sse_events(raw: str) -> list[tuple[str, dict]]:
    """解析原始 SSE 文本为 (event_name, payload) 列表。"""
    events = []
    current_event = ""
    for line in raw.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            payload_str = line[len("data: "):]
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                payload = {}
            events.append((current_event, payload))
    return events


def test_stream_first_event_is_trace(client, tmp_path):
    """第一条 SSE 事件应是 trace，包含 trace_id。"""
    with patch("fireagent.api.server.run_streaming_workflow", side_effect=_mock_streaming_events):
        resp = client.post("/chat/stream", json={"query": "你好"})
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert len(events) >= 1
    first_event_name, first_payload = events[0]
    assert first_event_name == "trace"
    assert "trace_id" in first_payload
    assert first_payload["trace_id"].startswith("trace_")


def test_stream_done_has_trace_id(client, tmp_path):
    """done 事件应包含与 trace 事件相同的 trace_id。"""
    with patch("fireagent.api.server.run_streaming_workflow", side_effect=_mock_streaming_events):
        resp = client.post("/chat/stream", json={"query": "你好"})
    events = _parse_sse_events(resp.text)
    trace_events = [(n, p) for n, p in events if n == "trace"]
    done_events = [(n, p) for n, p in events if n == "done"]
    assert len(trace_events) == 1
    assert len(done_events) == 1
    assert trace_events[0][1]["trace_id"] == done_events[0][1]["trace_id"]


def test_stream_creates_trace_file(client, tmp_path):
    """流式请求结束后应生成 trace 文件。"""
    with patch("fireagent.api.server.run_streaming_workflow", side_effect=_mock_streaming_events):
        resp = client.post("/chat/stream", json={"query": "测试"})
    events = _parse_sse_events(resp.text)
    done_events = [(n, p) for n, p in events if n == "done"]
    trace_id = done_events[0][1]["trace_id"]
    json_files = list(tmp_path.glob("**/*.json"))
    assert len(json_files) >= 1
    assert any(trace_id in f.name for f in json_files)


def test_stream_trace_has_api_steps(client, tmp_path):
    """流式 trace 中应包含 api_receive、api_response 等 API 层步骤。"""
    with patch("fireagent.api.server.run_streaming_workflow", side_effect=_mock_streaming_events):
        resp = client.post("/chat/stream", json={"query": "测试"})
    events = _parse_sse_events(resp.text)
    trace_id = [p["trace_id"] for n, p in events if n == "trace"][0]
    trace_file = [f for f in tmp_path.glob("**/*.json") if trace_id in f.name][0]
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    step_types = [s["step_type"] for s in data["steps"]]
    assert "api_receive" in step_types
    assert "api_response" in step_types


def test_stream_trace_final_status_success(client, tmp_path):
    """正常流式请求的 trace final_status 应为 success。"""
    with patch("fireagent.api.server.run_streaming_workflow", side_effect=_mock_streaming_events):
        resp = client.post("/chat/stream", json={"query": "测试"})
    events = _parse_sse_events(resp.text)
    trace_id = [p["trace_id"] for n, p in events if n == "trace"][0]
    trace_file = [f for f in tmp_path.glob("**/*.json") if trace_id in f.name][0]
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert data["final_status"] == "success"


def test_stream_exception_saves_failed_trace(client, tmp_path):
    """流式过程中异常时应保存 failed trace。"""
    def _failing_stream(*args, **kwargs):
        from fireagent.api.streaming import _sse_event
        yield _sse_event("stage", {"stage": "intent", "value": "chat", "route_decision": {}})
        raise RuntimeError("stream boom")

    with patch("fireagent.api.server.run_streaming_workflow", side_effect=_failing_stream):
        resp = client.post("/chat/stream", json={"query": "测试"})
    # TestClient 可能不会抛异常，但 trace 应该保存
    json_files = list(tmp_path.glob("**/*.json"))
    assert len(json_files) >= 1
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert data["final_status"] == "failed"
