"""TraceRecorder 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fireagent.observability import (
    AgentTrace,
    TraceRecorder,
    TraceStep,
    generate_trace_id,
    get_current_trace_recorder,
    trace_context,
)


@pytest.fixture
def recorder(tmp_path: Path) -> TraceRecorder:
    return TraceRecorder(trace_dir=tmp_path, enabled=True)


# ── generate_trace_id ──


def test_generate_trace_id_format():
    tid = generate_trace_id()
    assert tid.startswith("trace_")
    parts = tid.split("_")
    assert len(parts) == 4  # trace, date, time, hex
    assert len(parts[1]) == 8  # YYYYMMDD
    assert len(parts[2]) == 6  # HHMMSS
    assert len(parts[3]) == 8  # 8 hex chars


# ── start_trace ──


def test_start_trace_creates_agent_trace(recorder: TraceRecorder):
    trace = recorder.start_trace(endpoint="/chat", user_goal="测试问题")
    assert isinstance(trace, AgentTrace)
    assert trace.trace_id.startswith("trace_")
    assert trace.endpoint == "/chat"
    assert trace.user_goal == "测试问题"
    assert trace.start_time != ""


def test_start_trace_with_session_id(recorder: TraceRecorder):
    trace = recorder.start_trace(endpoint="/chat", user_goal="q", session_id="sess_123")
    assert trace.session_id == "sess_123"


# ── step ──


def test_step_success_status(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    with recorder.step("test_step", current_goal="测试步骤") as step:
        assert step.status == "running"
        assert step.step_id == 1
    assert step.status == "success"
    assert step.latency_ms >= 0
    assert step.end_time != ""


def test_step_auto_increment_id(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    with recorder.step("step_a") as s1:
        pass
    with recorder.step("step_b") as s2:
        pass
    assert s1.step_id == 1
    assert s2.step_id == 2


def test_step_exception_marks_failed(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    with pytest.raises(ValueError):
        with recorder.step("test_step") as step:
            raise ValueError("boom")
    assert step.status == "failed"
    assert "boom" in step.error


def test_step_manual_skip(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    with recorder.step("test_step") as step:
        step.status = "skipped"
        step.error = "service unavailable"
    assert step.status == "skipped"
    assert step.error == "service unavailable"


def test_step_tool_args_summary(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    with recorder.step("test_step", tool_args_summary={"top_k": 5}) as step:
        pass
    assert step.tool_args_summary == {"top_k": 5}


def test_step_risk_level(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    with recorder.step("test_step", risk_level="medium") as step:
        pass
    assert step.risk_level == "medium"


def test_step_without_start_raises(recorder: TraceRecorder):
    with pytest.raises(RuntimeError, match="start_trace"):
        with recorder.step("test"):
            pass


# ── update_trace ──


def test_update_trace_fields(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    recorder.update_trace(intent="rag", sub_intent="knowledge_qa", evidence_sufficient=True)
    assert recorder.trace.intent == "rag"
    assert recorder.trace.sub_intent == "knowledge_qa"
    assert recorder.trace.evidence_sufficient is True


def test_update_trace_ignores_unknown(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    recorder.update_trace(nonexistent_field="value")  # should not raise
    assert recorder.trace.intent == ""


# ── finalize ──


def test_finalize_sets_fields(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    result = recorder.finalize(
        final_status="success",
        message_id="msg_123",
        final_eval={"answer_chars": 100},
    )
    assert result.final_status == "success"
    assert result.message_id == "msg_123"
    assert result.end_time != ""
    assert result.latency_ms >= 0
    assert result.final_eval == {"answer_chars": 100}


def test_finalize_failed_status(recorder: TraceRecorder):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    result = recorder.finalize(final_status="failed", error="something broke")
    assert result.final_status == "failed"
    assert result.error_count >= 1


def test_finalize_before_start_raises(recorder: TraceRecorder):
    with pytest.raises(RuntimeError, match="start_trace"):
        recorder.finalize(final_status="success")


# ── save ──


def test_save_creates_json_file(recorder: TraceRecorder, tmp_path: Path):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    recorder.finalize(final_status="success")
    recorder.save()
    trace_id = recorder.trace.trace_id
    # file should be in tmp_path/<date>/<trace_id>.json
    json_files = list(tmp_path.glob("**/*.json"))
    assert len(json_files) == 1
    assert trace_id in json_files[0].name
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert data["trace_id"] == trace_id


def test_save_creates_index_jsonl(recorder: TraceRecorder, tmp_path: Path):
    recorder.start_trace(endpoint="/chat", user_goal="test")
    recorder.finalize(final_status="success")
    recorder.save()
    index_files = list(tmp_path.glob("**/index.jsonl"))
    assert len(index_files) == 1
    lines = index_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["trace_id"] == recorder.trace.trace_id
    assert entry["final_status"] == "success"


def test_save_disabled_noop(tmp_path: Path):
    recorder = TraceRecorder(trace_dir=tmp_path, enabled=False)
    recorder.start_trace(endpoint="/chat", user_goal="test")
    recorder.finalize(final_status="success")
    recorder.save()
    assert list(tmp_path.glob("**/*.json")) == []


def test_save_multiple_traces(recorder: TraceRecorder, tmp_path: Path):
    for i in range(3):
        recorder.start_trace(endpoint="/chat", user_goal=f"q{i}")
        recorder.finalize(final_status="success")
        recorder.save()
    json_files = list(tmp_path.glob("**/*.json"))
    index_files = list(tmp_path.glob("**/index.jsonl"))
    assert len(json_files) == 3
    assert len(index_files) == 1
    lines = index_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


# ── context var ──


def test_context_var_propagation(recorder: TraceRecorder):
    assert get_current_trace_recorder() is None
    with trace_context(recorder):
        assert get_current_trace_recorder() is recorder
    assert get_current_trace_recorder() is None


def test_context_var_nested():
    r1 = TraceRecorder(trace_dir=Path("/tmp/a"), enabled=True)
    r2 = TraceRecorder(trace_dir=Path("/tmp/b"), enabled=True)
    with trace_context(r1):
        assert get_current_trace_recorder() is r1
        with trace_context(r2):
            assert get_current_trace_recorder() is r2
        assert get_current_trace_recorder() is r1
    assert get_current_trace_recorder() is None


def test_full_lifecycle(recorder: TraceRecorder, tmp_path: Path):
    """完整生命周期测试。"""
    recorder.start_trace(endpoint="/chat", user_goal="隧道火灾烟气", session_id="s1")
    recorder.update_trace(intent="rag", model_version={"llm": "qwen3:8b"})

    with recorder.step("intent_router", current_goal="识别意图") as step:
        step.tool_result_summary = {"intent": "rag"}

    with recorder.step("dense_retrieve", current_goal="dense 检索") as step:
        step.tool_result_summary = {"count": 10}

    with recorder.step("answer_generate", current_goal="生成回答") as step:
        step.tool_result_summary = {"answer_chars": 500}

    recorder.finalize(
        final_status="success",
        message_id="msg_001",
        final_eval={"answer_chars": 500},
    )
    recorder.save()

    assert recorder.trace.final_status == "success"
    assert len(recorder.trace.steps) == 3
    assert all(s.status == "success" for s in recorder.trace.steps)

    # verify file
    trace_file = list(tmp_path.glob("**/*.json"))[0]
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert data["intent"] == "rag"
    assert len(data["steps"]) == 3
