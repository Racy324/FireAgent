"""FireAgent Trace 与 Step 数据结构定义。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


_CST = timezone(timedelta(hours=8))


def generate_trace_id() -> str:
    """生成 trace_id，格式：trace_YYYYMMDD_HHMMSS_8位hex。"""
    now = datetime.now(_CST)
    return f"trace_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    """返回当前时间的 ISO 8601 格式（带 +08:00 时区）。"""
    return datetime.now(_CST).isoformat()


class TraceStep(BaseModel):
    """单个链路步骤记录。"""

    step_id: int = 0
    step_type: str = ""
    current_goal: str = ""
    status: Literal["running", "success", "failed", "skipped"] = "running"
    context_refs: list[str] = Field(default_factory=list)
    model_input_summary: str = ""
    model_output_summary: str = ""
    tool_name: str = ""
    tool_args_summary: dict[str, Any] = Field(default_factory=dict)
    tool_result_summary: dict[str, Any] = Field(default_factory=dict)
    state_delta_summary: dict[str, Any] = Field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    latency_ms: int = 0
    tokens: Optional[int] = None
    risk_level: Literal["low", "medium", "high"] = "low"
    eval_result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class AgentTrace(BaseModel):
    """单次请求的完整 Trace。"""

    trace_id: str = ""
    request_id: str = ""
    user_id: str = "local"
    session_id: str = ""
    message_id: str = ""
    endpoint: str = ""
    user_goal: str = ""
    normalized_goal: str = ""
    agent_version: str = ""
    model_version: dict[str, Any] = Field(default_factory=dict)
    policy_version: dict[str, Any] = Field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    latency_ms: int = 0
    final_status: Literal["running", "success", "degraded", "failed"] = "running"
    intent: str = ""
    sub_intent: str = ""
    evidence_sufficient: bool = False
    used_citation_count: int = 0
    error_count: int = 0
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None
    final_eval: dict[str, Any] = Field(default_factory=dict)
    steps: list[TraceStep] = Field(default_factory=list)
