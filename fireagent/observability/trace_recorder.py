"""FireAgent Trace 记录器：创建 Trace、追加 Step、落盘保存。"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from fireagent import __version__
from fireagent.observability.trace_schema import AgentTrace, TraceStep, _now_iso, generate_trace_id


logger = logging.getLogger(__name__)

_CST = timezone(timedelta(hours=8))

_current_trace_recorder: ContextVar[Optional["TraceRecorder"]] = ContextVar(
    "_current_trace_recorder", default=None
)


def get_current_trace_recorder() -> Optional["TraceRecorder"]:
    """获取当前上下文中的 TraceRecorder 实例，无则返回 None。"""
    return _current_trace_recorder.get()


@contextmanager
def trace_context(recorder: "TraceRecorder") -> Iterator[None]:
    """设置当前上下文的 TraceRecorder，退出时恢复为 None。

    StreamingResponse 会在不同的 context 中运行 generator，导致
    token.reset() 抛出 ValueError。此处捕获该异常，回退到直接 set(None)。
    """
    token: Token = _current_trace_recorder.set(recorder)
    try:
        yield
    finally:
        try:
            _current_trace_recorder.reset(token)
        except ValueError:
            # StreamingResponse 跨 context 边界，token 无效
            _current_trace_recorder.set(None)


class TraceRecorder:
    """Trace 记录器，负责创建、更新和保存 Trace。"""

    def __init__(
        self,
        trace_dir: Path,
        enabled: bool = True,
        save_trace: bool = True,
        save_index: bool = True,
        max_summary_chars: int = 500,
    ) -> None:
        self._trace_dir = trace_dir
        self._enabled = enabled
        self._save_trace = save_trace
        self._save_index = save_index
        self._max_summary_chars = max_summary_chars
        self._trace: Optional[AgentTrace] = None

    @property
    def trace(self) -> Optional[AgentTrace]:
        return self._trace

    def start_trace(
        self,
        *,
        endpoint: str,
        user_goal: str,
        session_id: str = "",
        user_id: str = "local",
        request_id: str | None = None,
    ) -> AgentTrace:
        """创建新 Trace 并设置基本信息。"""
        trace_id = generate_trace_id()
        now = _now_iso()
        self._trace = AgentTrace(
            trace_id=trace_id,
            request_id=request_id or trace_id,
            user_id=user_id,
            session_id=session_id,
            endpoint=endpoint,
            user_goal=user_goal,
            agent_version=__version__,
            start_time=now,
        )
        return self._trace

    @contextmanager
    def step(
        self,
        step_type: str,
        current_goal: str = "",
        tool_name: str = "",
        context_refs: list[str] | None = None,
        tool_args_summary: dict[str, Any] | None = None,
        risk_level: str = "low",
    ) -> Iterator[TraceStep]:
        """创建并追加一个 Step，退出时自动计算耗时和状态。"""
        if self._trace is None:
            raise RuntimeError("TraceRecorder.start_trace() must be called before step()")

        step_obj = TraceStep(
            step_id=len(self._trace.steps) + 1,
            step_type=step_type,
            current_goal=current_goal,
            status="running",
            context_refs=context_refs or [],
            tool_name=tool_name,
            tool_args_summary=tool_args_summary or {},
            risk_level=risk_level,  # type: ignore[arg-type]
            start_time=_now_iso(),
        )
        self._trace.steps.append(step_obj)

        step_start = time.monotonic()
        exc_info: tuple = (None, None, None)
        try:
            yield step_obj
        except Exception:
            exc_info = sys.exc_info()
            raise
        finally:
            elapsed_ms = int((time.monotonic() - step_start) * 1000)
            step_obj.latency_ms = elapsed_ms
            step_obj.end_time = _now_iso()

            if exc_info[0] is not None:
                step_obj.status = "failed"
                step_obj.error = str(exc_info[1])[: self._max_summary_chars]
                if self._trace is not None:
                    self._trace.error_count += 1
            elif step_obj.status == "running":
                step_obj.status = "success"

    def update_trace(self, **fields: Any) -> None:
        """更新当前 Trace 的字段。"""
        if self._trace is None:
            return
        for key, value in fields.items():
            if hasattr(self._trace, key):
                setattr(self._trace, key, value)

    def finalize(
        self,
        *,
        final_status: str,
        message_id: str = "",
        final_eval: dict[str, Any] | None = None,
        error: str = "",
    ) -> AgentTrace:
        """结束 Trace，计算总耗时。"""
        if self._trace is None:
            raise RuntimeError("TraceRecorder.start_trace() must be called before finalize()")

        self._trace.end_time = _now_iso()
        self._trace.final_status = final_status  # type: ignore[assignment]
        if message_id:
            self._trace.message_id = message_id
        if final_eval is not None:
            self._trace.final_eval = final_eval
        if error:
            self._trace.error_count += 1

        # 计算总耗时
        try:
            start = datetime.fromisoformat(self._trace.start_time)
            end = datetime.fromisoformat(self._trace.end_time)
            self._trace.latency_ms = int((end - start).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass

        return self._trace

    def save(self) -> None:
        """将 Trace 保存到磁盘，受 save_trace / save_index 控制。"""
        if not self._enabled or self._trace is None:
            return

        try:
            today = datetime.now(_CST).strftime("%Y-%m-%d")
            day_dir = self._trace_dir / today
            day_dir.mkdir(parents=True, exist_ok=True)

            if self._save_trace:
                trace_file = day_dir / f"{self._trace.trace_id}.json"
                trace_file.write_text(
                    json.dumps(self._trace.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if self._save_index:
                index_entry = {
                    "trace_id": self._trace.trace_id,
                    "start_time": self._trace.start_time,
                    "end_time": self._trace.end_time,
                    "latency_ms": self._trace.latency_ms,
                    "final_status": self._trace.final_status,
                    "intent": self._trace.intent,
                    "sub_intent": self._trace.sub_intent,
                    "session_id": self._trace.session_id,
                    "error_count": self._trace.error_count,
                    "user_goal": self._trace.user_goal[:100],
                }
                index_file = day_dir / "index.jsonl"
                with index_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        except Exception:  # noqa: BLE001 - 落盘失败不影响业务
            logger.warning("Trace 落盘失败", exc_info=True)
