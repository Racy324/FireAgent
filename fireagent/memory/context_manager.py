"""短期上下文管理器：从 session state + 最近消息构建结构化 prompt 上下文。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from fireagent.memory.store import SQLiteMemoryStore
from fireagent.utils.config import ShortTermMemoryConfig


class ShortTermContextResult(BaseModel):
    """短期上下文构建结果。"""

    text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    total_chars: int = 0
    truncated: bool = False
    recent_message_ids: list[str] = Field(default_factory=list)
    pinned_message_ids: list[str] = Field(default_factory=list)


class ContextManager:
    """从 session state 和最近消息构建结构化短期上下文。"""

    def __init__(self, store: SQLiteMemoryStore, config: ShortTermMemoryConfig) -> None:
        self.store = store
        self.config = config

    def build(self, session_id: str, query: str = "") -> ShortTermContextResult:
        """构建短期上下文文本。

        Args:
            session_id: 会话 ID。
            query: 当前用户问题（仅用于调试，不注入上下文，避免重复）。
        """
        state = self.store.get_session_state(session_id)
        recent_messages = self.store.list_recent_messages(
            session_id=session_id,
            limit=self.config.recent_message_limit,
        )

        # 读取 pinned messages
        pinned_ids = state.pinned_message_ids if state else []
        pinned_messages: list[Any] = []
        if pinned_ids:
            pinned_messages = self.store.get_messages_by_ids(
                pinned_ids[: self.config.pinned_message_limit]
            )

        sections: list[tuple[str, str]] = []

        if state:
            sections.extend([
                ("当前任务目标", state.current_goal),
                ("当前步骤", state.current_step),
                ("当前约束", _format_list(state.constraints)),
                ("待完成事项", _format_list(state.pending_items)),
                ("已完成事项", _format_list(state.completed_items)),
                ("滚动摘要", state.rolling_summary),
                ("中间结果", _format_intermediate_results(state.intermediate_results)),
            ])

        # 关键消息插在最近对话之前
        if pinned_messages:
            sections.append(("关键消息", _format_messages(pinned_messages)))

        sections.append(("最近对话", _format_messages(recent_messages)))

        text, truncated = _fit_sections(sections, self.config.max_chars)

        return ShortTermContextResult(
            text=text,
            sections={title: body for title, body in sections if body.strip()},
            total_chars=len(text),
            truncated=truncated,
            recent_message_ids=[m.message_id for m in recent_messages],
            pinned_message_ids=pinned_ids,
        )

    def maybe_refresh_rolling_summary(self, session_id: str) -> str:
        """当消息数超过阈值时，用 extractive 方法刷新滚动摘要。

        策略（不依赖 LLM）：
        1. 保留最早用户目标。
        2. 保留最近 N 轮之外的用户问题标题。
        3. 保留明确的"完成/决定/约束"句子。
        4. 截断到 rolling_summary_max_chars。

        Returns:
            当前 rolling_summary（可能已更新）。
        """
        all_messages = self.store.list_messages(session_id, limit=500)
        if len(all_messages) < self.config.summarize_after_messages:
            # 消息数不足，不刷新
            state = self.store.get_session_state(session_id)
            return state.rolling_summary if state else ""

        # 取最近 N 轮之外的旧消息做压缩
        recent_limit = self.config.recent_message_limit * 2  # user+assistant 算一轮
        old_messages = all_messages[: max(0, len(all_messages) - recent_limit)]
        if not old_messages:
            state = self.store.get_session_state(session_id)
            return state.rolling_summary if state else ""

        summary = _extractive_summary(old_messages, self.config.rolling_summary_max_chars)

        # 持久化
        self.store.update_session_state(session_id, rolling_summary=summary)
        return summary


# ── extractive summary helpers ──

_DECISION_PATTERN = re.compile(
    r"(?:完成|已决定|结论|确定|约束|要求|不要|必须|最终).{4,}",
    re.IGNORECASE,
)


def _extractive_summary(messages: list[Any], max_chars: int) -> str:
    """从旧消息中提取关键信息作为滚动摘要（不调用 LLM）。"""
    lines: list[str] = []

    # 1. 最早用户问题作为目标
    for msg in messages:
        if msg.role == "user":
            first_q = " ".join(msg.content.strip().split())
            if len(first_q) > 80:
                first_q = first_q[:80].rstrip() + "..."
            lines.append(f"最初目标：{first_q}")
            break

    # 2. 收集用户问题标题（跳过最早那条）
    user_questions: list[str] = []
    seen_first = False
    for msg in messages:
        if msg.role == "user":
            if not seen_first:
                seen_first = True
                continue
            q = " ".join(msg.content.strip().split())
            if len(q) > 60:
                q = q[:60].rstrip() + "..."
            if q:
                user_questions.append(q)

    if user_questions:
        lines.append("历史问题：" + "；".join(user_questions[-8:]))  # 最多保留 8 条

    # 3. 提取包含决策/约束关键词的句子
    decisions: list[str] = []
    for msg in messages:
        text = msg.content
        for match in _DECISION_PATTERN.finditer(text):
            sentence = match.group(0).strip()
            if len(sentence) > 100:
                sentence = sentence[:100].rstrip() + "..."
            if sentence:
                decisions.append(sentence)

    if decisions:
        # 去重并限制数量
        seen: set[str] = set()
        unique: list[str] = []
        for d in decisions:
            key = d[:20]
            if key not in seen:
                seen.add(key)
                unique.append(d)
        lines.append("关键决定：" + "；".join(unique[-5:]))

    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "..."
    return summary


def _format_list(items: list[str]) -> str:
    """格式化列表为 markdown 条目。"""
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items if item.strip())


def _format_intermediate_results(results: list[dict[str, Any]]) -> str:
    """格式化中间结果为简洁文本。"""
    if not results:
        return ""
    lines: list[str] = []
    for item in results:
        step = item.get("step", item.get("action", ""))
        status = item.get("status", "")
        summary = item.get("summary", item.get("detail", ""))
        parts = [str(p) for p in [step, status, summary] if p]
        if parts:
            lines.append(f"- {'：'.join(parts)}")
    return "\n".join(lines)


def _format_messages(messages: list[Any]) -> str:
    """格式化最近消息。"""
    if not messages:
        return ""
    lines: list[str] = []
    for msg in messages:
        label = "用户" if msg.role == "user" else "助手" if msg.role == "assistant" else msg.role
        # 截断过长的单条消息
        content = msg.content
        if len(content) > 500:
            content = content[:500].rstrip() + "..."
        lines.append(f"{label}：{content}")
    return "\n".join(lines)


def _fit_sections(sections: list[tuple[str, str]], max_chars: int) -> tuple[str, bool]:
    """按优先级拼接各区域，超出预算时截断。

    Returns:
        (text, truncated)
    """
    output: list[str] = []
    used = 0
    truncated = False

    for title, body in sections:
        if not body.strip():
            continue
        block = f"# {title}\n{body.strip()}\n"
        if used + len(block) <= max_chars:
            output.append(block)
            used += len(block)
            continue

        # 尝试部分填充剩余空间
        remaining = max_chars - used
        if remaining > 200:
            output.append(block[:remaining].rstrip() + "\n[已截断]\n")
        truncated = True
        break

    return "\n".join(output).strip(), truncated
