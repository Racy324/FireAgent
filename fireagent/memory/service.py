"""会话历史服务层。"""

from __future__ import annotations

import re
from typing import Any

from fireagent.memory.context_manager import ContextManager, ShortTermContextResult
from fireagent.memory.schema import MessageRecord, SessionRecord, SessionStateRecord
from fireagent.memory.store import SQLiteMemoryStore
from fireagent.utils.config import FireAgentConfig, get_config


class MemoryService:
    """会话历史业务服务。"""

    def __init__(self, config: FireAgentConfig | None = None) -> None:
        self.config = config or get_config()
        self.store = SQLiteMemoryStore(self.config.memory.database_path)
        self.store.init_db()
        self._context_manager: ContextManager | None = None

    @property
    def context_manager(self) -> ContextManager:
        """懒加载短期上下文管理器。"""
        if self._context_manager is None:
            self._context_manager = ContextManager(
                store=self.store,
                config=self.config.memory.short_term,
            )
        return self._context_manager

    def get_or_create_session(self, session_id: str | None, query: str = "") -> SessionRecord:
        """读取或创建会话。"""
        title = _title_from_query(query)
        return self.store.get_or_create_session(session_id=session_id, title=title)

    def recent_context(self, session_id: str) -> str:
        """构建最近 N 轮上下文（兼容方法）。"""
        return self.store.build_recent_context(
            session_id=session_id,
            limit=self.config.memory.recent_message_limit,
        )

    def build_short_term_context(self, session_id: str, query: str = "") -> ShortTermContextResult:
        """构建结构化短期上下文。"""
        if not self.config.memory.short_term.enabled:
            # 回退到简单最近消息
            text = self.recent_context(session_id)
            return ShortTermContextResult(text=text, total_chars=len(text))
        return self.context_manager.build(session_id, query=query)

    def record_user_message(self, session_id: str, query: str) -> MessageRecord:
        """保存用户消息。"""
        return self.store.save_message(session_id=session_id, role="user", content=query)

    def record_assistant_message(
        self,
        session_id: str,
        answer: str,
        intent: str = "",
        citations: list[str] | None = None,
        debug: dict | None = None,
        metadata: dict | None = None,
    ) -> MessageRecord:
        """保存助手消息。"""
        return self.store.save_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            intent=intent,
            citations=citations or [],
            debug=debug or {},
            metadata=metadata or {},
        )

    def update_short_term_state_after_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> SessionStateRecord:
        """在一轮对话后更新短期状态。

        使用保守规则：只记录确定性较高的信息。
        """
        state = self.store.get_or_create_session_state(session_id)
        updates: dict[str, Any] = {}

        # 如果 current_goal 为空，用第一条问题生成短标题
        if not state.current_goal:
            updates["current_goal"] = _title_from_query(user_query, max_chars=60)

        # 如果用户问题包含动作词，更新 current_step
        if _has_action_words(user_query):
            updates["current_step"] = _extract_step_hint(user_query)

        # 如果用户明确表达约束
        constraint = _extract_constraint(user_query)
        if constraint:
            constraints = list(state.constraints)
            if constraint not in constraints:
                constraints.append(constraint)
                updates["constraints"] = constraints

        # 如果有元数据中的中间结果
        if metadata:
            result_entry: dict[str, Any] = {}
            if metadata.get("short_term_truncated"):
                result_entry["status"] = "上下文已截断"
            if metadata.get("short_term_chars"):
                result_entry["detail"] = f"上下文长度 {metadata['short_term_chars']} 字符"
            if result_entry:
                result_entry["step"] = "本轮对话"
                self.store.append_intermediate_result(
                    session_id,
                    result_entry,
                    limit=self.config.memory.short_term.intermediate_result_limit,
                )

        if updates:
            state = self.store.update_session_state(session_id, **updates)

        # 尝试刷新滚动摘要
        self.context_manager.maybe_refresh_rolling_summary(session_id)

        return state


_ACTION_PATTERN = re.compile(
    r"(实现|修改|检查|继续|下一步|按照文档|编写|创建|新增|删除|重构|优化|修复|部署)"
)
_CONSTRAINT_PATTERN = re.compile(
    r"(?:不要|不需要|以后再做|暂不|先不|不做)"
    r"(.{2,20})"
)


def _title_from_query(query: str, max_chars: int = 28) -> str:
    title = " ".join(query.strip().split())
    if not title:
        return "新会话"
    return title[:max_chars]


def _has_action_words(query: str) -> bool:
    return bool(_ACTION_PATTERN.search(query))


def _extract_step_hint(query: str, max_chars: int = 50) -> str:
    """从用户问题中提取步骤提示。"""
    normalized = " ".join(query.strip().split())
    return normalized[:max_chars]


def _extract_constraint(query: str) -> str:
    """如果用户明确表达约束，返回约束文本。"""
    match = _CONSTRAINT_PATTERN.search(query)
    if match:
        return match.group(0).strip()
    return ""

