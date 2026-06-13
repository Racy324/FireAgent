"""会话历史服务层。"""

from __future__ import annotations

from fireagent.memory.schema import MessageRecord, SessionRecord
from fireagent.memory.store import SQLiteMemoryStore
from fireagent.utils.config import FireAgentConfig, get_config


class MemoryService:
    """会话历史业务服务。"""

    def __init__(self, config: FireAgentConfig | None = None) -> None:
        self.config = config or get_config()
        self.store = SQLiteMemoryStore(self.config.memory.database_path)
        self.store.init_db()

    def get_or_create_session(self, session_id: str | None, query: str = "") -> SessionRecord:
        """读取或创建会话。"""
        title = _title_from_query(query)
        return self.store.get_or_create_session(session_id=session_id, title=title)

    def recent_context(self, session_id: str) -> str:
        """构建最近 N 轮上下文。"""
        return self.store.build_recent_context(
            session_id=session_id,
            limit=self.config.memory.recent_message_limit,
        )

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


def _title_from_query(query: str, max_chars: int = 28) -> str:
    title = " ".join(query.strip().split())
    if not title:
        return "新会话"
    return title[:max_chars]

