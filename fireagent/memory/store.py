"""SQLite-backed session history store."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fireagent.memory.schema import MessageRecord, SessionRecord, SessionStateRecord
from fireagent.utils.config import PROJECT_ROOT


class SQLiteMemoryStore:
    """SQLite 会话历史存储。"""

    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.database_path = path

    def init_db(self) -> None:
        """初始化数据库表。"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    user_id TEXT NOT NULL DEFAULT 'local',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    intent TEXT NOT NULL DEFAULT '',
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    debug_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                    ON sessions(user_id, deleted_at, updated_at);
                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                    ON messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS session_states (
                    session_id TEXT PRIMARY KEY,
                    current_goal TEXT NOT NULL DEFAULT '',
                    current_step TEXT NOT NULL DEFAULT '',
                    rolling_summary TEXT NOT NULL DEFAULT '',
                    completed_items_json TEXT NOT NULL DEFAULT '[]',
                    pending_items_json TEXT NOT NULL DEFAULT '[]',
                    constraints_json TEXT NOT NULL DEFAULT '[]',
                    intermediate_results_json TEXT NOT NULL DEFAULT '[]',
                    pinned_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_session_states_updated_at
                    ON session_states(updated_at);
                """
            )

    def create_session(self, title: str = "", user_id: str = "local") -> SessionRecord:
        """创建会话。"""
        self.init_db()
        now = _now()
        session = SessionRecord(
            session_id=f"sess_{uuid.uuid4().hex}",
            title=title.strip() or "新会话",
            user_id=user_id,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                (session_id, title, user_id, created_at, updated_at, deleted_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    session.session_id,
                    session.title,
                    session.user_id,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.metadata, ensure_ascii=False),
                ),
            )
        return session

    def get_session(self, session_id: str) -> SessionRecord | None:
        """读取单个未删除会话。"""
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND deleted_at IS NULL",
                (session_id,),
            ).fetchone()
        return _session_from_row(row) if row else None

    def get_or_create_session(
        self,
        session_id: str | None = None,
        title: str = "",
        user_id: str = "local",
    ) -> SessionRecord:
        """读取会话；不存在时创建新会话。"""
        if session_id:
            existing = self.get_session(session_id)
            if existing is not None:
                return existing
        return self.create_session(title=title, user_id=user_id)

    def list_sessions(self, user_id: str = "local", limit: int = 50) -> list[SessionRecord]:
        """按更新时间倒序列出会话。"""
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def update_session_title(self, session_id: str, title: str) -> SessionRecord | None:
        """更新会话标题。"""
        self.init_db()
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ? AND deleted_at IS NULL",
                (title.strip() or "新会话", now, session_id),
            )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        """软删除会话，并清理关联的 session_states。"""
        self.init_db()
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET deleted_at = ?, updated_at = ? WHERE session_id = ?",
                (now, now, session_id),
            )
            conn.execute(
                "DELETE FROM session_states WHERE session_id = ?",
                (session_id,),
            )

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = "",
        citations: list[str] | None = None,
        debug: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageRecord:
        """保存消息并更新会话时间。"""
        self.init_db()
        now = _now()
        message = MessageRecord(
            message_id=f"msg_{uuid.uuid4().hex}",
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
            token_count=_rough_token_count(content),
            intent=intent,
            citations=citations or [],
            debug=debug or {},
            metadata=metadata or {},
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages
                (message_id, session_id, role, content, created_at, token_count,
                 intent, citations_json, debug_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.role,
                    message.content,
                    message.created_at,
                    message.token_count,
                    message.intent,
                    json.dumps(message.citations, ensure_ascii=False),
                    json.dumps(message.debug, ensure_ascii=False),
                    json.dumps(message.metadata, ensure_ascii=False),
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
        return message

    def list_messages(self, session_id: str, limit: int = 100) -> list[MessageRecord]:
        """按创建时间升序读取会话消息。"""
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def list_recent_messages(self, session_id: str, limit: int = 8) -> list[MessageRecord]:
        """读取最近 N 条消息，按时间升序返回。"""
        return self.list_messages(session_id=session_id, limit=limit)

    def get_messages_by_ids(self, message_ids: list[str]) -> list[MessageRecord]:
        """按 message_id 列表读取消息，保持传入顺序。"""
        if not message_ids:
            return []
        self.init_db()
        placeholders = ",".join("?" for _ in message_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM messages WHERE message_id IN ({placeholders})",
                message_ids,
            ).fetchall()
        id_to_msg: dict[str, MessageRecord] = {}
        for row in rows:
            msg = _message_from_row(row)
            id_to_msg[msg.message_id] = msg
        return [id_to_msg[mid] for mid in message_ids if mid in id_to_msg]

    def build_recent_context(self, session_id: str, limit: int = 8) -> str:
        """把最近消息格式化为 prompt 可用的短期上下文。"""
        messages = self.list_recent_messages(session_id=session_id, limit=limit)
        lines: list[str] = []
        for message in messages:
            label = "用户" if message.role == "user" else "助手" if message.role == "assistant" else message.role
            lines.append(f"{label}：{message.content}")
        return "\n".join(lines)

    # ── Session State ──

    def get_session_state(self, session_id: str) -> SessionStateRecord | None:
        """读取会话状态。"""
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_states WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _state_from_row(row) if row else None

    def get_or_create_session_state(self, session_id: str) -> SessionStateRecord:
        """读取会话状态；不存在时基于已有 session 创建。

        如果 session 不存在或已被软删除，抛出 ValueError 避免孤儿 state。
        """
        existing = self.get_session_state(session_id)
        if existing is not None:
            return existing
        # 校验 session 确实存在且未删除
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"session {session_id} 不存在或已删除，无法创建 state。")
        now = _now()
        state = SessionStateRecord(session_id=session_id, updated_at=now)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO session_states
                (session_id, current_goal, current_step, rolling_summary,
                 completed_items_json, pending_items_json, constraints_json,
                 intermediate_results_json, pinned_message_ids_json,
                 updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.session_id,
                    state.current_goal,
                    state.current_step,
                    state.rolling_summary,
                    json.dumps(state.completed_items, ensure_ascii=False),
                    json.dumps(state.pending_items, ensure_ascii=False),
                    json.dumps(state.constraints, ensure_ascii=False),
                    json.dumps(state.intermediate_results, ensure_ascii=False),
                    json.dumps(state.pinned_message_ids, ensure_ascii=False),
                    state.updated_at,
                    json.dumps(state.metadata, ensure_ascii=False),
                ),
            )
        return self.get_session_state(session_id) or state

    def upsert_session_state(self, state: SessionStateRecord) -> SessionStateRecord:
        """整体写入或更新会话状态。"""
        self.init_db()
        now = _now()
        state.updated_at = now
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_states
                (session_id, current_goal, current_step, rolling_summary,
                 completed_items_json, pending_items_json, constraints_json,
                 intermediate_results_json, pinned_message_ids_json,
                 updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    current_goal = excluded.current_goal,
                    current_step = excluded.current_step,
                    rolling_summary = excluded.rolling_summary,
                    completed_items_json = excluded.completed_items_json,
                    pending_items_json = excluded.pending_items_json,
                    constraints_json = excluded.constraints_json,
                    intermediate_results_json = excluded.intermediate_results_json,
                    pinned_message_ids_json = excluded.pinned_message_ids_json,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    state.session_id,
                    state.current_goal,
                    state.current_step,
                    state.rolling_summary,
                    json.dumps(state.completed_items, ensure_ascii=False),
                    json.dumps(state.pending_items, ensure_ascii=False),
                    json.dumps(state.constraints, ensure_ascii=False),
                    json.dumps(state.intermediate_results, ensure_ascii=False),
                    json.dumps(state.pinned_message_ids, ensure_ascii=False),
                    state.updated_at,
                    json.dumps(state.metadata, ensure_ascii=False),
                ),
            )
        return state

    def update_session_state(self, session_id: str, **fields: Any) -> SessionStateRecord:
        """按字段更新会话状态。"""
        self.init_db()
        state = self.get_or_create_session_state(session_id)
        json_fields = {
            "completed_items", "pending_items", "constraints",
            "intermediate_results", "pinned_message_ids", "metadata",
        }
        for key, value in fields.items():
            if key in json_fields:
                setattr(state, key, value)
            elif hasattr(state, key):
                setattr(state, key, value)
        return self.upsert_session_state(state)

    def append_intermediate_result(
        self,
        session_id: str,
        result: dict[str, Any],
        limit: int = 8,
    ) -> SessionStateRecord:
        """追加一条中间结果，超过上限时移除最旧的。"""
        state = self.get_or_create_session_state(session_id)
        results = list(state.intermediate_results)
        results.append(result)
        if len(results) > limit:
            results = results[-limit:]
        state.intermediate_results = results
        return self.upsert_session_state(state)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 2) if text else 0


def _json_loads(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw) if raw else default
    except json.JSONDecodeError:
        return default


def _session_from_row(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        title=row["title"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
        metadata=_json_loads(row["metadata_json"], {}),
    )


def _message_from_row(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        message_id=row["message_id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        token_count=row["token_count"],
        intent=row["intent"],
        citations=_json_loads(row["citations_json"], []),
        debug=_json_loads(row["debug_json"], {}),
        metadata=_json_loads(row["metadata_json"], {}),
    )


def _state_from_row(row: sqlite3.Row) -> SessionStateRecord:
    return SessionStateRecord(
        session_id=row["session_id"],
        current_goal=row["current_goal"],
        current_step=row["current_step"],
        rolling_summary=row["rolling_summary"],
        completed_items=_json_loads(row["completed_items_json"], []),
        pending_items=_json_loads(row["pending_items_json"], []),
        constraints=_json_loads(row["constraints_json"], []),
        intermediate_results=_json_loads(row["intermediate_results_json"], []),
        pinned_message_ids=_json_loads(row["pinned_message_ids_json"], []),
        updated_at=row["updated_at"],
        metadata=_json_loads(row["metadata_json"], {}),
    )

