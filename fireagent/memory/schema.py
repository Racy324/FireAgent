"""会话历史和记忆系统数据结构。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryModel(BaseModel):
    """Memory 模块 Pydantic 基类。"""

    model_config = ConfigDict(extra="ignore")


class SessionRecord(MemoryModel):
    """会话记录。"""

    session_id: str
    title: str = ""
    user_id: str = "local"
    created_at: str
    updated_at: str
    deleted_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageRecord(MemoryModel):
    """消息记录。"""

    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    token_count: int = 0
    intent: str = ""
    citations: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
