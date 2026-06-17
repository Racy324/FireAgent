"""长期记忆数据结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MemoryType = Literal["semantic", "episodic", "procedural"]
MemoryStatus = Literal["active", "disabled"]


class LongTermMemoryRecord(BaseModel):
    """一条长期记忆。"""

    memory_id: str
    user_id: str = "local"
    session_id: str = ""
    source_message_ids: list[str] = Field(default_factory=list)
    content: str
    memory_type: MemoryType = "semantic"
    tags: list[str] = Field(default_factory=list)
    status: MemoryStatus = "active"
    importance: float = 0.0
    confidence: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str = ""
    access_count: int = 0
    half_life_days: float = 180.0
    source: str = "chat_turn"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryCandidate(BaseModel):
    """一条待判别的记忆候选。"""

    content: str
    memory_type: MemoryType
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTermMemorySearchResult(BaseModel):
    """长期记忆检索结果。"""

    memory: LongTermMemoryRecord
    semantic_score: float = 0.0
    decay_factor: float = 1.0
    final_score: float = 0.0
