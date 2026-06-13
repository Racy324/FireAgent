"""长期记忆对外服务层。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fireagent.memory.importance import extract_candidates
from fireagent.memory.long_term_schema import (
    LongTermMemoryRecord,
    LongTermMemorySearchResult,
)
from fireagent.memory.long_term_store import LongTermMemoryStore, new_memory_id
from fireagent.utils.config import FireAgentConfig, get_config

logger = logging.getLogger(__name__)


class LongTermMemoryService:
    """长期记忆业务服务。"""

    def __init__(
        self,
        config: Optional[FireAgentConfig] = None,
        store: Optional[LongTermMemoryStore] = None,
    ) -> None:
        self.config = config or get_config()
        self.store = store or LongTermMemoryStore(config=self.config)
        self._collection_initialized = False

    def _ensure_collection(self) -> None:
        """懒初始化 Qdrant collection，只执行一次。"""
        if self._collection_initialized:
            return
        try:
            self.store.create_collection()
            self._collection_initialized = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆 collection 创建失败: %s", exc)

    def retrieve_for_query(
        self,
        query: str,
        user_id: str = "local",
    ) -> list[LongTermMemorySearchResult]:
        """按 query 检索相关长期记忆。"""
        if not self.config.memory.long_term.enabled:
            return []
        self._ensure_collection()
        try:
            return self.store.search(
                query=query,
                user_id=user_id,
                top_k=self.config.memory.long_term.top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆检索失败: %s", exc)
            return []

    def format_for_prompt(
        self,
        results: list[LongTermMemorySearchResult],
    ) -> str:
        """将检索结果格式化为 prompt 注入文本。"""
        if not results:
            return ""
        lines: list[str] = []
        for r in results:
            mem = r.memory
            label = {"semantic": "semantic", "episodic": "episodic", "procedural": "procedural"}.get(
                mem.memory_type, mem.memory_type
            )
            lines.append(f"- [{label} | {r.final_score:.2f}] {mem.content}")
        text = "\n".join(lines)
        max_chars = self.config.memory.long_term.max_prompt_chars
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        return text

    def maybe_write_after_turn(
        self,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        user_query: str,
        assistant_answer: str,
        user_id: str = "local",
    ) -> list[LongTermMemoryRecord]:
        """在一轮对话后尝试写入长期记忆。

        只有重要性和置信度都超过阈值的候选才会被写入。
        """
        if not self.config.memory.long_term.enabled or not self.config.memory.long_term.write_enabled:
            return []

        self._ensure_collection()
        lt_cfg = self.config.memory.long_term
        candidates = extract_candidates(
            user_query,
            assistant_answer,
            max_candidates=lt_cfg.max_candidates_per_turn,
        )

        written: list[LongTermMemoryRecord] = []
        now = _now()

        for candidate in candidates:
            if candidate.importance < lt_cfg.importance_threshold:
                continue
            if candidate.confidence < lt_cfg.confidence_threshold:
                continue

            try:
                # 去重检查
                existing = self.store.find_duplicate(candidate.content, user_id=user_id)
                if existing is not None:
                    self.store.update_memory_fields(
                        existing.memory_id,
                        importance=max(existing.importance, candidate.importance),
                        confidence=max(existing.confidence, candidate.confidence),
                        tags=candidate.tags,
                    )
                    written.append(existing)
                    continue

                memory = LongTermMemoryRecord(
                    memory_id=new_memory_id(),
                    user_id=user_id,
                    session_id=session_id,
                    source_message_ids=[user_message_id, assistant_message_id],
                    content=candidate.content,
                    memory_type=candidate.memory_type,
                    tags=candidate.tags,
                    importance=candidate.importance,
                    confidence=candidate.confidence,
                    created_at=now,
                    updated_at=now,
                    half_life_days=self._half_life_for_type(candidate.memory_type),
                    source="chat_turn",
                    metadata={"reason": candidate.reason},
                )
                self.store.upsert_memory(memory)
                written.append(memory)
            except Exception as exc:  # noqa: BLE001
                logger.warning("长期记忆写入失败: %s", exc)

        return written

    def _half_life_for_type(self, memory_type: str) -> float:
        lt_cfg = self.config.memory.long_term
        mapping = {
            "semantic": lt_cfg.semantic_half_life_days,
            "episodic": lt_cfg.episodic_half_life_days,
            "procedural": lt_cfg.procedural_half_life_days,
        }
        return mapping.get(memory_type, lt_cfg.default_half_life_days)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
