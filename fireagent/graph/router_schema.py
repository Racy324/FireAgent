"""结构化意图路由结果。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Intent = Literal["chat", "reject", "emergency", "paper", "rag"]
RouteSource = Literal["hard_rule", "llm", "fallback_rule"]


class IntentRouteResult(BaseModel):
    """FireAgent 意图路由器的结构化输出。"""

    intent: Intent
    sub_intent: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    need_rag: bool = False
    need_memory: bool = False
    need_safety_notice: bool = False
    reason: str = ""
    source: RouteSource = "llm"
