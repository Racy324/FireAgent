"""长期记忆时间衰退和最终分数计算。"""

from __future__ import annotations

from datetime import datetime, timezone


def decay_factor(age_days: float, half_life_days: float) -> float:
    """计算时间衰退因子。

    半衰期时返回 0.5，越旧越小。
    """
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def importance_boost(importance: float) -> float:
    """重要性加成系数。"""
    return 0.75 + 0.5 * importance


def memory_final_score(
    semantic_score: float,
    importance: float,
    age_days: float,
    half_life_days: float,
) -> float:
    """计算长期记忆最终检索分数。

    final_score = semantic_score * decay_factor * importance_boost
    """
    decay = decay_factor(age_days, half_life_days)
    boost = importance_boost(importance)
    return semantic_score * decay * boost


def age_days_from_now(created_at: str) -> float:
    """计算从创建时间到现在的天数。"""
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - created
        return max(0.0, delta.total_seconds() / 86400)
    except (ValueError, TypeError):
        return 0.0
