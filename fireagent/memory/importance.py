"""规则化长期记忆候选抽取和重要性评分。"""

from __future__ import annotations

import re

from fireagent.memory.long_term_schema import MemoryCandidate, MemoryType


# ── 高价值信号 ──

_PREFERENCE_PATTERNS = re.compile(
    r"(以后都|默认|我希望|我不想|以后.*也|固定采用|一直用|始终|偏好)",
)
_PROJECT_FACT_PATTERNS = re.compile(
    r"(这个项目|本项目|FireAgent|fireagent).{0,20}(使用|采用|不用|不做|基于|部署在)",
)
_REUSE_SIGNAL_PATTERNS = re.compile(
    r"(以后其他项目|复用|通用|模板|标准做法|最佳实践|检查清单)",
)
_NEGATIVE_CONSTRAINT_PATTERNS = re.compile(
    r"(不要|不需要|不做|禁止|避免|不允许|暂不|先不).{2,30}",
)

# ── 低价值/禁止信号 ──

_SENSITIVE_PATTERNS = re.compile(
    r"(api[_ ]?key|token|密码|password|secret|身份证|银行卡|手机号|住址|ssh-rsa|-----BEGIN)",
    re.IGNORECASE,
)
_TRANSIENT_PATTERNS = re.compile(
    r"(报错了|报错|error|failed|临时|这次|刚才|debug|stacktrace|traceback|exception)",
    re.IGNORECASE,
)


def score_candidate(text: str) -> tuple[float, float, str, MemoryType, list[str]]:
    """对一段文本进行重要性评分。

    Returns:
        (importance, confidence, reason, memory_type, tags)
    """
    normalized = text.strip()
    if not normalized or len(normalized) < 6:
        return 0.0, 0.0, "文本过短", "semantic", []

    score = 0.0
    confidence = 0.6
    reasons: list[str] = []
    tags: list[str] = []
    memory_type: MemoryType = "semantic"

    # 高价值信号
    if _PREFERENCE_PATTERNS.search(normalized):
        score += 0.45
        confidence = 0.8
        reasons.append("明确偏好表达")
        tags.append("preference")
        memory_type = "semantic"

    if _PROJECT_FACT_PATTERNS.search(normalized):
        score += 0.25
        confidence = max(confidence, 0.75)
        reasons.append("项目事实")
        tags.append("project")

    if _REUSE_SIGNAL_PATTERNS.search(normalized):
        score += 0.20
        confidence = max(confidence, 0.7)
        reasons.append("可复用流程")
        tags.append("workflow")
        memory_type = "procedural"

    if _NEGATIVE_CONSTRAINT_PATTERNS.search(normalized):
        score += 0.20
        confidence = max(confidence, 0.7)
        reasons.append("明确约束")
        tags.append("constraint")

    # 低价值/禁止信号
    if _SENSITIVE_PATTERNS.search(normalized):
        score -= 0.70
        reasons.append("疑似敏感信息")

    if _TRANSIENT_PATTERNS.search(normalized):
        score -= 0.40
        reasons.append("疑似临时信息")

    # 没有任何高价值信号
    if not reasons:
        score = 0.0
        confidence = 0.3
        reasons.append("无明确高价值信号")

    importance = max(0.0, min(1.0, score))
    reason = "；".join(reasons)
    return importance, confidence, reason, memory_type, tags


def extract_candidates(
    user_query: str,
    assistant_answer: str = "",
    max_candidates: int = 2,
) -> list[MemoryCandidate]:
    """从一轮对话中抽取记忆候选。

    MVP-3 主要从 user_query 抽取，避免从助手回答误记编造内容。
    内容会规范化为结构化格式（如"用户偏好：..."）。
    """
    candidates: list[MemoryCandidate] = []

    # 分句处理用户问题
    sentences = _split_sentences(user_query)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 6:
            continue
        importance, confidence, reason, memory_type, tags = score_candidate(sentence)
        if importance > 0:
            normalized = _normalize_content(sentence, memory_type, tags)
            candidates.append(MemoryCandidate(
                content=normalized,
                memory_type=memory_type,
                tags=tags,
                importance=importance,
                confidence=confidence,
                reason=reason,
            ))

    # 按重要性排序，取 top N
    candidates.sort(key=lambda c: c.importance, reverse=True)
    return candidates[:max_candidates]


def _normalize_content(sentence: str, memory_type: str, tags: list[str]) -> str:
    """将原始句子规范化为结构化记忆格式。"""
    # 清理句尾标点
    cleaned = sentence.rstrip("。！？!?. ").strip()
    if not cleaned:
        cleaned = sentence

    # 根据 tags 添加前缀
    if "preference" in tags:
        return f"用户偏好：{cleaned}"
    if "project" in tags:
        return f"项目事实：{cleaned}"
    if "workflow" in tags:
        return f"工作流程：{cleaned}"
    if "constraint" in tags:
        return f"约束：{cleaned}"

    # 兜底：根据 memory_type
    type_prefix = {
        "semantic": "记忆",
        "episodic": "事件",
        "procedural": "流程",
    }
    prefix = type_prefix.get(memory_type, "记忆")
    return f"{prefix}：{cleaned}"


def _split_sentences(text: str) -> list[str]:
    """按中文标点和英文句号分句。"""
    parts = re.split(r"[。！？!?\n；;]", text)
    return [p.strip() for p in parts if p.strip()]
