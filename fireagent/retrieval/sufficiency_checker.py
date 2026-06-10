"""本地证据充分性判断模块。"""

from __future__ import annotations

from typing import Optional

from fireagent.retrieval.schema import RerankedRetrievalResult, SufficiencyResult
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.sparse_embedding import tokenize_text


TEMPORAL_KEYWORDS = (
    "最新",
    "政策",
    "标准",
    "规范",
    "法规",
    "事故",
    "2026",
    "今年",
    "最近",
    "现行",
    "当前",
)

STOP_TERMS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "在",
    "中",
    "对",
    "是",
    "有",
    "请",
    "问",
    "什么",
    "如何",
    "怎么",
}

DOMAIN_TERMS = (
    "一氧化碳",
    "烟气控制",
    "人员疏散",
    "应急疏散",
    "风险评估",
    "火灾",
    "烟气",
    "烟雾",
    "疏散",
    "逃生",
    "隧道",
    "森林",
    "排烟",
    "通风",
    "风险",
    "预警",
    "检测",
    "识别",
    "蔓延",
    "温度",
    "热流",
    "火焰",
    "燃烧",
    "安全",
    "消防",
    "应急",
    "标准",
    "规范",
    "法规",
)


class LocalEvidenceSufficiencyChecker:
    """规则化本地证据充分性判断器。

    MVP 阶段先使用分数、证据数量、查询词覆盖率和时效词触发规则；
    后续可在 LangGraph 中接入 LLM sufficiency checker。
    """

    def __init__(
        self,
        min_score: Optional[float] = None,
        min_evidence_count: Optional[int] = None,
        min_term_coverage: Optional[float] = None,
        config: FireAgentConfig | None = None,
    ) -> None:
        self.config = config or get_config()
        self.min_score = self.config.retrieval.sufficiency_min_score if min_score is None else min_score
        self.min_evidence_count = (
            self.config.retrieval.sufficiency_min_evidence
            if min_evidence_count is None
            else min_evidence_count
        )
        self.min_term_coverage = (
            self.config.retrieval.sufficiency_min_term_coverage
            if min_term_coverage is None
            else min_term_coverage
        )

    def check(self, query: str, candidates: list[RerankedRetrievalResult]) -> SufficiencyResult:
        """判断本地证据是否足以回答用户问题。"""
        top_score = max((candidate.final_score for candidate in candidates), default=0.0)
        evidence_count = len(candidates)
        query_terms = self._query_terms(query)
        evidence_text = "\n".join(candidate.text or str(candidate.payload.get("text", "")) for candidate in candidates)
        matched_terms = [term for term in query_terms if term in evidence_text]
        missing_terms = [term for term in query_terms if term not in matched_terms]
        coverage = len(matched_terms) / max(len(query_terms), 1)
        temporal_keywords = [keyword for keyword in TEMPORAL_KEYWORDS if keyword in query]

        if temporal_keywords:
            return SufficiencyResult(
                sufficient=False,
                needs_web=True,
                reason=f"问题包含时效词：{', '.join(temporal_keywords)}，需要联网补充最新资料。",
                top_score=top_score,
                evidence_count=evidence_count,
                matched_terms=matched_terms,
                missing_terms=missing_terms,
                metadata={"term_coverage": coverage, "temporal_keywords": temporal_keywords},
            )

        if evidence_count < self.min_evidence_count:
            return SufficiencyResult(
                sufficient=False,
                needs_web=True,
                reason=f"本地证据数量不足：{evidence_count} < {self.min_evidence_count}。",
                top_score=top_score,
                evidence_count=evidence_count,
                matched_terms=matched_terms,
                missing_terms=missing_terms,
                metadata={"term_coverage": coverage},
            )

        if top_score < self.min_score:
            return SufficiencyResult(
                sufficient=False,
                needs_web=True,
                reason=f"本地最高重排分数偏低：{top_score:.4f} < {self.min_score:.4f}。",
                top_score=top_score,
                evidence_count=evidence_count,
                matched_terms=matched_terms,
                missing_terms=missing_terms,
                metadata={"term_coverage": coverage},
            )

        if query_terms and coverage < self.min_term_coverage:
            return SufficiencyResult(
                sufficient=False,
                needs_web=True,
                reason=f"本地证据覆盖的关键查询词不足：{coverage:.2%} < {self.min_term_coverage:.2%}。",
                top_score=top_score,
                evidence_count=evidence_count,
                matched_terms=matched_terms,
                missing_terms=missing_terms,
                metadata={"term_coverage": coverage},
            )

        return SufficiencyResult(
            sufficient=True,
            needs_web=False,
            reason="本地证据数量、相关性分数和查询词覆盖率满足回答要求。",
            top_score=top_score,
            evidence_count=evidence_count,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            metadata={"term_coverage": coverage},
        )

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        """抽取用于覆盖率判断的查询词。"""
        terms: list[str] = []
        for term in DOMAIN_TERMS:
            if term in query and term not in terms:
                terms.append(term)
        for token in tokenize_text(query):
            if token.isascii() and len(token) > 1 and token not in STOP_TERMS and token not in terms:
                terms.append(token)
        if terms:
            return terms

        for token in tokenize_text(query):
            if token in STOP_TERMS:
                continue
            if token.isascii() and len(token) <= 1:
                continue
            if not token.isascii() and len(token) <= 1:
                continue
            if token not in terms:
                terms.append(token)
        return terms


def check_local_evidence_sufficiency(
    query: str,
    candidates: list[RerankedRetrievalResult],
) -> SufficiencyResult:
    """便捷函数：判断本地证据是否充分。"""
    return LocalEvidenceSufficiencyChecker().check(query=query, candidates=candidates)
