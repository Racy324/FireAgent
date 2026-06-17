"""FireAgent 评测数据结构。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationModel(BaseModel):
    """评测模块 Pydantic 基类。"""

    model_config = ConfigDict(extra="ignore")


class EvaluationCase(EvaluationModel):
    """单条 RAG 问答评测样本。"""

    case_id: str
    question: str
    reference_answer: str = ""
    reference_contexts: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    required_citation_substrings: list[str] = Field(default_factory=list)
    intent: str = "rag"
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationPrediction(EvaluationModel):
    """FireAgent 对评测样本生成的一次预测结果。"""

    case_id: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    intent: str = ""
    evidence_sufficient: bool = False
    errors: list[str] = Field(default_factory=list)
    latency_seconds: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Intent router 决策字段
    route_intent: str = ""
    route_sub_intent: str = ""
    route_confidence: Optional[float] = None
    route_source: str = ""
    route_reason: str = ""
    # Fallback 决策字段
    fallback_action: str = ""
    fallback_reason: str = ""
    # Used citation 字段
    used_citation_markers: list[str] = Field(default_factory=list)
    invalid_citation_markers: list[str] = Field(default_factory=list)
    candidate_citation_count: int = 0
    used_citation_count: int = 0


class ManualScore(EvaluationModel):
    """不依赖外部评测模型的规则化人工辅助评分。"""

    case_id: str
    answer_keyword_recall: float = 0.0
    context_keyword_recall: float = 0.0
    reference_overlap: float = 0.0
    citation_score: float = 0.0
    groundedness_proxy: float = 0.0
    completeness_proxy: float = 0.0
    safety_score: float = 1.0
    overall_score: float = 0.0
    missing_keywords: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Fallback 决策字段
    expected_action: str = ""
    actual_action: str = ""
    fallback_correct: Optional[bool] = None
    sufficiency_correct: Optional[bool] = None
    web_triggered: bool = False
    false_web: bool = False
    missed_web: bool = False


class EvaluationSummary(EvaluationModel):
    """一次评测运行的汇总结果。"""

    total_cases: int
    average_scores: dict[str, float] = Field(default_factory=dict)
    group_summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    passed: bool = True
    fail_under: Optional[float] = None
    run_dir: str = ""
    prediction_path: str = ""
    manual_report_path: str = ""
    ragas_report_path: str = ""
    errors: list[str] = Field(default_factory=list)
