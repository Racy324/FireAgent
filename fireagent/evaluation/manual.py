"""不依赖外部模型的 RAG 人工辅助评测指标。"""

from __future__ import annotations

import re
from statistics import mean

from fireagent.evaluation.schema import EvaluationCase, EvaluationPrediction, ManualScore


EMERGENCY_TERMS = ("119", "撤离", "安全", "报警", "疏散", "不要", "立即")


class ManualRAGEvaluator:
    """基于关键词、引用、上下文覆盖和安全提醒的轻量评测器。"""

    def score_case(self, case: EvaluationCase, prediction: EvaluationPrediction) -> ManualScore:
        """计算单条样本的规则化评分。"""
        answer = prediction.answer or ""
        context_text = "\n".join(prediction.contexts)
        citation_text = "\n".join(prediction.citations)

        answer_keyword_recall, missing_keywords = self._keyword_recall(case.expected_keywords, answer)
        context_keyword_recall, _ = self._keyword_recall(case.expected_keywords, context_text)
        reference_overlap = self._reference_overlap(case.reference_answer, answer)
        citation_score, missing_citations = self._citation_score(
            required=case.required_citation_substrings,
            citations=citation_text,
            contexts=context_text,
        )
        groundedness_proxy = self._groundedness_proxy(answer=answer, contexts=context_text)
        completeness_proxy = self._completeness_proxy(
            answer_keyword_recall=answer_keyword_recall,
            reference_overlap=reference_overlap,
            has_reference=bool(case.reference_answer.strip()),
        )
        safety_score = self._safety_score(case=case, answer=answer)

        notes: list[str] = []
        if missing_keywords:
            notes.append(f"答案缺少关键词：{'、'.join(missing_keywords)}")
        if missing_citations:
            notes.append(f"缺少预期引用线索：{'、'.join(missing_citations)}")
        if not prediction.citations:
            notes.append("预测结果没有引用。")
        if prediction.errors:
            notes.append(f"运行错误：{'；'.join(prediction.errors[:3])}")

        overall_score = self._overall_score(
            answer_keyword_recall=answer_keyword_recall,
            context_keyword_recall=context_keyword_recall,
            citation_score=citation_score,
            groundedness_proxy=groundedness_proxy,
            completeness_proxy=completeness_proxy,
            safety_score=safety_score,
        )

        return ManualScore(
            case_id=case.case_id,
            answer_keyword_recall=answer_keyword_recall,
            context_keyword_recall=context_keyword_recall,
            reference_overlap=reference_overlap,
            citation_score=citation_score,
            groundedness_proxy=groundedness_proxy,
            completeness_proxy=completeness_proxy,
            safety_score=safety_score,
            overall_score=overall_score,
            missing_keywords=missing_keywords,
            missing_citations=missing_citations,
            notes=notes,
        )

    def score_all(
        self,
        cases: list[EvaluationCase],
        predictions: list[EvaluationPrediction],
    ) -> list[ManualScore]:
        """对整批样本评分。"""
        prediction_by_id = {prediction.case_id: prediction for prediction in predictions}
        scores: list[ManualScore] = []
        for case in cases:
            prediction = prediction_by_id.get(case.case_id)
            if prediction is None:
                prediction = EvaluationPrediction(
                    case_id=case.case_id,
                    question=case.question,
                    answer="",
                    errors=["缺少预测结果。"],
                )
            scores.append(self.score_case(case, prediction))
        return scores

    @staticmethod
    def average_scores(scores: list[ManualScore]) -> dict[str, float]:
        """汇总各项指标均值。"""
        if not scores:
            return {}
        fields = [
            "answer_keyword_recall",
            "context_keyword_recall",
            "reference_overlap",
            "citation_score",
            "groundedness_proxy",
            "completeness_proxy",
            "safety_score",
            "overall_score",
        ]
        return {
            field: round(mean(float(getattr(score, field)) for score in scores), 4)
            for field in fields
        }

    @staticmethod
    def _keyword_recall(keywords: list[str], text: str) -> tuple[float, list[str]]:
        """统计预期关键词在文本中的召回率。"""
        expected = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not expected:
            return 1.0, []
        missing = [keyword for keyword in expected if keyword not in text]
        recall = (len(expected) - len(missing)) / len(expected)
        return round(recall, 4), missing

    @staticmethod
    def _reference_overlap(reference_answer: str, answer: str) -> float:
        """用分词重叠近似衡量答案与参考答案的一致性。"""
        reference_terms = _terms(reference_answer)
        if not reference_terms:
            return 1.0
        answer_terms = _terms(answer)
        if not answer_terms:
            return 0.0
        overlap = reference_terms & answer_terms
        return round(len(overlap) / len(reference_terms), 4)

    @staticmethod
    def _citation_score(
        required: list[str],
        citations: str,
        contexts: str,
    ) -> tuple[float, list[str]]:
        """检查引用是否存在，并匹配预期来源线索。"""
        haystack = f"{citations}\n{contexts}"
        required_items = [item.strip() for item in required if item.strip()]
        if required_items:
            missing = [item for item in required_items if item not in haystack]
            return round((len(required_items) - len(missing)) / len(required_items), 4), missing
        return (1.0 if citations.strip() else 0.0), []

    @staticmethod
    def _groundedness_proxy(answer: str, contexts: str) -> float:
        """用答案句子和上下文的词项重叠近似衡量 groundedness。"""
        sentences = [sentence.strip() for sentence in re.split(r"[。！？!?]\s*", answer) if sentence.strip()]
        if not sentences:
            return 0.0
        context_terms = _terms(contexts)
        if not context_terms:
            return 1.0 if "证据不足" in answer else 0.0

        supported = 0
        for sentence in sentences:
            sentence_terms = _terms(sentence)
            if not sentence_terms:
                continue
            overlap_ratio = len(sentence_terms & context_terms) / max(len(sentence_terms), 1)
            if overlap_ratio >= 0.25 or any(term in contexts for term in sentence_terms if len(term) >= 3):
                supported += 1
        return round(supported / len(sentences), 4)

    @staticmethod
    def _completeness_proxy(
        answer_keyword_recall: float,
        reference_overlap: float,
        has_reference: bool,
    ) -> float:
        """融合关键词召回和参考答案覆盖，近似衡量完整性。"""
        if has_reference:
            return round(0.6 * answer_keyword_recall + 0.4 * reference_overlap, 4)
        return answer_keyword_recall

    @staticmethod
    def _safety_score(case: EvaluationCase, answer: str) -> float:
        """应急类样本必须包含安全提醒。"""
        is_emergency = case.intent == "emergency" or "emergency" in case.tags or "应急" in case.tags
        if not is_emergency:
            return 1.0
        return 1.0 if any(term in answer for term in EMERGENCY_TERMS) else 0.0

    @staticmethod
    def _overall_score(
        answer_keyword_recall: float,
        context_keyword_recall: float,
        citation_score: float,
        groundedness_proxy: float,
        completeness_proxy: float,
        safety_score: float,
    ) -> float:
        """计算默认综合分。"""
        score = (
            0.20 * answer_keyword_recall
            + 0.15 * context_keyword_recall
            + 0.15 * citation_score
            + 0.25 * groundedness_proxy
            + 0.20 * completeness_proxy
            + 0.05 * safety_score
        )
        return round(score, 4)


def _terms(text: str) -> set[str]:
    """抽取中英文混合文本的粗粒度词项。"""
    terms: set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text):
        token = match.group(0).lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.add(token)
            for index in range(0, max(len(token) - 1, 0)):
                terms.add(token[index : index + 2])
        else:
            terms.add(token)
    return terms
