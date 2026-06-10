"""RAGAS 可选评测适配器。

RAGAS 版本演进较快，因此这里把导入和数据转换集中封装。默认评测流程
不依赖 RAGAS；只有用户显式选择 ``--mode ragas`` 或 ``--mode both`` 时才会
尝试导入并执行。
"""

from __future__ import annotations

from typing import Any

from fireagent.evaluation.schema import EvaluationCase, EvaluationPrediction


class RagasEvaluationError(RuntimeError):
    """RAGAS 评测不可用或执行失败时抛出的异常。"""


DEFAULT_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def evaluate_with_ragas(
    cases: list[EvaluationCase],
    predictions: list[EvaluationPrediction],
    metric_names: list[str] | None = None,
) -> dict[str, Any]:
    """使用 RAGAS 对预测结果进行评测。"""
    symbols = _load_ragas_symbols()
    evaluate = symbols["evaluate"]
    EvaluationDataset = symbols["EvaluationDataset"]
    SingleTurnSample = symbols["SingleTurnSample"]
    metrics = _build_metrics(metric_names or list(DEFAULT_RAGAS_METRICS), symbols)

    prediction_by_id = {prediction.case_id: prediction for prediction in predictions}
    samples = []
    for case in cases:
        prediction = prediction_by_id.get(case.case_id)
        if prediction is None:
            continue
        samples.append(
            SingleTurnSample(
                user_input=case.question,
                response=prediction.answer,
                retrieved_contexts=prediction.contexts,
                reference=case.reference_answer or None,
                reference_contexts=case.reference_contexts or None,
            )
        )

    if not samples:
        raise RagasEvaluationError("没有可用于 RAGAS 的预测样本。")

    dataset = EvaluationDataset(samples=samples)
    try:
        result = evaluate(dataset=dataset, metrics=metrics)
    except TypeError:
        # 兼容部分旧版本：evaluate(dataset, metrics=...)
        result = evaluate(dataset, metrics=metrics)
    except Exception as exc:  # noqa: BLE001 - RAGAS/LLM 配置错误需要原样包装。
        raise RagasEvaluationError(f"RAGAS 执行失败：{exc}") from exc
    return _result_to_dict(result)


def _load_ragas_symbols() -> dict[str, Any]:
    """延迟导入 RAGAS，避免默认流程强依赖。"""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except Exception as exc:  # noqa: BLE001
        raise RagasEvaluationError(
            "未安装或无法导入 RAGAS。请先运行：pip install -e \".[eval]\""
        ) from exc

    try:
        from ragas import metrics as ragas_metrics
    except Exception as exc:  # noqa: BLE001
        raise RagasEvaluationError("无法导入 RAGAS metrics 模块。") from exc

    return {
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "evaluate": evaluate,
        "metrics": ragas_metrics,
    }


def _build_metrics(metric_names: list[str], symbols: dict[str, Any]) -> list[Any]:
    """根据名称创建 RAGAS metric 对象。"""
    ragas_metrics = symbols["metrics"]
    metric_factories = {
        "faithfulness": ("Faithfulness", "faithfulness"),
        "answer_relevancy": ("ResponseRelevancy", "answer_relevancy"),
        "response_relevancy": ("ResponseRelevancy", "answer_relevancy"),
        "context_precision": ("LLMContextPrecisionWithReference", "context_precision"),
        "context_recall": ("LLMContextRecall", "context_recall"),
        "answer_correctness": ("AnswerCorrectness", "answer_correctness"),
    }

    metrics: list[Any] = []
    for metric_name in metric_names:
        normalized = metric_name.strip().lower()
        if not normalized:
            continue
        class_name, fallback_attr = metric_factories.get(normalized, ("", normalized))
        metric = _metric_from_module(ragas_metrics, class_name=class_name, attr_name=fallback_attr)
        if metric is None:
            raise RagasEvaluationError(f"当前 RAGAS 版本不支持指标：{metric_name}")
        metrics.append(metric)
    if not metrics:
        raise RagasEvaluationError("RAGAS 指标列表为空。")
    return metrics


def _metric_from_module(module: Any, class_name: str, attr_name: str) -> Any | None:
    """兼容类式 metric 和模块级单例 metric。"""
    if class_name and hasattr(module, class_name):
        metric_class = getattr(module, class_name)
        return metric_class()
    if hasattr(module, attr_name):
        metric = getattr(module, attr_name)
        return metric() if callable(metric) and isinstance(metric, type) else metric
    return None


def _result_to_dict(result: Any) -> dict[str, Any]:
    """把 RAGAS 返回对象转成普通 dict。"""
    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        records = dataframe.to_dict(orient="records")
        averages = {
            column: float(dataframe[column].mean())
            for column in dataframe.columns
            if dataframe[column].dtype.kind in {"f", "i"}
        }
        return {"records": records, "average_scores": averages}
    if hasattr(result, "dict"):
        return result.dict()
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}
