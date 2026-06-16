"""seed60 到 router eval 转换脚本测试。"""

from __future__ import annotations

import json

import pytest

from scripts.convert_rag_eval_to_router_eval import convert_rows, main


def test_convert_rows_preserves_seed60_metadata() -> None:
    """转换结果应保留 case、source intent、question_type 和期望行为。"""
    rows = [
        {
            "case_id": "case-web",
            "question": "最新消防政策是什么？",
            "intent": "web_fallback",
            "tags": ["policy"],
            "expected_behavior": {"expected_action": "use_web"},
            "metadata": {"question_type": "web_fallback"},
        }
    ]

    converted = convert_rows(rows)

    assert converted == [
        {
            "query": "最新消防政策是什么？",
            "expected_intent": "rag",
            "source_intent": "web_fallback",
            "source_case_id": "case-web",
            "source": "seed60_converted",
            "question_type": "web_fallback",
            "tags": ["policy"],
            "expected_behavior": {"expected_action": "use_web"},
            "notes": "converted from fireagent_benchmark_v1_seed60",
        }
    ]


def test_convert_rows_rejects_missing_question() -> None:
    """缺少 question 时应显式报错，避免生成空 query。"""
    with pytest.raises(ValueError, match="missing question"):
        convert_rows([{"case_id": "bad", "intent": "rag"}])


def test_convert_cli_writes_jsonl_and_summary(tmp_path, capsys) -> None:
    """CLI 应写出 JSONL，并在 --json 时打印汇总。"""
    input_path = tmp_path / "seed60.jsonl"
    output_path = tmp_path / "router.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "case_id": "case-emergency",
                "question": "楼道起火怎么办？",
                "intent": "emergency",
                "metadata": {"question_type": "emergency"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    main(["--input", str(input_path), "--output", str(output_path), "--json"])

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["expected_intent"] == "emergency"
    summary = json.loads(capsys.readouterr().out)
    assert summary == {"input_rows": 1, "output_rows": 1, "output": str(output_path)}
