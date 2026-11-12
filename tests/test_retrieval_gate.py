"""Offline tests for the free retrieval gate (no embedding model download, no OpenAI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import retrieval_gate as gate


def test_reference_facts_keeps_prices_percentages_and_counts() -> None:
    ref = "Starter is $49/month, Growth $149. Uptime is 99.9%, 1,000 leads, refunds within 30 days, 2 seats."
    assert gate.reference_facts(ref) == ["$49", "$149", "99.9%", "1,000", "30"]


def test_fact_in_text_needs_whole_numbers() -> None:
    assert gate.fact_in_text("30", "within 30 days")
    assert not gate.fact_in_text("30", "within 300 days")
    assert not gate.fact_in_text("30", "at 1.30 pm")
    assert gate.fact_in_text("$49", "Starter: $49/month")
    assert not gate.fact_in_text("$49", "Starter: $499/month")


def test_score_question_rank_and_missing_facts() -> None:
    row = {"id": "q", "question": "?", "reference": "Growth is $149, Starter $49.", "source": "pricing.md"}
    chunks = [
        {"title": "SLA", "content": "99.9% uptime"},
        {"title": "Pricing Plans", "content": "Growth plan: $149 per seat"},
    ]
    result = gate.score_question(row, chunks, "Pricing Plans")
    assert result["rank"] == 2
    assert result["retrieved"] == ["SLA", "Pricing Plans"]
    assert (result["facts"], result["facts_found"], result["missing_facts"]) == (2, 1, ["$49"])

    missed = gate.score_question(row, chunks[:1], "Pricing Plans")
    assert missed["rank"] is None


def test_summarize() -> None:
    results = [
        {"rank": 1, "facts": 2, "facts_found": 2},
        {"rank": 2, "facts": 1, "facts_found": 0},
        {"rank": None, "facts": 1, "facts_found": 1},
        {"rank": 1, "facts": 0, "facts_found": 0},
    ]
    assert gate.summarize(results) == {
        "top1_accuracy": 0.5,
        "hit_rate": 0.75,
        "mrr": 0.625,  # (1 + 0.5 + 0 + 1) / 4
        "fact_recall": 0.75,  # 3 of 4 facts
    }
    assert "fact_recall" not in gate.summarize([{"rank": 1, "facts": 0, "facts_found": 0}])
    assert gate.summarize([]) == {}


def test_thresholds_file_is_valid(tmp_path: Path) -> None:
    thresholds = gate.load_thresholds()
    assert set(thresholds) == set(gate.METRICS)
    assert all(0 < v <= 1 for v in thresholds.values())

    bad = tmp_path / "t.json"
    bad.write_text('{"hit_rte": 0.9}')
    with pytest.raises(ValueError, match="hit_rte"):
        gate.load_thresholds(bad)


def test_gate_failures_include_missing_metrics() -> None:
    thresholds = {"hit_rate": 0.96, "mrr": 0.9, "fact_recall": 0.97}
    assert gate.gate_failures({"hit_rate": 1.0, "mrr": 0.85}, thresholds) == ["mrr", "fact_recall"]
    assert gate.gate_failures({"hit_rate": 1.0, "mrr": 0.9, "fact_recall": 1.0}, thresholds) == []


def test_report_md_marks_gated_mode_and_misses() -> None:
    results = {
        "hybrid": [
            {"id": "pricing-starter", "rank": 1, "missing_facts": ["$49"]},
            {"id": "sla-uptime", "rank": 1, "missing_facts": []},
        ],
        "vector": [{"id": "sla-uptime", "rank": None, "missing_facts": []}],
    }
    summaries = {"hybrid": {"hit_rate": 1.0, "fact_recall": 0.5}, "vector": {"hit_rate": 0.5}}
    report = gate.report_md(summaries, {"fact_recall": 0.97}, results, gated_mode="hybrid")
    assert "❌ failed" in report
    assert "0.500 ❌ (floor 0.97)" in report
    assert "| pricing-starter | 1 | $49 |" in report
    assert "| sla-uptime | not retrieved | - |" in report

    scores_only = gate.report_md(summaries, {"fact_recall": 0.97}, results, gated_mode=None)
    assert scores_only.startswith("## Retrieval scores")
    assert "floor" not in scores_only


def test_parse_args_defaults_to_production_mode() -> None:
    args = gate.parse_args([])
    assert (args.mode, args.compare, args.no_gate) == ("hybrid", False, False)
