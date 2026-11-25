"""eval/judge.py with a fake judge model: calibration, pass/fail rule, report. No network."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval import judge
from tests.openai_fakes import completion_with_json

GOOD = {"groundedness": 5, "relevance": 5, "tone": 5, "next_step": 4, "unsupported_claims": [], "reason": "fine"}


def _fake_judge(bad_markers: tuple[str, ...] = ("$99", "any time", "99.999", "demo slots", "read it")):
    """Grades a reply bad when it contains a planted defect; records every prompt it saw."""
    seen: list[str] = []

    def create(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        seen.append(prompt)
        reply = prompt.split("REPLY: ", 1)[1]
        hit = next((m for m in bad_markers if m in reply), None)
        grade = dict(GOOD) if hit is None else {**GOOD, "groundedness": 2, "relevance": 3, "unsupported_claims": [hit]}
        return completion_with_json(grade)

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    return client, seen


def _run_file(tmp_path: Path) -> Path:
    rows = [
        {"n": 1, "status": 200, "text": "Price of Growth?", "priority": "Medium", "priority_source": "llm",
         "suggested_response": "Growth is $149 per month for up to 15 seats."},
        {"n": 2, "status": 200, "text": "Price?", "priority": "Medium", "priority_source": "llm",
         "suggested_response": "Growth is $99 per month."},
        {"n": 3, "status": 200, "text": "spam", "priority": "Low", "priority_source": "classifier",
         "suggested_response": "Thanks for reaching out!"},
        {"n": 4, "status": 502, "text": "x"},
    ]
    path = tmp_path / "live-20990101T000000Z.json"
    path.write_text(json.dumps({"rows": rows}))
    return path


def test_pass_rule_is_applied_in_code() -> None:
    assert judge.verdict(GOOD) == "pass"
    assert judge.verdict({**GOOD, "unsupported_claims": ["99% off"]}) == "fail"
    assert judge.verdict({**GOOD, "tone": 3}) == "fail"
    assert judge.verdict({**GOOD, "next_step": 2}) == "pass", "next_step is reported, not a gate"


def test_dry_run_sends_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    client, seen = _fake_judge()
    assert judge.main(["--out-dir", str(tmp_path), "--run", str(_run_file(tmp_path))], client=client) == 0
    out = capsys.readouterr().out
    assert seen == [] and "dry run" in out and "2 replies" in out


def test_full_run_calibrates_and_judges(tmp_path: Path) -> None:
    run = _run_file(tmp_path)
    client, seen = _fake_judge()
    assert judge.main(["--yes", "--out-dir", str(tmp_path)], client=client) == 0
    assert len(seen) == len(judge.CALIBRATION) + 2, "classifier replies and errors are skipped"
    assert all("COMPANY KNOWLEDGE:" in p and "Growth" in p for p in seen), "the knowledge base is the reference"

    data = json.loads(next(tmp_path.glob("judge-*.json")).read_text())
    assert data["source"] == str(run)
    cal = data["calibration"]
    assert (cal["caught"], cal["planted_bad"], cal["passed_good"]) == (5, 5, 4)
    assert data["summary"]["judged"] == 2 and data["summary"]["pass_rate"] == 0.5

    md = next(tmp_path.glob("judge-*.md")).read_text()
    assert "Caught **5 of 5** bad replies" in md and "trustworthy" in md
    assert "Growth is $99 per month." in md, "failed replies are listed with the claim"


def test_a_lenient_judge_is_flagged(tmp_path: Path) -> None:
    client, _ = _fake_judge(bad_markers=())  # passes everything
    assert judge.main(["--yes", "--calibrate-only", "--out-dir", str(tmp_path)], client=client) == 0
    md = next(tmp_path.glob("judge-*.md")).read_text()
    assert "Caught **0 of 5**" in md and "read its scores with care" in md


def test_needs_a_run_with_replies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "live-old.json").write_text(json.dumps({"rows": [{"status": 200, "priority": "High"}]}))
    assert judge.latest_run(tmp_path) is None
    judge.main(["--out-dir", str(tmp_path)])
    assert "run `python -m eval.live_run --yes` first" in capsys.readouterr().out
