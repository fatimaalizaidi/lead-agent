"""eval/live_run.py against a fake server (httpx.MockTransport): no network, no OpenAI."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from eval import live_run

LEADS = [
    {"text": "Budget approved for 40 seats, send the contract.", "label": "High"},
    {"text": "What does Growth include?", "label": "Medium"},
    {"text": "Cheap backlinks!", "label": "Low"},
    {"text": "Need it next week, 10 seats.", "label": "High"},
]
ANSWERS = {  # text -> (priority, source)
    LEADS[0]["text"]: ("High", "llm"),
    LEADS[1]["text"]: ("Low", "llm"),  # one wrong answer
    LEADS[2]["text"]: ("Low", "classifier"),
}


def _fake_server(calls: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/routing/stats":
            n = calls.count("/analyze-lead")
            return httpx.Response(200, json={"analyzed": n, "llm_calls_saved": min(n, 1)})
        assert request.headers["X-API-Key"] == "la_test"
        message = json.loads(request.content)["message"]
        if message == LEADS[3]["text"] and calls.count("/analyze-lead") == 4:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"error_code": "rate_limited"})
        priority, source = ANSWERS.get(message, ("High", "llm"))
        return httpx.Response(200, json={
            "priority": priority, "priority_source": source, "classifier_priority": "Low",
            "classifier_confidence": 0.8, "needs_human_review": False, "fallback_reason": None,
        })

    return handler


@pytest.fixture
def leads_file(tmp_path: Path) -> Path:
    path = tmp_path / "leads.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in LEADS))
    return path


def test_dry_run_sends_nothing(leads_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []
    client = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(_fake_server(calls)))
    assert live_run.main(["--file", str(leads_file)], client=client) == 0
    assert calls == [] and "dry run" in capsys.readouterr().out


def test_full_run_writes_report(leads_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []
    waits: list[float] = []
    client = httpx.Client(
        base_url="http://fake", headers={"X-API-Key": "la_test"}, transport=httpx.MockTransport(_fake_server(calls))
    )
    out = tmp_path / "out"
    code = live_run.main(["--file", str(leads_file), "--yes", "--out-dir", str(out)], client=client, sleep=waits.append)
    assert code == 0
    assert calls.count("/analyze-lead") == 5, "the 429 is retried once"
    assert 1.0 in waits, "waits for Retry-After"

    data = json.loads(next(out.glob("*.json")).read_text())
    s = data["summary"]
    assert (s["answered"], s["classifier_routed"], s["classifier_routed_wrong"], s["llm_calls"]) == (4, 1, 0, 3)
    assert s["system"]["accuracy"] == 0.75 and s["system"]["high_recall"] == 1.0
    assert data["routing_after"]["analyzed"] == 5

    md = next(out.glob("*.md")).read_text()
    assert "OpenAI calls saved: **1 of 4**" in md
    assert "| Medium | Low | llm | What does Growth include? |" in md
    assert len(next(out.glob("*.csv")).read_text().splitlines()) == 5


def test_needs_a_key(leads_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(live_run.KEY_ENV, "")
    monkeypatch.setattr(live_run, "ROOT", leads_file.parent)  # no .env there
    assert live_run.main(["--file", str(leads_file), "--yes"]) == 2


def test_limit_mixes_labels(leads_file: Path) -> None:
    assert [r["label"] for r in live_run.load_leads(leads_file, 3)] == ["High", "Medium", "Low"]
    assert len(live_run.load_leads(leads_file, None)) == 4
