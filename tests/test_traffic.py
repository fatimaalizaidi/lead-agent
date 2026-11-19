"""eval/traffic.py: varied lead mix, and a full run against a fake server (no network, no OpenAI)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx
import pytest

from eval import traffic


def test_runs_differ_and_mix_kinds() -> None:
    a = traffic.pick_leads(40, random.Random(1))
    b = traffic.pick_leads(40, random.Random(2))
    assert [x["text"] for x in a] != [x["text"] for x in b]
    kinds = {x["kind"] for x in traffic.pick_leads(200, random.Random(3))}
    assert kinds == {"generated", "handwritten", "roman_urdu", "injection"}
    assert all(x["label"] in ("High", "Medium", "Low", "blocked") for x in a)


def test_dry_run_and_missing_key_send_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    assert traffic.main(["--count", "2", "--seed", "1"]) == 0
    assert "dry run" in capsys.readouterr().out
    monkeypatch.setenv(traffic.live_run.KEY_ENV, "")
    monkeypatch.setattr(traffic.live_run, "ROOT", tmp_path)
    assert traffic.main(["--yes"]) == 0, "no secret yet: the scheduled run is a passing no-op"


def test_full_run_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    leads = [
        {"text": "Budget approved, 30 seats", "label": "High", "kind": "generated"},
        {"text": traffic.INJECTIONS[0], "label": "blocked", "kind": "injection"},
        {"text": "Cheap backlinks", "label": "Low", "kind": "handwritten"},
    ]
    monkeypatch.setattr(traffic, "pick_leads", lambda count, rng: leads)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/routing/stats":
            return httpx.Response(200, json={"analyzed": 10, "llm_calls_saved": 4, "saved_share": 0.4})
        message = json.loads(request.content)["message"]
        if message in traffic.INJECTIONS:
            return httpx.Response(400, json={"error_code": "prompt_injection_blocked"})
        low = "backlinks" in message
        return httpx.Response(200, json={
            "priority": "Low" if low else "High", "priority_source": "classifier" if low else "llm",
        })

    client = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    summary = tmp_path / "summary.md"
    assert traffic.main(["--yes", "--summary-file", str(summary)], client=client, sleep=lambda s: None) == 0
    md = summary.read_text()
    assert "Right: **2 of 2** leads" in md
    assert "Prompt injections blocked: 1 of 1" in md
    assert "Answered by the classifier without OpenAI: 1" in md
    assert "4 of 10 leads answered without OpenAI (40%)" in md
