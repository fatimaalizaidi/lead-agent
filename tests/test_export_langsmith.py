"""LangSmith → distillation dataset (finetune/export_langsmith.py). No network: runs are fakes
shaped like what ``langsmith.wrappers.wrap_openai`` records for a chat completion."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune import export_langsmith as ex
from finetune.generate_dataset import read_jsonl

T0 = datetime(2026, 11, 1, tzinfo=UTC)
TEST_LEAD = read_jsonl(ex.DATA_DIR / "test.jsonl")[0]["text"]


def user_content(message: str, email: str = "lead@acme.com") -> str:
    """Same format as main._analyze_with_openai."""
    return f"email: {email}\nmessage: {message}\n\nCOMPANY KNOWLEDGE (retrieved; use when relevant):\n[pricing] ..."


def llm_run(message: str, priority: str = "High", *, minutes: int = 0, history: int = 0, error=None, **kw):
    messages = [{"role": "system", "content": "You triage inbound leads."}]
    for i in range(history):
        messages += [{"role": "user", "content": f"earlier {i}"}, {"role": "assistant", "content": "reply"}]
    messages.append({"role": "user", "content": kw.get("content", user_content(message))})
    answer = json.dumps({"priority": priority, "suggested_response": "ok", "rationale": "r"})
    outputs = kw.get("outputs", {"choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}]})
    return SimpleNamespace(
        id=f"run-{message[:8]}-{minutes}",
        inputs=kw.get("inputs", {"messages": messages, "model": "gpt-4o-mini"}),
        outputs=outputs,
        error=error,
        start_time=T0 + timedelta(minutes=minutes),
    )


def test_parses_an_analyze_lead_run_and_drops_the_sender() -> None:
    row, reason = ex.parse_run(llm_run("Need 40 seats by Friday", "High"))
    assert reason == "ok"
    assert (row["text"], row["label"], row["source"]) == ("Need 40 seats by Friday", "High", "langsmith")
    assert "lead@acme.com" not in json.dumps(row)


def test_masks_pii_in_the_message() -> None:
    row, _ = ex.parse_run(llm_run("Mail jo@acme.io or call +1 (415) 555-0100, see https://acme.io/x. 200 seats."))
    assert row["text"] == "Mail [EMAIL] or call [PHONE], see [URL] 200 seats."


@pytest.mark.parametrize(
    ("run", "reason"),
    [
        (llm_run("x", error="Timeout"), "error"),
        (llm_run("x", inputs={}), "no_inputs"),  # LANGSMITH_HIDE_INPUTS
        (llm_run("x", content="You are the sales agent. Lead says hi"), "not_analyze_lead"),  # /multi-agent
        (llm_run("x", history=1), "followup_turn"),
        (llm_run("x", outputs={"choices": [{"message": {"content": "not json"}}]}), "unparseable_answer"),
        (llm_run("x", priority="Urgent"), "unparseable_answer"),
    ],
)
def test_skips_unusable_runs(run, reason: str) -> None:
    assert ex.parse_run(run) == (None, reason)


def test_followups_can_be_included() -> None:
    row, reason = ex.parse_run(llm_run("Still keen, 30 seats", history=1), include_followups=True)
    assert reason == "ok" and row["label"] == "High"


def test_other_output_shapes() -> None:
    answer = json.dumps({"priority": "Low"})
    for outputs in ({"message": {"content": answer}}, {"output": answer}, {"content": answer}):
        row, reason = ex.parse_run(llm_run("buy backlinks", outputs=outputs))
        assert (reason, row["label"]) == ("ok", "Low")


def test_collect_dedupes_and_never_leaks_the_test_set() -> None:
    runs = [llm_run("Need 40 seats", minutes=1), llm_run("need 40 seats", minutes=2), llm_run(TEST_LEAD, minutes=3)]
    result = ex.collect(runs, exclude={TEST_LEAD.lower()})
    assert [r["text"] for r in result.rows] == ["Need 40 seats"]
    assert result.skipped == {"duplicate_or_test": 2}
    assert result.fetched == 3


def test_time_split_puts_newest_in_test() -> None:
    rows = [ex.parse_run(llm_run(f"lead {i}", minutes=i))[0] for i in (5, 1, 4, 2, 3)]
    train, test = ex.split_by_time(rows, 0.4)
    assert [r["text"] for r in train] == ["lead 1", "lead 2", "lead 3"]
    assert [r["text"] for r in test] == ["lead 4", "lead 5"]


def test_main_writes_private_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = [llm_run(f"We need {n} seats this week", "High", minutes=n) for n in range(10)]
    runs += [llm_run("Spam backlinks", "Low", minutes=20), llm_run("x", inputs={})]
    assert ex.main(["--out-dir", str(tmp_path)], runs=runs) == 0

    production = read_jsonl(tmp_path / "production.jsonl")
    test = read_jsonl(tmp_path / "production_test.jsonl")
    mix = read_jsonl(tmp_path / "train_mix.jsonl")
    assert len(production) == 11 and len(test) == 2
    assert test[-1]["text"] == "Spam backlinks", "newest run lands in the test split"
    synthetic = read_jsonl(ex.DATA_DIR / "train.jsonl")
    assert len(mix) == len(synthetic) + 9
    assert not {r["text"] for r in test} & {r["text"] for r in mix}
    out = capsys.readouterr().out
    assert "skipped no_inputs: 1" in out and "Fewer than 100" in out


def test_main_explains_hidden_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ex.main(["--out-dir", str(tmp_path)], runs=[llm_run("x", inputs={})]) == 1
    assert "LANGSMITH_HIDE_INPUTS" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_main_needs_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import get_settings

    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    get_settings.cache_clear()
    assert ex.main([]) == 2


def test_private_dir_is_gitignored() -> None:
    gitignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
    assert "finetune/data/private/" in gitignore
