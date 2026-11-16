"""Build a distillation dataset from production traces: gpt-4o-mini's own priorities in LangSmith.

Every ``/analyze-lead`` call (synchronous or background job) leaves an LLM run in LangSmith: the
prompt (``email: …\\nmessage: …\\n\\nCOMPANY KNOWLEDGE…``) and the model's JSON answer, which
includes ``priority``. This script pulls those runs and turns each one into
``{"text": <lead message>, "label": <teacher priority>}``. The production LLM becomes the teacher
and real traffic becomes the training data: distillation from production logs.

- **Privacy:** lead messages are real customer text. Output goes to ``finetune/data/private/``
  (gitignored, never committed). Email addresses, phone numbers and URLs in the message are
  masked, and the sender's email is dropped entirely.
- **Clean labels:** only successful runs whose answer parses to High/Medium/Low. Follow-up turns
  (runs with earlier conversation turns) are skipped, because their priority depends on context
  the student never sees. Duplicates and any lead in the hand-written test set are dropped.
- **Honest evaluation:** the newest ``--test-share`` of runs (by time) become
  ``production_test.jsonl``, never trained on. That's the student's accuracy against the teacher
  on real, future-looking traffic. The rest are mixed with the synthetic set into ``train_mix.jsonl``.

Needs ``LANGSMITH_API_KEY`` (read-only use) and traces with inputs: not ``LANGSMITH_HIDE_INPUTS``.

    python -m finetune.export_langsmith --days 30
    .venv-train/bin/python -m finetune.train_lora \\
        --train-file finetune/data/private/train_mix.jsonl --out models/priority-prod
    python -m finetune.evaluate --model models/priority-prod --model finetune/model \\
        --test-file finetune/data/private/production_test.jsonl --no-gate
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from finetune.generate_dataset import DATA_DIR, LABELS, read_jsonl, write_jsonl

PRIVATE_DIR = DATA_DIR / "private"
# Must match the user message built in main._analyze_with_openai.
USER_CONTENT_RE = re.compile(r"\Aemail:[^\n]*\nmessage: (?P<message>.*?)\n\nCOMPANY KNOWLEDGE", re.DOTALL)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ()./-]{7,}\d(?!\w)")


def mask_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _URL_RE.sub("[URL]", text)
    return _PHONE_RE.sub("[PHONE]", text)


def _content(message: Any) -> str | None:
    """Text of a chat message in OpenAI ({role, content}) or LangChain ({type, data}) shape."""
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if content is None and isinstance(message.get("data"), dict):
        content = message["data"].get("content")
    if isinstance(content, list):  # content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else None


def _role(message: dict[str, Any]) -> str:
    role = message.get("role") or message.get("type") or ""
    return {"human": "user", "ai": "assistant"}.get(role, role)


def _answer_text(outputs: Any) -> str | None:
    """The model's reply from a wrapped-OpenAI run's outputs (several SDK shapes)."""
    if not isinstance(outputs, dict):
        return None
    choices = outputs.get("choices")
    if isinstance(choices, list) and choices:
        return _content(choices[0].get("message"))
    if isinstance(outputs.get("message"), dict):
        return _content(outputs["message"])
    for key in ("content", "output"):
        if isinstance(outputs.get(key), str):
            return outputs[key]
    return None


def parse_run(run: Any, *, include_followups: bool = False) -> tuple[dict[str, str] | None, str]:
    """(row, "ok") for a usable /analyze-lead LLM run, else (None, skip reason)."""
    if getattr(run, "error", None):
        return None, "error"
    inputs = getattr(run, "inputs", None) or {}
    messages = inputs.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "no_inputs"  # LANGSMITH_HIDE_INPUTS, or not a chat call
    user_messages = [m for m in messages if isinstance(m, dict) and _role(m) == "user"]
    if not user_messages:
        return None, "not_analyze_lead"
    match = USER_CONTENT_RE.match(_content(user_messages[-1]) or "")
    if not match:
        return None, "not_analyze_lead"  # /multi-agent, /agent-action or other LLM calls
    if len(user_messages) > 1 and not include_followups:
        return None, "followup_turn"
    try:
        answer = json.loads(_answer_text(getattr(run, "outputs", None)) or "")
        label = answer["priority"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, "unparseable_answer"
    if label not in LABELS:
        return None, "unparseable_answer"
    text = mask_pii(match.group("message").strip())
    if not text:
        return None, "empty_message"
    started = getattr(run, "start_time", None)
    return {
        "text": text,
        "label": label,
        "source": "langsmith",
        "run_id": str(getattr(run, "id", "")),
        "start_time": started.isoformat() if isinstance(started, datetime) else str(started or ""),
    }, "ok"


@dataclass
class ExportResult:
    rows: list[dict[str, str]] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    fetched: int = 0


def collect(runs: Iterable[Any], exclude: set[str], *, include_followups: bool = False) -> ExportResult:
    result = ExportResult()
    seen = set(exclude)
    for run in runs:
        result.fetched += 1
        row, reason = parse_run(run, include_followups=include_followups)
        if row is None:
            result.skipped[reason] += 1
            continue
        key = row["text"].lower()
        if key in seen:
            result.skipped["duplicate_or_test"] += 1
            continue
        seen.add(key)
        result.rows.append(row)
    return result


def split_by_time(rows: list[dict[str, str]], test_share: float) -> tuple[list[dict], list[dict]]:
    """Oldest runs → train, newest → test, so the test set looks like future traffic."""
    ordered = sorted(rows, key=lambda r: r["start_time"])
    n_test = round(len(ordered) * test_share)
    cut = len(ordered) - n_test
    return ordered[:cut], ordered[cut:]


def fetch_runs(project: str, days: int, limit: int | None) -> Iterable[Any]:
    from langsmith import Client

    return Client().list_runs(
        project_name=project,
        run_type="llm",
        error=False,
        start_time=datetime.now(UTC) - timedelta(days=days),
        limit=limit,
    )


def main(argv: list[str] | None = None, runs: Iterable[Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default=os.environ.get("LANGSMITH_PROJECT", "lead-agent"))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="max LLM runs to read")
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--include-followups", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=PRIVATE_DIR)
    args = parser.parse_args(argv)

    if runs is None:
        from config import get_settings

        settings = get_settings()
        if not settings.langsmith_api_key.strip():
            print("LANGSMITH_API_KEY is not set (.env or environment).")
            return 2
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.strip()
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        runs = fetch_runs(args.project, args.days, args.limit)

    test_texts = {r["text"].lower() for r in read_jsonl(DATA_DIR / "test.jsonl")}
    result = collect(runs, test_texts, include_followups=args.include_followups)
    print(f"read {result.fetched} LLM runs → {len(result.rows)} usable /analyze-lead leads")
    for reason, count in result.skipped.most_common():
        print(f"  skipped {reason}: {count}")
    if result.skipped.get("no_inputs") and not result.rows:
        print("Traces have no inputs: LANGSMITH_HIDE_INPUTS=true hides the lead text, so there's nothing to learn.")
    if not result.rows:
        return 1

    train_rows, test_rows = split_by_time(result.rows, args.test_share)
    print("labels:", dict(Counter(r["label"] for r in result.rows)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "production.jsonl", result.rows)
    write_jsonl(args.out_dir / "production_test.jsonl", test_rows)
    synthetic = read_jsonl(DATA_DIR / "train.jsonl")
    write_jsonl(args.out_dir / "train_mix.jsonl", synthetic + train_rows)
    print(
        f"wrote {args.out_dir}/: production.jsonl ({len(result.rows)}), production_test.jsonl ({len(test_rows)}, "
        f"newest), train_mix.jsonl ({len(synthetic)} synthetic + {len(train_rows)} production)"
    )
    if len(result.rows) < 100:
        print("Fewer than 100 production leads: keep mixing with the synthetic set and re-export as traffic grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
