"""Send a labelled set of test leads to the live API and write a stats report.

Every lead in ``finetune/data/test.jsonl`` (84 hand-written, fictional leads with a known
High / Medium / Low label, never used for training) is posted to ``POST /analyze-lead`` one at a
time, the way n8n would. The report then answers:

- **Is the system right?** Final priority vs the known label: accuracy, High recall, confusion.
- **Who decided?** How many leads the fine-tuned classifier answered alone (no OpenAI call), and
  whether any of those were wrong; GPT's accuracy on the rest.
- **How fast and how costly?** Latency p50 / p95 per path, human-review rate, errors, and an
  estimated OpenAI cost.

It calls the API directly, not the n8n webhook, so a test run doesn't post 25 Slack alerts or send
29 auto-reply emails.

**Costs a little:** each GPT-analysed lead is one gpt-4o-mini call (~$0.00015), so a full run is
about one US cent. Without ``--yes`` it only prints the plan (dry run).

The API key is read from the ``LEAD_AGENT_API_KEY`` environment variable (or a ``.env`` line with
that name). It is never printed. Requests are paced under the default ``RATE_LIMIT`` (20/minute
per key) and a 429 waits for ``Retry-After``.

    python -m eval.live_run                    # dry run: what would be sent, estimated cost
    python -m eval.live_run --yes              # all 84 leads, ~5 minutes
    python -m eval.live_run --yes --limit 10   # quick check
    python -m eval.live_run --yes --url http://127.0.0.1:8000 --pace 0

Reports go to ``eval/results/`` (gitignored): ``live-<time>.md``, ``.csv`` and ``.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from finetune.evaluate import metrics

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "https://lead-agent-fjxq.onrender.com"
DEFAULT_FILE = ROOT / "finetune" / "data" / "test.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
KEY_ENV = "LEAD_AGENT_API_KEY"
# Sender domain of every test / simulated lead; finetune/export_langsmith.py skips it.
SIMULATED_DOMAIN = "simulated.example"
COST_PER_LLM_CALL = 0.00015  # gpt-4o-mini, one /analyze-lead with RAG context (LangSmith average)
FIELDS = [
    "n", "gold", "priority", "priority_source", "classifier_priority", "classifier_confidence",
    "needs_human_review", "fallback_reason", "latency_ms", "status", "error_code", "text",
]


def load_leads(path: Path, limit: int | None) -> list[dict[str, str]]:
    """All leads, or ``limit`` of them taken round-robin across labels (the file is sorted by label)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not limit:
        return rows
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    mixed = [g[i] for i in range(max(map(len, groups.values()))) for g in groups.values() if i < len(g)]
    return mixed[:limit]


def api_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == KEY_ENV:
                key = value.strip().strip("\"'")
    return key


def wake(client: httpx.Client, tries: int = 6) -> bool:
    """Render's free plan sleeps: hit /health until it answers (up to ~3 minutes)."""
    for attempt in range(tries):
        try:
            if client.get("/health", timeout=60).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        print(f"  waiting for the server to wake up ({attempt + 1}/{tries})…")
        time.sleep(20)
    return False


def routing_stats(client: httpx.Client) -> dict[str, Any] | None:
    try:
        resp = client.get("/routing/stats", timeout=30)
        return resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def send(client: httpx.Client, n: int, lead: dict[str, str], *, sleep=time.sleep) -> dict[str, Any]:
    """POST one lead; retries 429 (Retry-After) and 502/503 (cold start, upstream) up to 3 times."""
    payload = {"email": f"live-test-{n}@{SIMULATED_DOMAIN}", "message": lead["text"]}
    row: dict[str, Any] = {"n": n, "gold": lead["label"], "text": lead["text"]}
    for attempt in range(4):
        started = time.perf_counter()
        try:
            resp = client.post("/analyze-lead", json=payload, timeout=90)
        except httpx.HTTPError as exc:
            row.update(status=0, error_code=type(exc).__name__)
            sleep(10)
            continue
        row["latency_ms"] = round((time.perf_counter() - started) * 1000)
        row["status"] = resp.status_code
        if resp.status_code == 200:
            body = resp.json()
            for key in ("priority", "priority_source", "classifier_priority", "classifier_confidence",
                        "needs_human_review", "fallback_reason"):
                row[key] = body.get(key)
            row["error_code"] = None
            return row
        try:
            row["error_code"] = resp.json().get("error_code")
        except ValueError:
            row["error_code"] = "non_json"
        if resp.status_code == 429 and attempt < 3:
            sleep(float(resp.headers.get("Retry-After", 30)))
        elif resp.status_code in (502, 503) and attempt < 3:
            sleep(20)
        else:
            return row
    return row


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == 200]
    by_source = {s: [r for r in ok if r.get("priority_source") == s] for s in ("classifier", "llm")}
    out: dict[str, Any] = {
        "sent": len(rows),
        "answered": len(ok),
        "errors": dict(Counter(r.get("error_code") or f"http_{r.get('status')}" for r in rows if r not in ok)),
        "system": metrics([r["gold"] for r in ok], [r["priority"] for r in ok]) if ok else None,
        "classifier_routed": len(by_source["classifier"]),
        "classifier_routed_wrong": sum(r["gold"] != r["priority"] for r in by_source["classifier"]),
        "llm_calls": len(by_source["llm"]),
        "llm_accuracy": (
            round(sum(r["gold"] == r["priority"] for r in by_source["llm"]) / len(by_source["llm"]), 4)
            if by_source["llm"] else None
        ),
        "needs_human_review": sum(bool(r.get("needs_human_review")) for r in ok),
        "estimated_cost_usd": round(len(by_source["llm"]) * COST_PER_LLM_CALL, 4),
        "latency_ms": {
            s: {"p50": _pct([r["latency_ms"] for r in rs], 0.5), "p95": _pct([r["latency_ms"] for r in rs], 0.95)}
            for s, rs in by_source.items()
        },
    }
    with_clf = [r for r in ok if r.get("classifier_priority")]
    if with_clf:
        out["classifier_alone"] = metrics([r["gold"] for r in with_clf], [r["classifier_priority"] for r in with_clf])
    return out


def report_md(summary: dict[str, Any], rows: list[dict[str, Any]], url: str, before, after) -> str:
    s = summary
    lines = [f"# Live test run: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC", "", f"Target: `{url}/analyze-lead`, "
             f"{s['sent']} labelled test leads, {s['answered']} answered.", ""]
    if s["system"]:
        m = s["system"]
        lines += [
            "## Is the system right? (final priority vs the known label)", "",
            "| Accuracy | High recall | High precision | Macro F1 |", "|---|---|---|---|",
            f"| {m['accuracy']:.1%} | {m['high_recall']:.1%} | {m['high_precision']:.1%} | {m['macro_f1']:.3f} |", "",
            "Confusion (rows = known label, columns = answer):", "",
            "| | High | Medium | Low |", "|---|---|---|---|",
            *[f"| **{g}** | " + " | ".join(str(m["confusion"][g][p]) for p in ("High", "Medium", "Low")) + " |"
              for g in ("High", "Medium", "Low")],
            "",
        ]
    lat = s["latency_ms"]
    lines += [
        "## Who decided?", "",
        "| Path | Leads | Wrong | Latency p50 | Latency p95 |", "|---|---|---|---|---|",
        f"| Fine-tuned classifier alone (no OpenAI) | {s['classifier_routed']} | {s['classifier_routed_wrong']} "
        f"| {lat['classifier']['p50'] or '–'} ms | {lat['classifier']['p95'] or '–'} ms |",
        f"| GPT-4o-mini + RAG | {s['llm_calls']} | "
        f"{s['llm_calls'] - round((s['llm_accuracy'] or 0) * s['llm_calls'])} "
        f"| {lat['llm']['p50'] or '–'} ms | {lat['llm']['p95'] or '–'} ms |",
        "",
        f"- OpenAI calls saved: **{s['classifier_routed']} of {s['answered']}** "
        f"({s['classifier_routed'] / s['answered']:.0%})" if s["answered"] else "- No answers.",
        f"- Estimated OpenAI cost of this run: **${s['estimated_cost_usd']}**",
        f"- Flagged for human review: {s['needs_human_review']}",
        f"- Errors: {s['errors'] or 'none'}",
    ]
    if s.get("classifier_alone"):
        c = s["classifier_alone"]
        lines += ["", f"For comparison, the classifier's own label on every lead: accuracy {c['accuracy']:.1%}, "
                      f"High recall {c['high_recall']:.1%}."]
    if before is not None and after is not None:
        lines += ["", f"`/routing/stats` on the server: {before['llm_calls_saved']}/{before['analyzed']} before → "
                      f"{after['llm_calls_saved']}/{after['analyzed']} after."]
    wrong = [r for r in rows if r.get("status") == 200 and r["gold"] != r["priority"]]
    if wrong:
        lines += ["", "## Leads it got wrong", "", "| Known | Answer | Decided by | Lead |", "|---|---|---|---|"]
        lines += [f"| {r['gold']} | {r['priority']} | {r.get('priority_source')} | {r['text'][:90]} |" for r in wrong]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=os.environ.get("LEAD_AGENT_URL", DEFAULT_URL))
    p.add_argument("--file", type=Path, default=DEFAULT_FILE, help="JSONL with text + label")
    p.add_argument("--limit", type=int, default=None, help="send only the first N leads")
    p.add_argument("--pace", type=float, default=3.2, help="seconds between requests (20/minute limit)")
    p.add_argument("--yes", action="store_true", help="really send (costs ~$0.00015 per GPT-analysed lead)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None, client: httpx.Client | None = None, sleep=time.sleep) -> int:
    args = parse_args(argv)
    leads = load_leads(args.file, args.limit)
    url = args.url.rstrip("/")
    estimate = round(len(leads) * COST_PER_LLM_CALL, 4)
    print(f"{len(leads)} leads from {args.file.name} → {url}/analyze-lead "
          f"(~{len(leads) * args.pace / 60:.1f} min, at most ~${estimate} OpenAI)")
    if not args.yes:
        print("dry run: nothing sent. Add --yes to run it.")
        return 0
    key = api_key()
    if not key and client is None:
        print(f"Set {KEY_ENV} (environment or .env) to an API key from the server's API_KEYS (the la_… part).")
        return 2

    own = client is None
    client = client or httpx.Client(base_url=url, headers={"X-API-Key": key})
    try:
        if not wake(client):
            print("The server never answered /health.")
            return 1
        before = routing_stats(client)
        rows = []
        for n, lead in enumerate(leads, 1):
            row = send(client, n, lead, sleep=sleep)
            rows.append(row)
            mark = "✓" if row.get("priority") == lead["label"] else ("✗" if row.get("status") == 200 else "!")
            got = row.get("priority") or row.get("error_code")
            print(f"  {n:3}/{len(leads)} {mark} known {lead['label']:<6} got {got!s:<10}"
                  f" via {row.get('priority_source') or '-':<10} {row.get('latency_ms', '-')} ms")
            if n < len(leads):
                sleep(args.pace)
        after = routing_stats(client)
    finally:
        if own:
            client.close()

    summary = summarize(rows)
    md = report_md(summary, rows, url, before, after)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"live-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    stem.with_suffix(".md").write_text(md)
    stem.with_suffix(".json").write_text(json.dumps({"url": url, "summary": summary, "routing_before": before,
                                                     "routing_after": after}, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("\n" + md)
    print(f"Saved {stem}.md / .csv / .json")
    return 0 if summary["answered"] else 1


if __name__ == "__main__":
    sys.exit(main())
