"""LLM-as-a-judge: grade the replies /analyze-lead wrote, against the knowledge base.

``eval/live_run.py`` checks the *priority* against known labels. It can't check the *reply*:
there is no single right wording. This script asks a judge model to grade each GPT-written
``suggested_response`` on a fixed rubric, using the full knowledge base as the source of truth:

| Criterion | 1–5 | What it checks |
|---|---|---|
| groundedness | | every specific fact (price, number, policy, timing, feature) is in the knowledge base |
| relevance | | answers what the lead actually asked |
| tone | | professional, friendly, concise |
| next_step | | moves the lead forward in a way that fits its priority |

The judge also lists every **unsupported claim** it finds. A reply **passes** when groundedness,
relevance and tone are all >= 4 and there are no unsupported claims. The pass/fail rule is
applied in code, not left to the judge.

**Calibration first.** An LLM judge can be lenient or miss things, so every run starts with
planted replies whose verdict is known (an invented price, a wrong refund promise, an off-topic
or rude answer, and good replies). The report says how many it got right. If the judge misses
planted defects, don't trust its scores.

Known limits: a judge from the same model family can favour that family's writing
(``--judge-model gpt-4o`` is a stronger, pricier second opinion), and scores are a guide for
finding bad replies, not a precise measurement.

Input: the newest ``eval/results/live-*.json`` with replies in it (``python -m eval.live_run --yes``
saves them). Leads answered by the classifier's fixed reply are skipped.

    python -m eval.judge                      # dry run: what would be judged, estimated cost
    python -m eval.judge --yes                # calibration + judge every GPT reply (~$0.02)
    python -m eval.judge --yes --calibrate-only
    python -m eval.judge --yes --run eval/results/live-....json --judge-model gpt-4o

Uses OPENAI_API_KEY from .env. Reports: ``eval/results/judge-<time>.md`` and ``.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
RESULTS_DIR = ROOT / "eval" / "results"
CRITERIA = ("groundedness", "relevance", "tone", "next_step")
COST_PER_CALL = {"gpt-4o-mini": 0.0003, "gpt-4o": 0.005}  # ~1.5k prompt tokens (KB + lead + reply)

JUDGE_PROMPT = """You are a strict evaluator of replies a sales assistant wrote to inbound leads.
COMPANY KNOWLEDGE below is the only source of truth about the company.

Score the REPLY from 1 (bad) to 5 (excellent) on:
- groundedness: every specific fact in the reply (price, number, discount, policy, timing, feature,
  SLA) appears in COMPANY KNOWLEDGE. Generic courtesy, offers to talk, and questions back to the
  lead are fine. Any invented or contradicted fact means 2 or lower.
- relevance: the reply addresses what the lead actually asked or needs.
- tone: professional, friendly and concise; no pressure, no rudeness.
- next_step: the reply moves the lead forward in a way that fits its PRIORITY
  (High: concrete step such as contract, call or onboarding; Medium: helpful info and an
  invitation; Low: polite and brief).

Low-priority leads (spam, gibberish, off-topic questions, insults) need no information:
a polite, brief reply that declines, redirects or asks what they need scores 4 or 5 on
groundedness, relevance and next_step, as long as it makes no claim about the company.
Saying the knowledge doesn't cover something and offering to confirm is grounded, not a defect.
Check arithmetic against the plan limits before calling a budget or seat claim wrong.

List every unsupported claim verbatim in unsupported_claims (empty list if none).
Give a one-sentence reason. Return JSON only."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "reply_grade",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [*CRITERIA, "unsupported_claims", "reason"],
            "properties": {
                **{c: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for c in CRITERIA},
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
        },
    },
}

# Planted replies with a known verdict: the judge must fail the bad ones and pass the good ones.
CALIBRATION = [
    {"text": "What does the Growth plan cost?", "priority": "Medium", "expect": "fail", "defect": "invented price",
     "reply": "Growth is $99 per month and includes unlimited seats. Want me to set it up?"},
    {"text": "Can I get a refund if I cancel my monthly plan after two months?", "priority": "Medium",
     "expect": "fail", "defect": "wrong refund promise",
     "reply": "Yes, we refund any unused days on monthly plans at any time, no questions asked."},
    {"text": "What uptime do you guarantee?", "priority": "Medium", "expect": "fail", "defect": "invented SLA",
     "reply": "We guarantee 99.999% uptime with a full refund for any outage."},
    {"text": "Do you offer a discount for paying annually?", "priority": "Medium", "expect": "fail",
     "defect": "off-topic", "reply": "Great question! We have demo slots on Thursday afternoon, want one?"},
    {"text": "How long does Enterprise onboarding take?", "priority": "Medium", "expect": "fail",
     "defect": "rude", "reply": "It's on our website. Please read it before emailing sales."},
    {"text": "What does the Growth plan cost, and is there an annual discount?", "priority": "Medium",
     "expect": "pass", "defect": None,
     "reply": "Growth is $149 per month for up to 15 seats, and annual billing saves 15%. "
              "Happy to walk you through it on a call."},
    {"text": "Budget approved for 40 seats, please send the contract today.", "priority": "High",
     "expect": "pass", "defect": None,
     "reply": "Thanks, great to hear! I'll get the contract over today and can set up a kickoff call "
              "for your 40 seats this week."},
    {"text": "Cheap SEO backlinks, reply for our price list!", "priority": "Low", "expect": "pass", "defect": None,
     "reply": "Thanks for reaching out, but we're not looking for SEO services right now."},
    {"text": "Can I get a trial account to explore?", "priority": "Medium", "expect": "pass", "defect": None,
     "reply": "Good question. I'll confirm trial options with the team and get back to you today; "
              "would a short call this week help in the meantime?"},
]


def knowledge_text() -> str:
    return "\n\n".join(p.read_text().strip() for p in sorted(KNOWLEDGE_DIR.glob("*.md")))


def verdict(grade: dict[str, Any]) -> str:
    ok = all(grade[c] >= 4 for c in ("groundedness", "relevance", "tone")) and not grade["unsupported_claims"]
    return "pass" if ok else "fail"


def judge_one(client: Any, model: str, kb: str, lead: str, priority: str, reply: str) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format=JUDGE_SCHEMA,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",
             "content": f"COMPANY KNOWLEDGE:\n{kb}\n\nLEAD: {lead}\nPRIORITY: {priority}\nREPLY: {reply}"},
        ],
    )
    grade = json.loads(resp.choices[0].message.content)
    grade["verdict"] = verdict(grade)
    return grade


def latest_run(results_dir: Path) -> Path | None:
    for path in sorted(results_dir.glob("live-*.json"), reverse=True):
        rows = json.loads(path.read_text()).get("rows") or []
        if any(r.get("suggested_response") for r in rows):
            return path
    return None


def replies_to_judge(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        r for r in run.get("rows", [])
        if r.get("status") == 200 and r.get("suggested_response") and r.get("priority_source") != "classifier"
    ]


def calibrate(client: Any, model: str, kb: str) -> dict[str, Any]:
    results = []
    for case in CALIBRATION:
        grade = judge_one(client, model, kb, case["text"], case["priority"], case["reply"])
        results.append({**case, "grade": grade, "correct": grade["verdict"] == case["expect"]})
    bad = [r for r in results if r["expect"] == "fail"]
    good = [r for r in results if r["expect"] == "pass"]
    return {
        "caught": sum(r["correct"] for r in bad), "planted_bad": len(bad),
        "passed_good": sum(r["correct"] for r in good), "planted_good": len(good),
        "cases": results,
    }


def summarize(graded: list[dict[str, Any]]) -> dict[str, Any]:
    if not graded:
        return {"judged": 0}
    return {
        "judged": len(graded),
        "pass_rate": round(sum(g["grade"]["verdict"] == "pass" for g in graded) / len(graded), 4),
        "averages": {c: round(mean(g["grade"][c] for g in graded), 2) for c in CRITERIA},
        "with_unsupported_claims": sum(bool(g["grade"]["unsupported_claims"]) for g in graded),
    }


def report_md(model: str, source: str | None, cal: dict[str, Any] | None, summary: dict, graded: list) -> str:
    lines = [f"# LLM-as-a-judge: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC", "", f"Judge: `{model}`."]
    if cal:
        trust = "trustworthy on these checks" if cal["caught"] == cal["planted_bad"] and \
            cal["passed_good"] == cal["planted_good"] else "**missed some planted cases: read its scores with care**"
        lines += ["", "## Calibration (planted replies with a known verdict)", "",
                  f"Caught **{cal['caught']} of {cal['planted_bad']}** bad replies and passed "
                  f"**{cal['passed_good']} of {cal['planted_good']}** good ones: {trust}.", "",
                  "| Planted | Expected | Judge | Unsupported claims |", "|---|---|---|---|"]
        for c in cal["cases"]:
            mark = "✓" if c["correct"] else "✗"
            lines.append(f"| {c['defect'] or 'good reply'} | {c['expect']} | {mark} {c['grade']['verdict']} "
                         f"| {'; '.join(c['grade']['unsupported_claims']) or '–'} |")
    if summary.get("judged"):
        a = summary["averages"]
        lines += ["", f"## Replies from `{source}`", "",
                  f"{summary['judged']} GPT-written replies judged. **Pass rate {summary['pass_rate']:.0%}**, "
                  f"{summary['with_unsupported_claims']} with an unsupported claim.", "",
                  "| Groundedness | Relevance | Tone | Next step |", "|---|---|---|---|",
                  f"| {a['groundedness']} | {a['relevance']} | {a['tone']} | {a['next_step']} |"]
        failed = [g for g in graded if g["grade"]["verdict"] == "fail"]
        if failed:
            lines += ["", "### Replies that failed", "", "| Lead | Reply | Judge's reason | Unsupported |",
                      "|---|---|---|---|"]
            for g in failed:
                lines.append(f"| {g['text'][:70]} | {g['suggested_response'][:110]} | {g['grade']['reason']} "
                             f"| {'; '.join(g['grade']['unsupported_claims']) or '–'} |")
    return "\n".join(line.replace("\n", " ") if line.startswith("|") else line for line in lines) + "\n"


def main(argv: list[str] | None = None, client: Any = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, help="a live-*.json from eval/live_run.py (default: newest with replies)")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--yes", action="store_true", help="call OpenAI (about $0.0003 per reply with gpt-4o-mini)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    run_path = None if args.calibrate_only else (args.run or latest_run(args.out_dir))
    rows = replies_to_judge(json.loads(run_path.read_text())) if run_path else []
    calls = len(CALIBRATION) + len(rows)
    cost = round(calls * COST_PER_CALL.get(args.judge_model, 0.005), 4)
    print(f"{len(CALIBRATION)} calibration cases + {len(rows)} replies"
          f"{f' from {run_path.name}' if run_path else ''} → judge {args.judge_model} (~${cost})")
    if not args.calibrate_only and not run_path:
        print("No live run with replies yet: run `python -m eval.live_run --yes` first (it saves the replies).")
    if not args.yes:
        print("dry run: nothing sent. Add --yes to run it.")
        return 0

    if client is None:
        from openai import OpenAI

        from config import get_settings

        client = OpenAI(api_key=get_settings().openai_api_key)
    kb = knowledge_text()
    cal = calibrate(client, args.judge_model, kb)
    graded = [{**r, "grade": judge_one(client, args.judge_model, kb, r["text"], r["priority"], r["suggested_response"])}
              for r in rows]
    summary = summarize(graded)
    md = report_md(args.judge_model, run_path.name if run_path else None, cal, summary, graded)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"judge-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    stem.with_suffix(".md").write_text(md)
    stem.with_suffix(".json").write_text(json.dumps(
        {"judge_model": args.judge_model, "source": str(run_path) if run_path else None,
         "calibration": cal, "summary": summary, "graded": graded}, indent=2))
    print("\n" + md + f"\nSaved {stem}.md / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
