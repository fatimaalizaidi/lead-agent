"""Score the RAG pipeline with Ragas: Faithfulness, Answer Relevancy, Context Recall/Precision.

Each golden question goes through the real Knowledge Agent (``multiagent.knowledge_node``):
the same retrieval (``rag.retrieve_context``), prompt and model the API uses. Ragas then
grades every (question, retrieved contexts, answer, reference) row with an LLM judge.

Usage (from the repo root, after ``pip install -r requirements-dev.txt``):

    python -m eval.run_ragas                      # current RAG_* settings
    python -m eval.run_ragas --mode vector        # vector | hybrid | hybrid-rerank
    python -m eval.run_ragas --compare            # all three modes side by side
    python -m eval.run_ragas --limit 5            # quick smoke run
    python -m eval.run_ragas --fail-under 0.8     # exit 1 if any metric mean < 0.8 (CI)

Costs OpenAI calls (answers + judge); results land in ``eval/results/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# mode -> (RAG_HYBRID, RAG_RERANKER). "configured" leaves .env / defaults untouched.
MODES: dict[str, tuple[bool, str] | None] = {
    "configured": None,
    "vector": (False, "none"),
    "hybrid": (True, "none"),
    "hybrid-rerank": (True, "auto"),
}
COMPARE_MODES = ["vector", "hybrid", "hybrid-rerank"]

# question -> (answer, retrieved contexts)
AnswerFn = Callable[[str], Awaitable[tuple[str, list[str]]]]


def load_golden(path: Path = GOLDEN_PATH, limit: int | None = None) -> list[dict[str, str]]:
    """Read the golden set; every row needs ``question`` and ``reference``."""
    rows: list[dict[str, str]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question") or not row.get("reference"):
            raise ValueError(f"{path.name}:{n} needs 'question' and 'reference'")
        rows.append(row)
    return rows[:limit] if limit else rows


def configure_mode(mode: str) -> None:
    """Point the cached app settings at one retrieval mode."""
    from config import get_settings

    preset = MODES[mode]
    if preset is None:
        return
    settings = get_settings()
    settings.rag_hybrid, settings.rag_reranker = preset


async def knowledge_agent_answer(question: str) -> tuple[str, list[str]]:
    """Run the real Knowledge Agent node; capture the contexts it retrieved."""
    import multiagent

    captured: list[dict[str, str]] = []
    original = multiagent.retrieve_context

    def capturing_retrieve(*args: Any, **kwargs: Any):
        chunks, score = original(*args, **kwargs)
        captured.extend(chunks)
        return chunks, score

    multiagent.retrieve_context = capturing_retrieve
    try:
        state = {
            "email": "eval@example.com",
            "message": question,
            # Fresh thread → no conversation memory leaks between questions.
            "thread_id": f"ragas-eval-{uuid.uuid4().hex}",
        }
        result = await multiagent.knowledge_node(state)  # type: ignore[arg-type]
    finally:
        multiagent.retrieve_context = original

    contexts = [f"{c['title']}\n{c['content']}" for c in captured]
    return result["agent_reply"], contexts


async def build_samples(golden: list[dict[str, str]], answer_fn: AnswerFn) -> list[dict[str, Any]]:
    """Ragas single-turn samples: user_input, retrieved_contexts, response, reference."""
    samples: list[dict[str, Any]] = []
    for i, row in enumerate(golden, start=1):
        answer, contexts = await answer_fn(row["question"])
        print(f"  [{i}/{len(golden)}] {row.get('id', row['question'][:40])}", file=sys.stderr)
        samples.append(
            {
                "user_input": row["question"],
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": row["reference"],
            }
        )
    return samples


def score_samples(samples: list[dict[str, Any]], judge_model: str):
    """Grade samples with Ragas; returns a pandas DataFrame (one row per sample)."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy

    from config import get_settings

    key = get_settings().openai_api_key
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy(), ContextRecall(), ContextPrecision()],
        llm=LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0, api_key=key)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=key)),
        show_progress=True,
    )
    return result.to_pandas()


METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def summarize(df) -> dict[str, float]:
    """Mean of each metric column present (NaN rows — judge failures — are skipped)."""
    return {
        col: round(float(df[col].mean()), 3)
        for col in METRIC_COLUMNS
        if col in df.columns and df[col].notna().any()
    }


def failing_metrics(summary: dict[str, float], threshold: float | None) -> list[str]:
    if threshold is None:
        return []
    return [name for name, value in summary.items() if value < threshold]


def print_table(summaries: dict[str, dict[str, float]]) -> None:
    cols = [c for c in METRIC_COLUMNS if any(c in s for s in summaries.values())]
    width = max(len(m) for m in summaries) + 2
    print("\n" + "mode".ljust(width) + "".join(c.rjust(20) for c in cols))
    for mode, summary in summaries.items():
        cells = "".join(
            (f"{summary[c]:.3f}" if c in summary else "-").rjust(20) for c in cols
        )
        print(mode.ljust(width) + cells)


def run(args: argparse.Namespace) -> int:
    from config import get_settings

    golden = load_golden(limit=args.limit)
    judge_model = args.judge_model or get_settings().openai_model
    modes = COMPARE_MODES if args.compare else [args.mode]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summaries: dict[str, dict[str, float]] = {}
    for mode in modes:
        configure_mode(mode)
        print(f"\n== {mode}: answering {len(golden)} questions", file=sys.stderr)
        # Ragas runs its own event loop, so answering and scoring stay in separate loops.
        samples = asyncio.run(build_samples(golden, knowledge_agent_answer))
        print(f"== {mode}: scoring with judge {judge_model}", file=sys.stderr)
        df = score_samples(samples, judge_model)
        df.insert(0, "id", [row.get("id", "") for row in golden])
        df.to_csv(RESULTS_DIR / f"{stamp}-{mode}.csv", index=False)
        summaries[mode] = summarize(df)

    (RESULTS_DIR / f"{stamp}-summary.json").write_text(
        json.dumps(
            {"judge_model": judge_model, "questions": len(golden), "modes": summaries},
            indent=2,
        ),
        encoding="utf-8",
    )
    print_table(summaries)
    print(f"\nPer-question scores: {RESULTS_DIR}/{stamp}-*.csv")

    failed = {m: failing_metrics(s, args.fail_under) for m, s in summaries.items()}
    failed = {m: names for m, names in failed.items() if names}
    if failed:
        print(f"\nBelow --fail-under {args.fail_under}: {failed}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=list(MODES), default="configured")
    parser.add_argument("--compare", action="store_true", help="run vector, hybrid, hybrid-rerank")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--judge-model", default=None, help="default: OPENAI_MODEL")
    parser.add_argument("--fail-under", type=float, default=None, help="exit 1 if a mean is lower")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import logging

    # One INFO line per OpenAI request drowns the progress output.
    for name in ("httpx", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
