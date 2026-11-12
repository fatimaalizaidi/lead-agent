"""Free retrieval quality gate: hit rate, MRR and fact recall over the golden set.

No LLM, no API key, no cost, so it runs in CI on every relevant pull request. It runs the real
retrieval pipeline (``rag.retrieve_context``: chunking, Chroma vector search, BM25, Reciprocal
Rank Fusion, article grouping, cut to ``RAG_K``) on all golden questions, then checks:

- **top1_accuracy**: share of questions whose source article is ranked first.
- **hit_rate**: share of questions whose source article is in the retrieved results at all.
- **mrr**: mean reciprocal rank of the source article (1.0 = always ranked first).
- **fact_recall**: share of the reference answers' key facts (prices, percentages, hours,
  day counts such as ``$149``, ``99.9%``, ``30``) that appear in the retrieved text. This is
  a cheap stand-in for Ragas context recall: if a fact isn't retrieved, the agent can't say it.

The only difference from production is the embedding model. Vector search uses Chroma's bundled
all-MiniLM-L6-v2 (ONNX, runs on CPU) instead of OpenAI embeddings, in a temporary index, so
``data/chroma`` is never touched. The gate catches regressions in chunking, keyword search,
fusion, grouping, ``RAG_K`` and the knowledge base itself. Answer quality (faithfulness,
relevancy) needs an LLM judge: see ``eval/run_ragas.py``.

Usage (from the repo root):

    python -m eval.retrieval_gate              # production mode (hybrid, no re-ranker)
    python -m eval.retrieval_gate --compare    # vector vs hybrid vs hybrid + re-rank
    python -m eval.retrieval_gate --no-gate    # print scores, never fail

Exits 1 if a metric is under its floor in ``eval/retrieval_thresholds.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from eval.run_ragas import EVAL_DIR, RESULTS_DIR, configure_mode, load_golden

THRESHOLDS_PATH = EVAL_DIR / "retrieval_thresholds.json"
METRICS = ["top1_accuracy", "hit_rate", "mrr", "fact_recall"]
COMPARE_MODES = ["vector", "hybrid", "hybrid-rerank"]
PRODUCTION_MODE = "hybrid"  # render.yaml: RAG_HYBRID=true, RAG_RERANKER=none

# "$149", "99.9%", "1,000", "30". Single digits are skipped below: they match almost anywhere.
_FACT_RE = re.compile(r"\$?\d+(?:[.,]\d+)*%?")


class LocalEmbeddings(Embeddings):
    """Chroma's bundled all-MiniLM-L6-v2 (ONNX, CPU); downloads ~80 MB once to ~/.cache/chroma."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._fn(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def reference_facts(reference: str) -> list[str]:
    """Key facts in a reference answer: numbers with 2+ digits, or with a ``$`` / ``%``."""
    facts: list[str] = []
    for match in _FACT_RE.findall(reference):
        fact = match.rstrip(".,")
        digits = sum(ch.isdigit() for ch in fact)
        if (digits >= 2 or "$" in fact or "%" in fact) and fact not in facts:
            facts.append(fact)
    return facts


def fact_in_text(fact: str, text: str) -> bool:
    """``30`` matches "30 days" but not "300" or "1.30"."""
    return re.search(rf"(?<![\d.,]){re.escape(fact)}(?![\d])", text) is not None


def score_question(row: dict[str, str], chunks: list[dict[str, str]], source_title: str) -> dict[str, Any]:
    retrieved = [chunk["title"] for chunk in chunks]
    rank = retrieved.index(source_title) + 1 if source_title in retrieved else None
    text = "\n".join(f"{chunk['title']}\n{chunk['content']}" for chunk in chunks)
    facts = reference_facts(row["reference"])
    missing = [fact for fact in facts if not fact_in_text(fact, text)]
    return {
        "id": row.get("id", ""),
        "source": row["source"],
        "rank": rank,
        "retrieved": retrieved,
        "facts": len(facts),
        "facts_found": len(facts) - len(missing),
        "missing_facts": missing,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    n = len(results)
    total_facts = sum(r["facts"] for r in results)
    summary = {
        "top1_accuracy": sum(r["rank"] == 1 for r in results) / n,
        "hit_rate": sum(r["rank"] is not None for r in results) / n,
        "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / n,
    }
    if total_facts:
        summary["fact_recall"] = sum(r["facts_found"] for r in results) / total_facts
    return {name: round(value, 3) for name, value in summary.items()}


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = [k for k in raw if not k.startswith("_") and k not in METRICS]
    if unknown:
        raise ValueError(f"{path.name}: unknown metrics {unknown}")
    return {k: float(v) for k, v in raw.items() if not k.startswith("_")}


def gate_failures(summary: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Metrics under their floor; a gated metric with no score fails too."""
    return [name for name, floor in thresholds.items() if name not in summary or summary[name] < floor]


def report_md(
    summaries: dict[str, dict[str, float]],
    thresholds: dict[str, float],
    results: dict[str, list[dict[str, Any]]],
    *,
    gated_mode: str | None,
) -> str:
    """Markdown for the GitHub Actions job summary."""
    failed = gate_failures(summaries[gated_mode], thresholds) if gated_mode else []
    questions = len(next(iter(results.values()), []))
    if gated_mode:
        title = f"## Retrieval gate: {'❌ failed' if failed else '✅ passed'}"
    else:
        title = "## Retrieval scores"
    lines = [
        title,
        "",
        f"{questions} golden questions · local embeddings (all-MiniLM-L6-v2) · no LLM, no API key",
        "",
        "| Mode | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for mode, summary in summaries.items():
        cells = []
        for name in METRICS:
            cell = f"{summary[name]:.3f}" if name in summary else "-"
            if mode == gated_mode and name in thresholds:
                cell += f" {'❌' if name in failed else '✅'} (floor {thresholds[name]:.2f})"
            cells.append(cell)
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    for mode, rows in results.items():
        misses = [r for r in rows if r["rank"] is None or r["missing_facts"]]
        if not misses:
            continue
        lines += ["", f"**{mode}: questions with a miss**", "", "| Question | Rank | Missing facts |", "|---|---|---|"]
        for r in misses:
            rank = r["rank"] or "not retrieved"
            lines.append(f"| {r['id']} | {rank} | {', '.join(r['missing_facts']) or '-'} |")
    return "\n".join(lines) + "\n"


def isolate_index(index_dir: Path) -> None:
    """Point rag at a temporary Chroma dir and local embeddings; reset its caches."""
    import rag

    embeddings = LocalEmbeddings()
    rag.CHROMA_DIR = index_dir
    rag.FINGERPRINT_PATH = index_dir / ".kb_fingerprint"
    rag._embeddings = lambda: embeddings  # type: ignore[assignment]
    rag._vectorstore = None
    rag._bm25_cache = None


def source_titles() -> dict[str, str]:
    """knowledge/<file>.md -> article title, as retrieval reports it."""
    import rag

    return {str(d.metadata["source"]): str(d.metadata["title"]) for d in rag._load_knowledge_documents()}


def run_mode(mode: str, golden: list[dict[str, str]], titles: dict[str, str]) -> list[dict[str, Any]]:
    import rag

    configure_mode(mode)
    results = []
    for row in golden:
        chunks, _score = rag.retrieve_context(row["question"])
        results.append(score_question(row, chunks, titles[row["source"]]))
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "retrieved": " | ".join(row["retrieved"]),
                             "missing_facts": " ".join(row["missing_facts"])})


def run(args: argparse.Namespace) -> int:
    # Settings require OPENAI_API_KEY, but nothing here calls OpenAI.
    os.environ.setdefault("OPENAI_API_KEY", "not-used-by-retrieval-gate")
    import logging

    logging.getLogger("rag").setLevel(logging.WARNING)

    golden = load_golden()
    modes = COMPARE_MODES if args.compare else [args.mode]
    gated_mode = None if args.no_gate else (PRODUCTION_MODE if PRODUCTION_MODE in modes else modes[0])
    thresholds = load_thresholds()

    results: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="retrieval-gate-") as tmp:
        isolate_index(Path(tmp))
        titles = source_titles()
        for mode in modes:
            print(f"== {mode}: retrieving for {len(golden)} questions", file=sys.stderr)
            results[mode] = run_mode(mode, golden, titles)

    summaries = {mode: summarize(rows) for mode, rows in results.items()}
    report = report_md(summaries, thresholds, results, gated_mode=gated_mode)
    print("\n" + report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for mode, rows in results.items():
        write_csv(RESULTS_DIR / f"{stamp}-retrieval-{mode}.csv", rows)
    (RESULTS_DIR / f"{stamp}-retrieval-gate.md").write_text(report, encoding="utf-8")

    if gated_mode and (failed := gate_failures(summaries[gated_mode], thresholds)):
        print(f"Retrieval gate failed ({gated_mode}): {failed}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=COMPARE_MODES, default=PRODUCTION_MODE)
    parser.add_argument("--compare", action="store_true", help="score all three retrieval modes")
    parser.add_argument("--no-gate", action="store_true", help="report scores, never exit 1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
