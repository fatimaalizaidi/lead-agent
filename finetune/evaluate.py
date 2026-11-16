"""Evaluate exported priority models on the hand-written test set (no torch, no API calls).

Uses ``classifier.PriorityClassifier``, the exact code the API serves, so the numbers here are
the numbers production gets. Reports accuracy, macro-F1, per-class precision/recall, the
confusion matrix, CPU latency and model size, and checks the floors in
``finetune/thresholds.json``. High recall has its own floor: a missed hot lead costs the most.

    python -m finetune.evaluate                                   # finetune/model (shipped)
    python -m finetune.evaluate --model models/priority-classifier --model models/priority-head-only
    python -m finetune.evaluate --report finetune/results/report.md

Exits 1 if the first model is under a floor (unless --no-gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from classifier import PriorityClassifier
from finetune.generate_dataset import DATA_DIR, LABELS, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = Path(__file__).resolve().parent / "thresholds.json"


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"accuracy": round(float(np.mean([g == p for g, p in zip(gold, pred, strict=True)])), 4)}
    f1s = []
    for label in LABELS:
        tp = sum(g == p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(p == label != g for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label != p for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        out[f"{label.lower()}_precision"] = round(precision, 4)
        out[f"{label.lower()}_recall"] = round(recall, 4)
    out["macro_f1"] = round(float(np.mean(f1s)), 4)
    pairs = list(zip(gold, pred, strict=True))
    out["confusion"] = {g: {p: pairs.count((g, p)) for p in LABELS} for g in LABELS}
    return out


def evaluate(model_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    clf = PriorityClassifier(model_dir)
    texts = [r["text"] for r in rows]
    preds = clf.predict_batch(texts)
    result = metrics([r["label"] for r in rows], [p.label for p in preds])

    clf.predict(texts[0])  # warm-up
    timings = []
    for text in texts:
        started = time.perf_counter()
        clf.predict(text)
        timings.append((time.perf_counter() - started) * 1000)
    result["latency_ms_p50"] = round(float(np.percentile(timings, 50)), 2)
    result["latency_ms_p95"] = round(float(np.percentile(timings, 95)), 2)
    result["size_mb"] = round((model_dir / "model.onnx").stat().st_size / 1e6, 1)
    result["meta"] = clf.meta
    result["errors"] = [
        {"text": r["text"], "label": r["label"], "pred": p.label, "confidence": p.confidence}
        for r, p in zip(rows, preds, strict=True)
        if r["label"] != p.label
    ]
    if all("teacher_label" in r for r in rows):
        agree = [r["teacher_label"] == p.label for r, p in zip(rows, preds, strict=True)]
        result["teacher_agreement"] = round(float(np.mean(agree)), 4)
    return result


def gate_failures(result: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    return [
        f"{name} {result[name]:.3f} < {floor:.3f}"
        for name, floor in thresholds.items()
        if not name.startswith("_") and result.get(name, 0.0) < floor
    ]


def report_md(results: dict[str, dict[str, Any]], n: int) -> str:
    cols = ["accuracy", "macro_f1", "high_recall", "high_precision", "low_recall", "latency_ms_p50", "size_mb"]
    lines = [
        f"# Priority classifier: hand-written test set ({n} leads)",
        "",
        "| Model | Trainable params | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for name, r in results.items():
        meta = r["meta"]
        params = f"{meta['trainable_params']:,} / {meta['total_params']:,}"
        lines.append(f"| {name} | {params} | " + " | ".join(str(r[c]) for c in cols) + " |")
    for name, r in results.items():
        lines += ["", f"## {name}", "", "Confusion matrix (rows = true label, columns = predicted):", ""]
        lines += ["| | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
        lines += [f"| **{g}** | " + " | ".join(str(r["confusion"][g][p]) for p in LABELS) + " |" for g in LABELS]
        if r["errors"]:
            lines += ["", "Misclassified:", ""]
            lines += [f"- `{e['label']}` → `{e['pred']}` ({e['confidence']:.2f}): {e['text']}" for e in r["errors"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, action="append", help="model dir (repeatable)")
    parser.add_argument("--test-file", type=Path, default=DATA_DIR / "test.jsonl")
    parser.add_argument("--report", type=Path, help="write a Markdown report here")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args(argv)

    dirs = args.model or [ROOT / "finetune" / "model"]
    rows = read_jsonl(args.test_file)
    results = {}
    for model_dir in dirs:
        r = evaluate(model_dir, rows)
        name = f"{r['meta']['mode']} ({model_dir.name})"
        results[name] = r
        summary = {k: v for k, v in r.items() if k not in {"confusion", "errors", "meta"}}
        print(name, json.dumps(summary))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_md(results, len(rows)))
        print(f"report → {args.report}")

    if args.no_gate:
        return 0
    first = next(iter(results.values()))
    failures = gate_failures(first, json.loads(THRESHOLDS.read_text()))
    for failure in failures:
        print(f"GATE FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
