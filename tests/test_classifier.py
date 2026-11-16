"""Fine-tuned priority classifier: the shipped ONNX model, POST /classify-priority, the quality gate,
and the offline parts of the training pipeline (dataset generation, metrics, teacher labelling).

Runs the real int8 model in finetune/model (a few ms per lead, CPU, no torch, no OpenAI).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import AsyncClient

import classifier
from finetune import evaluate, generate_dataset, label_with_teacher
from tests.openai_fakes import completion_with_json, make_async_openai_client

DEMO_KEY = "la_demo-test-secret-0123456789abcdef"


@pytest.fixture(autouse=True)
def _fresh_classifier() -> Iterator[None]:
    classifier.reset()
    yield
    classifier.reset()


# ---- the shipped model ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Budget is approved for 40 seats. Please send the contract today, we start Monday.", "High"),
        ("Just researching tools for later this year. What does the Growth plan include?", "Medium"),
        ("Cheap SEO backlinks and guest posts, reply for our price list!", "Low"),
    ],
)
def test_shipped_model_predicts_clear_cases(text: str, label: str) -> None:
    prediction = classifier.get_classifier().predict(text)
    assert prediction.label == label
    assert set(prediction.scores) == {"High", "Medium", "Low"}
    assert abs(sum(prediction.scores.values()) - 1) < 1e-3


def test_shipped_model_passes_quality_gate() -> None:
    """The CI gate: the committed model must meet finetune/thresholds.json on the hand-written set."""
    result = evaluate.evaluate(classifier.model_dir(), generate_dataset.read_jsonl(evaluate.DATA_DIR / "test.jsonl"))
    thresholds = json.loads(evaluate.THRESHOLDS.read_text())
    assert evaluate.gate_failures(result, thresholds) == [], result


# ---- endpoint ------------------------------------------------------------------------------------


async def test_classify_priority_endpoint(client: AsyncClient) -> None:
    resp = await client.post("/classify-priority", json={"message": "Ready to buy 25 seats, send the invoice."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["priority"] == "High"
    assert 0 < body["confidence"] <= 1
    assert body["model"] == "lora-all-MiniLM-L6-v2"
    assert body["latency_ms"] >= 0


async def test_classify_priority_validates_input(client: AsyncClient) -> None:
    assert (await client.post("/classify-priority", json={"message": "   "})).status_code == 400
    assert (await client.post("/classify-priority", json={})).status_code == 400


async def test_classify_priority_requires_auth_when_keys_set(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config import get_settings

    monkeypatch.setenv("API_KEYS", f"demo:{DEMO_KEY}")
    get_settings.cache_clear()
    payload = {"message": "Pricing for 20 seats?"}
    assert (await client.post("/classify-priority", json=payload)).status_code == 401
    ok = await client.post("/classify-priority", json=payload, headers={"X-API-Key": DEMO_KEY})
    assert ok.status_code == 200


async def test_missing_model_returns_503(client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from config import get_settings

    monkeypatch.setenv("PRIORITY_MODEL_DIR", str(tmp_path))
    get_settings.cache_clear()
    resp = await client.post("/classify-priority", json={"message": "hello"})
    assert (resp.status_code, resp.json()["error_code"]) == (503, "classifier_unavailable")
    assert (await client.get("/health")).status_code == 200


# ---- training pipeline (offline parts) ------------------------------------------------------------


def test_dataset_is_reproducible_and_disjoint_from_test() -> None:
    test_texts = {r["text"].lower() for r in generate_dataset.read_jsonl(evaluate.DATA_DIR / "test.jsonl")}
    first = generate_dataset.generate(20, seed=1, exclude=test_texts)
    assert first == generate_dataset.generate(20, seed=1, exclude=test_texts)
    assert {r["label"] for r in first} == set(generate_dataset.LABELS)
    assert not {r["text"].lower() for r in first} & test_texts

    committed = generate_dataset.read_jsonl(evaluate.DATA_DIR / "train.jsonl")
    assert not {r["text"].lower() for r in committed} & test_texts, "test leads must never be trained on"


def test_metrics() -> None:
    gold = ["High", "High", "Medium", "Low"]
    pred = ["High", "Medium", "Medium", "Low"]
    m = evaluate.metrics(gold, pred)
    assert m["accuracy"] == 0.75
    assert (m["high_recall"], m["high_precision"], m["medium_precision"]) == (0.5, 1.0, 0.5)
    assert m["confusion"]["High"] == {"High": 1, "Medium": 1, "Low": 0}
    assert evaluate.gate_failures(m, {"accuracy": 0.8, "_comment": 0}) == ["accuracy 0.750 < 0.800"]


def test_teacher_labelling_with_fake_openai() -> None:
    client = make_async_openai_client(lambda *a, **k: completion_with_json({"priority": "Medium"}))
    rows = [{"text": "Tell me more", "label": "Medium"}, {"text": "hi", "label": "Low"}]
    out = asyncio.run(label_with_teacher.label_rows(rows, client, "gpt-4o-mini", "prompt", concurrency=2))
    assert [r["teacher_label"] for r in out] == ["Medium", "Medium"]
    assert [r["label"] for r in out] == ["Medium", "Low"], "rubric labels are kept"


def test_teacher_labelling_dry_run_spends_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    assert label_with_teacher.main([str(evaluate.DATA_DIR / "test.jsonl")]) == 0
    assert "dry run" in capsys.readouterr().out
