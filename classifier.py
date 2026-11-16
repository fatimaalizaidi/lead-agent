"""Local lead-priority classifier: the fine-tuned student model, served without OpenAI.

``finetune/train_lora.py`` fine-tunes all-MiniLM-L6-v2 with LoRA and exports an int8 ONNX model
(~23 MB) plus its tokenizer to ``PRIORITY_MODEL_DIR``. This module runs it with onnxruntime and
tokenizers, which the app already ships for Chroma, so there's no torch in production.
A prediction takes a few milliseconds on CPU and costs nothing.

``POST /classify-priority`` exposes it. A cheap first pass in n8n can drop or park Low leads and
send only High/Medium ones to ``/analyze-lead`` for an LLM-drafted reply.
See docs/fine-tuning.md.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from config import get_settings
from errors import AppError


class ClassifierUnavailableError(AppError):
    status_code = 503
    error_code = "classifier_unavailable"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


class PriorityClassifier:
    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.labels: list[str] = self.meta["labels"]
        self.version = f"{self.meta['mode']}-{self.meta['base_model'].rsplit('/', 1)[-1]}"
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=int(self.meta["max_length"]))
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # one small request at a time; don't fight uvicorn for CPU
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        encodings = self.tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        (logits,) = self.session.run(["logits"], {k: v for k, v in feed.items() if k in self._inputs})
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        results = []
        for row in probs:
            best = int(row.argmax())
            scores = {label: round(float(p), 4) for label, p in zip(self.labels, row, strict=True)}
            results.append(Prediction(self.labels[best], round(float(row[best]), 4), scores))
        return results

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]


_lock = threading.Lock()
_classifier: PriorityClassifier | None = None
_load_failed_for: str | None = None


def model_dir() -> Path:
    """PRIORITY_MODEL_DIR; a relative path is relative to the app directory, not the cwd."""
    path = Path(get_settings().priority_model_dir).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def get_classifier() -> PriorityClassifier:
    """Load once (thread-safe); 503 when the model files are missing or broken."""
    global _classifier, _load_failed_for
    if _classifier is not None:
        return _classifier
    path = model_dir()
    with _lock:
        if _classifier is not None:
            return _classifier
        if _load_failed_for == str(path):
            raise ClassifierUnavailableError("The priority model is not available on this server.")
        if not (path / "model.onnx").is_file():
            _load_failed_for = str(path)
            logger.warning("priority classifier: no model at {}", path)
            raise ClassifierUnavailableError("The priority model is not available on this server.")
        try:
            _classifier = PriorityClassifier(path)
        except Exception as exc:
            _load_failed_for = str(path)
            logger.warning("priority classifier failed to load ({})", type(exc).__name__)
            raise ClassifierUnavailableError("The priority model failed to load.") from exc
        logger.info("priority classifier loaded ({})", _classifier.version)
        return _classifier


def reset() -> None:
    """Drop the cached model (tests, or after replacing the files)."""
    global _classifier, _load_failed_for
    with _lock:
        _classifier = None
        _load_failed_for = None
