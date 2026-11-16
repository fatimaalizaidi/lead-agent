"""Fine-tune a small encoder to classify lead priority (High / Medium / Low) with LoRA.

Student: ``sentence-transformers/all-MiniLM-L6-v2`` (22M parameters) with a 3-way classification
head. LoRA trains rank-8 adapters on the attention query/value projections plus the head, about
1% of the weights, so it trains in minutes on a laptop CPU/Apple GPU or a free Colab T4.

After training, the adapters are merged into the base weights and the model is exported to ONNX
and quantised to int8 (~23 MB). The API serves it with onnxruntime + tokenizers (``classifier.py``),
which the app already ships for Chroma, so production needs no torch.

``--mode head-only`` freezes the encoder and trains only the head (a "linear probe"): the baseline
that shows what fine-tuning the encoder adds.

Needs the training extras (not used by the app or CI):
    python -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt
    .venv-train/bin/python -m finetune.train_lora               # LoRA → models/priority-classifier
    .venv-train/bin/python -m finetune.train_lora --mode head-only --out models/priority-head-only
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune.generate_dataset import DATA_DIR, LABELS, read_jsonl

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 128
ROOT = Path(__file__).resolve().parent.parent


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(mode: str, rank: int) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    if mode == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # also keeps the new classifier head trainable
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, config)
    else:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")
    return model


def batches(rows: list[dict], tokenizer, batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [rows[i] for i in order[start : start + batch_size]]
        enc = tokenizer(
            [r["text"] for r in chunk], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        enc["labels"] = torch.tensor([LABELS.index(r["label"]) for r in chunk])
        yield enc


@torch.no_grad()
def accuracy(model, rows, tokenizer, device) -> float:
    model.eval()
    correct = 0
    for enc in batches(rows, tokenizer, 64, False, random.Random(0)):
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = enc.pop("labels")
        correct += (model(**enc).logits.argmax(-1) == labels).sum().item()
    return correct / len(rows)


def export_onnx(model: torch.nn.Module, tokenizer, out: Path) -> Path:
    """Merged fp32 model → ONNX → dynamic int8 quantisation (CPU-friendly, ~4x smaller)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = model.to("cpu").eval()
    sample = tokenizer(["Pricing for 20 seats?"], return_tensors="pt")
    fp32 = out / "model-fp32.onnx"
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {name: {0: "batch", 1: "sequence"} for name in names}
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(fp32),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={**axes, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    final = out / "model.onnx"
    quantize_dynamic(str(fp32), str(final), weight_type=QuantType.QInt8)
    fp32.unlink()
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["lora", "head-only"], default="lora")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "priority-classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None, help="default: 5e-4 for LoRA, 2e-3 for head-only")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-file", type=Path, default=DATA_DIR / "train.jsonl")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device()
    lr = args.lr or (5e-4 if args.mode == "lora" else 2e-3)

    train = read_jsonl(args.train_file)
    val = read_jsonl(DATA_DIR / "val.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = build_model(args.mode, args.rank).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"device={device} mode={args.mode} trainable={trainable:,}/{total:,} ({trainable / total:.2%})")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)

    best_acc, best_state = -1.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for enc in batches(train, tokenizer, args.batch_size, True, rng):
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = model(**enc).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        val_acc = accuracy(model, val, tokenizer, device)
        print(f"epoch {epoch}: loss={np.mean(losses):.4f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    if args.mode == "lora":
        model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, tokenizer, args.out)
    tokenizer.backend_tokenizer.save(str(args.out / "tokenizer.json"))
    meta = {
        "base_model": BASE_MODEL,
        "mode": args.mode,
        "labels": list(LABELS),
        "max_length": MAX_LENGTH,
        "lora_rank": args.rank if args.mode == "lora" else None,
        "trainable_params": trainable,
        "total_params": total,
        "epochs": args.epochs,
        "learning_rate": lr,
        "train_examples": len(train),
        "train_file": args.train_file.name,
        "best_val_accuracy": round(best_acc, 4),
        "train_seconds": round(train_seconds, 1),
        "device": str(device),
        "size_mb": round(onnx_path.stat().st_size / 1e6, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
