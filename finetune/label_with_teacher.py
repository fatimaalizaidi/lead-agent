"""Optional: relabel a dataset with the production LLM (the "teacher") for true distillation.

The shipped student is trained on rubric labels (``generate_dataset.py``). This script asks
``OPENAI_MODEL`` with the app's own ``SYSTEM_PROMPT`` to label each lead, and writes the file
back with a ``teacher_label`` field. Then:

- train on teacher labels:   python -m finetune.train_lora --train-file finetune/data/train.teacher.jsonl
- measure agreement:         python -m finetune.evaluate --test-file finetune/data/test.teacher.jsonl

Costs money (your OPENAI_API_KEY): about 200 input + 5 output tokens per lead, roughly $0.04 for
the 1,200 training leads on gpt-4o-mini. It prints the estimate and does nothing without --yes.

    python -m finetune.label_with_teacher finetune/data/test.jsonl --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from finetune.generate_dataset import LABELS, read_jsonl, write_jsonl

TOKENS_IN, TOKENS_OUT = 200, 5
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini USD per 1M tokens

SCHEMA = {
    "name": "priority",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"priority": {"type": "string", "enum": list(LABELS)}},
        "required": ["priority"],
        "additionalProperties": False,
    },
}


async def label_rows(rows: list[dict], client, model: str, system_prompt: str, concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        async with sem:
            completion = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=20,
                response_format={"type": "json_schema", "json_schema": SCHEMA},
                messages=[
                    {"role": "system", "content": system_prompt + "\nReturn only the priority."},
                    {"role": "user", "content": row["text"]},
                ],
            )
            label = json.loads(completion.choices[0].message.content)["priority"]
            return {**row, "teacher_label": label}

    return await asyncio.gather(*(one(r) for r in rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--yes", action="store_true", help="actually call OpenAI (costs money)")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    rows = read_jsonl(args.file)
    cost = len(rows) * (TOKENS_IN * PRICE_IN + TOKENS_OUT * PRICE_OUT) / 1e6
    print(f"{len(rows)} leads → about ${cost:.3f} on gpt-4o-mini")
    if not args.yes:
        print("dry run: add --yes to call OpenAI")
        return 0

    from config import get_settings
    from main import SYSTEM_PROMPT, _get_async_client

    labelled = asyncio.run(
        label_rows(rows, _get_async_client(), get_settings().openai_model, SYSTEM_PROMPT, args.concurrency)
    )
    out = args.file.with_suffix(".teacher.jsonl")
    write_jsonl(out, labelled)
    agree = sum(r["teacher_label"] == r["label"] for r in labelled) / len(labelled)
    print(f"wrote {out} (teacher agrees with rubric labels on {agree:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
