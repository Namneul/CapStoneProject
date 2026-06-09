"""
Train a KoELECTRA token-classification model for hesitation detection.

Expected dataset format:
[
  {"tokens": ["어", "저는", "음"], "labels": ["FILLER", "O", "FILLER"]}
]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing training dependency. Install torch and transformers first, "
        "for example: pip install torch transformers accelerate"
    ) from exc

try:
    from training.label_schema import ID_TO_LABEL, LABELS, LABEL_TO_ID
except ModuleNotFoundError:
    from label_schema import ID_TO_LABEL, LABELS, LABEL_TO_ID


IGNORE_INDEX = -100
DEFAULT_MODEL = "monologg/koelectra-base-v3-discriminator"
DEFAULT_OUTPUT_DIR = "models/hesitation_detector"


@dataclass
class TokenSample:
    tokens: list[str]
    labels: list[str]


class HesitationTokenDataset(Dataset):
    def __init__(self, samples: list[TokenSample], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoded.word_ids()
        labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(IGNORE_INDEX)
            elif word_id != previous_word_id:
                labels.append(LABEL_TO_ID[sample.labels[word_id]])
            else:
                labels.append(IGNORE_INDEX)
            previous_word_id = word_id
        encoded["labels"] = labels
        return encoded


def load_samples(path: str | Path) -> list[TokenSample]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Training dataset must be a JSON list.")

    samples = []
    for idx, item in enumerate(data):
        tokens = [str(token) for token in item["tokens"]]
        labels = [str(label).upper() for label in item["labels"]]
        if len(tokens) != len(labels):
            raise ValueError(f"Token/label length mismatch at sample {idx}.")
        unknown = sorted(set(labels) - set(LABELS))
        if unknown:
            raise ValueError(f"Unknown labels at sample {idx}: {unknown}")
        samples.append(TokenSample(tokens=tokens, labels=labels))
    if not samples:
        raise ValueError("No training samples found.")
    return samples


def split_samples(
    samples: list[TokenSample],
    validation_ratio: float,
    seed: int,
) -> tuple[list[TokenSample], list[TokenSample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) > 1 else 0
    eval_samples = shuffled[:eval_size]
    train_samples = shuffled[eval_size:]
    if not train_samples:
        train_samples = eval_samples
        eval_samples = []
    return train_samples, eval_samples


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = logits.argmax(axis=-1)

    total = 0
    correct = 0
    hesitation_total = 0
    hesitation_correct = 0
    for pred_row, label_row in zip(predictions, labels):
        for pred, label in zip(pred_row, label_row):
            if label == IGNORE_INDEX:
                continue
            total += 1
            correct += int(pred == label)
            if ID_TO_LABEL[int(label)] != "O":
                hesitation_total += 1
                hesitation_correct += int(pred == label)

    return {
        "token_accuracy": correct / total if total else 0.0,
        "hesitation_token_accuracy": hesitation_correct / hesitation_total if hesitation_total else 0.0,
    }


def train(args: argparse.Namespace) -> str:
    samples = load_samples(args.dataset)
    train_samples, eval_samples = split_samples(samples, args.validation_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = HesitationTokenDataset(train_samples, tokenizer, args.max_length)
    eval_dataset = HesitationTokenDataset(eval_samples, tokenizer, args.max_length) if eval_samples else None
    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = _build_training_args(args, eval_dataset is not None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {"labels": LABELS, "label2id": LABEL_TO_ID, "id2label": ID_TO_LABEL},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return str(output_dir)


def _build_training_args(args: argparse.Namespace, has_eval: bool):
    common = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "report_to": [],
        "seed": args.seed,
    }
    try:
        return TrainingArguments(
            **common,
            eval_strategy="epoch" if has_eval else "no",
        )
    except TypeError:
        return TrainingArguments(
            **common,
            evaluation_strategy="epoch" if has_eval else "no",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KoELECTRA hesitation token classifier.")
    parser.add_argument("--dataset", required=True, help="Converted JSON dataset path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save the model")
    parser.add_argument("--base-model", default=DEFAULT_MODEL, help="HuggingFace base model")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = train(args)
    print(f"model saved -> {output_dir}")


if __name__ == "__main__":
    main()
