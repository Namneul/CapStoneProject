"""
Convert hesitation/disfluency annotations to token-classification JSON.

Canonical output:
[
  {"tokens": ["어", "저는", "음"], "labels": ["FILLER", "O", "FILLER"]}
]

Supported inputs:
- json/jsonl records already containing tokens + labels
- json/jsonl records containing text + annotations
- csv/tsv rows with sentence_id, token, label columns
- simple Switchboard-like transcript text using markers:
  [text]        -> RESTART
  {text}        -> SELF_CORRECTION
  <pause>       -> PAUSE
  filler words  -> FILLER
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

try:
    from training.label_schema import LABELS
except ModuleNotFoundError:
    from label_schema import LABELS


DEFAULT_FILLERS = {
    "어",
    "음",
    "으음",
    "음...",
    "어...",
    "그",
    "그니까",
    "그러니까",
    "저",
    "저기",
    "약간",
    "뭐",
}

PAUSE_TOKENS = {"<pause>", "[pause]", "(pause)", "...", "…"}
SELF_CORRECTION_MARKERS = {"아니", "다시", "정정하면", "말하자면", "말씀드리면"}
RESTART_MARKERS = {"처음부터", "그러면"}
VALID_LABELS = set(LABELS)


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    input_format: str = "auto",
) -> list[dict[str, list[str]]]:
    path = Path(input_path)
    fmt = _detect_format(path, input_format)

    if fmt in {"json", "jsonl"}:
        samples = _convert_json_records(_read_json_records(path, fmt))
    elif fmt in {"csv", "tsv"}:
        samples = _convert_table(path, delimiter="," if fmt == "csv" else "\t")
    elif fmt in {"switchboard", "podcastfillers", "text"}:
        samples = _convert_text_lines(path)
    else:
        raise ValueError(f"Unsupported format: {input_format}")

    _validate_samples(samples)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    return samples


def _detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested.lower()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    return "text"


def _read_json_records(path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "samples" in data:
            data = data["samples"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list, or a dict with data/samples.")
    return data


def _convert_json_records(records: list[dict[str, Any]]) -> list[dict[str, list[str]]]:
    samples = []
    for record in records:
        if "tokens" in record and "labels" in record:
            samples.append({
                "tokens": [str(token) for token in record["tokens"]],
                "labels": [_normalize_label(label) for label in record["labels"]],
            })
            continue

        if isinstance(record.get("words"), list):
            tokens = []
            labels = []
            for word in record["words"]:
                if isinstance(word, str):
                    tokens.append(word)
                    labels.append("O")
                    continue
                token = word.get("word", word.get("token", ""))
                if not token:
                    continue
                label = word.get("label", "FILLER" if word.get("is_filler") else "O")
                tokens.append(str(token))
                labels.append(_normalize_label(label))
            _apply_heuristic_labels(tokens, labels)
            samples.append({"tokens": tokens, "labels": labels})
            continue

        text = str(record.get("text", "")).strip()
        if not text:
            continue
        tokens = _tokenize(text)
        labels = ["O"] * len(tokens)

        for annotation in record.get("annotations", record.get("spans", [])):
            label = _normalize_label(annotation.get("label", annotation.get("type", "O")))
            if "token_start" in annotation and "token_end" in annotation:
                start = int(annotation["token_start"])
                end = int(annotation["token_end"])
                for idx in range(max(0, start), min(len(labels), end)):
                    labels[idx] = label
            elif "text" in annotation:
                _label_matching_tokens(tokens, labels, str(annotation["text"]), label)

        _apply_heuristic_labels(tokens, labels)
        samples.append({"tokens": tokens, "labels": labels})
    return samples


def _convert_table(path: Path, delimiter: str) -> list[dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = set(reader.fieldnames or [])
        if not {"token", "label"}.issubset(fieldnames):
            raise ValueError("CSV/TSV input must include token and label columns.")

        for row in reader:
            sentence_id = row.get("sentence_id") or row.get("utterance_id") or "0"
            item = grouped.setdefault(sentence_id, {"tokens": [], "labels": []})
            item["tokens"].append(str(row["token"]))
            item["labels"].append(_normalize_label(row["label"]))
    return list(grouped.values())


def _convert_text_lines(path: Path) -> list[dict[str, list[str]]]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens, labels = _parse_marked_text(line)
            samples.append({"tokens": tokens, "labels": labels})
    return samples


def _parse_marked_text(text: str) -> tuple[list[str], list[str]]:
    tokens: list[str] = []
    labels: list[str] = []
    pieces = re.findall(r"\[[^\]]+\]|\{[^}]+\}|<pause>|\S+", text)
    for piece in pieces:
        if piece.lower() in PAUSE_TOKENS:
            tokens.append(piece)
            labels.append("PAUSE")
        elif piece.startswith("[") and piece.endswith("]"):
            inner_tokens = _tokenize(piece[1:-1])
            tokens.extend(inner_tokens)
            labels.extend(["RESTART"] * len(inner_tokens))
        elif piece.startswith("{") and piece.endswith("}"):
            inner_tokens = _tokenize(piece[1:-1])
            tokens.extend(inner_tokens)
            labels.extend(["SELF_CORRECTION"] * len(inner_tokens))
        else:
            tokens.append(piece)
            labels.append("O")

    _apply_heuristic_labels(tokens, labels)
    return tokens, labels


def _apply_heuristic_labels(tokens: list[str], labels: list[str]) -> None:
    correction_window = 0
    restart_window = 0
    for idx, token in enumerate(tokens):
        normalized = token.strip(".,!?\"'()[]{}")
        if labels[idx] == "O" and normalized in DEFAULT_FILLERS:
            labels[idx] = "FILLER"
        if labels[idx] == "O" and token.lower() in PAUSE_TOKENS:
            labels[idx] = "PAUSE"
        if labels[idx] == "O" and normalized in SELF_CORRECTION_MARKERS:
            correction_window = 2
            labels[idx] = "SELF_CORRECTION"
            continue
        if labels[idx] == "O" and normalized in RESTART_MARKERS:
            restart_window = 1
            labels[idx] = "RESTART"
            continue
        if labels[idx] == "O" and correction_window > 0:
            labels[idx] = "SELF_CORRECTION"
            correction_window -= 1
            continue
        if labels[idx] == "O" and restart_window > 0:
            labels[idx] = "RESTART"
            restart_window -= 1
            continue


def _label_matching_tokens(tokens: list[str], labels: list[str], phrase: str, label: str) -> None:
    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return
    width = len(phrase_tokens)
    for start in range(0, len(tokens) - width + 1):
        if tokens[start:start + width] == phrase_tokens:
            for idx in range(start, start + width):
                labels[idx] = label


def _tokenize(text: str) -> list[str]:
    return text.split()


def _normalize_label(label: Any) -> str:
    normalized = str(label).strip().upper()
    alias = {
        "NONE": "O",
        "NORMAL": "O",
        "FALSE_START": "RESTART",
        "REPETITION": "RESTART",
        "REPAIR": "SELF_CORRECTION",
        "CORRECTION": "SELF_CORRECTION",
        "DISCOURSE_MARKER": "FILLER",
    }.get(normalized, normalized)
    if alias not in VALID_LABELS:
        raise ValueError(f"Unknown label: {label}")
    return alias


def _validate_samples(samples: list[dict[str, list[str]]]) -> None:
    if not samples:
        raise ValueError("No samples were converted.")
    for idx, sample in enumerate(samples):
        if len(sample["tokens"]) != len(sample["labels"]):
            raise ValueError(f"Token/label length mismatch at sample {idx}.")
        for label in sample["labels"]:
            _normalize_label(label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert hesitation annotations to token-classification JSON.")
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "json", "jsonl", "csv", "tsv", "switchboard", "podcastfillers", "text"],
        help="Input format",
    )
    args = parser.parse_args()

    samples = convert_file(args.input, args.output, args.format)
    print(f"converted {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
