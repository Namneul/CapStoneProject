"""
Inference module for hesitation detection.

Primary path: load a trained token-classification model from
models/hesitation_detector/.

Development fallback: pass allow_fallback=True to use rule-based counts when a
trained model is not available yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from training.label_schema import HESITATION_LABELS, ID_TO_LABEL, LABELS
except ModuleNotFoundError:
    from label_schema import HESITATION_LABELS, ID_TO_LABEL, LABELS


DEFAULT_MODEL_DIR = "models/hesitation_detector"
DEFAULT_FILLERS = {"어", "음", "으음", "그", "그니까", "그러니까", "저", "저기", "약간", "뭐"}
SELF_CORRECTION_MARKERS = {"아니", "다시", "정정하면", "말하자면", "말씀드리면"}
RESTART_MARKERS = {"처음부터", "다시", "그러면", "그러니까"}
PAUSE_MARKERS = {"...", "…", "<pause>", "[pause]"}
COUNT_KEYS = {
    "FILLER": "filler_count",
    "PAUSE": "pause_count",
    "SELF_CORRECTION": "self_correction_count",
    "RESTART": "restart_count",
}


class HesitationDetector:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        allow_fallback: bool = False,
        max_length: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.allow_fallback = allow_fallback
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

        if self.model_dir.exists():
            self._load_model()
        elif not allow_fallback:
            raise FileNotFoundError(
                f"Trained hesitation model not found: {self.model_dir}. "
                "Train it first or set allow_fallback=True for rule-based development inference."
            )

    def predict(self, text: str) -> dict[str, Any]:
        tokens = _tokenize(text)
        if not tokens:
            return _empty_result(tokens=[], labels=[])

        if self.model is None or self.tokenizer is None:
            labels = _heuristic_labels(tokens)
        else:
            labels = self._predict_labels(tokens)

        counts = _count_label_spans(labels)
        score = _hesitation_score(tokens, counts)
        return {
            **counts,
            "hesitation_score": round(score, 2),
            "tokens": tokens,
            "labels": labels,
        }

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ModuleNotFoundError as exc:
            if self.allow_fallback:
                return
            raise RuntimeError(
                "Missing inference dependency. Install torch and transformers first."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.eval()

    def _predict_labels(self, tokens: list[str]) -> list[str]:
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoded.word_ids(batch_index=0)
        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]

        word_logits: dict[int, list[Any]] = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            word_logits.setdefault(word_id, []).append(logits[idx])

        labels = ["O"] * len(tokens)
        for word_id, pieces in word_logits.items():
            stacked = self.torch.stack(pieces)
            prediction_id = int(stacked.mean(dim=0).argmax().item())
            labels[word_id] = _label_from_id(prediction_id)
        return labels


def _count_label_spans(labels: list[str]) -> dict[str, int]:
    counts = {
        "filler_count": 0,
        "pause_count": 0,
        "self_correction_count": 0,
        "restart_count": 0,
    }
    previous = "O"
    for label in labels:
        if label != "O" and label != previous:
            counts[COUNT_KEYS[label]] += 1
        previous = label
    return counts


def _hesitation_score(tokens: list[str], counts: dict[str, int]) -> float:
    if not tokens:
        return 0.0
    weighted = (
        counts["filler_count"] * 1.0
        + counts["pause_count"] * 1.2
        + counts["self_correction_count"] * 1.6
        + counts["restart_count"] * 1.8
    )
    density = weighted / max(1, len(tokens))
    return min(1.0, density * 2.4)


def _heuristic_labels(tokens: list[str]) -> list[str]:
    labels = []
    correction_window = 0
    restart_window = 0
    for token in tokens:
        cleaned = token.strip(".,!?\"'()[]{}")
        lower = cleaned.lower()
        if lower in PAUSE_MARKERS or token in PAUSE_MARKERS:
            labels.append("PAUSE")
            continue
        if cleaned in DEFAULT_FILLERS:
            labels.append("FILLER")
            continue
        if cleaned in SELF_CORRECTION_MARKERS:
            correction_window = 2
            labels.append("SELF_CORRECTION")
            continue
        if cleaned in RESTART_MARKERS:
            restart_window = 1
            labels.append("RESTART")
            continue
        if correction_window > 0:
            labels.append("SELF_CORRECTION")
            correction_window -= 1
            continue
        if restart_window > 0:
            labels.append("RESTART")
            restart_window -= 1
            continue
        labels.append("O")
    return labels


def _label_from_id(label_id: int) -> str:
    if label_id in ID_TO_LABEL:
        return ID_TO_LABEL[label_id]
    as_str = str(label_id)
    if as_str in ID_TO_LABEL:
        return ID_TO_LABEL[as_str]
    return LABELS[label_id] if 0 <= label_id < len(LABELS) else "O"


def _empty_result(tokens: list[str], labels: list[str]) -> dict[str, Any]:
    return {
        "filler_count": 0,
        "pause_count": 0,
        "self_correction_count": 0,
        "restart_count": 0,
        "hesitation_score": 0.0,
        "tokens": tokens,
        "labels": labels,
    }


def _tokenize(text: str) -> list[str]:
    return text.split()


def detect_hesitation(
    text: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    detector = HesitationDetector(model_dir=model_dir, allow_fallback=allow_fallback)
    return detector.predict(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hesitation detection for one text.")
    parser.add_argument("text", help="Input STT text")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    result = detect_hesitation(args.text, args.model_dir, allow_fallback=args.allow_fallback)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
