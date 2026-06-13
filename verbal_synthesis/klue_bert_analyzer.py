"""TensorFlow inference for the AI Hub KLUE-BERT disfluency model."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("O", "FIL-B", "REP-B", "PS-B", "WR-B")
TAG_NAMES = {
    "FIL": "filler",
    "REP": "repeat",
    "PS": "pause",
    "WR": "word_error",
}
MAX_SEQUENCE_LENGTH = 128


class KlueBertAnalyzer:
    """Load the TensorFlow checkpoint distributed with the AI Hub model."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        base_model_name: str = "klue/bert-base",
    ):
        path = Path(model_path)
        checkpoint_path = path / "cp.ckpt"
        if not checkpoint_path.with_suffix(".ckpt.index").exists():
            raise FileNotFoundError(
                f"KLUE-BERT 체크포인트를 찾을 수 없습니다: {checkpoint_path}"
            )

        try:
            import numpy as np
            import tensorflow as tf
            from transformers import BertConfig, BertTokenizer, TFBertModel
        except ImportError as exc:
            raise RuntimeError(
                "AI Hub KLUE-BERT 실행에는 tensorflow와 transformers가 필요합니다. "
                "pip install -r requirements.txt를 실행하세요."
            ) from exc

        self.np = np
        self.tf = tf
        self.confidence_threshold = confidence_threshold
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)

        class TFBertForTokenClassification(tf.keras.Model):
            def __init__(self):
                super().__init__()
                config = BertConfig.from_pretrained(base_model_name)
                self.bert = TFBertModel(config, name="bert")
                self.dropout = tf.keras.layers.Dropout(0.1)
                self.classifier = tf.keras.layers.Dense(
                    len(LABELS),
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                    name="classifier",
                )

            def call(self, inputs, training=False):
                input_ids, attention_mask, token_type_ids = inputs
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    training=training,
                )
                sequence_output = self.dropout(
                    outputs.last_hidden_state, training=training
                )
                return self.classifier(sequence_output)

        self.model = TFBertForTokenClassification()
        dummy = tf.zeros((1, MAX_SEQUENCE_LENGTH), dtype=tf.int32)
        self.model([dummy, dummy, dummy], training=False)
        status = self.model.load_weights(str(checkpoint_path))
        status.expect_partial()

    def analyze(self, text: str) -> dict[str, Any]:
        words = _split_words_with_offsets(text)
        if not words:
            return _empty_result()

        spans = []
        for batch in self._make_batches(words):
            logits = self.model(batch["inputs"], training=False)
            probabilities = self.tf.nn.softmax(logits, axis=-1).numpy()[0]

            for token_position, word in batch["first_tokens"]:
                label_index = int(self.np.argmax(probabilities[token_position]))
                label = LABELS[label_index]
                score = float(probabilities[token_position, label_index])
                if label == "O" or score < self.confidence_threshold:
                    continue

                tag = label.split("-", 1)[0]
                spans.append({
                    "tag": tag,
                    "name": TAG_NAMES[tag],
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["text"],
                    "confidence": round(score, 4),
                })

        counts = Counter(span["tag"] for span in spans)
        return {
            "backend": "aihub-klue-bert-tensorflow",
            "counts": {
                name: counts.get(tag, 0)
                for tag, name in TAG_NAMES.items()
            },
            "spans": spans,
        }

    def _make_batches(self, words: list[dict]):
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        current_ids = []
        first_tokens = []

        for word in words:
            token_ids = self.tokenizer.encode(word["text"], add_special_tokens=False)
            if not token_ids:
                continue

            if current_ids and len(current_ids) + len(token_ids) > MAX_SEQUENCE_LENGTH - 2:
                yield self._build_batch(
                    cls_id, sep_id, pad_id, current_ids, first_tokens
                )
                current_ids = []
                first_tokens = []

            if len(token_ids) > MAX_SEQUENCE_LENGTH - 2:
                token_ids = token_ids[:MAX_SEQUENCE_LENGTH - 2]

            first_tokens.append((len(current_ids) + 1, word))
            current_ids.extend(token_ids)

        if current_ids:
            yield self._build_batch(
                cls_id, sep_id, pad_id, current_ids, first_tokens
            )

    def _build_batch(
        self,
        cls_id: int,
        sep_id: int,
        pad_id: int,
        token_ids: list[int],
        first_tokens: list[tuple[int, dict]],
    ) -> dict:
        input_ids = [cls_id, *token_ids, sep_id]
        attention_mask = [1] * len(input_ids)
        padding = MAX_SEQUENCE_LENGTH - len(input_ids)
        input_ids.extend([pad_id] * padding)
        attention_mask.extend([0] * padding)
        token_type_ids = [0] * MAX_SEQUENCE_LENGTH

        return {
            "inputs": [
                self.tf.constant([input_ids], dtype=self.tf.int32),
                self.tf.constant([attention_mask], dtype=self.tf.int32),
                self.tf.constant([token_type_ids], dtype=self.tf.int32),
            ],
            "first_tokens": first_tokens,
        }


def _split_words_with_offsets(text: str) -> list[dict]:
    return [
        {
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
        }
        for match in re.finditer(r"\S+", text)
    ]


def _empty_result() -> dict[str, Any]:
    return {
        "backend": "aihub-klue-bert-tensorflow",
        "counts": {name: 0 for name in TAG_NAMES.values()},
        "spans": [],
    }
