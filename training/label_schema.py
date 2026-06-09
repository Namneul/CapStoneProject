"""Shared label schema for hesitation token classification."""

LABELS = [
    "O",
    "FILLER",
    "PAUSE",
    "SELF_CORRECTION",
    "RESTART",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

HESITATION_LABELS = [label for label in LABELS if label != "O"]
