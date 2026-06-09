# Hesitation Detection

This folder contains the independent training and inference pipeline for STT-text hesitation detection.

## Convert Dataset

```bash
python3 training/dataset_converter.py \
  --input data/raw_hesitation.json \
  --output data/hesitation_token_classification.json
```

Canonical dataset format:

```json
[
  {
    "tokens": ["어", "저는", "음", "프로젝트를"],
    "labels": ["FILLER", "O", "FILLER", "O"]
  }
]
```

Labels:

- `O`
- `FILLER`
- `PAUSE`
- `SELF_CORRECTION`
- `RESTART`

## Train

```bash
pip install -r training/requirements_hesitation.txt

python3 training/train_hesitation_model.py \
  --dataset data/hesitation_token_classification.json \
  --output-dir models/hesitation_detector
```

Base model defaults to `monologg/koelectra-base-v3-discriminator`.

## Inference

```bash
python3 training/hesitation_detector.py \
  "어 저는 음 프로젝트를 진행했고 아니 다시 말씀드리면 사용자 경험 개선이 목표였습니다"
```

During development, before a trained model exists:

```bash
python3 training/hesitation_detector.py \
  "어 저는 음 프로젝트를 진행했고 아니 다시 말씀드리면 사용자 경험 개선이 목표였습니다" \
  --allow-fallback
```

This module is intentionally not connected to `orchestrator.py` yet.
