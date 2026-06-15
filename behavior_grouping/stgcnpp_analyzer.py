"""Optional STGCN++ action recognition for nonverbal analysis.

The AI Hub bundle ships a fine-tuned MMAction2 STGCN++ checkpoint.  This module
keeps that dependency optional: the rest of the interview pipeline still works
when mmengine/mmcv/mmaction are not installed, while environments that have the
OpenMMLab stack can add skeleton action scores to the output JSON.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR))
DEFAULT_MMACTION_ROOT = (
    PROJECT_ROOT
    / "models"
    / "1.모델"
    / "1.모델소스코드"
    / "모델2_비언어적_STGCN++"
    / "mmaction2"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "1.모델"
    / "2.AI학습모델파일"
    / "모델2_비언어적_STGCN++"
    / "checkpoint.pth"
)

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
MEDIAPIPE_TO_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def collect_pose_result(results, frame_shape: tuple[int, int, int]) -> dict | None:
    """Convert one MediaPipe Holistic frame into MMAction2 skeleton input."""
    if results.pose_landmarks is None:
        return None

    height, width = frame_shape[:2]
    landmarks = results.pose_landmarks.landmark
    keypoints = np.zeros((1, len(MEDIAPIPE_TO_COCO), 2), dtype=np.float32)
    scores = np.zeros((1, len(MEDIAPIPE_TO_COCO)), dtype=np.float32)

    for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO):
        lm = landmarks[mp_idx]
        keypoints[0, coco_idx] = (lm.x * width, lm.y * height)
        scores[0, coco_idx] = float(getattr(lm, "visibility", 1.0) or 0.0)

    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
    }


def unavailable_result(reason: str) -> dict[str, Any]:
    return {
        "backend": "stgcnpp-mmaction2-unavailable",
        "status": "unavailable",
        "reason": reason,
    }


def analyze_pose_sequence(
    pose_results: list[dict],
    img_shape: tuple[int, int],
    checkpoint_path: str | os.PathLike[str] | None = None,
    mmaction_root: str | os.PathLike[str] | None = None,
    device: str | None = None,
    topk: int = 5,
) -> dict[str, Any]:
    """Run STGCN++ on collected COCO-format skeleton frames."""
    if os.getenv("STGCNPP_ENABLED", "1").lower() in {"0", "false", "no"}:
        return unavailable_result("STGCNPP_ENABLED is disabled")

    if len(pose_results) < 2:
        return unavailable_result(f"not enough pose frames: {len(pose_results)}")

    checkpoint = Path(
        checkpoint_path or os.getenv("STGCNPP_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    )
    root = Path(mmaction_root or os.getenv("STGCNPP_MMACTION_ROOT", str(DEFAULT_MMACTION_ROOT)))

    if not checkpoint.exists():
        return unavailable_result(f"checkpoint not found: {checkpoint}")
    if not root.exists():
        return unavailable_result(f"mmaction2 source not found: {root}")

    try:
        model = _load_model(str(checkpoint), str(root), device)
        from mmaction.apis import inference_skeleton
    except Exception as exc:  # dependency errors are common on local laptops
        return unavailable_result(f"{type(exc).__name__}: {exc}")

    try:
        result = inference_skeleton(model, pose_results, img_shape)
        scores = result.pred_score.detach().cpu().numpy().astype(float)
    except Exception as exc:
        return unavailable_result(f"inference failed: {type(exc).__name__}: {exc}")

    order = np.argsort(scores)[::-1][:topk]
    top_predictions = [
        {
            "class_index": int(idx),
            "label": f"class_{int(idx):02d}",
            "score": round(float(scores[idx]), 4),
        }
        for idx in order
    ]

    return {
        "backend": "stgcnpp-mmaction2",
        "status": "ok",
        "num_pose_frames": len(pose_results),
        "img_shape": [int(img_shape[0]), int(img_shape[1])],
        "predicted_class": top_predictions[0]["class_index"] if top_predictions else None,
        "confidence": top_predictions[0]["score"] if top_predictions else 0.0,
        "top_predictions": top_predictions,
        "note": "Class names were not included in the model bundle, so labels are numeric.",
    }


@lru_cache(maxsize=1)
def _load_model(checkpoint_path: str, mmaction_root: str, device: str | None):
    root = str(Path(mmaction_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch
    from mmengine import Config
    from mmengine.logging.history_buffer import HistoryBuffer
    from mmaction.apis import init_recognizer

    # Import modules so their registry entries exist before Config composition.
    import mmaction.datasets.transforms  # noqa: F401
    import mmaction.models  # noqa: F401

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            HistoryBuffer,
            np.core.multiarray._reconstruct,
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            type(np.dtype("float64")),
            type(np.dtype("float32")),
            type(np.dtype("int64")),
            type(np.dtype("int32")),
            getattr,
        ])

    if device is None:
        device = os.getenv("STGCNPP_DEVICE")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = Config(
        dict(
            default_scope="mmaction",
            model=dict(
                type="RecognizerGCN",
                backbone=dict(
                    type="STGCN",
                    gcn_adaptive="init",
                    gcn_with_res=True,
                    tcn_type="mstcn",
                    graph_cfg=dict(layout="coco", mode="spatial"),
                ),
                cls_head=dict(type="GCNHead", num_classes=20, in_channels=256),
            ),
            test_pipeline=[
                dict(type="GenSkeFeat", dataset="coco", feats=["jm"]),
                dict(
                    type="UniformSampleFrames",
                    clip_len=10,
                    num_clips=10,
                    test_mode=True,
                ),
                dict(type="PoseDecode"),
                dict(type="FormatGCNInput", num_person=2),
                dict(type="PackActionInputs"),
            ],
        )
    )
    return init_recognizer(config, checkpoint_path, device=device)
