"""
output.py
클러스터링 결과를 Orchestrator가 받을 JSON 형식으로 변환
"""

from typing import List, Tuple, Optional
import json
import numpy as np
from clustering import ClusterResult

MIN_RATIO = 0.05
MIN_DURATION = 1.0


def build_output(
    result: ClusterResult,
    timestamps: List[float],
    raw_vectors: List[np.ndarray],
) -> dict:
    labels = result.labels
    n = len(labels)
    total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    # 노이즈 클러스터 ID 먼저 추려내기
    valid_cluster_ids = set()
    clusters = []
    for cid in range(result.n_clusters):
        mask = labels == cid
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        ratio = round(float(np.sum(mask) / n), 3)
        segments = _get_continuous_segments(indices, timestamps)
        duration = round(sum(e - s for s, e in segments), 2)

        if ratio < MIN_RATIO or duration < MIN_DURATION:
            print(f"클러스터 {cid} 제거 (ratio={ratio}, duration={duration}s)")
            continue

        valid_cluster_ids.add(cid)
        feat_means = _compute_feature_hints(indices, raw_vectors)

        clusters.append({
            "cluster_id": int(cid),
            "ratio": ratio,
            "timestamps": [[round(s, 2), round(e, 2)] for s, e in segments],
            "duration_seconds": duration,
            "feature_means": feat_means,
        })

    clusters.sort(key=lambda x: x["ratio"], reverse=True)

    # 유효한 클러스터에 속한 프레임만 frame_sequence에 포함
    raw_sequence = [
        {
            "timestamp": round(timestamps[i], 2),
            "cluster_id": int(labels[i])
        }
        for i in range(n)
        if labels[i] in valid_cluster_ids
    ]

    frame_sequence = []
    for i, frame in enumerate(raw_sequence):
        if 0 < i < len(raw_sequence) - 1:
            prev_id = raw_sequence[i - 1]["cluster_id"]
            next_id = raw_sequence[i + 1]["cluster_id"]
            if frame["cluster_id"] != prev_id and prev_id == next_id:
                frame = {"timestamp": frame["timestamp"], "cluster_id": prev_id}
        frame_sequence.append(frame)

    return {
        "total_duration": round(total_duration, 2),
        "n_clusters": len(clusters),
        "pca_explained_variance": round(result.explained_variance, 3),
        "clusters": clusters,
        "frame_sequence": frame_sequence,
    }


def _get_continuous_segments(
    indices: np.ndarray,
    timestamps: List[float],
    gap_threshold: float = 2.0,
) -> List[Tuple[float, float]]:
    if len(indices) == 0:
        return []

    segments = []
    seg_start = indices[0]
    seg_end = indices[0]

    for idx in indices[1:]:
        if timestamps[idx] - timestamps[seg_end] <= gap_threshold:
            seg_end = idx
        else:
            segments.append((timestamps[seg_start], timestamps[seg_end]))
            seg_start = idx
            seg_end = idx

    segments.append((timestamps[seg_start], timestamps[seg_end]))
    return segments


def _compute_feature_hints(
    indices: np.ndarray,
    raw_vectors: List[np.ndarray],
) -> dict:
    FACE_DIM = 468 * 3
    POSE_DIM = 33 * 3
    LH_DIM = 21 * 3
    RH_DIM = 21 * 3

    vecs = np.stack([raw_vectors[i] for i in indices])

    face_active = float(np.mean(np.any(vecs[:, :FACE_DIM] != 0, axis=1)))
    pose_active = float(np.mean(np.any(vecs[:, FACE_DIM:FACE_DIM+POSE_DIM] != 0, axis=1)))
    lh_start = FACE_DIM + POSE_DIM
    lh_active = float(np.mean(np.any(vecs[:, lh_start:lh_start+LH_DIM] != 0, axis=1)))
    rh_start = lh_start + LH_DIM
    rh_active = float(np.mean(np.any(vecs[:, rh_start:rh_start+RH_DIM] != 0, axis=1)))

    return {
        "face_detected_ratio": round(face_active, 3),
        "pose_detected_ratio": round(pose_active, 3),
        "left_hand_detected_ratio": round(lh_active, 3),
        "right_hand_detected_ratio": round(rh_active, 3),
    }


def to_json_string(output: dict) -> str:
    return json.dumps(output, ensure_ascii=False, indent=2)