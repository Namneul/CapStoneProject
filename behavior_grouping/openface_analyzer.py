"""
OpenFace-based facial analysis for uploaded videos.

OpenFace is treated as an external feature extractor. This module runs the
FeatureExtraction binary when available, parses the generated CSV, and converts
it into the same face_metrics shape used by the rest of the project.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GAZE_X_THRESHOLD = 0.25
DEFAULT_GAZE_Y_THRESHOLD = 0.25
DEFAULT_MIN_CONFIDENCE = 0.75
DEFAULT_BLINK_AU_THRESHOLD = 1.0


@dataclass
class OpenFaceConfig:
    feature_extraction_bin: Optional[str] = None
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    gaze_x_threshold: float = DEFAULT_GAZE_X_THRESHOLD
    gaze_y_threshold: float = DEFAULT_GAZE_Y_THRESHOLD
    blink_au_threshold: float = DEFAULT_BLINK_AU_THRESHOLD


def analyze_video_with_openface(
    video_path: str,
    output_dir: str = "result/openface",
    config: Optional[OpenFaceConfig] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> dict:
    """
    Run OpenFace FeatureExtraction on a video and return summarized face metrics.

    Raises:
        FileNotFoundError: when the video or FeatureExtraction binary is missing.
        RuntimeError: when OpenFace fails to produce a CSV.
    """
    cfg = config or OpenFaceConfig()
    csv_path = run_openface_feature_extraction(
        video_path=video_path,
        output_dir=output_dir,
        executable=cfg.feature_extraction_bin,
        extra_args=extra_args,
    )
    return summarize_openface_csv(csv_path, config=cfg)


def run_openface_feature_extraction(
    video_path: str,
    output_dir: str,
    executable: Optional[str] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> str:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    feature_extraction = find_feature_extraction_binary(executable)
    if feature_extraction is None:
        raise FileNotFoundError(
            "OpenFace FeatureExtraction 실행 파일을 찾을 수 없습니다. "
            "OPENFACE_FEATURE_EXTRACTION_BIN 환경변수에 경로를 지정하거나 PATH에 추가하세요."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in out_dir.glob("*.csv")}

    command = [
        feature_extraction,
        "-f",
        str(video),
        "-out_dir",
        str(out_dir),
        "-q",
    ]
    if extra_args:
        command.extend(str(arg) for arg in extra_args)

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenFace 분석 실패\n"
            f"command: {' '.join(command)}\n"
            f"stderr: {completed.stderr.strip()}"
        )

    expected = out_dir / f"{video.stem}.csv"
    if expected.exists():
        return str(expected)

    after = {p.resolve() for p in out_dir.glob("*.csv")}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if created:
        return str(created[0])

    raise RuntimeError(f"OpenFace CSV 결과를 찾을 수 없습니다: {out_dir}")


def find_feature_extraction_binary(explicit_path: Optional[str] = None) -> Optional[str]:
    candidates = [
        explicit_path,
        os.getenv("OPENFACE_FEATURE_EXTRACTION_BIN"),
        "FeatureExtraction",
        str(PROJECT_ROOT / "tools" / "openface" / "OpenFace_2.2.0_win_x64" / "FeatureExtraction.exe"),
        "/usr/local/bin/FeatureExtraction",
        "/opt/OpenFace/build/bin/FeatureExtraction",
        "/opt/OpenFace/OpenFace/build/bin/FeatureExtraction",
        "/app/OpenFace/build/bin/FeatureExtraction",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) if os.path.basename(candidate) == candidate else candidate
        if resolved and os.path.exists(resolved) and os.access(resolved, os.X_OK):
            return resolved
    return None


def summarize_openface_csv(csv_path: str, config: Optional[OpenFaceConfig] = None) -> dict:
    cfg = config or OpenFaceConfig()
    rows = _read_openface_rows(csv_path)
    total = len(rows)
    valid = [r for r in rows if _is_successful_frame(r, cfg.min_confidence)]
    valid_count = len(valid)

    if total == 0:
        return _empty_metrics("openface")

    timestamps = [_float(r, "timestamp") for r in valid]
    duration = _duration_seconds(rows)
    gaze_forward = [
        r for r in valid
        if abs(_float(r, "gaze_angle_x")) <= cfg.gaze_x_threshold
        and abs(_float(r, "gaze_angle_y")) <= cfg.gaze_y_threshold
    ]
    blink_flags = [_blink_flag(r, cfg.blink_au_threshold) for r in valid]
    blink_count = _count_rising_edges(blink_flags)

    smile_values = [
        (_float(r, "AU06_r") + _float(r, "AU12_r")) / 2.0
        for r in valid
    ]
    mouth_values = [
        max(_float(r, "AU25_r"), _float(r, "AU26_r"))
        for r in valid
    ]
    brow_values = [
        max(_float(r, "AU01_r"), _float(r, "AU04_r"))
        for r in valid
    ]

    yaw_values = [_pose_degrees(_float(r, "pose_Ry")) for r in valid]
    pitch_values = [_pose_degrees(_float(r, "pose_Rx")) for r in valid]
    roll_values = [_pose_degrees(_float(r, "pose_Rz")) for r in valid]
    gaze_x_values = [_float(r, "gaze_angle_x") for r in valid]
    gaze_y_values = [_float(r, "gaze_angle_y") for r in valid]

    return {
        "backend": "openface",
        "source_csv": str(csv_path),
        "sample_count": total,
        "valid_sample_count": valid_count,
        "face_detected_ratio": _ratio(valid_count, total),
        "eye_contact_ratio": _ratio(len(gaze_forward), valid_count),
        "gaze_away_ratio": _ratio(valid_count - len(gaze_forward), valid_count),
        "blink_count": int(blink_count),
        "blink_per_minute": round((blink_count / duration * 60.0) if duration > 0 else 0.0, 2),
        "gaze": {
            "avg_abs_gaze_angle_x": _mean_abs(gaze_x_values),
            "avg_abs_gaze_angle_y": _mean_abs(gaze_y_values),
            "gaze_x_variability": _std(gaze_x_values),
            "gaze_y_variability": _std(gaze_y_values),
        },
        "expression": {
            "smile_ratio": _ratio(sum(1 for v in smile_values if v >= 1.0), len(smile_values)),
            "avg_smile_score": _mean(smile_values),
            "avg_mouth_open": _mean(mouth_values),
            "avg_brow_tension": _mean(brow_values),
            "expression_stability": _expression_stability(smile_values, mouth_values, brow_values),
            "au_means": _au_means(valid),
        },
        "head_pose": {
            "avg_yaw": _mean(yaw_values),
            "avg_pitch": _mean(pitch_values),
            "avg_roll": _mean(roll_values),
            "yaw_variability": _std(yaw_values),
            "pitch_variability": _std(pitch_values),
        },
        "timeline": _build_timeline(valid, cfg),
    }


def _read_openface_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return [{k.strip(): v for k, v in row.items()} for row in reader]


def _is_successful_frame(row: dict, min_confidence: float) -> bool:
    success = _float(row, "success", default=1.0)
    confidence = _float(row, "confidence", default=1.0)
    return success >= 1.0 and confidence >= min_confidence


def _duration_seconds(rows: list[dict]) -> float:
    timestamps = [_float(r, "timestamp") for r in rows if "timestamp" in r]
    if len(timestamps) >= 2:
        return max(timestamps[-1] - timestamps[0], 0.0)
    return 0.0


def _blink_flag(row: dict, threshold: float) -> bool:
    if "AU45_c" in row and row["AU45_c"] != "":
        return _float(row, "AU45_c") >= 1.0
    if "AU45_r" in row and row["AU45_r"] != "":
        return _float(row, "AU45_r") >= threshold
    return False


def _count_rising_edges(flags: list[bool]) -> int:
    count = 0
    previous = False
    for flag in flags:
        if flag and not previous:
            count += 1
        previous = flag
    return count


def _pose_degrees(value: float) -> float:
    return math.degrees(value)


def _au_means(rows: list[dict]) -> dict:
    au_columns = [
        "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
        "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
        "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
    ]
    return {
        column: _mean([_float(r, column) for r in rows if column in r])
        for column in au_columns
        if any(column in r for r in rows)
    }


def _build_timeline(rows: list[dict], config: OpenFaceConfig) -> list[dict]:
    timeline = []
    for row in rows:
        gaze_x = _float(row, "gaze_angle_x")
        gaze_y = _float(row, "gaze_angle_y")
        timeline.append({
            "timestamp": round(_float(row, "timestamp"), 3),
            "confidence": round(_float(row, "confidence", default=1.0), 3),
            "eye_contact": (
                abs(gaze_x) <= config.gaze_x_threshold
                and abs(gaze_y) <= config.gaze_y_threshold
            ),
            "blink": _blink_flag(row, config.blink_au_threshold),
            "gaze_angle_x": round(gaze_x, 4),
            "gaze_angle_y": round(gaze_y, 4),
            "yaw": round(_pose_degrees(_float(row, "pose_Ry")), 3),
            "pitch": round(_pose_degrees(_float(row, "pose_Rx")), 3),
            "roll": round(_pose_degrees(_float(row, "pose_Rz")), 3),
            "smile_au": round((_float(row, "AU06_r") + _float(row, "AU12_r")) / 2.0, 4),
            "mouth_open_au": round(max(_float(row, "AU25_r"), _float(row, "AU26_r")), 4),
            "brow_tension_au": round(max(_float(row, "AU01_r"), _float(row, "AU04_r")), 4),
        })
    return timeline


def _empty_metrics(backend: str) -> dict:
    return {
        "backend": backend,
        "sample_count": 0,
        "valid_sample_count": 0,
        "face_detected_ratio": 0.0,
        "eye_contact_ratio": 0.0,
        "gaze_away_ratio": 0.0,
        "blink_count": 0,
        "blink_per_minute": 0.0,
        "gaze": {},
        "expression": {},
        "head_pose": {},
        "timeline": [],
    }


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator / denominator), 3) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def _mean_abs(values: list[float]) -> float:
    return round(float(np.mean(np.abs(values))), 4) if values else 0.0


def _std(values: list[float]) -> float:
    return round(float(np.std(values)), 4) if values else 0.0


def _expression_stability(*series: list[float]) -> float:
    values = [value for item in series for value in item]
    if not values:
        return 0.0
    return round(1.0 / (1.0 + float(np.std(values))), 3)
