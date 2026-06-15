"""
Analyze an uploaded interview video file.

This is the server-style entry point: it expects a completed video file instead
of opening a webcam. For now it focuses on OpenFace facial metrics; verbal audio
and full web API integration can be layered on top of this.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".cache" / "matplotlib"),
)

import cv2
import mediapipe as mp

from behavior_grouping.openface_analyzer import (
    OpenFaceConfig,
    analyze_video_with_openface,
)
from behavior_grouping.stgcnpp_analyzer import analyze_pose_sequence, collect_pose_result


mp_holistic = mp.solutions.holistic


def analyze_uploaded_video(
    video_path: str,
    output_dir: str = "result/uploaded",
    openface_bin: str | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    face_metrics = analyze_video_with_openface(
        video_path=video_path,
        output_dir=str(out_dir / "openface"),
        config=OpenFaceConfig(feature_extraction_bin=openface_bin),
    )
    stgcnpp_action = _analyze_stgcnpp_from_video(video_path)

    result = {
        "video_path": video_path,
        "analysis_mode": "uploaded_file",
        "face_metrics": face_metrics,
        "stgcnpp_action": stgcnpp_action,
    }

    with open(out_dir / "analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def _analyze_stgcnpp_from_video(video_path: str, sample_interval: float = 0.2) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "backend": "stgcnpp-mmaction2-unavailable",
            "status": "unavailable",
            "reason": f"video source could not be opened: {video_path}",
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    last_sample_time = -sample_interval
    pose_results = []
    img_shape = None

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = frame_idx / fps
            frame_idx += 1
            if elapsed - last_sample_time < sample_interval:
                continue

            last_sample_time = elapsed
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            pose_result = collect_pose_result(results, frame.shape)
            if pose_result is not None:
                pose_results.append(pose_result)
                img_shape = frame.shape[:2]

    cap.release()
    return analyze_pose_sequence(pose_results, img_shape or (0, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an uploaded video with OpenFace.")
    parser.add_argument("video_path", help="Path to uploaded video file")
    parser.add_argument("--output-dir", default="result/uploaded", help="Directory for JSON and OpenFace outputs")
    parser.add_argument("--openface-bin", default=None, help="Path to OpenFace FeatureExtraction binary")
    args = parser.parse_args()

    result = analyze_uploaded_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        openface_bin=args.openface_bin,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
