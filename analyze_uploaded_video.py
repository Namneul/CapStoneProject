"""
Analyze an uploaded interview video file.

This is the server-style entry point: it expects a completed video file instead
of opening a webcam. For now it focuses on OpenFace facial metrics; verbal audio
and full web API integration can be layered on top of this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from behavior_grouping.openface_analyzer import (
    OpenFaceConfig,
    analyze_video_with_openface,
)


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

    result = {
        "video_path": video_path,
        "analysis_mode": "uploaded_file",
        "face_metrics": face_metrics,
    }

    with open(out_dir / "analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


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
