"""
RecruitView 데이터셋 샘플 1개로 얼굴 감지율 및 특징값 확인
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import tempfile
import cv2
import mediapipe as mp
from datasets import load_dataset

mp_holistic = mp.solutions.holistic


def test_sample():
    print("스트리밍으로 샘플 1개 로드 중...")
    from datasets import Video
    ds = load_dataset("AI4A-lab/RecruitView", streaming=True)
    ds = ds.cast_column("video", Video(decode=False))
    item = next(iter(ds["train"]))

    print("\n[데이터 구조]")
    for k, v in item.items():
        if k == "video":
            print(f"  video: {v}")
        elif k == "transcript":
            print(f"  transcript: {str(v)[:80]}...")
        elif k == "gemini_summary":
            print(f"  gemini_summary: {str(v)[:60]}...")
        else:
            print(f"  {k}: {v}")

    # 영상 → 임시 파일
    video = item["video"]
    tmp_path = None

    if isinstance(video, bytes) and len(video) > 0:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video)
            tmp_path = f.name
    elif isinstance(video, dict):
        if video.get("bytes"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video["bytes"])
                tmp_path = f.name
        elif video.get("path"):
            tmp_path = video["path"]
            print(f"  video path: {tmp_path}")

    if not tmp_path or not os.path.exists(tmp_path):
        # hf:// 경로를 fsspec으로 스트리밍 읽기
        hf_path = video.get("path") if isinstance(video, dict) else None
        if hf_path and hf_path.startswith("hf://"):
            print(f"  fsspec으로 영상 로드 중: {hf_path}")
            import fsspec
            with fsspec.open(hf_path, "rb") as f:
                video_bytes = f.read()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                tmp_path = f.name
        else:
            print(f"  영상 파일 접근 불가 — video 구조: {video}")
            return

    print("\n[MediaPipe 얼굴 감지]")
    cap = cv2.VideoCapture(tmp_path)
    total, face_detected = 0, 0

    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if holistic.process(rgb).face_landmarks:
                face_detected += 1

    cap.release()
    os.remove(tmp_path)

    print(f"  전체 프레임: {total}")
    print(f"  얼굴 감지: {face_detected} ({face_detected/total*100:.1f}%)" if total else "  프레임 없음")


if __name__ == "__main__":
    test_sample()