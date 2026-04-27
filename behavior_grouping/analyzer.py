"""
analyzer.py
웹캠 영상을 받아 비언어적 표현 패턴을 분석하고 JSON을 반환하는 메인 파이프라인
"""

from typing import Optional, Union
import cv2
import time
import numpy as np
import mediapipe as mp

from landmarks import extract_normalized_vector
from clustering import run_clustering
from output import build_output, to_json_string

mp_holistic = mp.solutions.holistic


def analyze_video(
    source: Union[int, str] = 0,
    n_clusters: int = 3,
    pca_components: int = 30,
    sample_interval: float = 0.5,
    duration: Optional[float] = None,
) -> dict:
    """
    웹캠 또는 영상 파일을 분석해 클러스터링 결과 JSON을 반환한다.

    Args:
        source: 웹캠 인덱스(0) 또는 영상 파일 경로
        n_clusters: 클러스터 수
        pca_components: PCA 축소 차원
        sample_interval: 샘플링 간격(초)
        duration: 녹화 시간(초), None이면 수동 종료
    """
    vectors = []
    timestamps = []

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")

    start_time = time.time()
    last_sample_time = -sample_interval  # 첫 프레임 즉시 샘플링

    print("분석 시작 — 'q' 키를 누르면 종료합니다." if duration is None else f"{duration}초 동안 분석합니다.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time

            # 지정 시간 초과 시 종료
            if duration is not None and elapsed >= duration:
                break

            # 샘플링 간격 체크
            if elapsed - last_sample_time < sample_interval:
                # 디스플레이만 업데이트
                cv2.imshow("면접 분석 중 (q: 종료)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            last_sample_time = elapsed

            # MediaPipe 처리
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            prev_vector = None

            vec = extract_normalized_vector(results, prev_vector)
            if vec is not None:
                prev_vector = vec
                vectors.append(vec)
                timestamps.append(round(elapsed, 3))
                print(f"\r수집: {len(vectors)}개 샘플 ({elapsed:.1f}초)", end="", flush=True)

            # 화면 표시
            annotated = _draw_overlay(frame, results, len(vectors), elapsed)
            cv2.imshow("면접 분석 중 (q: 종료)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n총 {len(vectors)}개 샘플 수집 완료")

    if len(vectors) < n_clusters:
        raise ValueError(f"샘플이 너무 적습니다 ({len(vectors)}개). 최소 {n_clusters}개 필요.")

    # 클러스터링
    print("클러스터링 중...")
    result = run_clustering(vectors)

    # JSON 생성
    output = build_output(result, timestamps, vectors)
    return output


def _draw_overlay(frame, results, sample_count: int, elapsed: float):
    """화면에 감지 상태 오버레이 표시"""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    face_ok = results.face_landmarks is not None
    pose_ok = results.pose_landmarks is not None
    lh_ok = results.left_hand_landmarks is not None
    rh_ok = results.right_hand_landmarks is not None

    items = [
        ("얼굴", face_ok),
        ("포즈", pose_ok),
        ("왼손", lh_ok),
        ("오른손", rh_ok),
    ]
    for i, (label, ok) in enumerate(items):
        color = (80, 200, 80) if ok else (80, 80, 200)
        cv2.putText(overlay, f"{label}: {'O' if ok else 'X'}", (10, 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.putText(overlay, f"샘플: {sample_count}  시간: {elapsed:.1f}s",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return overlay


if __name__ == "__main__":
    import json

    output = analyze_video(
        source=0,
        n_clusters=3,
        pca_components=30,
        sample_interval=0.5,
    )

    print("\n=== 분석 결과 ===")
    print(to_json_string(output))

    # 파일 저장
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\nresult.json 저장 완료")
