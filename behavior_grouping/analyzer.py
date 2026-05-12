"""
analyzer.py
웹캠 영상을 받아 비언어적 표현 패턴을 분석하고 JSON을 반환하는 메인 파이프라인
"""
import threading
from typing import Optional, Union, List
import cv2
import os
import time
import numpy as np
import mediapipe as mp

from behavior_grouping.landmarks import extract_normalized_vector, compute_ear, get_raw_nose_y
from behavior_grouping.clustering import run_clustering
from behavior_grouping.output import build_output, to_json_string

mp_holistic = mp.solutions.holistic


def analyze_video(
    source: Union[int, str] = 0,
    pca_components: int = 30,
    sample_interval: float = 0.5,
    duration: Optional[float] = None,
    output_dir: str = "result",
    frames_per_cluster: int = 3,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    vectors = []
    timestamps = []
    raw_frames = []     # 프레임 이미지 메모리에 저장

    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")

    start_time = time.time()
    last_sample_time = -sample_interval
    prev_vector = None
    blink_count = 0
    prev_blink = 0.0
    prev_nose_y = 0.0

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

            if duration is not None and elapsed >= duration:
                break

            if elapsed - last_sample_time < sample_interval:
                cv2.imshow("면접 분석 중 (q: 종료)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or (stop_event is not None and stop_event.is_set()):
                    if key == ord('q') and stop_event is not None:
                        stop_event.set()
                    break
                continue

            last_sample_time = elapsed

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # 깜빡임 상태 갱신 (얼굴 감지 시에만)
            _, _, _, current_blink = compute_ear(results)
            if results.face_landmarks is not None:
                if prev_blink == 0.0 and current_blink == 1.0:
                    blink_count += 1
                prev_blink = current_blink
            blink_rate = blink_count / elapsed if elapsed > 0.0 else 0.0

            # prev_nose_y 초기화: 첫 유효 프레임에서 현재 값으로 세팅해 첫 head_nod 오염 방지
            if prev_nose_y == 0.0 and results.face_landmarks is not None:
                prev_nose_y = get_raw_nose_y(results)

            vec = extract_normalized_vector(
                results, prev_vector,
                blink_rate=blink_rate,
                prev_nose_y=prev_nose_y,
            )
            if vec is not None:
                prev_vector = vec
                if results.face_landmarks is not None:
                    prev_nose_y = get_raw_nose_y(results)
                vectors.append(vec)
                timestamps.append(round(elapsed, 3))
                raw_frames.append(frame.copy())     # 프레임 저장
                print(f"\r수집: {len(vectors)}개 샘플 ({elapsed:.1f}초)", end="", flush=True)

            annotated = _draw_overlay(frame, results, len(vectors), elapsed)
            cv2.imshow("면접 분석 중 (q: 종료)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or (stop_event is not None and stop_event.is_set()):
                if key == ord('q') and stop_event is not None:
                    stop_event.set()
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS에서 창이 실제로 닫히려면 필요
    print(f"\n총 {len(vectors)}개 샘플 수집 완료")

    if len(vectors) < 2:
        raise ValueError(f"샘플이 너무 적습니다 ({len(vectors)}개).")

    print("클러스터링 중...")
    result = run_clustering(vectors, pca_components=pca_components)

    # 대표 프레임 저장
    representative_frames = _save_representative_frames(
        result.labels, raw_frames, timestamps, output_dir, frames_per_cluster
    )

    output = build_output(result, timestamps, vectors, representative_frames)
    return output


def _save_representative_frames(
    labels: np.ndarray,
    raw_frames: List[np.ndarray],
    timestamps: List[float],
    output_dir: str,
    frames_per_cluster: int,
) -> dict:
    """
    각 클러스터에서 고르게 분포된 대표 프레임을 추출해 저장한다.
    반환값: { cluster_id: [이미지 경로, ...] }
    """
    n_clusters = len(set(labels))
    representative_frames = {}

    for cid in range(n_clusters):
        indices = np.where(labels == cid)[0]
        if len(indices) == 0:
            continue

        # 클러스터 내에서 고르게 분포된 인덱스 선택
        step = max(1, len(indices) // frames_per_cluster)
        selected = indices[::step][:frames_per_cluster]

        paths = []
        for idx in selected:
            filename = f"cluster_{cid}_frame_{timestamps[idx]:.2f}.jpg"
            filepath = os.path.join(output_dir, "frames", filename)
            cv2.imwrite(filepath, raw_frames[idx])
            paths.append(filepath)

        representative_frames[int(cid)] = paths
        print(f"클러스터 {cid}: 대표 프레임 {len(paths)}장 저장")

    return representative_frames


def _draw_overlay(frame, results, sample_count: int, elapsed: float):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    face_ok = results.face_landmarks is not None
    pose_ok = results.pose_landmarks is not None
    lh_ok = results.left_hand_landmarks is not None
    rh_ok = results.right_hand_landmarks is not None

    items = [("얼굴", face_ok), ("포즈", pose_ok), ("왼손", lh_ok), ("오른손", rh_ok)]
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
        output_dir="result",
        frames_per_cluster=3,
    )

    print("\n=== 분석 결과 ===")
    print(to_json_string(output))

    with open("result/result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\nresult/result.json 저장 완료")