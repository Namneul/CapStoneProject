"""
analyzer.py
웹캠 영상을 받아 비언어적 표현 패턴을 분석하고 JSON을 반환하는 메인 파이프라인
"""
import threading
from typing import Optional, Union, List
import os
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "matplotlib"),
)

import cv2
import numpy as np
import mediapipe as mp

from flask import Flask, Response
from flask_cors import CORS

flask_app = Flask(__name__)
CORS(flask_app)
latest_frame = None

@flask_app.route('/video_feed')
def video_feed():
    def gen():
        global latest_frame
        
        # Create a black placeholder frame
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Waiting for camera...", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, blank_buffer = cv2.imencode('.jpg', blank_frame)
        blank_bytes = blank_buffer.tobytes()

        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    flask_app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=start_flask, daemon=True)
flask_thread.start()


from behavior_grouping.landmarks import extract_normalized_vector, compute_ear, get_raw_nose_y
from behavior_grouping.clustering import run_clustering
from behavior_grouping.output import build_output, to_json_string
from behavior_grouping.face_metrics import FaceMetricTracker
from behavior_grouping.behavior_state import infer_behavior_state
from behavior_grouping.exporter import export_analysis_result
from behavior_grouping.stgcnpp_analyzer import analyze_pose_sequence, collect_pose_result

mp_holistic = mp.solutions.holistic
WINDOW_SIZE = 5.0


def analyze_video(
    source: Union[int, str] = 0,
    pca_components: int = 30,
    sample_interval: float = 0.5,
    duration: Optional[float] = None,
    output_dir: str = "result",
    frames_per_cluster: int = 3,
    stop_event: Optional[threading.Event] = None,
    headless: bool = False,   # True면 cv2 창 없이 실행 (영상 파일 배치 처리용)
    export_json: bool = True,
    state_window_size: float = WINDOW_SIZE,
) -> dict:
    vectors = []
    timestamps = []
    raw_frames = []
    face_tracker = FaceMetricTracker()
    stgcnpp_pose_results = []
    stgcnpp_img_shape = None
    state_timeline = []
    next_state_time = state_window_size if state_window_size > 0 else float("inf")

    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")

    # 영상 파일이면 FPS 기반 타임스탬프 사용 (웹캠은 wall-clock)
    is_file = isinstance(source, str)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    start_time = time.time()
    last_sample_time = -sample_interval
    prev_vector = None
    blink_count = 0
    prev_blink = 0.0
    prev_nose_y = 0.0
    elapsed = 0.0

    if not headless:
        print("분석 시작 — 'q' 키를 누르면 종료합니다." if duration is None else f"{duration}초 동안 분석합니다.")
    else:
        print(f"분석 시작 (headless): {source}")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        retry_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                if retry_count > 10:
                    print("카메라 프레임을 읽을 수 없습니다.")
                    break
                time.sleep(0.5)
                continue
            
            retry_count = 0 # reset on success

            elapsed = frame_idx / video_fps if is_file else time.time() - start_time
            frame_idx += 1

            if duration is not None and elapsed >= duration:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            _, _, _, current_blink = compute_ear(results)
            face_tracker.update(
                results,
                frame.shape,
                elapsed,
                current_blink,
            )
            pose_result = collect_pose_result(results, frame.shape)
            if pose_result is not None:
                stgcnpp_pose_results.append(pose_result)
                stgcnpp_img_shape = frame.shape[:2]
            while state_window_size > 0 and elapsed >= next_state_time:
                window_start = max(0.0, next_state_time - state_window_size)
                window_metrics = face_tracker.get_window_metrics(window_start, next_state_time)
                if window_metrics["samples_analyzed"] > 0:
                    state_timeline.append({
                        "time": round(next_state_time, 2),
                        "window_start": window_metrics["window_start"],
                        "window_end": window_metrics["window_end"],
                        "metrics": window_metrics,
                        "state": infer_behavior_state(window_metrics),
                    })
                next_state_time += state_window_size

            # 깜빡임 상태 갱신 (얼굴 감지 시에만)
            if results.face_landmarks is not None:
                if prev_blink == 0.0 and current_blink == 1.0:
                    blink_count += 1
                prev_blink = current_blink
            blink_rate = blink_count / elapsed if elapsed > 0.0 else 0.0

            # prev_nose_y 초기화: 첫 유효 프레임에서 현재 값으로 세팅해 첫 head_nod 오염 방지
            if prev_nose_y == 0.0 and results.face_landmarks is not None:
                prev_nose_y = get_raw_nose_y(results)

            should_sample = elapsed - last_sample_time >= sample_interval
            if not should_sample:
                annotated = _draw_overlay(frame, results, len(vectors), elapsed)
                global latest_frame
                latest_frame = annotated

                if not headless:
                    # cv2.imshow("면접 분석 중 (q: 종료)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or (stop_event is not None and stop_event.is_set()):
                        if key == ord('q') and stop_event is not None:
                            stop_event.set()
                        break
                continue

            last_sample_time = elapsed

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
            latest_frame = annotated

            if not headless:
                # cv2.imshow("면접 분석 중 (q: 종료)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or (stop_event is not None and stop_event.is_set()):
                    if key == ord('q') and stop_event is not None:
                        stop_event.set()
                    break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    # Clear stream when done
    latest_frame = None
    print(f"\n총 {len(vectors)}개 샘플 수집 완료")

    if state_window_size > 0 and elapsed > 0:
        last_window_start = max(0.0, elapsed - state_window_size)
        should_add_final = not state_timeline or elapsed - state_timeline[-1]["time"] >= sample_interval
        if should_add_final:
            window_metrics = face_tracker.get_window_metrics(last_window_start, elapsed)
            if window_metrics["samples_analyzed"] > 0:
                state_timeline.append({
                    "time": round(elapsed, 2),
                    "window_start": window_metrics["window_start"],
                    "window_end": window_metrics["window_end"],
                    "metrics": window_metrics,
                    "state": infer_behavior_state(window_metrics),
                })

    if len(vectors) < 2:
        raise ValueError(f"샘플이 너무 적습니다 ({len(vectors)}개).")

    print("클러스터링 중...")
    result = run_clustering(vectors, pca_components=pca_components)

    # 대표 프레임 저장
    representative_frames = _save_representative_frames(
        result.labels, raw_frames, timestamps, output_dir, frames_per_cluster
    )

    output = build_output(result, timestamps, vectors, representative_frames)
    output["cluster_reliability"] = _cluster_reliability(output.get("clustering_metrics", {}))

    face_metrics = face_tracker.summarize(elapsed)
    behavior_state = infer_behavior_state(face_metrics)

    output["face_metrics"] = face_metrics
    output["behavior_state"] = behavior_state
    output["stgcnpp_action"] = analyze_pose_sequence(
        stgcnpp_pose_results,
        stgcnpp_img_shape or (0, 0),
    )
    output["state_timeline"] = state_timeline
    output["state_transitions"] = _detect_state_transitions(state_timeline)

    if export_json:
        export_path = export_analysis_result(
            output,
            output_dir=output_dir,
            filename="analysis_result.json",
            metadata={
                "source": str(source),
                "sample_interval": sample_interval,
                "frames_per_cluster": frames_per_cluster,
                "state_window_size": state_window_size,
            },
        )
        print(f"\n분석 결과 JSON 저장 완료: {export_path}")

    return output


def _detect_state_transitions(state_timeline: list[dict]) -> list[dict]:
    transitions = []
    previous = None
    for item in state_timeline:
        current_state = item.get("state", {}).get("state")
        if previous is not None and current_state != previous["state"]:
            transitions.append({
                "from": previous["state"],
                "to": current_state,
                "time": item.get("time"),
                "window_start": item.get("window_start"),
                "window_end": item.get("window_end"),
            })
        previous = {
            "state": current_state,
            "time": item.get("time"),
        }
    return transitions


def _cluster_reliability(metrics: dict) -> dict:
    silhouette = float(metrics.get("silhouette_score", 0.0) or 0.0)
    davies_bouldin = float(metrics.get("davies_bouldin_index", 999.0) or 999.0)
    reliable = silhouette >= 0.15 and davies_bouldin <= 1.5
    if reliable:
        reason = "cluster separation is acceptable"
    elif silhouette < 0.15:
        reason = "silhouette_score is too low for strong behavior-pattern claims"
    else:
        reason = "davies_bouldin_index is too high for strong behavior-pattern claims"
    return {
        "reliable": reliable,
        "silhouette_score": round(silhouette, 4),
        "davies_bouldin_index": round(davies_bouldin, 4),
        "reason": reason,
    }


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
    output = analyze_video(
        source=0,
        output_dir="result",
        frames_per_cluster=3,
    )

    print("\n=== 분석 결과 ===")
    print(to_json_string(output))

    export_path = export_analysis_result(output, output_dir="result", filename="result.json")
    print(f"\n{export_path} 저장 완료")
