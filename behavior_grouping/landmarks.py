"""
landmarks.py
MediaPipe Holistic으로 랜드마크 추출 후 정규화된 특징 벡터 반환
"""

from typing import Optional, Tuple
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def compute_ear(results) -> Tuple[float, float, float, float]:
    """
    EAR(Eye Aspect Ratio) 계산.
    반환: (left_ear, right_ear, avg_ear, blink)
    blink: avg_ear < 0.2이면 1.0
    """
    if not results.face_landmarks:
        return 0.0, 0.0, 0.0, 0.0

    lm = results.face_landmarks.landmark
    # 왼눈: top=159, bottom=145, outer=33, inner=133
    # 오른눈: top=386, bottom=374, outer=263, inner=362
    left_ear = _single_ear(lm, 159, 145, 33, 133)
    right_ear = _single_ear(lm, 386, 374, 263, 362)
    avg_ear = (left_ear + right_ear) / 2.0
    blink = 1.0 if avg_ear < 0.2 else 0.0
    return float(left_ear), float(right_ear), float(avg_ear), float(blink)


def get_raw_nose_y(results) -> float:
    """FaceMesh 1번(코끝) 원본 y좌표 반환 — head_nod 계산용"""
    if results.face_landmarks:
        return float(results.face_landmarks.landmark[1].y)
    return 0.0


def extract_normalized_vector(
    results,
    prev_vector: Optional[np.ndarray] = None,
    blink_rate: float = 0.0,
    prev_nose_y: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Holistic 결과에서 정규화된 특징 벡터를 추출한다.

    정규화 기준:
    - 얼굴: 코끝(랜드마크 1번)을 원점으로, 양 광대 거리로 스케일 정규화
    - 포즈: 양 어깨 중점을 원점으로, 어깨 너비로 스케일 정규화
    - 손: 손목(0번)을 원점으로, 손 크기로 스케일 정규화

    반환값: 1D numpy 배열
      [얼굴 1404 + 포즈 99 + 왼손 63 + 오른손 63]  (current, 1629)
      + 위 변화량 (velocity, 1629)
      + EAR 4차원
      + 행동 특징 6차원
    감지 실패 시 None 반환
    """
    face = _extract_face(results)
    pose = _extract_pose(results)
    left_hand = _extract_hand(results.left_hand_landmarks)
    right_hand = _extract_hand(results.right_hand_landmarks)

    if face is None and pose is None:
        return None

    parts = []
    parts.append(face if face is not None else np.zeros(468 * 3))
    parts.append(pose if pose is not None else np.zeros(33 * 3))
    parts.append(left_hand if left_hand is not None else np.zeros(21 * 3))
    parts.append(right_hand if right_hand is not None else np.zeros(21 * 3))

    current = np.concatenate(parts).astype(np.float32)

    # velocity: 이전 벡터의 랜드마크 구간(앞쪽 len(current) 차원)과의 차이
    if prev_vector is not None:
        velocity = current - prev_vector[:len(current)]
    else:
        velocity = np.zeros_like(current)

    ear_features = np.array(compute_ear(results), dtype=np.float32)
    behavioral = _compute_behavioral_features(results, blink_rate, prev_nose_y)

    return np.concatenate([current, velocity, ear_features, behavioral])


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────

def _single_ear(lm, top_idx: int, bottom_idx: int, outer_idx: int, inner_idx: int) -> float:
    top = np.array([lm[top_idx].x, lm[top_idx].y])
    bottom = np.array([lm[bottom_idx].x, lm[bottom_idx].y])
    outer = np.array([lm[outer_idx].x, lm[outer_idx].y])
    inner = np.array([lm[inner_idx].x, lm[inner_idx].y])
    return float(np.linalg.norm(top - bottom) / (np.linalg.norm(outer - inner) + 1e-6))


def _compute_behavioral_features(
    results,
    blink_rate: float,
    prev_nose_y: float,
) -> np.ndarray:
    """
    6가지 행동 지표 계산.
    순서: [blink_rate, brow_furrow, body_lean, head_nod, shoulder_tension, hand_to_face]
    감지 실패 항목은 0.0으로 채움.
    """
    features = np.zeros(6, dtype=np.float32)

    has_face = results.face_landmarks is not None
    has_pose = results.pose_landmarks is not None

    # 0: blink_rate — analyzer.py에서 계산해 전달받음
    features[0] = float(blink_rate)

    # 1: brow_furrow — 눈썹 안쪽 거리 / 얼굴 너비
    if has_face:
        lm = results.face_landmarks.landmark
        brow_l = np.array([lm[105].x, lm[105].y])
        brow_r = np.array([lm[334].x, lm[334].y])
        brow_dist = np.linalg.norm(brow_l - brow_r)
        face_width = np.linalg.norm(
            np.array([lm[234].x, lm[234].y]) - np.array([lm[454].x, lm[454].y])
        ) + 1e-6
        features[1] = float(brow_dist / face_width)

    # 2: body_lean — 코끝 z와 어깨 중점 z의 차이 (양수: 앞으로 기울임)
    if has_face and has_pose:
        nose_z = results.face_landmarks.landmark[1].z
        pose_lm = results.pose_landmarks.landmark
        shoulder_z = (pose_lm[11].z + pose_lm[12].z) / 2.0
        features[2] = float(nose_z - shoulder_z)

    # 3: head_nod — 코끝 y의 프레임 간 변화 절댓값
    if has_face:
        nose_y = results.face_landmarks.landmark[1].y
        features[3] = float(abs(nose_y - prev_nose_y))

    # 4: shoulder_tension — 어깨 y 평균 / 엉덩이 y 평균 (값이 작을수록 어깨가 올라감)
    if has_pose:
        pose_lm = results.pose_landmarks.landmark
        shoulder_y = (pose_lm[11].y + pose_lm[12].y) / 2.0
        hip_y = (pose_lm[23].y + pose_lm[24].y) / 2.0
        features[4] = float(shoulder_y / (hip_y + 1e-6))

    # 5: hand_to_face — 손목(포즈 15, 16)이 얼굴 bbox 안에 있으면 1.0
    if has_face and has_pose:
        face_lm = results.face_landmarks.landmark
        xs = [l.x for l in face_lm]
        ys = [l.y for l in face_lm]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pose_lm = results.pose_landmarks.landmark
        for wrist_idx in (15, 16):
            wx, wy = pose_lm[wrist_idx].x, pose_lm[wrist_idx].y
            if x_min <= wx <= x_max and y_min <= wy <= y_max:
                features[5] = 1.0
                break

    return features


def _extract_face(results) -> Optional[np.ndarray]:
    if not results.face_landmarks:
        return None

    lm = results.face_landmarks.landmark
    pts = np.array([[l.x, l.y, l.z] for l in lm])  # (468, 3)

    # 코끝(1번)을 원점으로
    origin = pts[1]
    pts = pts - origin

    # 양 광대(234, 454번) 거리로 스케일 정규화
    scale = np.linalg.norm(pts[234] - pts[454]) + 1e-6
    pts = pts / scale

    return pts.flatten()


def _extract_pose(results) -> Optional[np.ndarray]:
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    pts = np.array([[l.x, l.y, l.z] for l in lm])  # (33, 3)

    # 양 어깨(11, 12번) 중점을 원점으로
    shoulder_mid = (pts[11] + pts[12]) / 2
    pts = pts - shoulder_mid

    # 어깨 너비로 스케일 정규화
    scale = np.linalg.norm(pts[11] - pts[12]) + 1e-6
    pts = pts / scale

    return pts.flatten()


def _extract_hand(hand_landmarks) -> Optional[np.ndarray]:
    if not hand_landmarks:
        return None

    lm = hand_landmarks.landmark
    pts = np.array([[l.x, l.y, l.z] for l in lm])  # (21, 3)

    # 손목(0번)을 원점으로
    pts = pts - pts[0]

    # 손 크기(손목~중지 끝 거리)로 스케일 정규화
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale

    return pts.flatten()