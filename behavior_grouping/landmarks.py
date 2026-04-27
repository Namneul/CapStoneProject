"""
landmarks.py
MediaPipe Holistic으로 랜드마크 추출 후 정규화된 특징 벡터 반환
"""

from typing import Optional
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def extract_normalized_vector(results, prev_vector=None) -> "Optional[np.ndarray]":
    """
    Holistic 결과에서 정규화된 특징 벡터를 추출한다.

    정규화 기준:
    - 얼굴: 코끝(랜드마크 1번)을 원점으로, 양 광대 거리로 스케일 정규화
    - 포즈: 양 어깨 중점을 원점으로, 어깨 너비로 스케일 정규화
    - 손: 손목(0번)을 원점으로, 손 크기로 스케일 정규화

    반환값: 1D numpy 배열 (얼굴 + 포즈 + 손 특징 concat)
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

    # 이전 프레임이 있으면 변화량 concat, 없으면 0으로
    if prev_vector is not None:
        velocity = current - prev_vector[:len(current)]
    else:
        velocity = np.zeros_like(current)

    return np.concatenate([current, velocity])


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
