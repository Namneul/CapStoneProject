"""
Face-focused interview metrics.

This module keeps facial analysis separate from clustering so the web/API layer
can consume stable metrics, and the implementation can later be swapped with a
dedicated gaze or expression model.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import math
import cv2
import numpy as np

from behavior_grouping.landmarks import compute_ear


EYE_CLOSED_EAR = 0.2
EYE_OPEN_EAR = 0.24
GAZE_YAW_THRESHOLD = 15.0
GAZE_PITCH_THRESHOLD = 14.0


@dataclass
class FaceFrameMetric:
    timestamp: float
    face_detected: bool
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    blink_event: bool = False
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    gaze_forward: bool = False
    smile_score: float = 0.0
    mouth_open: float = 0.0
    brow_tension: float = 0.0


@dataclass
class FaceMetricTracker:

    def __init__(self):

        self.blink_timestamps = deque(maxlen=300)

        self.gaze_history = deque(maxlen=300)

        self.nose_history = deque(maxlen=300)

        self.smile_history = deque(maxlen=300)
        self.brow_history = deque(maxlen=300)
        self.mouth_history = deque(maxlen=300)
        self.yaw_history = deque(maxlen=300)
        self.pitch_history = deque(maxlen=300)

    def update(self, results, frame_shape, elapsed, current_blink):

        if results.face_landmarks is None:

            return

        face = results.face_landmarks.landmark

        # blink 기록

        if current_blink == 1.0:

            self.blink_timestamps.append(elapsed)

        # 코 위치 추적

        nose = face[1]

        self.nose_history.append((nose.x, nose.y))

        # 시선 안정성용 눈동자 중심값

        left_eye = face[468]   # iris

        right_eye = face[473]

        gaze_center_x = (left_eye.x + right_eye.x) / 2

        self.gaze_history.append(gaze_center_x)

        yaw, pitch, roll = estimate_head_pose(results.face_landmarks, frame_shape)
        smile_score, mouth_open, brow_tension = compute_expression_proxies(results.face_landmarks)

        self.smile_history.append(smile_score)
        self.brow_history.append(brow_tension)
        self.mouth_history.append(mouth_open)

        if yaw is not None:
            self.yaw_history.append(yaw)

        if pitch is not None:
            self.pitch_history.append(pitch)

    def summarize(self, current_time):

        recent_blinks = [

            t for t in self.blink_timestamps

            if current_time - t <= 10

        ]

        blink_rate_10s = len(recent_blinks) * 6

        gaze_stability = 0.0

        if len(self.gaze_history) > 5:

            gaze_stability = float(np.std(self.gaze_history))

        head_movement_variance = 0.0

        if len(self.nose_history) > 5:

            ys = [y for _, y in self.nose_history]

            head_movement_variance = float(np.var(ys))

        smile_mean = float(np.mean(self.smile_history)) if self.smile_history else 0.0
        smile_std = float(np.std(self.smile_history)) if self.smile_history else 0.0
        brow_tension_mean = float(np.mean(self.brow_history)) if self.brow_history else 0.0
        mouth_open_mean = float(np.mean(self.mouth_history)) if self.mouth_history else 0.0
        yaw_variance = float(np.var(self.yaw_history)) if len(self.yaw_history) > 5 else 0.0
        pitch_variance = float(np.var(self.pitch_history)) if len(self.pitch_history) > 5 else 0.0

        return {
            "blink_rate_10s": blink_rate_10s,
            "gaze_stability": gaze_stability,
            "head_movement_variance": head_movement_variance,
            "smile_mean": smile_mean,
            "smile_std": smile_std,
            "brow_tension_mean": brow_tension_mean,
            "mouth_open_mean": mouth_open_mean,
            "yaw_variance": yaw_variance,
            "pitch_variance": pitch_variance,
        }


def estimate_head_pose(face_landmarks, frame_shape: tuple[int, int, int]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    h, w = frame_shape[:2]
    lm = face_landmarks.landmark

    image_points = np.array([
        _point_2d(lm[1], w, h),    # nose tip
        _point_2d(lm[152], w, h),  # chin
        _point_2d(lm[33], w, h),   # left eye outer corner
        _point_2d(lm[263], w, h),  # right eye outer corner
        _point_2d(lm[61], w, h),   # left mouth corner
        _point_2d(lm[291], w, h),  # right mouth corner
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    focal_length = float(w)
    camera_matrix = np.array([
        [focal_length, 0.0, w / 2.0],
        [0.0, focal_length, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rotation_vec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles
    return _normalize_angle(yaw), _normalize_angle(pitch), _normalize_angle(roll)


def compute_expression_proxies(face_landmarks) -> tuple[float, float, float]:
    lm = face_landmarks.landmark
    face_width = _dist(lm[234], lm[454]) + 1e-6
    mouth_width = _dist(lm[61], lm[291])
    mouth_open = _dist(lm[13], lm[14]) / face_width
    mouth_corner_lift = ((lm[61].y + lm[291].y) / 2.0) - ((lm[13].y + lm[14].y) / 2.0)
    smile_score = (mouth_width / face_width) + max(0.0, mouth_corner_lift * 3.0)

    brow_gap = _dist(lm[105], lm[334]) / face_width
    brow_tension = max(0.0, 1.0 - brow_gap)
    return float(smile_score), float(mouth_open), float(brow_tension)


def _point_2d(point, width: int, height: int) -> tuple[float, float]:
    return float(point.x * width), float(point.y * height)


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _normalize_angle(value: float) -> float:
    if value > 180.0:
        return value - 360.0
    if value < -180.0:
        return value + 360.0
    return float(value)


def _ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator / denominator), 3) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def _std(values: list[float]) -> float:
    return round(float(np.std(values)), 4) if values else 0.0
