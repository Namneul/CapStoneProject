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
HEAD_NOD_DELTA_THRESHOLD = 0.012
HEAD_NOD_PITCH_DELTA_THRESHOLD = 4.0
HEAD_POSE_MAX_ABS_DEG = 60.0
GAZE_CALIBRATION_FRAMES = 10


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
        self.roll_history = deque(maxlen=300)
        self.gaze_forward_history = deque(maxlen=300)
        self.head_nod_history = deque(maxlen=300)
        self.prev_nose_y = None
        self.prev_pitch = None
        self.prev_nose_delta = None
        self.total_frame_count = 0
        self.face_frame_count = 0
        self.frame_records = []
        self.yaw_calibration = deque(maxlen=GAZE_CALIBRATION_FRAMES)
        self.pitch_calibration = deque(maxlen=GAZE_CALIBRATION_FRAMES)

    def update(self, results, frame_shape, elapsed, current_blink):
        self.total_frame_count += 1

        if results.face_landmarks is None:
            self.frame_records.append({
                "timestamp": elapsed,
                "face_detected": False,
            })

            return

        self.face_frame_count += 1
        face = results.face_landmarks.landmark

        # blink 기록

        if current_blink == 1.0:

            self.blink_timestamps.append(elapsed)

        nose = face[1]
        prev_nose_y = self.prev_nose_y
        self.nose_history.append((nose.x, nose.y))
        self.prev_nose_y = nose.y

        # 시선 안정성용 눈동자 중심값

        gaze_center_x = _estimate_eye_center_x(face)

        self.gaze_history.append(gaze_center_x)

        yaw, pitch, roll = _valid_head_pose(estimate_head_pose(results.face_landmarks, frame_shape))
        smile_score, mouth_open, brow_tension = compute_expression_proxies(results.face_landmarks)

        self.smile_history.append(smile_score)
        self.brow_history.append(brow_tension)
        self.mouth_history.append(mouth_open)

        gaze_forward = False
        if yaw is not None:
            self.yaw_history.append(yaw)

        if pitch is not None:
            self.pitch_history.append(pitch)
        if roll is not None:
            self.roll_history.append(roll)

        if yaw is not None and pitch is not None:
            self.yaw_calibration.append(yaw)
            self.pitch_calibration.append(pitch)
            gaze_forward = _is_gaze_forward(
                yaw,
                pitch,
                self.yaw_calibration,
                self.pitch_calibration,
            )
            self.gaze_forward_history.append(gaze_forward)

        head_nod_event = False
        if prev_nose_y is not None:
            nose_delta_signed = nose.y - prev_nose_y
            nose_delta = abs(nose_delta_signed)
            pitch_delta = abs(pitch - self.prev_pitch) if pitch is not None and self.prev_pitch is not None else 0.0
            direction_changed = (
                self.prev_nose_delta is not None
                and np.sign(nose_delta_signed) != 0
                and np.sign(self.prev_nose_delta) != 0
                and np.sign(nose_delta_signed) != np.sign(self.prev_nose_delta)
            )
            head_nod_event = (
                direction_changed
                and (
                    nose_delta >= HEAD_NOD_DELTA_THRESHOLD
                    or pitch_delta >= HEAD_NOD_PITCH_DELTA_THRESHOLD
                )
            )
            self.head_nod_history.append(float(head_nod_event))
            self.prev_nose_delta = nose_delta_signed

        if pitch is not None:
            self.prev_pitch = pitch

        self.frame_records.append({
            "timestamp": elapsed,
            "face_detected": True,
            "blink_event": current_blink == 1.0,
            "gaze_center_x": gaze_center_x,
            "nose_x": nose.x,
            "nose_y": nose.y,
            "gaze_forward": gaze_forward,
            "head_nod": float(head_nod_event) if prev_nose_y is not None else 0.0,
            "smile_score": smile_score,
            "mouth_open": mouth_open,
            "brow_tension": brow_tension,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        })

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
            xs = [x for x, _ in self.nose_history]
            ys = [y for _, y in self.nose_history]
            head_movement_variance = float(np.var(xs) + np.var(ys))

        smile_mean = float(np.mean(self.smile_history)) if self.smile_history else 0.0
        smile_std = float(np.std(self.smile_history)) if self.smile_history else 0.0
        brow_tension_mean = float(np.mean(self.brow_history)) if self.brow_history else 0.0
        mouth_open_mean = float(np.mean(self.mouth_history)) if self.mouth_history else 0.0
        yaw_variance = float(np.var(self.yaw_history)) if len(self.yaw_history) > 5 else 0.0
        pitch_variance = float(np.var(self.pitch_history)) if len(self.pitch_history) > 5 else 0.0
        roll_variance = float(np.var(self.roll_history)) if len(self.roll_history) > 5 else 0.0
        pose_valid_ratio = _ratio(len(self.gaze_forward_history), self.face_frame_count)
        eye_contact_ratio = _ratio(sum(self.gaze_forward_history), len(self.gaze_forward_history))
        if pose_valid_ratio < 0.3:
            eye_contact_ratio = None
        gaze_away_ratio = round(1.0 - eye_contact_ratio, 3) if eye_contact_ratio is not None else None
        head_nod = float(np.mean(self.head_nod_history)) if self.head_nod_history else 0.0
        face_detected_ratio = _ratio(self.face_frame_count, self.total_frame_count)

        return {
            "blink_rate_10s": blink_rate_10s,
            "blink_per_minute": blink_rate_10s,
            "face_detected_ratio": face_detected_ratio,
            "head_pose_valid_ratio": pose_valid_ratio,
            "eye_contact_ratio": eye_contact_ratio,
            "gaze_away_ratio": gaze_away_ratio,
            "gaze_stability": gaze_stability,
            "head_movement_variance": head_movement_variance,
            "head_nod": head_nod,
            "smile_mean": smile_mean,
            "smile_std": smile_std,
            "brow_tension_mean": brow_tension_mean,
            "mouth_open_mean": mouth_open_mean,
            "yaw_variance": yaw_variance,
            "pitch_variance": pitch_variance,
            "roll_variance": roll_variance,
            "frames_analyzed": len(self.gaze_history),
        }

    def get_window_metrics(self, start_time: float, end_time: float) -> dict:
        records = [
            r for r in self.frame_records
            if start_time <= r["timestamp"] <= end_time
        ]
        face_records = [r for r in records if r.get("face_detected")]
        duration = max(0.001, end_time - start_time)

        blink_count = sum(
            1 for t in self.blink_timestamps
            if start_time <= t <= end_time
        )
        blink_rate_5s = blink_count * (60.0 / duration)

        gaze_values = [
            r["gaze_center_x"] for r in face_records
            if r.get("gaze_center_x") is not None
        ]
        nose_values = [
            (r["nose_x"], r["nose_y"]) for r in face_records
            if r.get("nose_x") is not None and r.get("nose_y") is not None
        ]
        yaw_values = [r["yaw"] for r in face_records if r.get("yaw") is not None]
        pitch_values = [r["pitch"] for r in face_records if r.get("pitch") is not None]
        roll_values = [r["roll"] for r in face_records if r.get("roll") is not None]
        smile_values = [r["smile_score"] for r in face_records]
        brow_values = [r["brow_tension"] for r in face_records]
        mouth_values = [r["mouth_open"] for r in face_records]
        gaze_forward_values = [r["gaze_forward"] for r in face_records if "gaze_forward" in r]
        head_nod_values = [r["head_nod"] for r in face_records if "head_nod" in r]

        head_movement_variance = 0.0
        if len(nose_values) > 1:
            xs = [x for x, _ in nose_values]
            ys = [y for _, y in nose_values]
            head_movement_variance = float(np.var(xs) + np.var(ys))

        pose_valid_ratio = _ratio(len(gaze_forward_values), len(face_records))
        eye_contact_ratio = _ratio(sum(gaze_forward_values), len(gaze_forward_values))
        if pose_valid_ratio < 0.3:
            eye_contact_ratio = None
        gaze_away_ratio = round(1.0 - eye_contact_ratio, 3) if eye_contact_ratio is not None else None

        return {
            "window_start": round(start_time, 2),
            "window_end": round(end_time, 2),
            "window_duration": round(duration, 2),
            "blink_rate_5s": blink_rate_5s,
            "blink_rate_10s": blink_rate_5s,
            "blink_per_minute": blink_rate_5s,
            "face_detected_ratio": _ratio(len(face_records), len(records)),
            "head_pose_valid_ratio": pose_valid_ratio,
            "eye_contact_ratio": eye_contact_ratio,
            "gaze_away_ratio": gaze_away_ratio,
            "gaze_stability": float(np.std(gaze_values)) if len(gaze_values) > 1 else 0.0,
            "head_movement_variance": head_movement_variance,
            "head_nod": float(np.mean(head_nod_values)) if head_nod_values else 0.0,
            "smile_mean": float(np.mean(smile_values)) if smile_values else 0.0,
            "smile_std": float(np.std(smile_values)) if len(smile_values) > 1 else 0.0,
            "brow_tension_mean": float(np.mean(brow_values)) if brow_values else 0.0,
            "mouth_open_mean": float(np.mean(mouth_values)) if mouth_values else 0.0,
            "yaw_variance": float(np.var(yaw_values)) if len(yaw_values) > 1 else 0.0,
            "pitch_variance": float(np.var(pitch_values)) if len(pitch_values) > 1 else 0.0,
            "roll_variance": float(np.var(roll_values)) if len(roll_values) > 1 else 0.0,
            "frames_analyzed": len(face_records),
            "samples_analyzed": len(records),
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


def _estimate_eye_center_x(landmarks) -> float:
    # FaceMesh with iris refinement has 478 landmarks; Holistic usually has 468.
    if len(landmarks) > 473:
        return float((landmarks[468].x + landmarks[473].x) / 2.0)

    eye_indices = (33, 133, 159, 145, 263, 362, 386, 374)
    return float(np.mean([landmarks[idx].x for idx in eye_indices]))


def _valid_head_pose(
    pose: tuple[Optional[float], Optional[float], Optional[float]],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    yaw, pitch, roll = pose
    values = (yaw, pitch, roll)
    if any(value is None for value in values):
        return None, None, None
    if any(abs(float(value)) > HEAD_POSE_MAX_ABS_DEG for value in values):
        return None, None, None
    return yaw, pitch, roll


def _is_gaze_forward(
    yaw: float,
    pitch: float,
    yaw_calibration,
    pitch_calibration,
) -> bool:
    if len(yaw_calibration) >= 3 and len(pitch_calibration) >= 3:
        base_yaw = float(np.median(yaw_calibration))
        base_pitch = float(np.median(pitch_calibration))
        return (
            abs(yaw - base_yaw) <= GAZE_YAW_THRESHOLD
            and abs(pitch - base_pitch) <= GAZE_PITCH_THRESHOLD
        )
    return abs(yaw) <= GAZE_YAW_THRESHOLD and abs(pitch) <= GAZE_PITCH_THRESHOLD


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
