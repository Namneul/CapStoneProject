"""
Interpret face metrics as higher-level behavior state scores.

Basis used for the current rule-based priors:
- Naim et al. (2018) reports that "focused" is strongly related to
  overall interview ratings, and that multimodal cues outperform a
  single modality.
- Martin-Raugh et al. (2022) reports stronger meta-analytic associations
  for eye contact and head movement than for smiling alone.

The numeric ranges below are still calibration defaults. They should be
retuned with labeled project data before being treated as final thresholds.
"""

from __future__ import annotations

from typing import Any


CALIBRATION = {
    "blink_rate_10s": {"low": 8.0, "typical": 16.0, "high": 28.0},
    "gaze_stability": {"stable": 0.012, "unstable": 0.045},
    "head_movement_variance": {"stable": 0.00025, "unstable": 0.003},
    "pose_variance": {"stable": 20.0, "unstable": 180.0},
    "smile_mean": {"low": 0.18, "high": 0.42},
    "smile_std": {"stable": 0.035, "unstable": 0.16},
    "brow_tension_mean": {"low": 0.48, "high": 0.72},
    "mouth_open_mean": {"low": 0.018, "high": 0.07},
    "speech_rate": {"low": 2.0, "high": 4.0},
    "filler_count": {"low": 2.0, "high": 8.0},
    "silence_time": {"low": 2.0, "high": 6.0},
}

PAPER_PRIORS = {
    "focused": "Naim2018: focused trait correlates strongly with overall interview rating.",
    "eye_contact": "MartinRaugh2022: eye contact is one of the strongest dynamic nonverbal interview cues.",
    "head_movement": "MartinRaugh2022: head movement/nodding is positive, but current tracker only measures instability.",
    "smile": "Naim2018: smile helps interview outcomes/friendliness; MartinRaugh2022 finds smiling-alone effects unstable.",
}


def infer_behavior_state(metrics: dict[str, Any]) -> dict[str, Any]:
    blink = _metric(metrics, "blink_rate_10s", "blink_per_minute")
    gaze = _metric(metrics, "gaze_stability", ("gaze", "gaze_x_variability"))
    head = _metric(metrics, "head_movement_variance")
    smile = _metric(metrics, "smile_mean", ("expression", "avg_smile_score"))
    smile_std = _metric(metrics, "smile_std")
    brow = _metric(metrics, "brow_tension_mean", ("expression", "avg_brow_tension"))
    mouth_open = _metric(metrics, "mouth_open_mean", ("expression", "avg_mouth_open"))
    yaw_var = _metric(metrics, "yaw_variance", ("head_pose", "yaw_variability"))
    pitch_var = _metric(metrics, "pitch_variance", ("head_pose", "pitch_variability"))
    roll_var = _metric(metrics, "roll_variance", ("head_pose", "roll_variability"))
    frames = int(_metric(metrics, "frames_analyzed", "valid_sample_count", "sample_count"))
    eye_contact_ratio = _optional_number(metrics.get("eye_contact_ratio"))
    gaze_away_ratio = _optional_number(metrics.get("gaze_away_ratio"))
    eye_contact_missing = (
        "eye_contact_ratio" in metrics
        and metrics.get("eye_contact_ratio") is None
        and "gaze_away_ratio" in metrics
        and metrics.get("gaze_away_ratio") is None
    )
    head_nod = _optional_number(metrics.get("head_nod"))
    speech_rate = _optional_number(metrics.get("speech_rate"))
    filler_count = _optional_number(metrics.get("filler_count"))
    silence_time = _optional_number(metrics.get("silence_time"))
    pause_count = _optional_number(metrics.get("pause_count"))

    gaze_stability_score = 1.0 - _scale(
        gaze,
        CALIBRATION["gaze_stability"]["stable"],
        CALIBRATION["gaze_stability"]["unstable"],
    )
    eye_contact_score = _eye_contact_score(
        eye_contact_ratio=eye_contact_ratio,
        gaze_away_ratio=gaze_away_ratio,
        fallback_score=gaze_stability_score,
        allow_fallback=not eye_contact_missing,
    )
    head_control_score = 1.0 - _scale(
        head,
        CALIBRATION["head_movement_variance"]["stable"],
        CALIBRATION["head_movement_variance"]["unstable"],
    )
    head_movement_score = _head_movement_score(head_nod, head_control_score)
    pose_stability_score = 1.0 - _scale(
        yaw_var + pitch_var + roll_var,
        CALIBRATION["pose_variance"]["stable"],
        CALIBRATION["pose_variance"]["unstable"],
    )
    blink_regularity_score = 1.0 - _distance_from_range(
        blink,
        CALIBRATION["blink_rate_10s"]["low"],
        CALIBRATION["blink_rate_10s"]["high"],
        tolerance=16.0,
    )
    smile_presence_score = _scale(
        smile,
        CALIBRATION["smile_mean"]["low"],
        CALIBRATION["smile_mean"]["high"],
    )
    expression_stability_score = 1.0 - _scale(
        smile_std,
        CALIBRATION["smile_std"]["stable"],
        CALIBRATION["smile_std"]["unstable"],
    )
    brow_tension_score = _scale(
        brow,
        CALIBRATION["brow_tension_mean"]["low"],
        CALIBRATION["brow_tension_mean"]["high"],
    )
    mouth_tension_score = _scale(
        mouth_open,
        CALIBRATION["mouth_open_mean"]["low"],
        CALIBRATION["mouth_open_mean"]["high"],
    )
    blink_stress_score = _scale(
        blink,
        CALIBRATION["blink_rate_10s"]["typical"],
        CALIBRATION["blink_rate_10s"]["high"],
    )
    verbal_scores = _verbal_scores(
        speech_rate=speech_rate,
        filler_count=filler_count,
        silence_time=silence_time,
        pause_count=pause_count,
    )

    focused = _weighted_mean(
        (eye_contact_score, 0.40),
        (gaze_stability_score, 0.25),
        (head_control_score, 0.20),
        (pose_stability_score, 0.10),
        (blink_regularity_score, 0.05),
    )
    engagement = _weighted_mean(
        (focused, 0.35),
        (head_movement_score, 0.20),
        (smile_presence_score, 0.15),
        (expression_stability_score, 0.15),
        (1.0 - mouth_tension_score, 0.15),
    )
    if verbal_scores is not None:
        engagement = _weighted_mean((engagement, 0.70), (verbal_scores["fluency"], 0.30))

    nervous = _weighted_mean(
        (blink_stress_score, 0.25),
        (1.0 - eye_contact_score, 0.20),
        (1.0 - gaze_stability_score, 0.20),
        (1.0 - head_control_score, 0.15),
        (brow_tension_score, 0.20),
        (mouth_tension_score, 0.10),
    )
    if verbal_scores is not None:
        nervous = _weighted_mean((nervous, 0.75), (verbal_scores["disfluency"], 0.25))

    confidence = _weighted_mean(
        (focused, 0.35),
        (engagement, 0.25),
        (smile_presence_score, 0.15),
        (1.0 - nervous, 0.25),
    )

    scores = {
        "focused": round(focused, 2),
        "engagement": round(engagement, 2),
        "nervous": round(nervous, 2),
        "confidence": round(confidence, 2),
    }
    state, state_confidence = _choose_state(scores)
    measurement_quality = _measurement_quality(metrics, eye_contact_missing)
    state_confidence = min(state_confidence, measurement_quality)

    return {
        "state": state,
        "confidence": round(state_confidence, 2),
        "measurement_quality": round(measurement_quality, 2),
        "scores": scores,
        "evidence": _build_evidence(
            scores=scores,
            blink=blink,
            gaze=gaze,
            head=head,
            eye_contact_score=eye_contact_score,
            eye_contact_ratio=eye_contact_ratio,
            head_nod=head_nod,
            smile=smile,
            brow=brow,
            mouth_open=mouth_open,
            pose_variance=yaw_var + pitch_var + roll_var,
            frames=frames,
            verbal_scores=verbal_scores,
            speech_rate=speech_rate,
            filler_count=filler_count,
            silence_time=silence_time,
        ),
    }


def _choose_state(scores: dict[str, float]) -> tuple[str, float]:
    if scores["nervous"] >= 0.62 and scores["nervous"] >= scores["confidence"]:
        return "nervous", scores["nervous"]
    if scores["confidence"] >= 0.62:
        return "confident", scores["confidence"]
    if scores["focused"] >= 0.68 and scores["engagement"] >= 0.50:
        return "focused", (scores["focused"] + scores["engagement"]) / 2.0
    if scores["engagement"] >= 0.62:
        return "engaged", scores["engagement"]
    return "neutral", max(0.50, max(scores.values()) * 0.8)


def _build_evidence(
    *,
    scores: dict[str, float],
    blink: float,
    gaze: float,
    head: float,
    eye_contact_score: float,
    eye_contact_ratio: float | None,
    head_nod: float | None,
    smile: float,
    brow: float,
    mouth_open: float,
    pose_variance: float,
    frames: int,
    verbal_scores: dict[str, float] | None,
    speech_rate: float | None,
    filler_count: float | None,
    silence_time: float | None,
) -> list[str]:
    evidence = [
        f"paper_basis: {PAPER_PRIORS['focused']}",
        f"paper_basis: {PAPER_PRIORS['eye_contact']}",
        f"paper_basis: {PAPER_PRIORS['head_movement']}",
        f"focused={scores['focused']:.2f} prioritizes eye contact, gaze stability, and controlled head pose",
        f"engagement={scores['engagement']:.2f} combines focus, head movement cue, smile, and expression stability",
        f"nervous={scores['nervous']:.2f} reflects blink, low eye-contact proxy, gaze/head instability, brow tension, and mouth opening",
        f"confidence={scores['confidence']:.2f} balances focus, engagement, smile, and low nervousness",
    ]

    if eye_contact_ratio is not None:
        evidence.append(f"eye_contact_ratio observed: {eye_contact_ratio:.2f}")
    else:
        evidence.append(f"eye_contact_proxy_from_gaze: {eye_contact_score:.2f}")

    if head_nod is not None:
        evidence.append(f"head_nod cue used: {head_nod:.3f}")
    else:
        evidence.append("head_nod not available; head movement currently treated as controlled stability, not positive nodding")

    if blink >= CALIBRATION["blink_rate_10s"]["high"]:
        evidence.append(f"blink_rate_10s high: {blink:.1f}")
    elif blink <= CALIBRATION["blink_rate_10s"]["low"]:
        evidence.append(f"blink_rate_10s low: {blink:.1f}")

    if gaze >= CALIBRATION["gaze_stability"]["unstable"]:
        evidence.append(f"gaze_stability unstable: {gaze:.4f}")
    else:
        evidence.append(f"gaze_stability stable/moderate: {gaze:.4f}")

    if head >= CALIBRATION["head_movement_variance"]["unstable"]:
        evidence.append(f"head_movement_variance high: {head:.5f}")

    if pose_variance >= CALIBRATION["pose_variance"]["unstable"]:
        evidence.append(f"head_pose_variance high: {pose_variance:.2f}")

    if smile >= CALIBRATION["smile_mean"]["high"]:
        evidence.append(f"smile_mean supportive: {smile:.2f}")
    if brow >= CALIBRATION["brow_tension_mean"]["high"]:
        evidence.append(f"brow_tension_mean elevated: {brow:.2f}")
    if mouth_open >= CALIBRATION["mouth_open_mean"]["high"]:
        evidence.append(f"mouth_open_mean elevated: {mouth_open:.3f}")
    if frames < 10:
        evidence.append(f"low frame count: {frames}")

    if verbal_scores is not None:
        evidence.append(f"verbal_fluency={verbal_scores['fluency']:.2f} from speech_rate/fillers/silence")
        if speech_rate is not None:
            evidence.append(f"speech_rate observed: {speech_rate:.2f}")
        if filler_count is not None:
            evidence.append(f"filler_count observed: {filler_count:.0f}")
        if silence_time is not None:
            evidence.append(f"silence_time observed: {silence_time:.2f}")

    return evidence


def _measurement_quality(metrics: dict[str, Any], eye_contact_missing: bool) -> float:
    quality = 1.0
    face_ratio = _optional_number(metrics.get("face_detected_ratio"))
    if face_ratio is not None:
        quality *= _clamp(face_ratio, 0.0, 1.0)

    pose_ratio = _optional_number(metrics.get("head_pose_valid_ratio"))
    if pose_ratio is not None and pose_ratio < 0.5:
        quality *= 0.7

    if eye_contact_missing:
        quality *= 0.85

    frames = _optional_number(metrics.get("frames_analyzed"))
    if frames is not None and frames < 10:
        quality *= 0.8

    return max(0.3, quality)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(metrics: dict[str, Any], *keys: str | tuple[str, str]) -> float:
    for key in keys:
        if isinstance(key, tuple):
            parent, child = key
            parent_value = metrics.get(parent)
            if isinstance(parent_value, dict) and child in parent_value:
                return _number(parent_value.get(child))
        elif key in metrics:
            return _number(metrics.get(key))
    return 0.0


def _optional_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _eye_contact_score(
    *,
    eye_contact_ratio: float | None,
    gaze_away_ratio: float | None,
    fallback_score: float,
    allow_fallback: bool = True,
) -> float:
    if eye_contact_ratio is not None:
        return _clamp(eye_contact_ratio)
    if gaze_away_ratio is not None:
        return 1.0 - _clamp(gaze_away_ratio)
    return fallback_score if allow_fallback else 0.5


def _head_movement_score(head_nod: float | None, head_control_score: float) -> float:
    if head_nod is None:
        return head_control_score
    nod_score = _scale(head_nod, 0.01, 0.08)
    return _weighted_mean((nod_score, 0.55), (head_control_score, 0.45))


def _verbal_scores(
    *,
    speech_rate: float | None,
    filler_count: float | None,
    silence_time: float | None,
    pause_count: float | None,
) -> dict[str, float] | None:
    if all(value is None for value in (speech_rate, filler_count, silence_time, pause_count)):
        return None

    speech_score = 0.5
    if speech_rate is not None:
        speech_score = 1.0 - _distance_from_range(
            speech_rate,
            CALIBRATION["speech_rate"]["low"],
            CALIBRATION["speech_rate"]["high"],
            tolerance=2.0,
        )

    filler_score = 0.5
    if filler_count is not None:
        filler_score = 1.0 - _scale(
            filler_count,
            CALIBRATION["filler_count"]["low"],
            CALIBRATION["filler_count"]["high"],
        )

    silence_score = 0.5
    if silence_time is not None:
        silence_score = 1.0 - _scale(
            silence_time,
            CALIBRATION["silence_time"]["low"],
            CALIBRATION["silence_time"]["high"],
        )

    pause_score = 0.5
    if pause_count is not None:
        pause_score = 1.0 - _scale(pause_count, 2.0, 10.0)

    fluency = _weighted_mean(
        (speech_score, 0.35),
        (filler_score, 0.30),
        (silence_score, 0.25),
        (pause_score, 0.10),
    )
    return {
        "fluency": fluency,
        "disfluency": 1.0 - fluency,
    }


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _distance_from_range(value: float, low: float, high: float, tolerance: float) -> float:
    if low <= value <= high:
        return 0.0
    if tolerance <= 0.0:
        return 1.0
    distance = low - value if value < low else value - high
    return _clamp(distance / tolerance)


def _weighted_mean(*items: tuple[float, float]) -> float:
    total_weight = sum(weight for _, weight in items)
    if total_weight <= 0.0:
        return 0.0
    return _clamp(sum(_clamp(value) * weight for value, weight in items) / total_weight)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
