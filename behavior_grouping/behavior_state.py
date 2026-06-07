def infer_behavior_state(metrics):
    blink = metrics["blink_rate_10s"]
    gaze = metrics["gaze_stability"]
    head = metrics["head_movement_variance"]

    if blink > 25 and gaze > 0.03:
        return {
            "state": "nervous",
            "confidence": 0.82
        }

    if blink < 12 and gaze < 0.015:
        return {
            "state": "confident",
            "confidence": 0.76
        }

    return {
        "state": "neutral",
        "confidence": 0.50
    }