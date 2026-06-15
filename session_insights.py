"""Session-level timeline insights for interview analysis.

This layer connects transcript time ranges with nonverbal state windows so the
UI can explain when a candidate's delivery changed during a question.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_LIMIT = 12
FILLER_WORDS = ("음", "어", "저기", "그니까", "그러니까", "뭐", "약간")
TRANSCRIPT_CHECK_TERMS = {
    "존용": "문맥상 '존중'으로 말하려던 부분일 수 있어 발음 또는 STT 전사 확인이 필요합니다.",
    "근형": "문맥상 '균형'으로 말하려던 부분일 수 있어 받침/모음 발음이 흐려졌을 가능성이 있습니다.",
    "운전한": "문맥상 '온전한'으로 말하려던 부분일 수 있어 초성/모음 발음 확인이 필요합니다.",
    "실망 경험에 싸움": "문맥상 자연스럽지 않은 전사 결과라 발음 뭉개짐 또는 말 더듬기 확인이 필요합니다.",
    "자전거를 담아": "문맥상 자연스럽지 않은 전사 결과라 발음 또는 문장 연결 확인이 필요합니다.",
}


def build_session_insights(
    *,
    question: str,
    verbal: dict[str, Any],
    nonverbal: dict[str, Any],
    situation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transcript_segments = _normalize_segments(
        verbal.get("transcript_segments"),
        verbal.get("text", ""),
        verbal.get("duration"),
    )
    state_timeline = _normalize_timeline(nonverbal.get("state_timeline", []))
    transitions = _normalize_transitions(nonverbal.get("state_transitions", []))
    language_spans = verbal.get("language_analysis", {}).get("spans", [])

    highlighted_segments = [
        _annotate_segment(segment, state_timeline, language_spans)
        for segment in transcript_segments
    ]
    change_points = _build_change_points(transitions, highlighted_segments, state_timeline)
    focus_windows = _build_focus_windows(state_timeline, highlighted_segments)

    return {
        "version": 1,
        "situation": (situation or {}).get("name"),
        "question": question,
        "transcript_segments": highlighted_segments,
        "state_timeline": state_timeline,
        "change_points": change_points,
        "focus_windows": focus_windows,
        "question_analysis": {
            "label": _question_label(change_points, highlighted_segments),
            "first_nervous_time": _first_nervous_time(state_timeline),
            "answer_duration": _segment_end(highlighted_segments),
            "summary": _session_summary(change_points, highlighted_segments),
        },
        "vlm_review_candidates": _vlm_candidates(focus_windows, nonverbal),
    }


def attach_history_insights(
    payload: dict[str, Any],
    *,
    output_dir: str | Path = "result",
    history_filename: str = "session_history.json",
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / history_filename

    history = _read_history(history_path)
    entry = _history_entry(payload)
    if entry:
        history.append(entry)
        history = history[-HISTORY_LIMIT:]
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    payload["session_history"] = {
        "count": len(history),
        "habit_summary": summarize_habits(history),
        "history_path": str(history_path),
    }
    return payload


def summarize_habits(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "patterns": [],
            "summary": "아직 누적 세션이 없어 반복 습관을 판단할 수 없습니다.",
        }

    patterns = []
    frequent_reasons = _count_reasons(history)
    for reason, count in frequent_reasons[:5]:
        if count >= 2:
            pattern_type, label = _habit_label(reason)
            patterns.append({
                "type": pattern_type,
                "label": label,
                "description": _habit_description(reason, count),
                "strength": round(count / len(history), 2),
            })

    if not patterns:
        summary = "반복된 추임새나 특정 행동 신호는 아직 뚜렷하지 않습니다. 세션이 더 쌓이면 공통 패턴을 더 안정적으로 볼 수 있습니다."
    else:
        summary = "최근 세션에서 반복된 추임새, 반복 표현, 시선/긴장 관련 행동 신호를 습관 후보로 정리했습니다."

    return {
        "patterns": patterns,
        "summary": summary,
    }


def _normalize_segments(
    segments: list[dict[str, Any]] | None,
    text: str,
    duration: float | None,
) -> list[dict[str, Any]]:
    if segments:
        normalized = []
        cursor = 0
        for index, segment in enumerate(segments):
            segment_text = str(segment.get("text", "")).strip()
            start_char = text.find(segment_text, cursor) if segment_text else -1
            if start_char < 0:
                start_char = cursor
            end_char = start_char + len(segment_text)
            cursor = max(cursor, end_char)
            normalized.append({
                "id": index,
                "start": _round(segment.get("start", 0.0)),
                "end": _round(segment.get("end", segment.get("start", 0.0))),
                "text": segment_text,
                "char_start": start_char,
                "char_end": end_char,
            })
        return normalized

    safe_duration = float(duration or 0.0)
    return [{
        "id": 0,
        "start": 0.0,
        "end": _round(safe_duration),
        "text": text,
        "char_start": 0,
        "char_end": len(text),
    }]


def _normalize_timeline(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline = []
    for item in items:
        state = item.get("state", {})
        scores = state.get("scores", {})
        timeline.append({
            "start": _round(item.get("window_start", 0.0)),
            "end": _round(item.get("window_end", item.get("time", 0.0))),
            "time": _round(item.get("time", item.get("window_end", 0.0))),
            "state": state.get("state", "neutral"),
            "confidence": _round(state.get("confidence", 0.0)),
            "measurement_quality": _round(state.get("measurement_quality", 0.0)),
            "scores": {
                "focused": _round(scores.get("focused", 0.0)),
                "engagement": _round(scores.get("engagement", 0.0)),
                "nervous": _round(scores.get("nervous", 0.0)),
                "confidence": _round(scores.get("confidence", 0.0)),
            },
            "evidence": state.get("evidence", [])[:5],
        })
    return timeline


def _normalize_transitions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "from": item.get("from"),
            "to": item.get("to"),
            "time": _round(item.get("time", 0.0)),
            "start": _round(item.get("window_start", 0.0)),
            "end": _round(item.get("window_end", item.get("time", 0.0))),
        }
        for item in items
    ]


def _annotate_segment(
    segment: dict[str, Any],
    timeline: list[dict[str, Any]],
    language_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    windows = [_window for _window in timeline if _overlaps(segment, _window)]
    scores = _average_scores(windows)
    cues = _segment_cues(segment, language_spans, scores)
    level = _highlight_level(scores, cues)
    details = _segment_feedback_details(segment, cues, scores, windows)
    return {
        **segment,
        "state_scores": scores,
        "highlight": level,
        "reasons": cues,
        "feedback_details": details,
        "overlapping_states": [
            {
                "start": item["start"],
                "end": item["end"],
                "state": item["state"],
                "scores": item["scores"],
            }
            for item in windows
        ],
    }


def _segment_feedback_details(
    segment: dict[str, Any],
    cues: list[str],
    scores: dict[str, float],
    windows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    text = segment.get("text", "")
    details = []
    for cue in cues:
        if cue == "추임새 포함":
            fillers = [word for word in FILLER_WORDS if word in text]
            details.append({
                "title": "추임새 사용",
                "evidence": f"구간 안에서 {', '.join(fillers) if fillers else '추임새'} 표현이 감지되었습니다.",
                "feedback": "바로 다음 말을 찾기 어렵다면 추임새를 넣기보다 0.5초 정도 멈추고 핵심 명사부터 다시 시작하는 편이 더 안정적으로 들립니다.",
            })
        elif cue == "반복 표현 포함":
            details.append({
                "title": "반복 표현",
                "evidence": "같은 단어나 비슷한 표현이 짧은 간격으로 반복되었습니다.",
                "feedback": "반복된 표현은 말 더듬기처럼 들릴 수 있으니, 문장을 다시 시작할 때는 앞 문장을 끊고 더 짧은 문장으로 바꾸는 연습이 좋습니다.",
            })
        elif cue == "긴장 행동 신호":
            details.append({
                "title": "긴장 행동 신호",
                "evidence": f"이 구간과 겹치는 비언어 창의 nervous 평균이 {scores.get('nervous', 0):.2f}로 기준보다 높았습니다.",
                "feedback": "긴장 신호가 올라가는 구간에서는 문장을 길게 이어가기보다 결론 한 문장, 근거 한 문장으로 나눠 말하면 안정적으로 보입니다.",
            })
        elif cue == "자신감 점수 하락":
            details.append({
                "title": "자신감 신호 저하",
                "evidence": f"confidence 평균이 {scores.get('confidence', 0):.2f}로 낮게 나타났습니다.",
                "feedback": "시작 문장에 '저는 ~라고 생각합니다'처럼 주장을 먼저 고정하면 뒤 문장의 흔들림이 줄어듭니다.",
            })
        elif cue == "시선/집중 안정도 저하":
            states = ", ".join(
                f"{item.get('start'):.1f}~{item.get('end'):.1f}초 focused {item.get('scores', {}).get('focused', 0):.2f}"
                for item in windows[:2]
            )
            details.append({
                "title": "시선/집중 안정도",
                "evidence": states or "겹치는 비언어 분석 창에서 focused 점수가 낮게 나타났습니다.",
                "feedback": "핵심 문장을 말할 때는 카메라 또는 면접관 방향을 1~2초 유지하고, 생각할 때만 시선을 잠깐 이동하는 식으로 리듬을 나누세요.",
            })
    return details


def _segment_cues(
    segment: dict[str, Any],
    language_spans: list[dict[str, Any]],
    scores: dict[str, float],
) -> list[str]:
    cues = []
    if scores.get("nervous", 0.0) >= 0.65:
        cues.append("긴장 행동 신호")
    if scores.get("confidence", 1.0) <= 0.4:
        cues.append("자신감 점수 하락")

    segment_spans = [
        span for span in language_spans
        if _char_overlaps(
            segment.get("char_start", 0),
            segment.get("char_end", 0),
            span.get("start", 0),
            span.get("end", 0),
        )
    ]
    if any(span.get("name") == "filler" for span in segment_spans):
        cues.append("추임새 포함")
    if any(span.get("name") == "repeat" for span in segment_spans):
        cues.append("반복 표현 포함")
    cues.extend(_textual_cues(segment.get("text", "")))

    if (
        scores.get("focused", 1.0) <= 0.32
        and any(cue in cues for cue in ("긴장 행동 신호", "자신감 점수 하락"))
    ):
        cues.append("시선/집중 안정도 저하")
    return _unique(cues)


def _highlight_level(scores: dict[str, float], cues: list[str]) -> str:
    nervous = scores.get("nervous", 0.0)
    confidence = scores.get("confidence", 1.0)
    speech_cues = {"추임새 포함", "반복 표현 포함"}
    has_speech_issue = any(cue in speech_cues for cue in cues)
    if nervous >= 0.75 or (nervous >= 0.68 and confidence <= 0.4):
        return "high"
    if has_speech_issue or nervous >= 0.65 or len(cues) >= 2:
        return "medium"
    return "none"


def _textual_cues(text: str) -> list[str]:
    cues = []
    compact = " ".join(text.split())
    tokens = compact.split()
    if any(word in tokens or word in compact for word in FILLER_WORDS):
        cues.append("추임새 포함")
    if _has_repeated_expression(tokens):
        cues.append("반복 표현 포함")
    return _unique(cues)


def _has_repeated_expression(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens[:-1]):
        if token == tokens[index + 1]:
            return True
        if index + 2 < len(tokens) and token == tokens[index + 2]:
            return True
    text = " ".join(tokens)
    repeated_phrases = ("위에서는 위에서", "일과 삶", "지원하도록 지원", "삶 일과 삶")
    return any(phrase in text for phrase in repeated_phrases)


def _build_change_points(
    transitions: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    change_points = []
    for transition in transitions:
        related_segment = _segment_at(segments, transition["time"])
        related_window = _window_at(timeline, transition["time"])
        if not _is_meaningful_transition(transition, related_window):
            continue
        change_points.append({
            **transition,
            "kind": "state_transition",
            "segment_id": related_segment.get("id") if related_segment else None,
            "segment_text": related_segment.get("text") if related_segment else "",
            "label": _transition_label(transition),
            "evidence": _change_evidence(transition, related_window),
            "feedback": _change_feedback(transition),
        })

    previous = None
    for item in timeline:
        scores = item.get("scores", {})
        if previous:
            prev_scores = previous.get("scores", {})
            nervous_delta = scores.get("nervous", 0.0) - prev_scores.get("nervous", 0.0)
            confidence_delta = scores.get("confidence", 0.0) - prev_scores.get("confidence", 0.0)
            if nervous_delta >= 0.18 or confidence_delta <= -0.18:
                related_segment = _segment_at(segments, item["time"])
                change_points.append({
                    "kind": "score_shift",
                    "from": previous.get("state"),
                    "to": item.get("state"),
                    "time": item["time"],
                    "start": item["start"],
                    "end": item["end"],
                    "segment_id": related_segment.get("id") if related_segment else None,
                    "segment_text": related_segment.get("text") if related_segment else "",
                    "label": "긴장 상승" if nervous_delta >= 0.18 else "자신감 하락",
                    "nervous_delta": _round(nervous_delta),
                    "confidence_delta": _round(confidence_delta),
                    "evidence": (
                        f"이전 분석 창 대비 nervous {nervous_delta:+.2f}, "
                        f"confidence {confidence_delta:+.2f} 변화가 있었습니다."
                    ),
                    "feedback": "이 변화가 나타난 발화 앞뒤에서는 문장 길이를 줄이고, 핵심 단어를 먼저 말하는 방식이 좋습니다.",
                })
        previous = item

    return sorted(change_points, key=lambda item: item.get("time", 0.0))


def _build_focus_windows(
    timeline: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    windows = []
    for item in timeline:
        scores = item.get("scores", {})
        nervous = scores.get("nervous", 0.0)
        confidence = scores.get("confidence", 0.0)
        if nervous < 0.58 and confidence > 0.45:
            continue
        related = [
            {
                "id": segment["id"],
                "text": segment["text"],
                "highlight": segment["highlight"],
            }
            for segment in segments
            if _overlaps(segment, item)
        ]
        windows.append({
            "start": item["start"],
            "end": item["end"],
            "state": item["state"],
            "scores": scores,
            "reasons": _window_reasons(item),
            "segments": related,
        })
    return windows


def _window_reasons(item: dict[str, Any]) -> list[str]:
    scores = item.get("scores", {})
    reasons = []
    if scores.get("nervous", 0.0) >= 0.58:
        reasons.append("긴장 신호가 평균보다 높음")
    if scores.get("confidence", 1.0) <= 0.45:
        reasons.append("자신감 신호가 낮음")
    if scores.get("focused", 1.0) <= 0.45:
        reasons.append("집중/시선 관련 신호가 약함")
    return reasons


def _vlm_candidates(focus_windows: list[dict[str, Any]], nonverbal: dict[str, Any]) -> list[dict[str, Any]]:
    frames = []
    for cluster in nonverbal.get("clusters", []):
        frames.extend(cluster.get("representative_frames", []) or [])
    return [
        {
            "start": window["start"],
            "end": window["end"],
            "reason": ", ".join(window["reasons"]),
            "frame_refs": frames[:3],
        }
        for window in focus_windows[:5]
    ]


def _history_entry(payload: dict[str, Any]) -> dict[str, Any] | None:
    insights = payload.get("session_insights") or {}
    if not insights:
        return None
    question_analysis = insights.get("question_analysis", {})
    focus_windows = insights.get("focus_windows", [])
    highlighted_reasons = [
        reason
        for segment in insights.get("transcript_segments", [])
        if segment.get("highlight") != "none"
        for reason in segment.get("reasons", [])
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "situation": insights.get("situation"),
        "question": insights.get("question"),
        "first_nervous_time": question_analysis.get("first_nervous_time"),
        "highlight_count": sum(
            1 for item in insights.get("transcript_segments", [])
            if item.get("highlight") != "none"
        ),
        "focus_window_count": len(focus_windows),
        "reasons": highlighted_reasons + [
            reason
            for window in focus_windows
            for reason in window.get("reasons", [])
        ],
    }


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _count_reasons(history: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts = {}
    for item in history:
        for reason in set(item.get("reasons", [])):
            counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def _habit_label(reason: str) -> tuple[str, str]:
    if "추임새" in reason:
        return "filler", "추임새 반복"
    if "반복 표현" in reason:
        return "repeat", "반복 표현 습관"
    if "집중" in reason or "시선" in reason:
        return "nonverbal_focus", "시선/집중 신호 반복"
    if "긴장" in reason:
        return "nonverbal_tension", "긴장 행동 신호 반복"
    if "자신감" in reason:
        return "nonverbal_confidence", "자신감 저하 신호 반복"
    return "behavior_cue", "특정 행동 신호 반복"


def _habit_description(reason: str, count: int) -> str:
    if "추임새" in reason:
        return f"'음', '어', '그니까' 같은 filler words 사용이 {count}개 세션에서 반복되었습니다."
    if "반복 표현" in reason:
        return f"같은 단어나 표현을 짧은 간격으로 다시 말하는 패턴이 {count}개 세션에서 반복되었습니다."
    if "시선" in reason or "집중" in reason:
        return f"핵심 문장을 말할 때 시선 또는 집중 안정도가 낮아지는 행동이 {count}개 세션에서 반복되었습니다."
    if "긴장" in reason:
        return f"답변 중 특정 구간에서 긴장 신호가 올라가는 행동 패턴이 {count}개 세션에서 반복되었습니다."
    if "자신감" in reason:
        return f"말끝이나 설명 전환 구간에서 자신감 신호가 낮아지는 패턴이 {count}개 세션에서 반복되었습니다."
    return f"{reason} 패턴이 {count}개 세션에서 반복되었습니다."


def _unique(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _question_label(change_points: list[dict[str, Any]], segments: list[dict[str, Any]]) -> str:
    if change_points:
        first = change_points[0]
        label = first.get("label", "상태 변화")
        marker = "이" if str(label).endswith("전환") else "가"
        return f"{first.get('time', 0):.1f}초 부근부터 {label}{marker} 관찰됨"
    highlighted = [segment for segment in segments if segment.get("highlight") != "none"]
    if highlighted:
        first = highlighted[0]
        return f"{first.get('start', 0):.1f}초 발화 구간에서 피드백 후보가 관찰됨"
    return "큰 상태 변화 없이 비교적 안정적으로 진행됨"


def _session_summary(change_points: list[dict[str, Any]], segments: list[dict[str, Any]]) -> str:
    if change_points:
        first = change_points[0]
        text = first.get("segment_text") or "해당 발화 구간"
        return f"{first.get('time', 0):.1f}초 전후, '{text}' 부분과 함께 {first.get('label')} 신호가 나타났습니다."
    highlighted = [segment for segment in segments if segment.get("highlight") != "none"]
    if highlighted:
        return f"총 {len(highlighted)}개 발화 구간이 피드백 후보로 표시되었습니다."
    return "이번 답변에서는 강한 긴장 상승 구간이 뚜렷하게 잡히지 않았습니다."


def _first_nervous_time(timeline: list[dict[str, Any]]) -> float | None:
    for item in timeline:
        if item.get("scores", {}).get("nervous", 0.0) >= 0.62:
            return item.get("time")
    return None


def _segment_at(segments: list[dict[str, Any]], time_value: float) -> dict[str, Any] | None:
    for segment in segments:
        if segment.get("start", 0.0) <= time_value <= segment.get("end", 0.0):
            return segment
    return None


def _window_at(timeline: list[dict[str, Any]], time_value: float) -> dict[str, Any] | None:
    for item in timeline:
        if item.get("start", 0.0) <= time_value <= item.get("end", 0.0):
            return item
    return None


def _average_scores(windows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("focused", "engagement", "nervous", "confidence")
    if not windows:
        return {key: 0.0 for key in keys}
    return {
        key: _round(sum(item.get("scores", {}).get(key, 0.0) for item in windows) / len(windows))
        for key in keys
    }


def _transition_label(transition: dict[str, Any]) -> str:
    to_state = transition.get("to")
    if to_state == "nervous":
        return "긴장 상태로 전환"
    if to_state == "confident":
        return "자신감 회복"
    if to_state == "focused":
        return "집중 상태로 전환"
    return f"{transition.get('from')}에서 {to_state}로 전환"


def _is_meaningful_transition(
    transition: dict[str, Any],
    window: dict[str, Any] | None,
) -> bool:
    to_state = transition.get("to")
    if to_state in {"nervous", "confident"}:
        return True
    if to_state == "engaged" and window:
        return window.get("scores", {}).get("engagement", 0.0) >= 0.68
    if not window:
        return False
    scores = window.get("scores", {})
    return scores.get("nervous", 0.0) >= 0.68 or scores.get("confidence", 1.0) <= 0.4


def _change_evidence(transition: dict[str, Any], window: dict[str, Any] | None) -> str:
    if not window:
        return "앞뒤 구간의 행동 흐름이 달라져 변화점으로 표시했습니다."
    scores = window.get("scores", {})
    cues = []
    if scores.get("focused", 1.0) <= 0.35:
        cues.append("시선이나 머리 자세가 안정적으로 유지되지 않았습니다")
    if scores.get("nervous", 0.0) >= 0.62:
        cues.append("긴장으로 보이는 깜빡임/표정/시선 신호가 함께 올라왔습니다")
    if scores.get("confidence", 1.0) <= 0.45:
        cues.append("말하는 안정감이 앞뒤 구간보다 낮게 잡혔습니다")
    if not cues:
        cues.append("앞뒤 구간보다 행동 흐름이 달라졌습니다")
    return f"{transition.get('start', 0):.0f}~{transition.get('end', 0):.0f}초 구간에서 " + " 그리고 ".join(cues[:2]) + "."


def _change_feedback(transition: dict[str, Any]) -> str:
    to_state = transition.get("to")
    if to_state == "nervous":
        return "이 구간은 문장을 짧게 끊고 결론을 먼저 말하면 더 안정적으로 보입니다."
    if to_state == "confident":
        return "이때의 말하기 리듬을 답변 앞부분에도 쓰면 시작이 더 안정됩니다."
    if to_state == "engaged":
        return "구체 사례를 말하면서 흐름이 좋아진 구간입니다. 이 방식은 유지해도 좋습니다."
    return "흔들리는 구간은 더 짧은 문장으로 정리하는 편이 좋습니다."


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return max(left.get("start", 0.0), right.get("start", 0.0)) <= min(
        left.get("end", 0.0),
        right.get("end", 0.0),
    )


def _char_overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def _segment_end(segments: list[dict[str, Any]]) -> float:
    return max((segment.get("end", 0.0) for segment in segments), default=0.0)


def _round(value: Any, digits: int = 2) -> float:
    try:
        return round(float(value or 0.0), digits)
    except (TypeError, ValueError):
        return 0.0
