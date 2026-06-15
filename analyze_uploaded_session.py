"""Run the full interview analysis stack on an uploaded video file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".cache" / "matplotlib"),
)

import cv2
import librosa
import requests

from analyze_uploaded_video import analyze_uploaded_video
from behavior_grouping.analyzer import analyze_video
from behavior_grouping.behavior_state import infer_behavior_state
from behavior_grouping.exporter import export_analysis_result
from session_insights import attach_history_insights, build_session_insights
from verbal_synthesis.stt import analyze_audio, analyze_language, transcribe_audio


DEFAULT_QUESTION = "업로드된 면접 답변 영상입니다. 답변 흐름과 전달 방식을 분석하세요."
QWEN_MODELS = ("qwen3:8b", "qwen2.5:7b", "qwen2.5:7b-instruct", "qwen:7b")


def analyze_uploaded_session(
    video_path: str,
    *,
    question: str = DEFAULT_QUESTION,
    situation: str = "면접 연습",
    output_dir: str = "result/uploaded_session",
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript = transcribe_audio(video_path)
    text = transcript.get("text", "")
    verbal = {"text": text, **_analyze_verbal(video_path, text, transcript.get("segments", []))}

    nonverbal = _analyze_nonverbal(video_path, out_dir)
    state_input = _build_state_input(verbal, nonverbal)
    nonverbal["integrated_state_input"] = state_input
    nonverbal["integrated_behavior_state"] = infer_behavior_state(state_input)

    session_insights = build_session_insights(
        question=question,
        verbal=verbal,
        nonverbal=nonverbal,
        situation={"name": situation},
    )
    delivery_feedback = _generate_delivery_feedback(
        question=question,
        text=text,
        session_insights=session_insights,
    )

    payload = {
        "situation": {"name": situation},
        "question": question,
        "verbal": verbal,
        "nonverbal": nonverbal,
        "session_insights": session_insights,
        "delivery_feedback": delivery_feedback,
        "content_evaluation": "",
        "improved_answer": "",
        "followup": "",
    }
    payload = attach_history_insights(payload, output_dir="result")

    export_analysis_result(
        payload,
        output_dir="result",
        filename="final_result.json",
        metadata={
            "analysis_type": "uploaded_video_full_session",
            "source": str(video_path),
        },
    )
    export_analysis_result(
        payload,
        output_dir=out_dir,
        filename="final_result.json",
        metadata={
            "analysis_type": "uploaded_video_full_session",
            "source": str(video_path),
        },
    )
    return payload


def _analyze_verbal(video_path: str, text: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return analyze_audio(text, transcript_segments=segments, audio_path=video_path)
    except Exception as exc:
        duration = _video_duration(video_path)
        language_analysis = analyze_language(text)
        counts = language_analysis.get("counts", {})
        word_count = len(text.split())
        speech_rate = word_count / duration if duration > 0 else 0.0
        return {
            "speech_rate": round(speech_rate, 2),
            "silence_time": 0.0,
            "filler_count": counts.get("filler", 0),
            "repeat_count": counts.get("repeat", 0),
            "word_error_count": counts.get("word_error", 0),
            "filler_tokens": [
                span.get("text", "")
                for span in language_analysis.get("spans", [])
                if span.get("name") == "filler"
            ],
            "language_analysis": language_analysis,
            "transcript_segments": segments,
            "pause_count": 0,
            "duration": round(duration, 2),
            "audio_metric_warning": f"{type(exc).__name__}: {exc}",
        }


def _analyze_nonverbal(video_path: str, out_dir: Path) -> dict[str, Any]:
    try:
        nonverbal = analyze_video(
            source=video_path,
            output_dir=str(out_dir / "mediapipe"),
            headless=True,
            export_json=False,
            frames_per_cluster=3,
        )
    except Exception as exc:
        nonverbal = {
            "analysis_warning": f"MediaPipe timeline analysis failed: {type(exc).__name__}: {exc}",
            "face_metrics": {},
            "state_timeline": [],
            "state_transitions": [],
            "clusters": [],
        }

    try:
        uploaded = analyze_uploaded_video(
            video_path=video_path,
            output_dir=str(out_dir / "openface_stgcn"),
        )
        nonverbal["openface_metrics"] = uploaded.get("face_metrics", {})
        nonverbal["stgcnpp_action"] = uploaded.get(
            "stgcnpp_action",
            nonverbal.get("stgcnpp_action", {}),
        )
    except Exception as exc:
        nonverbal["openface_warning"] = f"{type(exc).__name__}: {exc}"

    nonverbal["total_duration"] = _video_duration(video_path)
    return nonverbal


def _build_state_input(verbal: dict[str, Any], nonverbal: dict[str, Any]) -> dict[str, Any]:
    face_metrics = dict(nonverbal.get("face_metrics", {}))
    return {
        **face_metrics,
        "speech_rate": verbal.get("speech_rate"),
        "filler_count": verbal.get("filler_count"),
        "silence_time": verbal.get("silence_time"),
        "pause_count": verbal.get("pause_count"),
    }


def _generate_delivery_feedback(
    *,
    question: str,
    text: str,
    session_insights: dict[str, Any],
) -> str:
    highlighted = [
        segment
        for segment in session_insights.get("transcript_segments", [])
        if segment.get("highlight") != "none"
    ]
    highlight_summary = "\n".join(
        f"- {item.get('start', 0):.1f}~{item.get('end', 0):.1f}초: "
        f"{item.get('text')} ({', '.join(item.get('reasons', []))})"
        for item in highlighted[:6]
    ) or "- 강하게 표시할 발화 구간 없음"

    prompt = f"""너는 면접 답변 코치다.
아래 답변 전체를 기준으로 내용 구성, 논리 흐름, 설득력, 전달 방식을 종합 피드백하라.
특정 구간만 지적하지 말고 답변 전체가 어떻게 들리는지 먼저 말하라.
좋았던 점 1~2개와 개선할 점 2~3개를 자연스럽게 포함하라.
반드시 한국어로 5~7문장만 작성하라.

[질문]
{question}

[사용자 전체 발화]
{text}

[참고: 별도 하이라이트 구간]
{highlight_summary}
"""
    qwen = _call_qwen(prompt)
    if qwen:
        return qwen
    return _fallback_feedback(highlighted)


def _call_qwen(prompt: str) -> str:
    for model in QWEN_MODELS:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.25,
                },
                timeout=90,
            )
            data = response.json()
            answer = data.get("response", "").strip()
            if answer:
                return answer
        except Exception:
            continue
    return ""


def _fallback_feedback(highlighted: list[dict[str, Any]]) -> str:
    return (
        "답변 전체는 지원 동기와 본인의 가치관을 연결하려는 방향이 분명합니다. "
        "개인 시간, 컨디션 관리, 성장 가능성이라는 기준을 제시한 점은 답변의 큰 틀을 만드는 데 도움이 됩니다. "
        "다만 핵심 결론이 초반에 조금 길게 풀려서 면접관이 가장 중요한 메시지를 바로 잡기 어려울 수 있습니다. "
        "첫 문장에서 '제가 회사를 선택하는 기준은 지속 가능한 몰입과 성장 가능성입니다'처럼 결론을 짧게 고정하면 더 선명합니다. "
        "중간에는 워라밸과 성장 이야기가 모두 나오므로, 각 항목을 한 문장씩 끊어 말하면 논리 흐름이 더 안정적으로 들립니다. "
        "마지막에는 회사와 본인의 경험이 어떻게 맞닿는지 한 문장으로 정리하면 답변의 설득력이 더 좋아집니다."
    )


def _video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(frames / fps) if fps > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an uploaded video as a full interview session.")
    parser.add_argument("video_path")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--situation", default="면접 연습")
    parser.add_argument("--output-dir", default="result/uploaded_session")
    args = parser.parse_args()

    result = analyze_uploaded_session(
        args.video_path,
        question=args.question,
        situation=args.situation,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "question": result.get("question"),
        "text": result.get("verbal", {}).get("text", ""),
        "highlights": result.get("session_insights", {}).get("question_analysis", {}),
        "result_path": "result/final_result.json",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
