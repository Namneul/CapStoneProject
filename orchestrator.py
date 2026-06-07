"""
orchestrator.py
언어적 + 비언어적 분석을 종합해서 최종 피드백 생성
"""

import json
import os
import threading
import requests

from verbal_synthesis.stt import record_audio, speech_to_text, analyze_audio
from verbal_synthesis.LLM import (
    generate_question, generate_followup,
    evaluate_answer_content, generate_improved_answer,
)
from behavior_grouping.analyzer import analyze_video


SITUATION_CONTEXT = {
    "1": {
        "name": "취업/입학 면접",
        "focus": "눈맞춤, 자세 안정성, 말의 명확성, 자신감",
        "avoid": "시선 회피, 과도한 제스처, 추임새, 긴 침묵",
    },
    "2": {
        "name": "발표/프레젠테이션",
        "focus": "전달력, 제스처 활용, 목소리 속도, 청중과의 눈맞춤",
        "avoid": "단조로운 톤, 너무 빠른 말 속도, 몸 흔들림",
    },
    "3": {
        "name": "소개팅/대인관계",
        "focus": "자연스러운 미소, 편안한 자세, 적절한 리액션",
        "avoid": "경직된 자세, 긴장된 표정, 과도한 침묵",
    },
    "4": {
        "name": "기타",
        "focus": "상황에 맞는 전반적인 커뮤니케이션",
        "avoid": "부자연스러운 비언어적 표현",
    },
}


def call_llm_orchestrator(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
        }
    )
    data = response.json()
    if "response" not in data:
        return "생성 실패"
    return data["response"].strip()


def select_situation() -> dict:
    print("\n어떤 상황을 연습하시겠습니까?")
    print("1. 취업 / 입학 면접")
    print("2. 발표 / 프레젠테이션")
    print("3. 소개팅 / 대인관계")
    print("4. 기타 (직접 입력)")
    choice = input("번호를 선택하세요: ").strip()

    if choice in ("1", "2", "3"):
        return SITUATION_CONTEXT[choice]

    # 4번 또는 잘못된 입력 → 직접 입력 후 LLM으로 focus/avoid 생성
    custom = input("상황을 직접 입력해주세요: ").strip()
    if not custom:
        return SITUATION_CONTEXT["4"]

    print("상황 분석 중...")
    raw = call_llm_orchestrator(
        f"아래 상황에서 커뮤니케이션할 때 중점적으로 신경 써야 할 항목과 피해야 할 행동을 각각 짧게 나열해라.\n"
        f"형식: 중점 항목: ...\n피해야 할 행동: ...\n\n상황: {custom}"
    )

    focus, avoid = SITUATION_CONTEXT["4"]["focus"], SITUATION_CONTEXT["4"]["avoid"]
    for line in raw.splitlines():
        if line.startswith("중점 항목:"):
            focus = line.split(":", 1)[1].strip()
        elif line.startswith("피해야 할 행동:"):
            avoid = line.split(":", 1)[1].strip()

    return {"name": custom, "focus": focus, "avoid": avoid}

EVALUATION_STANDARDS = {
    "취업/입학 면접": {
        "speech_rate": {"good": (2.5, 3.5), "too_fast": 3.5, "too_slow": 2.5},
        "silence": {"good": 3.0, "too_long": 5.0},
        "filler": {"good": 3, "too_many": 6},
        "gaze": "면접관과 70% 이상 눈맞춤 유지 권장",
        "posture": "상체를 약간 앞으로 기울여 관심 표현",
        "gesture": "절제된 제스처, 과도한 손동작 지양",
    },
    "발표/프레젠테이션": {
        "speech_rate": {"good": (2.8, 4.0), "too_fast": 4.0, "too_slow": 2.8},
        "silence": {"good": 2.0, "too_long": 4.0},
        "filler": {"good": 2, "too_many": 5},
        "gaze": "청중 전체를 고루 바라보며 시선 분산",
        "posture": "똑바로 서서 자신감 있는 자세 유지",
        "gesture": "내용을 강조하는 적극적인 제스처 권장",
    },
    "소개팅/대인관계": {
        "speech_rate": {"good": (2.0, 3.0), "too_fast": 3.5, "too_slow": 1.5},
        "silence": {"good": 4.0, "too_long": 8.0},
        "filler": {"good": 5, "too_many": 10},
        "gaze": "자연스러운 눈맞춤, 응시보다는 편안한 시선",
        "posture": "편안하고 자연스러운 자세",
        "gesture": "자연스러운 손동작, 과도하지 않게",
    },
    "기타": {
        "speech_rate": {"good": (2.0, 3.5), "too_fast": 3.5, "too_slow": 2.0},
        "silence": {"good": 3.0, "too_long": 6.0},
        "filler": {"good": 4, "too_many": 8},
        "gaze": "상황에 맞는 자연스러운 시선 처리",
        "posture": "상황에 맞는 편안한 자세",
        "gesture": "자연스러운 제스처",
    },
}


def orchestrate(situation: dict, question: str, verbal: dict, nonverbal: dict) -> str:
    std = EVALUATION_STANDARDS.get(situation['name'], EVALUATION_STANDARDS["기타"])
    speech_rate = verbal['speech_rate']
    good_range = std['speech_rate']['good']

    # 말 속도 평가
    if speech_rate > std['speech_rate']['too_fast']:
        rate_eval = f"말 속도가 권장 범위({good_range[0]}~{good_range[1]}단어/초)보다 빠름"
    elif speech_rate < std['speech_rate']['too_slow']:
        rate_eval = f"말 속도가 권장 범위({good_range[0]}~{good_range[1]}단어/초)보다 느림"
    else:
        rate_eval = "말 속도가 권장 범위 내로 적절함"

    # 침묵 평가
    if verbal['silence_time'] > std['silence']['too_long']:
        silence_eval = f"침묵 시간이 권장 기준({std['silence']['too_long']}초)을 초과하여 긺"
    else:
        silence_eval = "침묵 시간 적절"

    # 추임새 평가
    if verbal['filler_count'] > std['filler']['too_many']:
        filler_eval = f"추임새가 권장 기준({std['filler']['too_many']}회)을 초과하여 과다"
    else:
        filler_eval = "추임새 횟수 적절"

    # 비언어 클러스터 요약
    cluster_summary = ""
    for c in nonverbal.get("clusters", []):
        cid = c["cluster_id"]
        ratio = c["ratio"]
        fm = c["feature_means"]

        hand_activity = (fm['left_hand_detected_ratio'] + fm['right_hand_detected_ratio']) / 2
        if hand_activity > 0.6:
            hand_desc = "손 제스처 활발"
        elif hand_activity > 0.3:
            hand_desc = "손 제스처 간헐적"
        else:
            hand_desc = "손 제스처 거의 없음"

        face_desc = "얼굴 정면 유지" if fm['face_detected_ratio'] > 0.8 else "얼굴 자주 이탈"

        cluster_summary += f"- 패턴 {cid} (전체 {ratio*100:.0f}% 차지): {face_desc}, {hand_desc}\n"

    face_metrics = nonverbal.get("face_metrics", {})
    expression = face_metrics.get("expression", {})
    head_pose = face_metrics.get("head_pose", {})
    face_summary = (
        f"- 얼굴 감지율: {face_metrics.get('face_detected_ratio', 0) * 100:.0f}%\n"
        f"- 정면 응시 추정 비율: {face_metrics.get('eye_contact_ratio', 0) * 100:.0f}%\n"
        f"- 시선 이탈 추정 비율: {face_metrics.get('gaze_away_ratio', 0) * 100:.0f}%\n"
        f"- 눈 깜빡임: 분당 {face_metrics.get('blink_per_minute', 0):.1f}회\n"
        f"- 미소/표정 안정성: 미소 비율 {expression.get('smile_ratio', 0) * 100:.0f}%, "
        f"표정 안정도 {expression.get('expression_stability', 0):.2f}\n"
        f"- 머리 방향 평균: 좌우 {head_pose.get('avg_yaw', 0):.1f}도, "
        f"상하 {head_pose.get('avg_pitch', 0):.1f}도\n"
    )
    behavior_state = nonverbal.get("behavior_state", {})

    behavior_summary = (
        f"- 현재 행동 상태: {behavior_state.get('state', 'neutral')}\n"
        f"- 상태 신뢰도: {behavior_state.get('confidence', 0):.2f}\n"
    )

    prompt = f"""너는 {situation['name']} 상황 전문 커뮤니케이션 코치이다.

답변의 내용 평가는 이미 완료되었다. 여기서는 말하는 방식과 비언어적 표현만 평가해라.
잘한 점은 칭찬하고, 개선할 점은 구체적인 방법을 제시해라.
수치를 직접 언급하지 말고 자연스러운 한국어로 작성해라.
사용자의 행동 변화 흐름과 긴장 상태를 고려하여 평가해라.
특정 순간의 비언어적 변화가 전달력에 어떤 영향을 주었는지 설명해라.

반드시 아래 형식으로만 출력해라:
1. 전달 방식 평가 (말속도·침묵·추임새·비언어 통합)
2. 종합 피드백 및 개선 방향

각 항목은 4~5문장으로 작성해라.

[상황]
{situation['name']}

[이 상황의 권장 기준]
- 시선: {std['gaze']}
- 자세: {std['posture']}
- 제스처: {std['gesture']}
- 권장 말 속도: {good_range[0]}~{good_range[1]} 단어/초
- 권장 침묵: {std['silence']['good']}초 이하
- 권장 추임새: {std['filler']['good']}회 이하

[면접 질문]
{question}

[답변 내용]
{verbal['text']}

[언어적 분석 결과]
- {rate_eval}
- {silence_eval}
- {filler_eval}

[비언어적 분석 결과]
- 총 분석 시간: {nonverbal.get('total_duration', 0):.1f}초
- 얼굴/시선/표정 분석:
{face_summary}
- 행동 상태 분석:
{behavior_summary}
- 감지된 행동 패턴:
{cluster_summary}
"""
    return call_llm_orchestrator(prompt)


if __name__ == "__main__":
    situation = select_situation()
    print(f"\n선택된 상황: {situation['name']}")

    print("\n👉 Enter 누르면 시작")
    input()

    question = generate_question()
    print(f"\n질문:\n{question}")

    print("\n🎤 답변 준비되면 Enter 누르세요")
    input()

    stop_event = threading.Event()
    result_container = {}

    def record_and_stop():
        record_audio(stop_event=stop_event)

    record_thread = threading.Thread(target=record_and_stop)
    record_thread.start()

    try:
        result = analyze_video(
            source=0,
            output_dir="result",
            frames_per_cluster=3,
            stop_event=stop_event,
        )
        result_container["nonverbal"] = result
    except Exception as e:
        print(f"비언어 분석 오류: {e}")
        result_container["nonverbal"] = {}

    record_thread.join()

    print("\n음성 변환 중...")
    text = speech_to_text()
    print(f"\n답변: {text}")

    if text.strip() == "":
        print("음성이 인식되지 않았습니다.")
        exit()

    verbal = {"text": text, **analyze_audio(text)}
    nonverbal = result_container.get("nonverbal") or {}

    print("\n답변 내용 평가 중... (exaone)")
    content_eval = evaluate_answer_content(question, text, situation["name"])
    print("\n=== 답변 내용 평가 ===")
    print(content_eval)

    print("\n개선된 답변 예시 생성 중... (exaone)")
    improved = generate_improved_answer(question, text, situation["name"])
    print("\n=== 개선된 답변 예시 ===")
    print(improved)

    print("\n전달 방식 평가 중... (qwen3)")
    delivery_feedback = orchestrate(situation, question, verbal, nonverbal)
    print("\n=== 전달 방식 피드백 ===")
    print(delivery_feedback)

    followup = generate_followup(text)
    print(f"\n꼬리 질문:\n{followup}")

    os.makedirs("result", exist_ok=True)
    with open("result/final_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "situation": situation,
            "question": question,
            "verbal": verbal,
            "nonverbal": nonverbal,
            "content_evaluation": content_eval,
            "improved_answer": improved,
            "delivery_feedback": delivery_feedback,
            "followup": followup,
        }, f, ensure_ascii=False, indent=2)

    print("\nresult/final_result.json 저장 완료")
