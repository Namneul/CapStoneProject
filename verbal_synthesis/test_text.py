"""
test_text.py
음성 없이 키보드 입력으로 언어 분석 파이프라인을 테스트하는 스크립트
"""

from verbal_synthesis.LLM import generate_question, generate_followup
from verbal_synthesis.stt import analyze_audio

DUMMY_WAV = "recorded.wav"


def analyze_text_only(text: str) -> dict:
    """recorded.wav 없이 텍스트만으로 간이 분석 (테스트용)"""
    word_count = len(text.split())
    fillers = ["음", "어", "그", "그니까", "약간"]
    filler_count = sum(text.count(f) for f in fillers)
    # 실제 말 속도·침묵은 오디오 기반이므로 텍스트 테스트에서는 추정값 사용
    estimated_rate = round(word_count / 5, 2)
    return {
        "speech_rate": estimated_rate,
        "silence_time": 0.0,
        "filler_count": filler_count,
        "duration": 0.0,
    }


if __name__ == "__main__":
    print("👉 Enter 누르면 테스트 시작")
    input()

    question = generate_question()

    while True:
        print(f"\n질문:\n{question}")
        print("\n답변을 입력하세요:")
        text = input()

        if text.strip() == "":
            print("입력 없음. 종료합니다.")
            break

        stats = analyze_text_only(text)
        print(f"\n분석 결과:")
        print(f"  말 속도(추정): {stats['speech_rate']} 단어/초")
        print(f"  추임새: {stats['filler_count']}회")

        question = generate_followup(text)

        print("\n계속하려면 Enter / 종료하려면 q 입력")
        if input().lower() == "q":
            print("테스트 종료")
            break