import warnings
warnings.filterwarnings("ignore")

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "exaone3.5:7.8b"


def call_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
        }
    )
    data = response.json()
    if "response" not in data:
        print("LLM 오류:", data)
        return "생성 실패"
    return data["response"].strip()


def generate_question() -> str:
    prompt = """한국어 인성 면접 질문을 1개 생성해라.

규칙:
- 자기소개, 지원동기, 성격 관련 질문 중 하나
- 질문만 출력, 부연 설명 없이

예시:
자기소개를 해보세요.
"""
    return call_llm(prompt)


def evaluate_answer_content(question: str, text: str, situation_name: str) -> str:
    prompt = f"""{situation_name} 상황에서 아래 질문에 대한 답변 내용을 평가해라.

평가 기준:
- 질문 연관성: 질문에 적절히 답했는가
- 논리적 구성: 내용이 체계적으로 전달되는가
- 구체성: 추상적이지 않고 구체적인 내용을 담고 있는가

반드시 아래 형식으로만 출력해라:
잘한 점: (1~2문장)
개선할 점: (1~2문장)

[질문]
{question}

[답변]
{text}
"""
    return call_llm(prompt)


def generate_improved_answer(question: str, text: str, situation_name: str) -> str:
    prompt = f"""{situation_name} 상황에서 아래 답변을 더 효과적으로 개선한 예시를 작성해라.
원래 답변의 핵심 내용은 유지하되, 구성과 표현만 개선해라.
3~5문장으로 자연스럽게 작성해라. 설명 없이 개선된 답변만 출력해라.

[질문]
{question}

[원래 답변]
{text}
"""
    return call_llm(prompt)


def generate_followup(text: str) -> str:
    prompt = f"""아래 답변을 바탕으로 자연스럽게 이어지는 꼬리 질문을 1개만 생성해라.
질문만 출력, 부연 설명 없이.

[지원자 답변]
{text}
"""
    return call_llm(prompt)


if __name__ == "__main__":
    q = generate_question()
    print("질문:", q)