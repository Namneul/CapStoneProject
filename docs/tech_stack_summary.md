# GTA 면접 분석 시스템 기술 스택

## Frontend
- React 18
- Vite
- React Router
- Tailwind CSS
- lucide-react 아이콘
- Express 프록시 서버와 연동

## Frontend Server
- Node.js
- Express
- CORS
- concurrently
- `npm run dev:all`로 Vite 프론트와 Express 서버를 함께 실행

## Backend / Analysis
- Python 3.11
- Flask / Flask-CORS
- OpenCV
- NumPy
- scikit-learn
- FINCH clustering
- librosa / scipy / sounddevice

## Speech / Language
- OpenAI Whisper
- AI Hub KLUE-BERT 기반 언어 모델
- TensorFlow 2.15
- Hugging Face Transformers
- filler / repeat / pause 계열 언어 분석

## Nonverbal / Vision
- MediaPipe Holistic
- OpenFace 2.2.0
- STGCN++ / MMAction2
- PyTorch
- MMEngine / MMCV-lite

## LLM Feedback
- Ollama 연동 구조
- Qwen 계열 모델 우선 시도
  - qwen3:8b
  - qwen2.5:7b
  - qwen2.5:7b-instruct
  - qwen:7b
- 일부 기존 코드에는 exaone3.5:7.8b 호출 구조도 존재
- Ollama 모델 미응답 시 fallback 피드백 생성

## Data / Result
- JSON 기반 결과 저장
- 주요 출력 파일
  - result/final_result.json
  - result/uploaded_session/final_result.json
  - result/session_history.json
- 세션 인사이트
  - 발화 하이라이트
  - 상태 변화점
  - 반복 습관 후보
  - 전체 답변 피드백

## Current Runtime URLs
- Frontend: http://localhost:5173
- Express API: http://localhost:3001
- Flask video feed: http://localhost:5001/video_feed
- Latest result page: http://localhost:5173/result-latest
