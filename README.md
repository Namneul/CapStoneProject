# AI 면접 연습 서비스 (Capstone Project)

이 프로젝트는 React 프론트엔드와 Python 백엔드(OpenCV, Whisper, Ollama)를 연동한 **AI 화상 면접 연습 및 분석 서비스**입니다. 
면접 상황을 선택하면 LLM이 질문을 던지고, 사용자의 답변(음성/비디오)을 분석하여 개선점과 피드백을 제공합니다.

## 🚀 팀원들을 위한 로컬 실행 가이드 (Getting Started)

이 프로젝트를 처음 깃허브에서 Clone 받으신 분들은 아래 순서대로 세팅을 진행해 주세요!

### 1. 프론트엔드 및 중계 서버 세팅 (Node.js)
React 클라이언트와 Express 중계 서버(`server.js`) 구동을 위한 패키지 설치가 필요합니다.
Node.js가 설치되어 있어야 합니다.

```bash
# 프론트엔드 디렉토리로 이동
cd gta-frontend

# 필요한 모든 패키지(express, cors, react 등) 한 번에 설치
npm install
```

### 2. 파이썬 백엔드 환경 세팅 (Python 3)
백엔드 로직 구동을 위한 파이썬 라이브러리(OpenCV, Whisper, Mediapipe 등) 설치가 필요합니다.
(Python 3.10 이상 권장)

```bash
# 1. 가상환경 생성 (원하는 이름으로 생성 가능, 예: venv)
python -m venv venv

# 2. 가상환경 활성화 (Windows 기준)
venv\Scripts\activate

# 3. 필수 라이브러리 일괄 설치
pip install -r requirements.txt
```
*(주의: 한글 이름이 포함된 폴더 경로에서 가상환경을 실행하면 OpenCV/Mediapipe C++ 코어 버그가 발생할 수 있습니다. 띄어쓰기나 한글이 없는 경로 사용을 권장합니다.)*

### 3. 필수 외부 프로그램 설치 (중요 ⭐)

#### A. FFmpeg 설치
사용자의 음성(`.wav`)을 Whisper 모델이 인식할 수 있도록 변환해 주는 오디오 처리 프로그램입니다.
- **Windows**: PowerShell 관리자 권한으로 아래 명령어를 실행하여 설치합니다.
  ```powershell
  winget install --id=Gyan.FFmpeg -e
  ```
- **또는 수동 설치**: `ffmpeg.exe`와 `ffprobe.exe`를 다운로드 받아 프로젝트의 가장 바깥 폴더(최상단 루트)에 넣어주세요.

#### B. Ollama 및 언어 모델 설치
질문 생성 및 답변 평가를 위해 로컬 LLM을 사용합니다.
- [Ollama 홈페이지](https://ollama.com/)에서 다운로드 후 설치
- 터미널을 열고 아래 명령어로 모델을 다운로드합니다.
  ```bash
  ollama run exaone3.5:7.8b
  ```

---

## 🏃 실행 방법

모든 세팅이 끝났다면, 앞으로 프로젝트를 실행하실 때는 아래 명령어 **하나만** 입력하시면 됩니다!

```bash
# 1. Ollama 백그라운드 실행 확인 (시스템 트레이 아이콘 확인)

# 2. 프론트엔드 및 백엔드 서버 동시 실행
cd gta-frontend
npm run dev:all
```

- 실행 후 브라우저에서 `http://localhost:5173` 로 접속하시면 됩니다.
- React 페이지에서 면접을 시작하면 파이썬 OpenCV 카메라 창이 별도로 팝업되어 분석을 진행합니다.
