# 실행방법

## 기존 로컬 콘솔 실행

- 터미널에 ollama serve
- 새 터미널에 ollama pull llama3:8b
- python orchestrator.py
- 면접 후 q 누르면 종료. 터미널 및 json 파일로 최종 평가 생성

## KLUE-BERT 언어 분석 모델

AI Hub에서 받은 TensorFlow KLUE-BERT 모델로 간투어(FIL), 반복(REP),
발화 오류(WR)를 탐지한다. 모델은 아래 경로에 배치한다.

```text
models/aihub-klue-bert/2.AI학습모델파일/모델1_언어적_KLUE-BERT/
  checkpoint
  cp.ckpt.data-00000-of-00001
  cp.ckpt.index
```

이 모델의 라벨 순서는 `O`, `FIL-B`, `REP-B`, `PS-B`, `WR-B`이다.
다른 경로를 사용할 때:

```powershell
$env:KLUE_BERT_MODEL_DIR="C:\path\to\klue-bert-model"
python orchestrator.py
```

## 웹 배포 방향

실시간 웹캠 피드백 대신 업로드 후 분석 방식으로 간다.

1. 브라우저에서 면접 영상을 녹화한다.
2. 녹화가 끝난 파일을 서버에 업로드한다.
3. 서버에서 OpenFace FeatureExtraction을 실행한다.
4. OpenFace CSV를 `face_metrics` JSON으로 변환한다.
5. 음성/STT 분석 결과와 합쳐 최종 피드백을 생성한다.

## OpenFace 얼굴 분석 실행

OpenFace `FeatureExtraction` 실행 파일이 PATH에 있거나, 환경변수로 지정되어 있어야 한다.

```bash
export OPENFACE_FEATURE_EXTRACTION_BIN=/path/to/FeatureExtraction
.venv/bin/python analyze_uploaded_video.py face_videos/김준원-1.mp4 --output-dir result/uploaded
```

결과는 `result/uploaded/analysis_result.json`에 저장된다.
