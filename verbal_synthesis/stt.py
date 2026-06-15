import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import librosa

from verbal_synthesis.klue_bert_analyzer import KlueBertAnalyzer


fs = 44100
recording = []
_whisper_model = None
_klue_bert_model = None


def callback(indata, frames, time, status):
    recording.append(indata.copy())


def record_audio(stop_event=None):
    global recording
    recording = []

    stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)
    stream.start()
    print("🔴 녹음 중... (Enter 누르면 종료)")

    if stop_event is not None:
        while not stop_event.is_set():
            time.sleep(0.1)
    else:
        sys.stdin.readline()

    stream.stop()
    stream.close()

    if not recording:
        raise ValueError("녹음된 데이터가 없습니다.")

    audio = np.concatenate(recording, axis=0)
    audio = np.squeeze(audio)
    audio = (audio * 32767).astype(np.int16)
    wav.write("recorded.wav", fs, audio)


def transcribe_audio(audio_path: str = "recorded.wav") -> dict:
    global _whisper_model
    model_name = os.getenv("WHISPER_MODEL", "small")
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    result = _whisper_model.transcribe(
        audio_path,
        language="ko",
        task="transcribe",
        fp16=False,
        temperature=0.0,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,
        initial_prompt=(
            "한국어 면접 답변입니다. 음, 어, 저, 그러니까, 그니까, 약간 같은 "
            "추임새와 망설임 표현도 생략하지 말고 그대로 받아쓰기하세요."
        ),
    )
    return {
        "text": result.get("text", "").strip(),
        "segments": [
            {
                "id": segment.get("id", index),
                "start": round(float(segment.get("start", 0.0)), 2),
                "end": round(float(segment.get("end", 0.0)), 2),
                "text": segment.get("text", "").strip(),
            }
            for index, segment in enumerate(result.get("segments", []))
            if segment.get("text", "").strip()
        ],
        "language": result.get("language"),
    }


def speech_to_text() -> str:
    return transcribe_audio()["text"]


def speech_to_text_result() -> dict:
    return transcribe_audio()


def analyze_language(text: str) -> dict:
    """Detect fillers, repetitions, and word errors with fine-tuned KLUE-BERT."""
    global _klue_bert_model

    model_dir = os.getenv(
        "KLUE_BERT_MODEL_DIR",
        os.path.join(
            "models",
            "1.모델",
            "2.AI학습모델파일",
            "모델1_언어적_KLUE-BERT",
        ),
    )
    if _klue_bert_model is None:
        try:
            _klue_bert_model = KlueBertAnalyzer(model_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            return _empty_language_analysis(str(exc))
    return _klue_bert_model.analyze(text)


def _empty_language_analysis(reason: str) -> dict:
    return {
        "backend": "aihub-klue-bert-unavailable",
        "reason": reason,
        "counts": {
            "filler": 0,
            "repeat": 0,
            "pause": 0,
            "word_error": 0,
        },
        "spans": [],
    }


def analyze_audio(
    text: str,
    transcript_segments: list[dict] | None = None,
    audio_path: str = "recorded.wav",
) -> dict:
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=20)
    speech_time = sum((end - start) for start, end in intervals) / sr
    silence_time = duration - speech_time
    word_count = len(text.split())
    speech_rate = word_count / speech_time if speech_time > 0 else 0
    language_analysis = analyze_language(text)
    language_counts = language_analysis["counts"]
    filler_tokens = [
        span["text"]
        for span in language_analysis["spans"]
        if span["name"] == "filler"
    ]
    pause_count = _count_long_pauses(intervals, sr)

    return {
        "speech_rate": round(speech_rate, 2),
        "silence_time": round(silence_time, 2),
        "filler_count": language_counts["filler"],
        "repeat_count": language_counts["repeat"],
        "word_error_count": language_counts["word_error"],
        "filler_tokens": filler_tokens,
        "language_analysis": language_analysis,
        "transcript_segments": transcript_segments or [],
        "pause_count": pause_count,
        "duration": round(duration, 2),
    }


def _count_long_pauses(intervals, sr, threshold_seconds: float = 0.7) -> int:
    if len(intervals) < 2:
        return 0
    count = 0
    previous_end = intervals[0][1]
    for start, end in intervals[1:]:
        gap = (start - previous_end) / sr
        if gap >= threshold_seconds:
            count += 1
        previous_end = end
    return count


if __name__ == "__main__":
    record_audio()
    text = speech_to_text()
    print("\n결과:", text)
    print(analyze_audio(text))
