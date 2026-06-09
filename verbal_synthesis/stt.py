import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
import time
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import librosa

fs = 44100
recording = []
_whisper_model = None

FILLER_WORDS = {
    "어",
    "음",
    "으음",
    "저",
    "저기",
    "그니까",
    "그러니까",
    "약간",
    "뭐",
}


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


def speech_to_text() -> str:
    global _whisper_model
    model_name = os.getenv("WHISPER_MODEL", "small")
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    result = _whisper_model.transcribe(
        "recorded.wav",
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
    return result["text"]


def analyze_audio(text: str) -> dict:
    y, sr = librosa.load("recorded.wav")
    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=20)
    speech_time = sum((end - start) for start, end in intervals) / sr
    silence_time = duration - speech_time
    word_count = len(text.split())
    speech_rate = word_count / speech_time if speech_time > 0 else 0
    tokens = _tokenize_korean_stt(text)
    filler_tokens = [token for token in tokens if token in FILLER_WORDS]
    filler_count = len(filler_tokens)
    pause_count = _count_long_pauses(intervals, sr)

    return {
        "speech_rate": round(speech_rate, 2),
        "silence_time": round(silence_time, 2),
        "filler_count": filler_count,
        "filler_tokens": filler_tokens,
        "pause_count": pause_count,
        "duration": round(duration, 2),
    }


def _tokenize_korean_stt(text: str) -> list[str]:
    return [
        token
        for token in re.split(r"[\s,.!?;:\"'()\[\]{}]+", text.strip())
        if token
    ]


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
    print(analyze_audio(text))
