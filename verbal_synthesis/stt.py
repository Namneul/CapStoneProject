import warnings
warnings.filterwarnings("ignore")

import sys
import time
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import librosa

fs = 44100
recording = []


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
    model = whisper.load_model("base")
    result = model.transcribe("recorded.wav", language="ko")
    return result["text"]


def analyze_audio(text: str) -> dict:
    y, sr = librosa.load("recorded.wav")
    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=20)
    speech_time = sum((end - start) for start, end in intervals) / sr
    silence_time = duration - speech_time
    word_count = len(text.split())
    speech_rate = word_count / speech_time if speech_time > 0 else 0
    fillers = ["음", "어", "그", "그니까", "약간"]
    filler_count = sum(text.count(f) for f in fillers)

    return {
        "speech_rate": round(speech_rate, 2),
        "silence_time": round(silence_time, 2),
        "filler_count": filler_count,
        "duration": round(duration, 2),
    }


if __name__ == "__main__":
    record_audio()
    text = speech_to_text()
    print("\n결과:", text)
    print(analyze_audio(text))