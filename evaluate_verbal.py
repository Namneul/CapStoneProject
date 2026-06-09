"""
evaluate_verbal.py
공적말하기 데이터셋 영상을 STT로 분석하고 전문가 JSON 라벨과 비교

흐름:
  1. mp4 → ffmpeg → wav 변환
  2. Whisper(base)로 STT → 텍스트
  3. librosa로 발화 시간 분석 → 말 속도 / 추임새 / 단어 수
  4. 전문가 JSON(evaluations 평균값)과 비교
  5. verbal_comparison.csv 저장 + 터미널 요약 출력

실행:
    python evaluate_verbal.py           # 전체 분석
    python evaluate_verbal.py --cache   # 캐시 사용 (재실행 시)
"""

import sys
import os
import csv
import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import re
import numpy as np
import librosa
from faster_whisper import WhisperModel

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
JSON_DIR  = BASE_DIR / "labels"
CACHE_DIR = BASE_DIR / "verbal_cache"
CSV_PATH  = BASE_DIR / "verbal_comparison.csv"

# 단어 단위로 매칭할 추임새 목록 (한국어 발표 대표 추임새)
FILLERS = {"음", "어", "아", "저", "뭐", "그", "그니까", "그러니까", "약간", "이제", "일단"}

# Whisper에 추임새를 전사하도록 유도하는 초기 프롬프트
WHISPER_PROMPT = "음, 어, 아, 저, 그, 뭐 같은 추임새가 포함된 한국어 발표 내용입니다."


# ──────────────────────────────────────────────
# 전문가 JSON 파싱
# ──────────────────────────────────────────────

def load_expert(vid_id: str) -> Optional[dict]:
    path = JSON_DIR / f"{vid_id}_presentation.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    evals = d.get("evaluations", [])
    if not evals:
        return None

    def avg(key_path):
        vals = []
        for e in evals:
            obj = e
            for k in key_path:
                obj = obj.get(k, {})
            if isinstance(obj, (int, float)):
                vals.append(obj)
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "expert_speed":    round(avg(["voice_speed",  "voc_speed"]),        4),
        "expert_filler":   round(avg(["filler_words", "filler_words_cnt"]), 2),
        "expert_repeat":   round(avg(["repeat",       "repeat_cnt"]),       2),
        "expert_word_cnt": d["script"].get("word_cnt", 0),
        "expert_grade":    d["average"].get("eval_grade", ""),
    }


# ──────────────────────────────────────────────
# mp4 → wav 변환
# ──────────────────────────────────────────────

def mp4_to_wav(mp4_path: str, wav_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-ar", "44100", "-ac", "1",
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ──────────────────────────────────────────────
# STT + 음성 분석
# ──────────────────────────────────────────────

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("  Whisper 모델 로드 중...", flush=True)
        _whisper_model = WhisperModel("base", device="auto", compute_type="int8")
    return _whisper_model


def transcribe(wav_path: str) -> str:
    model = get_whisper_model()
    segments, _ = model.transcribe(wav_path, language="ko", initial_prompt=WHISPER_PROMPT)
    return " ".join(seg.text for seg in segments)


def count_syllables(text: str) -> int:
    """한글 글자 수 = 음절 수 (전문가 voc_speed 단위: 음절/초와 일치)"""
    return sum(1 for c in text if '가' <= c <= '힣')


def analyze_audio(wav_path: str, text: str) -> dict:
    y, sr = librosa.load(wav_path, sr=44100)
    duration = librosa.get_duration(y=y, sr=sr)
    # top_db=30: 20보다 완화해 배경음을 침묵으로 잘못 분류하는 이상치 방지
    intervals = librosa.effects.split(y, top_db=30)
    speech_time = sum((end - start) for start, end in intervals) / sr
    silence_time = round(duration - speech_time, 2)

    words = text.split()
    word_cnt = len(words)
    syllable_cnt = count_syllables(text)
    # 전문가 voc_speed 단위(음절/초)에 맞춰 음절 기반으로 계산
    speech_rate = round(syllable_cnt / duration, 4) if duration > 0 else 0.0
    filler_cnt = sum(1 for w in re.sub(r'[^\w\s]', '', text).split() if w in FILLERS)

    return {
        "sys_speed":        speech_rate,      # 음절/초 (전문가와 동일 단위)
        "sys_syllable_cnt": syllable_cnt,
        "sys_filler":       filler_cnt,
        "sys_word_cnt":     word_cnt,
        "sys_duration":     round(duration, 2),
        "silence_time":     silence_time,
        "text":             text,
    }


# ──────────────────────────────────────────────
# 영상 하나 처리 (캐시 포함)
# ──────────────────────────────────────────────

def process_video(vid_id: str, use_cache: bool) -> Optional[dict]:
    cache_path = CACHE_DIR / f"{vid_id}.json"

    if use_cache:
        if not cache_path.exists():
            return None
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if "text" in cached:
            text = cached["text"]
            cached["sys_filler"] = sum(
                1 for w in re.sub(r'[^\w\s]', '', text).split() if w in FILLERS
            )
        return cached

    mp4_path = VIDEO_DIR / f"{vid_id}.mp4"
    if not mp4_path.exists():
        print(f"  영상 없음: {mp4_path}")
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        print(f"  wav 변환 중...", end=" ", flush=True)
        if not mp4_to_wav(str(mp4_path), wav_path):
            print("ffmpeg 실패")
            return None

        print(f"STT 중...", end=" ", flush=True)
        text = transcribe(wav_path)

        print(f"음성 분석 중...", end=" ", flush=True)
        sys_result = analyze_audio(wav_path, text)
        print("완료")

        CACHE_DIR.mkdir(exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(sys_result, f, ensure_ascii=False, indent=2)

        return sys_result

    except Exception as e:
        print(f"오류: {e}")
        return None
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ──────────────────────────────────────────────
# 전체 분석 + 비교
# ──────────────────────────────────────────────

def list_video_ids() -> list:
    ids = []
    for mp4 in sorted(VIDEO_DIR.glob("*.mp4")):
        vid_id = mp4.stem
        if (JSON_DIR / f"{vid_id}_presentation.json").exists():
            ids.append(vid_id)
    return ids


def build_rows(vid_ids: list, use_cache: bool) -> list:
    rows = []
    for vid_id in vid_ids:
        print(f"\n[{vid_id}]")
        expert = load_expert(vid_id)
        if expert is None:
            print("  라벨 없음 — 건너뜀")
            continue

        sys_r = process_video(vid_id, use_cache)
        if sys_r is None:
            continue

        e_speed  = expert["expert_speed"]
        s_speed  = sys_r["sys_speed"]
        e_filler = expert["expert_filler"]
        s_filler = sys_r["sys_filler"]
        e_words  = expert["expert_word_cnt"]
        s_words  = sys_r["sys_word_cnt"]

        rows.append({
            "video_id":           vid_id,
            # 말 속도 (단위: 음절/초)
            "expert_speed":       round(e_speed,  4),
            "sys_speed":          round(s_speed,  4),
            "speed_error":        round(abs(s_speed - e_speed), 4),
            # 추임새
            "expert_filler":      round(e_filler, 2),
            "sys_filler":         s_filler,
            "filler_error":       abs(s_filler - round(e_filler)),
            # 단어 수
            "expert_word_cnt":    e_words,
            "sys_word_cnt":       s_words,
            "word_error_rate":    round(abs(s_words - e_words) / e_words, 4) if e_words else 0,
            # 음절 수 (참고)
            "sys_syllable_cnt":   sys_r.get("sys_syllable_cnt", 0),
            # 참고
            "expert_repeat":      round(expert["expert_repeat"], 2),
            "expert_grade":       expert["expert_grade"],
            "sys_duration":       sys_r["sys_duration"],
        })
    return rows


# ──────────────────────────────────────────────
# CSV 저장
# ──────────────────────────────────────────────

def save_csv(rows: list):
    if not rows:
        return
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 저장: {CSV_PATH}")


# ──────────────────────────────────────────────
# 요약 출력
# ──────────────────────────────────────────────

SEP = "=" * 70

def print_summary(rows: list):
    n = len(rows)
    if n == 0:
        print("분석된 영상이 없습니다.")
        return

    speed_mae  = sum(r["speed_error"]    for r in rows) / n
    filler_mae = sum(r["filler_error"]   for r in rows) / n
    wer_mean   = sum(r["word_error_rate"]for r in rows) / n

    print(f"\n{SEP}")
    print("Summary — Verbal Analysis vs Expert Labels")
    print(SEP)
    print(f"  처리된 영상 수   : {n}개")
    print(f"  말 속도 MAE      : {speed_mae:.4f}  (단어/초 단위 절대 오차 평균)")
    print(f"  추임새 MAE       : {filler_mae:.2f}  (횟수 절대 오차 평균)")
    print(f"  단어 수 WER 평균 : {wer_mean:.4f}  ({wer_mean*100:.1f}%)")
    print(SEP)

    print(f"\n{'Video':<40} {'E.Spd':>6} {'S.Spd':>6} {'SpdErr':>7}  "
          f"{'E.Fil':>6} {'S.Fil':>6} {'FilErr':>7}  "
          f"{'E.Wrd':>6} {'S.Wrd':>6} {'WER':>7}  {'Grd':>4}")
    print("-" * 110)
    for r in rows:
        print(f"{r['video_id']:<40} "
              f"{r['expert_speed']:>6.3f} {r['sys_speed']:>6.3f} {r['speed_error']:>7.4f}  "
              f"{r['expert_filler']:>6.1f} {r['sys_filler']:>6} {r['filler_error']:>7}  "
              f"{r['expert_word_cnt']:>6} {r['sys_word_cnt']:>6} {r['word_error_rate']:>7.4f}  "
              f"{r['expert_grade']:>4}")
    print(SEP)


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    use_cache = "--cache" in sys.argv
    CACHE_DIR.mkdir(exist_ok=True)

    vid_ids = list_video_ids()
    print(f"분석 대상 영상: {len(vid_ids)}개  (캐시: {'사용' if use_cache else '미사용'})")

    rows = build_rows(vid_ids, use_cache)
    if not rows:
        print("처리된 영상이 없습니다.")
        sys.exit(1)

    print_summary(rows)
    save_csv(rows)