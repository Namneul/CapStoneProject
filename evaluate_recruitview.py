"""
evaluate_recruitview.py
RecruitView 데이터셋 2011개 영상을 분석하고 논문용 성능 표 생성

분석 항목:
  1. 감정 표현 탐지  → brow_furrow vs facial_expression (binarized)
  2. 자세/태도 탐지  → blink_rate + head_nod vs confidence_score (binarized)
  3. 전체 품질 예측  → shoulder_tension vs overall_performance (binarized)
  4. 추임새 탐지     → transcript filler count vs speaking_skills (binarized)

실행:
    python evaluate_recruitview.py              # 전체 분석
    python evaluate_recruitview.py --cache      # 캐시 사용 (재실행 시)
    python evaluate_recruitview.py --n 10       # 처음 N개만 (테스트용)
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys, os, re, json, csv, tempfile, warnings
warnings.filterwarnings("ignore")

import cv2
from pathlib import Path
from datasets import load_dataset, Video

sys.path.insert(0, str(Path(__file__).parent))
from behavior_grouping.analyzer import analyze_video

# ── 경로 설정 ──
CACHE_DIR  = Path("recruitview_cache")
FRAMES_DIR = CACHE_DIR / "frames"
CSV_PATH   = Path("recruitview_comparison.csv")

# ── 예측 임계값 (face_labels 그리드서치 최적화값 재사용) ──
BROW_THR  = 0.62
BLINK_THR = 0.20
NOD_THR   = 0.0242
SHLD_THR  = 0.62

# ── GT 이진화 임계값 (z-score 기준, 0 = 평균) ──
GT_THR = 0.0

# ── 영어 추임새 ──
FILLERS_EN = {"um", "uh", "umm", "uhh", "hmm", "er", "like"}
FILLER_THR = 3


# ──────────────────────────────────────────────
# transcript 분석
# ──────────────────────────────────────────────

def parse_transcript(text: str) -> dict:
    if not text:
        return {"word_cnt": 0, "filler_cnt": 0, "speech_rate": 0.0}

    clean = re.sub(r'\[\d+:\d+\s*-\s*\d+:\d+\]', '', text).strip()
    words = clean.split()
    word_cnt = len(words)
    filler_cnt = sum(
        1 for w in re.sub(r'[^\w\s]', '', clean).lower().split()
        if w in FILLERS_EN
    )

    timestamps = re.findall(r'\[(\d+):(\d+)\s*-\s*(\d+):(\d+)\]', text)
    speech_rate = 0.0
    if timestamps:
        t_start = int(timestamps[0][0]) * 60 + int(timestamps[0][1])
        t_end   = int(timestamps[-1][2]) * 60 + int(timestamps[-1][3])
        duration = t_end - t_start
        if duration > 0:
            speech_rate = round(word_cnt / duration * 60, 2)

    return {"word_cnt": word_cnt, "filler_cnt": filler_cnt, "speech_rate": speech_rate}


# ──────────────────────────────────────────────
# 영상 → 비언어 특징값 추출
# ──────────────────────────────────────────────

def extract_video_features(hf_path: str) -> dict:
    import fsspec

    with fsspec.open(hf_path, "rb") as f:
        video_bytes = f.read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name

    try:
        result = analyze_video(
            source=tmp_path,
            headless=True,
            output_dir=str(FRAMES_DIR),
            pca_components=10,
            frames_per_cluster=1,
        )
        clusters = result.get("clusters", [])
        if not clusters:
            return None

        total_ratio = sum(c["ratio"] for c in clusters)
        feat_names = ["brow_furrow", "blink_rate", "shoulder_tension", "head_nod", "body_lean"]
        return {
            feat: round(
                sum(c["feature_means"].get(feat, 0) * c["ratio"] for c in clusters) / total_ratio, 6
            )
            for feat in feat_names
        }
    except Exception as e:
        print(f"  오류: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ──────────────────────────────────────────────
# 예측 / GT 이진화
# ──────────────────────────────────────────────

def predict(features: dict) -> dict:
    return {
        "pred_emotion": "긍정" if features.get("brow_furrow", 0) >= BROW_THR else "부정",
        "pred_posture":  "긍정" if features.get("head_nod", 1) <= NOD_THR else "부정",
        "pred_quality":  "상" if features.get("shoulder_tension", 1) <= SHLD_THR else "하",
    }


def binarize_gt(item: dict) -> dict:
    return {
        "gt_emotion":  "긍정" if item["facial_expression"]   >= GT_THR else "부정",
        "gt_posture":  "긍정" if item["confidence_score"]    >= GT_THR else "부정",
        "gt_quality":  "상"   if item["overall_performance"] >= GT_THR else "하",
        "gt_filler":   "많음" if item["speaking_skills"]     <  GT_THR else "적음",
    }


# ──────────────────────────────────────────────
# 영상 하나 처리
# ──────────────────────────────────────────────

def process_item(item: dict, use_cache: bool):
    vid_id = item["id"]
    cache_path = CACHE_DIR / f"{vid_id}.json"

    if use_cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        # 임계값 변경 시 예측 재계산
        preds = predict(cached)
        cached.update(preds)
        return cached

    hf_path = item["video"].get("path") if isinstance(item["video"], dict) else None
    if not hf_path:
        print("  영상 경로 없음 — 건너뜀")
        return None

    print(f"  영상 분석 중...", end=" ", flush=True)
    features = extract_video_features(hf_path)
    if features is None:
        return None
    print("완료", flush=True)

    verbal = parse_transcript(item.get("transcript", ""))
    preds  = predict(features)
    gt     = binarize_gt(item)

    row = {
        "id":       vid_id,
        "duration": item.get("duration", ""),
        **features,
        **verbal,
        **preds,
        "pred_filler": "많음" if verbal["filler_cnt"] >= FILLER_THR else "적음",
        **gt,
        "facial_expression":   round(item["facial_expression"],   4),
        "confidence_score":    round(item["confidence_score"],    4),
        "overall_performance": round(item["overall_performance"], 4),
        "speaking_skills":     round(item["speaking_skills"],     4),
        "interview_score":     round(item["interview_score"],     4),
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    return row


# ──────────────────────────────────────────────
# 지표 계산 / 출력
# ──────────────────────────────────────────────

def compute_metrics(rows, pred_col, gt_col, pos_label):
    tp = sum(1 for r in rows if r[gt_col] == pos_label and r[pred_col] == pos_label)
    fp = sum(1 for r in rows if r[gt_col] != pos_label and r[pred_col] == pos_label)
    tn = sum(1 for r in rows if r[gt_col] != pos_label and r[pred_col] != pos_label)
    fn = sum(1 for r in rows if r[gt_col] == pos_label and r[pred_col] != pos_label)
    n  = tp + fp + tn + fn
    acc  = (tp + tn) / n       if n          else 0
    prec = tp / (tp + fp)      if tp + fp    else 0
    rec  = tp / (tp + fn)      if tp + fn    else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


SEP = "=" * 60

def print_summary(rows):
    n = len(rows)
    print(f"\n{SEP}")
    print(f"행동 분석 성능 평가 결과  (N={n})")
    print(SEP)

    tasks = [
        ("감정 표현 탐지", "pred_emotion", "gt_emotion", "긍정"),
        ("자세/태도 탐지", "pred_posture", "gt_posture", "긍정"),
        ("전체 품질 예측", "pred_quality", "gt_quality", "상"),
    ]

    print(f"\n{'평가 항목':<16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
    print("-" * 58)
    for name, pred_col, gt_col, pos in tasks:
        m = compute_metrics(rows, pred_col, gt_col, pos)
        print(f"{name:<16} {m['acc']:>10.4f} {m['prec']:>10.4f} {m['rec']:>10.4f} {m['f1']:>10.4f}")
    print(SEP)


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    use_cache = "--cache" in sys.argv
    n_limit   = None
    if "--n" in sys.argv:
        n_limit = int(sys.argv[sys.argv.index("--n") + 1])

    CACHE_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)

    print("RecruitView 스트리밍 로드 중...")
    ds = load_dataset("AI4A-lab/RecruitView", streaming=True)
    ds = ds.cast_column("video", Video(decode=False))

    rows = []
    for i, item in enumerate(ds["train"]):
        if n_limit and i >= n_limit:
            break
        label = f"{item['id']}  ({i+1}" + (f"/{n_limit}" if n_limit else "") + ")"
        print(f"\n[{label}]")
        row = process_item(item, use_cache)
        if row:
            rows.append(row)
            print(f"  샘플 누적: {len(rows)}개")

    if not rows:
        print("처리된 영상이 없습니다.")
        sys.exit(1)

    print_summary(rows)

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 저장: {CSV_PATH}")