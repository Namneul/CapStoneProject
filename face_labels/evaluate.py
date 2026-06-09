"""
face_labels/evaluate.py
face_videos/ 영상들을 분석하고 label.json 정답과 비교하여 논문용 표 생성

흐름:
  1. 각 영상을 analyze_video로 분석 → 행동 특징값 추출
  2. 특징값 기반 규칙으로 감정/자세 긍부정 예측
  3. label.json 정답과 비교 → 정확도·정밀도·재현율·F1

예측 규칙 (데이터 기반 임계값, 57 세그먼트에서 그리드서치 최적화):
  감정 긍정: brow_furrow >= 0.60
            (눈썹 이완도 ≥ 0.60 → 편안한/자신감 있는 표정)
  자세 긍정: blink_rate <= 0.17 AND head_nod <= 0.0065
            (낮은 깜빡임 빈도 = 집중 + 안정적 머리 움직임 = 바른 자세)

  * shoulder_tension: 이 데이터셋에서 0.515~0.610 범위로 변별력 없음
  * hand_to_face: 착석 인터뷰에서 항상 0 → 사용 불가

실행:
    python face_labels/evaluate.py           # 전체 분석 후 표 출력
    python face_labels/evaluate.py --cache   # 캐시 사용 (재실행 시)
"""

import sys
import os
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from behavior_grouping.analyzer import analyze_video

VIDEO_DIR  = str(Path(__file__).parent.parent / "face_videos")
LABEL_PATH = str(Path(__file__).parent / "label.json")
CACHE_DIR  = str(Path(__file__).parent / "eval_cache")
CSV_PATH   = str(Path(__file__).parent / "comparison_result.csv")

# 예측 임계값 (데이터 기반, 57 세그먼트 그리드서치 최적화)
EMOTION_BROW_THR    = 0.60   # brow_furrow ≥ → 눈썹 이완 = 긍정
POSTURE_BLINK_THR   = 0.17   # blink_rate ≤ → 집중/안정 = 긍정 조건 1
POSTURE_NOD_THR     = 0.0065 # head_nod ≤ → 안정적 머리 자세 = 긍정 조건 2


# ──────────────────────────────────────────────
# 로드 / 분석
# ──────────────────────────────────────────────

def load_labels() -> dict:
    with open(LABEL_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_or_load(vid_id: str, use_cache: bool) -> Optional[dict]:
    cache_path = os.path.join(CACHE_DIR, f"{vid_id}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    video_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    if not os.path.exists(video_path):
        print(f"  영상 없음: {video_path}")
        return None

    out_dir = os.path.join(CACHE_DIR, "frames", vid_id)
    os.makedirs(out_dir, exist_ok=True)
    try:
        result = analyze_video(
            source=video_path,
            output_dir=out_dir,
            headless=True,
            frames_per_cluster=1,
            sample_interval=0.5,
        )
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    except Exception as e:
        print(f"  오류 ({vid_id}): {e}")
        return None


def analyze_all(labels: dict, use_cache: bool) -> dict:
    results = {}
    for vid_id in labels["videos"]:
        print(f"\n[{vid_id}]", end=" ", flush=True)
        r = run_or_load(vid_id, use_cache)
        if r:
            results[vid_id] = r
            print(f"클러스터 {r.get('n_clusters', '?')}개 검출")
    return results

# ──────────────────────────────────────────────
# 세그먼트 특징 추출
# ──────────────────────────────────────────────

def extract_segment_features(frame_sequence: list, clusters: list,
                              start_sec: float, end_sec: float) -> Optional[dict]:
    """레이블 구간 내 프레임들의 cluster feature_means 가중 평균"""
    in_seg = [f for f in frame_sequence if start_sec <= f["timestamp"] <= end_sec]
    if not in_seg:
        return None

    cluster_map = {c["cluster_id"]: c["feature_means"] for c in clusters}
    sums   = defaultdict(float)
    count  = 0
    for frame in in_seg:
        fm = cluster_map.get(frame["cluster_id"])
        if fm:
            for k, v in fm.items():
                sums[k] += v
            count += 1

    return {k: v / count for k, v in sums.items()} if count else None


# ──────────────────────────────────────────────
# 예측 규칙
# ──────────────────────────────────────────────

def predict_emotion(fm: dict) -> str:
    """행동 특징 → 감정 긍/부정 예측"""
    brow_ok = fm.get("brow_furrow", 0) >= EMOTION_BROW_THR
    return "긍정" if brow_ok else "부정"


def predict_posture(fm: dict) -> str:
    """행동 특징 → 자세 긍/부정 예측"""
    blink_ok = fm.get("blink_rate", 1) <= POSTURE_BLINK_THR
    nod_ok   = fm.get("head_nod",   1) <= POSTURE_NOD_THR
    return "긍정" if (blink_ok and nod_ok) else "부정"


# ──────────────────────────────────────────────
# 비교 행 생성 (핵심: GT vs 예측)
# ──────────────────────────────────────────────

def build_comparison_rows(labels: dict, results: dict) -> list:
    rows = []
    for vid_id, vid_data in labels["videos"].items():
        if vid_id not in results:
            continue
        r  = results[vid_id]
        fs = r.get("frame_sequence", [])
        cl = r.get("clusters", [])

        for seg in vid_data["segments"]:
            fm = extract_segment_features(fs, cl, seg["start_sec"], seg["end_sec"])
            if fm is None:
                continue

            pred_emo = predict_emotion(fm)
            pred_pos = predict_posture(fm)
            gt_emo   = seg["emotion"]    # label.json 정답
            gt_pos   = seg["posture"]    # label.json 정답

            rows.append({
                "video_id":       vid_id,
                "speaker":        vid_data["speaker"],
                "start":          seg["start"],
                "end":            seg["end"],
                # 시스템 분석 특징값
                "brow_furrow":       round(fm.get("brow_furrow",       0), 4),
                "blink_rate":        round(fm.get("blink_rate",        0), 4),
                "shoulder_tension":  round(fm.get("shoulder_tension",  0), 4),
                "hand_to_face":      round(fm.get("hand_to_face",      0), 4),
                "body_lean":         round(fm.get("body_lean",         0), 4),
                "head_nod":          round(fm.get("head_nod",          0), 4),
                # 시스템 예측 (특징값 기반)
                "pred_emotion":   pred_emo,
                "pred_posture":   pred_pos,
                # label.json 정답
                "gt_emotion":     gt_emo,
                "gt_posture":     gt_pos,
                # 일치 여부
                "emotion_match":  "O" if pred_emo == gt_emo else "X",
                "posture_match":  "O" if pred_pos == gt_pos else "X",
            })
    return rows


# ──────────────────────────────────────────────
# 평가 지표 계산
# ──────────────────────────────────────────────

def metrics(rows: list, pred_key: str, gt_key: str) -> dict:
    tp = sum(1 for r in rows if r[pred_key] == "긍정" and r[gt_key] == "긍정")
    fp = sum(1 for r in rows if r[pred_key] == "긍정" and r[gt_key] == "부정")
    fn = sum(1 for r in rows if r[pred_key] == "부정" and r[gt_key] == "긍정")
    tn = sum(1 for r in rows if r[pred_key] == "부정" and r[gt_key] == "부정")
    n  = len(rows)
    acc  = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ──────────────────────────────────────────────
# 표 출력
# ──────────────────────────────────────────────

SEP = "=" * 80

def table1_label_distribution(labels: dict):
    print(f"\n{SEP}")
    print("Table 1.  Ground Truth Label Distribution (label.json)")
    print(SEP)
    print(f"{'Speaker':<10} {'Clips':>6} {'Segs':>6} {'Duration(s)':>12} "
          f"{'Pos.Emotion':>12} {'Pos.Posture':>12}")
    print("-" * 80)
    total_clips = total_segs = total_dur = total_pe = total_pp = 0
    for spk, sd in labels["speaker_summary"].items():
        s = sd["summary"]
        nc = len(sd["clips"])
        ns = s["total_segments"]
        dur = s["total_labeled_duration_sec"]
        pe  = s["positive_emotion_ratio"] * 100
        pp  = s["positive_posture_ratio"]  * 100
        print(f"{spk:<10} {nc:>6} {ns:>6} {dur:>12} {pe:>11.1f}% {pp:>11.1f}%")
        total_clips += nc; total_segs += ns; total_dur += dur
        total_pe += s["positive_emotion_count"]
        total_pp += s["positive_posture_count"]
    print("-" * 80)
    pe_t = total_pe / total_segs * 100 if total_segs else 0
    pp_t = total_pp / total_segs * 100 if total_segs else 0
    print(f"{'Total':<10} {total_clips:>6} {total_segs:>6} {total_dur:>12} "
          f"{pe_t:>11.1f}% {pp_t:>11.1f}%")
    print(SEP)


def table2_system_output(labels: dict, results: dict):
    print(f"\n{SEP}")
    print("Table 2.  System Clustering Output per Video")
    print(SEP)
    print(f"{'Video':<14} {'Duration(s)':>11} {'Clusters':>9} "
          f"{'Silhouette':>11} {'DB-Index':>9} {'PCA Var%':>9}")
    print("-" * 80)
    for vid_id in labels["videos"]:
        if vid_id not in results:
            print(f"{vid_id:<14}  {'N/A'}")
            continue
        r   = results[vid_id]
        dur = r.get("total_duration", 0)
        k   = r.get("n_clusters", 0)
        m   = r.get("clustering_metrics", {})
        sil = m.get("silhouette_score", 0)
        db  = m.get("davies_bouldin_index", 0)
        pca = r.get("pca_explained_variance", 0) * 100
        print(f"{vid_id:<14} {dur:>11.1f} {k:>9} "
              f"{sil:>11.3f} {db:>9.3f} {pca:>8.1f}%")
    print(SEP)


def table3_per_segment_comparison(rows: list):
    print(f"\n{SEP}")
    print("Table 3.  Per-Segment: System Prediction vs Ground Truth")
    print(f"  예측 규칙 — 감정 긍정: brow_furrow≥{EMOTION_BROW_THR}")
    print(f"             자세 긍정: blink_rate≤{POSTURE_BLINK_THR} AND head_nod≤{POSTURE_NOD_THR}")
    print(SEP)
    print(f"{'Video':<14} {'Time':>13}  "
          f"{'furrow':>7} {'blink':>7} {'shld':>7} {'h2f':>5}  "
          f"{'P-Emo':>6} {'G-Emo':>6} {'E?':>3}  "
          f"{'P-Pos':>6} {'G-Pos':>6} {'P?':>3}")
    print("-" * 80)
    for r in rows:
        t = f"{r['start']}-{r['end']}"
        print(f"{r['video_id']:<14} {t:>13}  "
              f"{r['brow_furrow']:>7.3f} {r['blink_rate']:>7.3f} "
              f"{r['shoulder_tension']:>7.3f} {r['hand_to_face']:>5.2f}  "
              f"{r['pred_emotion']:>6} {r['gt_emotion']:>6} {r['emotion_match']:>3}  "
              f"{r['pred_posture']:>6} {r['gt_posture']:>6} {r['posture_match']:>3}")
    print(SEP)


def table4_metrics(rows: list):
    emo = metrics(rows, "pred_emotion", "gt_emotion")
    pos = metrics(rows, "pred_posture", "gt_posture")

    print(f"\n{SEP}")
    print("Table 4.  Evaluation Metrics")
    print(SEP)
    print(f"{'':15} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}  "
          f"{'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("-" * 80)
    for name, m in [("감정 (Emotion)", emo), ("자세 (Posture)", pos)]:
        print(f"{name:<15} {m['accuracy']:>10.3f} {m['precision']:>10.3f} "
              f"{m['recall']:>8.3f} {m['f1']:>8.3f}  "
              f"{m['tp']:>4} {m['tn']:>4} {m['fp']:>4} {m['fn']:>4}")
    print(SEP)

    # 혼동 행렬
    print("\n  Confusion Matrix — 감정")
    print(f"              GT 긍정   GT 부정")
    print(f"  예측 긍정   {emo['tp']:>7}   {emo['fp']:>7}")
    print(f"  예측 부정   {emo['fn']:>7}   {emo['tn']:>7}")
    print("\n  Confusion Matrix — 자세")
    print(f"              GT 긍정   GT 부정")
    print(f"  예측 긍정   {pos['tp']:>7}   {pos['fp']:>7}")
    print(f"  예측 부정   {pos['fn']:>7}   {pos['tn']:>7}")


def table5_per_speaker(rows: list):
    speakers = sorted({r["speaker"] for r in rows})
    print(f"\n{SEP}")
    print("Table 5.  Per-Speaker Accuracy")
    print(SEP)
    print(f"{'Speaker':<10} {'Segs':>5}  "
          f"{'Emo Acc':>8} {'Emo F1':>7}  "
          f"{'Pos Acc':>8} {'Pos F1':>7}")
    print("-" * 80)
    for spk in speakers:
        spk_rows = [r for r in rows if r["speaker"] == spk]
        if not spk_rows:
            continue
        emo = metrics(spk_rows, "pred_emotion", "gt_emotion")
        pos = metrics(spk_rows, "pred_posture",  "gt_posture")
        print(f"{spk:<10} {len(spk_rows):>5}  "
              f"{emo['accuracy']:>8.3f} {emo['f1']:>7.3f}  "
              f"{pos['accuracy']:>8.3f} {pos['f1']:>7.3f}")
    print(SEP)


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
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    use_cache = "--cache" in sys.argv
    os.makedirs(CACHE_DIR, exist_ok=True)

    labels  = load_labels()
    results = analyze_all(labels, use_cache)

    rows = build_comparison_rows(labels, results)
    if not rows:
        print("분석된 세그먼트가 없습니다.")
        sys.exit(1)

    table1_label_distribution(labels)
    table2_system_output(labels, results)
    table3_per_segment_comparison(rows)
    table4_metrics(rows)
    table5_per_speaker(rows)
    save_csv(rows)

    print(f"\n완료 — 총 {len(rows)}개 세그먼트 비교")
