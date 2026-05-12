"""
test_pipeline.py
웹캠 없이 더미 벡터로 전체 파이프라인 검증
"""
import sys
sys.path.insert(0, '/home/claude/nonverbal')

import numpy as np
from clustering import run_clustering
from output import build_output, to_json_string

np.random.seed(42)

# 더미 데이터: 3개 클러스터 패턴으로 120개 샘플 생성
def make_dummy_vectors(n=120):
    dim = 468*3 + 33*3 + 21*3 + 21*3  # 1629
    vecs = []
    for i in range(n):
        seg = i // 40  # 0,1,2 세 구간
        base = np.zeros(dim)
        if seg == 0:
            # 얼굴 + 포즈만 활성
            base[:468*3] = np.random.randn(468*3) * 0.1 + 0.5
            base[468*3:468*3+33*3] = np.random.randn(33*3) * 0.1
        elif seg == 1:
            # 얼굴 + 오른손 활성
            base[:468*3] = np.random.randn(468*3) * 0.1 - 0.5
            rh_start = 468*3 + 33*3 + 21*3
            base[rh_start:] = np.random.randn(21*3) * 0.2 + 1.0
        else:
            # 포즈 + 양손 활성
            base[468*3:468*3+33*3] = np.random.randn(33*3) * 0.15 - 1.0
            lh_start = 468*3 + 33*3
            base[lh_start:lh_start+21*3] = np.random.randn(21*3) * 0.2
            rh_start = lh_start + 21*3
            base[rh_start:] = np.random.randn(21*3) * 0.2
        vecs.append(base.astype(np.float32))
    return vecs

vectors = make_dummy_vectors(120)
timestamps = [i * 0.5 for i in range(120)]  # 0.5초 간격, 총 60초

print(f"벡터 수: {len(vectors)}, 차원: {vectors[0].shape[0]}")

# 클러스터링
result = run_clustering(vectors, pca_components=30)
print(f"PCA 설명 분산: {result.explained_variance:.3f}")
print(f"클러스터 분포: {np.bincount(result.labels)}")

# JSON 출력
output = build_output(result, timestamps, vectors)
print("\n=== Orchestrator 출력 JSON ===")
print(to_json_string(output))

# 검증
assert len(output["clusters"]) >= 1
assert abs(sum(c["ratio"] for c in output["clusters"]) - 1.0) < 0.01
assert all(len(c["timestamps"]) > 0 for c in output["clusters"])
print("\n모든 검증 통과")
