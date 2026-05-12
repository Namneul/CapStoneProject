"""
clustering.py
수집된 특징 벡터들을 PCA로 차원 축소 후 FINCH 클러스터링
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from dataclasses import dataclass
from typing import List
from finch import FINCH


@dataclass
class ClusterResult:
    labels: np.ndarray          # 프레임별 클러스터 ID
    n_clusters: int
    pca_vectors: np.ndarray     # PCA 축소된 벡터 (시각화용)
    cluster_centers_pca: np.ndarray  # PCA 공간에서의 중심점
    explained_variance: float   # PCA가 설명하는 분산 비율
    inertia: float              # 클러스터 내 거리 제곱합
    silhouette: float           # 실루엣 점수 (높을수록 분리 명확)
    davies_bouldin: float       # Davies-Bouldin 지수 (낮을수록 좋음)


def run_clustering(
    vectors: List[np.ndarray],
    pca_components: int = 30,
) -> ClusterResult:
    """
    특징 벡터 리스트를 받아 FINCH 클러스터링 결과를 반환한다.
    클러스터 수는 FINCH가 자동으로 결정한다.

    Args:
        vectors: 프레임별 정규화된 특징 벡터 리스트
        pca_components: PCA 축소 차원 (기본 30)
    """
    # float64로 변환: float32는 PCA 내부 행렬 연산에서 오버플로우 유발
    X = np.stack(vectors).astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(pca_components, X.shape[0] - 1, X.shape[1])
    # svd_solver='full': 고차원 희소 데이터에서 randomized SVD 수치 불안정 방지
    pca = PCA(n_components=n_comp, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)
    explained = float(np.sum(pca.explained_variance_ratio_))

    # FINCH: k 없이 자동 파티셔닝. 마지막 파티션(가장 적은 클러스터 수) 사용
    c, num_clust, _ = FINCH(X_pca)
    labels = c[:, -1]
    n_clusters = int(num_clust[-1])
    print(f"FINCH 클러스터 수: {n_clusters} (전체 파티션: {num_clust})")

    # 클러스터 중심점 계산
    centers = np.array([
        X_pca[labels == k].mean(axis=0) for k in range(n_clusters)
    ])

    # inertia: 각 샘플과 자기 클러스터 중심 사이 거리 제곱합
    inertia = float(sum(
        np.sum((X_pca[labels == k] - centers[k]) ** 2)
        for k in range(n_clusters)
    ))

    sil = silhouette_score(X_pca, labels) if n_clusters >= 2 else 0.0
    db = davies_bouldin_score(X_pca, labels) if n_clusters >= 2 else 0.0

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        pca_vectors=X_pca,
        cluster_centers_pca=centers,
        explained_variance=explained,
        inertia=inertia,
        silhouette=float(sil),
        davies_bouldin=float(db),
    )