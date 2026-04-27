"""
clustering.py
수집된 특징 벡터들을 PCA로 차원 축소 후 K-Means 클러스터링
"""

from typing import List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClusterResult:
    labels: np.ndarray          # 프레임별 클러스터 ID
    n_clusters: int
    pca_vectors: np.ndarray     # PCA 축소된 벡터 (시각화용)
    cluster_centers_pca: np.ndarray  # PCA 공간에서의 중심점
    explained_variance: float   # PCA가 설명하는 분산 비율

def find_optimal_k(X_pca: np.ndarray, k_min: int = 2, k_max: int = 6) -> int:
    k_range = list(range(k_min, min(k_max + 1, X_pca.shape[0])))
    if len(k_range) == 1:
        return k_range[0]

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)

    # 정규화된 기울기 변화율로 elbow 찾기
    inertias_norm = [v / inertias[0] for v in inertias]
    deltas = [inertias_norm[i] - inertias_norm[i+1] for i in range(len(inertias_norm)-1)]
    second_deltas = [deltas[i] - deltas[i+1] for i in range(len(deltas)-1)]

    # 기울기가 가장 크게 꺾이는 지점
    if second_deltas:
        elbow_idx = second_deltas.index(max(second_deltas)) + 1
    else:
        elbow_idx = deltas.index(max(deltas))

    optimal_k = k_range[elbow_idx]
    print(f"최적 클러스터 수: {optimal_k} (inertias: {[round(v,1) for v in inertias]})")
    return optimal_k

def run_clustering(
    vectors: List[np.ndarray],
    n_clusters: Optional[int] = None,  # None이면 자동 결정
    pca_components: int = 30,
) -> ClusterResult:
    """
       특징 벡터 리스트를 받아 클러스터링 결과를 반환한다.

       Args:
           vectors: 프레임별 정규화된 특징 벡터 리스트
           n_clusters: 클러스터 수 (기본 3)
           pca_components: PCA 축소 차원 (기본 30)
       """
    X = np.stack(vectors)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(pca_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = float(np.sum(pca.explained_variance_ratio_))

    # n_clusters가 None이면 자동 결정
    if n_clusters is None:
        n_clusters = find_optimal_k(X_pca)
    else:
        n_clusters = min(n_clusters, X.shape[0])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        pca_vectors=X_pca,
        cluster_centers_pca=kmeans.cluster_centers_,
        explained_variance=explained,
    )
