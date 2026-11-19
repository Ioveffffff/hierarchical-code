import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def augment_entity(entity, noise_ratio=0.1):
    """
    对实体进行轻微扰动增强，用于实例对比学习。
    这里可以是简单的字符随机替换、插入、删除等。
    """
    import random
    entity_str = str(entity)
    n = max(1, int(len(entity_str) * noise_ratio))
    chars = list(entity_str)
    for _ in range(n):
        idx = random.randint(0, len(chars)-1)
        chars[idx] = chr(random.randint(19968, 40869))  # 随机汉字
    return ''.join(chars)


def build_hierarchical_labels(entities, fine_labels, n_coarse_clusters=5, use_cosine=False):
    """
    使用KMeans/层次聚类构建粗粒度标签
    """
    # 简单把实体embedding当作one-hot向量
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='char')
    X = vectorizer.fit_transform(entities).toarray().astype(np.float32)

    if use_cosine:
        X = normalize(X, axis=1)

    kmeans = KMeans(n_clusters=n_coarse_clusters, random_state=42)
    coarse_labels = kmeans.fit_predict(X)
    return coarse_labels


def build_hierarchical_labels_hybrid(entities, fine_labels, embeddings, n_coarse_clusters=5, use_cosine=False):
    """
    规则 + embedding混合构建粗粒度标签
    """
    if use_cosine:
        embeddings = normalize(embeddings, axis=1)
    # 用AgglomerativeClustering做初步分簇
    clustering = AgglomerativeClustering(n_clusters=n_coarse_clusters, linkage='average', affinity='cosine' if use_cosine else 'euclidean')
    coarse_labels = clustering.fit_predict(embeddings)
    return coarse_labels


def build_coarse_labels_with_acrl(embeddings, fine_labels, n_coarse_clusters=5,
                                  alpha=0.5, n_acrl_iterations=3, use_cosine=False,
                                  verbose=False, initial_labels=None):
    """
    基于ACRL算法优化粗粒度标签
    embeddings: np.array, N x D
    fine_labels: np.array, N
    initial_labels: 初始粗粒度标签
    """
    N, D = embeddings.shape
    if use_cosine:
        embeddings = normalize(embeddings, axis=1)

    if initial_labels is None:
        kmeans = KMeans(n_clusters=n_coarse_clusters, random_state=42)
        coarse_labels = kmeans.fit_predict(embeddings)
    else:
        coarse_labels = initial_labels.copy()

    for it in range(n_acrl_iterations):
        if verbose:
            print(f"  ACRL迭代 {it+1}/{n_acrl_iterations}")
        # 计算每个粗类内细粒度纯度
        coarse_to_fine_counts = {}
        for c in np.unique(coarse_labels):
            mask = coarse_labels == c
            fine_in_cluster = fine_labels[mask]
            counts = np.bincount(fine_in_cluster)
            coarse_to_fine_counts[c] = counts

        # 根据当前簇中心重新分配样本。
        for i in range(N):
            distances = []
            for c in np.unique(coarse_labels):
                mask = coarse_labels == c
                if mask.sum() == 0:
                    distances.append(np.inf)
                    continue
                center = embeddings[mask].mean(axis=0)
                if use_cosine:
                    dist = 1 - np.dot(embeddings[i], center) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(center))
                else:
                    dist = np.linalg.norm(embeddings[i] - center)
                # 融合细粒度信息
                fine_idx = fine_labels[i]
                fine_count = coarse_to_fine_counts[c][fine_idx] if fine_idx < len(coarse_to_fine_counts[c]) else 0
                dist = alpha * dist - (1 - alpha) * fine_count
                distances.append(dist)
            coarse_labels[i] = int(np.argmin(distances))

    # 最终簇中心
    acrl_centers = np.array([embeddings[coarse_labels == c].mean(axis=0) for c in np.unique(coarse_labels)])
    return coarse_labels, acrl_centers


def evaluate_coarse_quality(coarse_labels, fine_labels):
    """
    简单评估粗粒度标签质量: 计算簇内细粒度纯度
    """
    purity_list = []
    for c in np.unique(coarse_labels):
        mask = coarse_labels == c
        fine_in_cluster = fine_labels[mask]
        if len(fine_in_cluster) == 0:
            continue
        counts = np.bincount(fine_in_cluster)
        purity = counts.max() / counts.sum()
        purity_list.append(purity)
    avg_purity = np.mean(purity_list)
    print(f"  粗粒度簇平均纯度: {avg_purity:.4f}")
    return avg_purity
