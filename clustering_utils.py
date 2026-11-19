import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import warnings

warnings.filterwarnings('ignore')


def normalize_embeddings(embeddings):
    """L2归一化embeddings（用于余弦距离）"""
    return normalize(embeddings, norm='l2', axis=1)


def cluster_acc(y_true, y_pred):
    """
    计算聚类准确率（使用匈牙利算法匹配）
    """
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size


def compute_metrics(true_labels, pred_labels, embeddings):
    """
    计算聚类评估指标

    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
        embeddings: 特征向量（用于计算轮廓系数）

    Returns:
        dict: 包含 NMI, ARI, ACC, Silhouette 的字典
    """
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    acc = cluster_acc(true_labels, pred_labels)

    n_clusters = len(np.unique(pred_labels))
    if n_clusters > 1 and n_clusters < len(pred_labels):
        sil = silhouette_score(embeddings, pred_labels, metric='euclidean')
    else:
        sil = -1.0

    return {
        'NMI': float(nmi),
        'ARI': float(ari),
        'ACC': float(acc),
        'Silhouette': float(sil)
    }


def gap_statistic(X, max_k=20, n_refs=10, metric='euclidean'):
    """
    Gap Statistic 自动确定最优簇数

    Args:
        X: 数据矩阵 (n_samples, n_features)
        max_k: 最大簇数
        n_refs: 参考数据集数量
        metric: 距离度量 ('euclidean' 或 'cosine')

    Returns:
        optimal_k: 最优簇数
    """
    gaps = np.zeros(max_k - 1)
    s_k = np.zeros(max_k - 1)

    if metric == 'cosine':
        X = normalize_embeddings(X)

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wk = _compute_wk(X, labels)

        ref_wks = []
        for _ in range(n_refs):
            ref_data = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            if metric == 'cosine':
                ref_data = normalize_embeddings(ref_data)
            ref_labels = KMeans(n_clusters=k, random_state=None, n_init=10).fit_predict(ref_data)
            ref_wks.append(_compute_wk(ref_data, ref_labels))

        ref_wks = np.array(ref_wks)
        gap = np.log(ref_wks.mean()) - np.log(wk)
        s_k[k - 2] = np.sqrt(1 + 1.0 / n_refs) * ref_wks.std()
        gaps[k - 2] = gap

    # 选择满足 Gap(k) >= Gap(k+1) - s_{k+1} 的最小 k
    for k in range(len(gaps) - 1):
        if gaps[k] >= gaps[k + 1] - s_k[k + 1]:
            return k + 2

    return max_k


def _compute_wk(X, labels):
    """计算簇内平方距离和"""
    wk = 0.0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        if len(cluster_points) > 1:
            center = cluster_points.mean(axis=0)
            wk += np.sum((cluster_points - center) ** 2)
    return wk


def evaluate_clustering(embeddings, true_labels, n_clusters, method='hierarchical_gap', use_cosine=True):
    """
    通用聚类评估函数（向后兼容）

    Args:
        embeddings: 特征向量
        true_labels: 真实标签
        n_clusters: 簇数
        method: 聚类方法 ('hierarchical_gap', 'hierarchical_silhouette', 'kmeans')
        use_cosine: 是否使用余弦距离

    Returns:
        dict: 评估结果
    """
    if method == 'hierarchical_gap':
        return evaluate_clustering_hierarchical_auto(
            embeddings, true_labels, n_clusters=n_clusters,
            method='average' if use_cosine else 'ward',
            auto_cut_method='gap',
            use_cosine=use_cosine
        )
    elif method == 'hierarchical_silhouette':
        return evaluate_clustering_hierarchical_auto(
            embeddings, true_labels, n_clusters=n_clusters,
            method='average' if use_cosine else 'ward',
            auto_cut_method='silhouette',
            use_cosine=use_cosine
        )
    elif method == 'kmeans':
        return evaluate_clustering_kmeans(
            embeddings, true_labels, n_clusters,
            use_cosine=use_cosine
        )
    else:
        return evaluate_clustering_hierarchical_auto(
            embeddings, true_labels, n_clusters=n_clusters,
            method='average' if use_cosine else 'ward',
            auto_cut_method='gap',
            use_cosine=use_cosine
        )


def evaluate_clustering_hierarchical_auto(
        embeddings,
        true_labels,
        n_clusters=None,
        method='ward',
        auto_cut_method='gap',
        max_k=20,
        use_cosine=False
):
    """
    层次聚类 + 自动确定簇数

    Args:
        embeddings: 特征向量
        true_labels: 真实标签
        n_clusters: 指定簇数（如果为None则自动确定）
        method: 链接方法 ('ward', 'average', 'complete', 'single')
        auto_cut_method: 自动切割方法 ('gap' 或 'silhouette')
        max_k: 最大簇数
        use_cosine: 是否使用余弦距离

    Returns:
        dict: 评估结果
    """
    if use_cosine:
        metric = 'cosine'
        embeddings_for_clustering = embeddings  # AgglomerativeClustering 会处理
    else:
        metric = 'euclidean'
        embeddings_for_clustering = embeddings

    if n_clusters is None:
        if auto_cut_method == 'gap':
            optimal_k = gap_statistic(embeddings, max_k=max_k, metric=metric)
        elif auto_cut_method == 'silhouette':
            optimal_k = _find_optimal_k_silhouette(embeddings, max_k=max_k, metric=metric)
        else:
            raise ValueError(f"Unknown auto_cut_method: {auto_cut_method}")
    else:
        optimal_k = n_clusters

    # 注意：ward 方法只支持欧氏距离
    if method == 'ward' and use_cosine:
        print(f"   Ward 方法不支持余弦距离，自动切换为 average 方法")
        method = 'average'

    clustering = AgglomerativeClustering(
        n_clusters=optimal_k,
        linkage=method,
        metric=metric
    )
    pred_labels = clustering.fit_predict(embeddings_for_clustering)

    metrics = compute_metrics(true_labels, pred_labels, embeddings)

    return {
        **metrics,
        'pred_labels': pred_labels,
        'optimal_k': optimal_k,
        'method': f'hierarchical_{auto_cut_method}',
        'linkage_method': method,
        'distance_metric': metric  # 记录使用的距离度量
    }


def _find_optimal_k_silhouette(X, max_k=20, metric='euclidean'):
    """使用轮廓系数确定最优簇数"""
    if metric == 'cosine':
        X = normalize_embeddings(X)
        metric_for_silhouette = 'euclidean'
    else:
        metric_for_silhouette = metric

    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(n_clusters=k, linkage='average')
        labels = clustering.fit_predict(X)
        score = silhouette_score(X, labels, metric=metric_for_silhouette)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def evaluate_clustering_kmeans(
        embeddings,
        true_labels,
        n_clusters,
        use_cosine=False
):
    """
    KMeans 聚类评估

    Args:
        embeddings: 特征向量
        true_labels: 真实标签
        n_clusters: 簇数
        use_cosine: 是否使用余弦距离

    Returns:
        dict: 评估结果
    """
    # 如果使用余弦距离，必须先归一化
    if use_cosine:
        embeddings_to_use = normalize_embeddings(embeddings)
        metric = 'cosine'
    else:
        embeddings_to_use = embeddings
        metric = 'euclidean'

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings_to_use)

    metrics = compute_metrics(true_labels, pred_labels, embeddings_to_use)

    return {
        **metrics,
        'pred_labels': pred_labels,
        'optimal_k': n_clusters,
        'method': 'kmeans',
        'distance_metric': metric  # 记录使用的距离度量
    }


def compute_acrl_cluster_centers(embeddings, labels, alpha=0.5, verbose=True):
    """
    ACRL: 自适应簇代表学习 (Adaptive Cluster Representative Learning)

    同时考虑空间聚合性和语义一致性，计算更鲁棒的簇中心

    数学原理：
    1. 聚合度权重: w_agg(v_i) = exp(-d_i / σ)
       其中 d_i 为样本到簇均值的欧氏距离，σ 为簇内距离标准差

    2. 语义一致性权重: w_sem(v_i) = exp(sim(v_i, μ)) / Z
       其中 sim 为余弦相似度，Z 为归一化常数

    3. 融合权重: w(v_i) = α * w_agg(v_i) + (1-α) * w_sem(v_i)

    4. 自适应簇中心: c̃ = Σ w(v_i) * v_i

    Args:
        embeddings: 特征向量 (n_samples, n_features)
        labels: 聚类标签 (n_samples,)
        alpha: 平衡系数 [0,1]
               - α=1: 完全基于空间聚合性
               - α=0: 完全基于语义一致性
               - α=0.5: 平衡（推荐）
        verbose: 是否打印详细信息

    Returns:
        cluster_centers: 自适应簇中心 (n_clusters, n_features)
        cluster_info: 每个簇的详细信息
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_features = embeddings.shape[1]

    cluster_centers = np.zeros((n_clusters, n_features), dtype=np.float32)
    cluster_info = {}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"ACRL 自适应簇代表学习")
        print(f"{'=' * 70}")
        print(f"   簇数: {n_clusters}")
        print(f"   特征维度: {n_features}")
        print(f"   平衡系数 α: {alpha:.2f}")
        print(f"{'=' * 70}\n")

    for idx, label in enumerate(unique_labels):
        cluster_mask = (labels == label)
        cluster_samples = embeddings[cluster_mask]
        n_samples = len(cluster_samples)

        if n_samples == 0:
            continue

        mu = cluster_samples.mean(axis=0)

        # 聚合度权重
        # d_i: 每个样本到簇均值的欧氏距离
        distances = np.linalg.norm(cluster_samples - mu, axis=1)

        # σ: 簇内距离标准差（避免除零）
        sigma = distances.std() + 1e-8

        # w_agg(v_i) = exp(-d_i / σ)
        w_agg = np.exp(-distances / sigma)

        # 语义一致性权重
        mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
        samples_norm = cluster_samples / (np.linalg.norm(cluster_samples, axis=1, keepdims=True) + 1e-8)

        # sim(v_i, μ): 余弦相似度
        cos_sim = np.dot(samples_norm, mu_norm)

        # w_sem(v_i) = exp(sim(v_i, μ)) / Z
        w_sem_unnorm = np.exp(cos_sim)
        Z = w_sem_unnorm.sum() + 1e-8  # 归一化常数
        w_sem = w_sem_unnorm / Z

        # w(v_i) = α * w_agg(v_i) + (1-α) * w_sem(v_i)
        w_combined = alpha * w_agg + (1 - alpha) * w_sem

        w_combined = w_combined / (w_combined.sum() + 1e-8)

        # c̃ = Σ w(v_i) * v_i
        cluster_center = np.sum(w_combined[:, np.newaxis] * cluster_samples, axis=0)
        cluster_centers[idx] = cluster_center

        cluster_info[int(label)] = {
            'n_samples': int(n_samples),
            'mean_distance': float(distances.mean()),
            'std_distance': float(sigma),
            'mean_cosine_sim': float(cos_sim.mean()),
            'weight_entropy': float(-np.sum(w_combined * np.log(w_combined + 1e-8))),
            'top_weight': float(w_combined.max()),
            'mean_weight': float(w_combined.mean())
        }

        if verbose and idx < 5:  # 只打印前5个簇
            print(f"簇 {label}:")
            print(f"   样本数: {n_samples}")
            print(f"   平均距离: {distances.mean():.4f} ± {sigma:.4f}")
            print(f"   平均余弦相似度: {cos_sim.mean():.4f}")
            print(f"   权重熵: {cluster_info[int(label)]['weight_entropy']:.4f} "
                  f"(越高越均匀)")
            print(f"   最大权重: {w_combined.max():.4f} "
                  f"(越低说明权重越分散)")
            print()

    if verbose:
        print(f"{'=' * 70}")
        print(f"ACRL 簇中心计算完成")
        print(f"{'=' * 70}\n")

    return cluster_centers, cluster_info


def acrl_kmeans(embeddings, n_clusters, alpha=0.5, max_iter=10,
                use_cosine=False, verbose=False):
    """
    ACRL-KMeans聚类算法（在迭代中使用ACRL计算簇中心）

    策略：
        1. 使用标准KMeans初始化
        2. 迭代使用ACRL计算簇中心
        3. 基于新簇中心重新分配样本
        4. 重复2-3直到收敛或达到最大迭代次数

    Args:
        embeddings: ndarray - 嵌入向量
        n_clusters: int - 簇数量
        alpha: float - ACRL平衡系数 (0~1)
        max_iter: int - 最大迭代次数（建议3-10次）
        use_cosine: bool - 是否使用余弦距离
        verbose: bool - 是否打印详细信息

    Returns:
        labels: ndarray - 最终簇标签
        acrl_centers: ndarray - ACRL簇中心
        n_iterations: int - 实际迭代次数
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"ACRL-KMeans 聚类")
        print(f"{'=' * 70}")
        print(f"   样本数: {len(embeddings)}")
        print(f"   簇数: {n_clusters}")
        print(f"   α: {alpha}")
        print(f"   最大迭代: {max_iter}")
        print(f"   距离度量: {'余弦' if use_cosine else '欧氏'}")
        print(f"{'=' * 70}\n")

    if verbose:
        print("  [1/3] 使用标准KMeans初始化...")

    if use_cosine:
        normalized_emb = normalize_embeddings(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_emb)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

    if verbose:
        print(f"    初始化完成\n")
        print(f"  [2/3] ACRL迭代优化（最多{max_iter}次）...")

    for iteration in range(max_iter):
        acrl_centers, cluster_info = compute_acrl_cluster_centers(
            embeddings, labels, alpha=alpha, verbose=False
        )

        if use_cosine:
            distances = cosine_distances(embeddings, acrl_centers)
        else:
            distances = euclidean_distances(embeddings, acrl_centers)

        new_labels = np.argmin(distances, axis=1)

        changed = (new_labels != labels).sum()
        change_rate = changed / len(labels) * 100
        labels = new_labels

        if verbose:
            print(f"    迭代 {iteration + 1}: {changed} 个样本改变标签 ({change_rate:.2f}%)")

        if changed == 0:
            if verbose:
                print(f"    收敛于第 {iteration + 1} 次迭代")
            break

    if verbose:
        print(f"\n  [3/3] 完成！")
        print(f"{'=' * 70}\n")

    return labels, acrl_centers, iteration + 1


def evaluate_clustering_acrl_kmeans(embeddings, true_labels, n_clusters,
                                    alpha=0.5, max_iter=10, use_cosine=False,
                                    verbose=False):
    """
    使用ACRL-KMeans进行聚类评估

    Args:
        embeddings: 嵌入向量
        true_labels: 真实标签
        n_clusters: 簇数量
        alpha: ACRL平衡系数
        max_iter: 最大迭代次数
        use_cosine: 是否使用余弦距离
        verbose: 是否打印详细信息

    Returns:
        dict: 评估结果
    """
    pred_labels, acrl_centers, n_iter = acrl_kmeans(
        embeddings, n_clusters, alpha=alpha,
        max_iter=max_iter, use_cosine=use_cosine, verbose=verbose
    )

    metrics = compute_metrics(true_labels, pred_labels, embeddings)

    if verbose:
        print(f"ACRL-KMeans 结果:")
        print(f"   NMI: {metrics['NMI']:.4f}")
        print(f"   ACC: {metrics['ACC']:.4f}")
        print(f"   ARI: {metrics['ARI']:.4f}")
        print(f"   迭代次数: {n_iter}\n")

    return {
        **metrics,
        'pred_labels': pred_labels,
        'optimal_k': n_clusters,
        'n_iterations': n_iter,
        'alpha': alpha,
        'method': 'acrl_kmeans',
        'distance_metric': 'cosine' if use_cosine else 'euclidean'
    }


def build_coarse_labels_with_acrl(embeddings, fine_labels, n_coarse_clusters,
                                  alpha=0.5, n_acrl_iterations=2,
                                  use_cosine=True, verbose=True,
                                  initial_labels=None):
    """
    使用ACRL混合策略构建粗粒度标签(用于Stage1→Stage2)
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    fine_labels = np.asarray(fine_labels)

    fine_labels_min = fine_labels.min()
    fine_labels_normalized = fine_labels - fine_labels_min
    n_fine = len(np.unique(fine_labels_normalized))

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"使用ACRL混合策略构建粗粒度标签")
        print(f"{'=' * 70}\n")
        print(f"  目标簇数: {n_coarse_clusters}")
        print(f"  ACRL α: {alpha}")
        print(f"  优化次数: {n_acrl_iterations}")
        print(f"  距离度量: {'余弦' if use_cosine else '欧氏'}")
        print(f"  细粒度类别数: {n_fine} (标签范围: {fine_labels.min()}-{fine_labels.max()})\n")

    if initial_labels is not None:
        coarse_labels = np.asarray(initial_labels)
        if coarse_labels.min() > 0:
            coarse_labels = coarse_labels - coarse_labels.min()
        actual_n_clusters = len(np.unique(coarse_labels))
        if verbose:
            print(f"  [1/3] 使用提供的初始标签...")
            print(f"    初始标签有 {actual_n_clusters} 个簇")
    else:
        if verbose:
            print("  [1/3] 使用层次聚类初始化...")

        if use_cosine:
            normalized_emb = normalize_embeddings(embeddings)
            Z = linkage(normalized_emb, method='average', metric='cosine')
        else:
            Z = linkage(embeddings, method='average', metric='euclidean')

        coarse_labels = fcluster(Z, n_coarse_clusters, criterion='maxclust') - 1
        actual_n_clusters = len(np.unique(coarse_labels))

        if verbose:
            print(f"    初始聚类完成,得到 {actual_n_clusters} 个簇")

    if verbose:
        print(f"\n  [2/3] ACRL迭代优化({n_acrl_iterations}次)...")

    for iteration in range(n_acrl_iterations):
        if verbose:
            print(f"\n    --- 第 {iteration + 1} 次优化 ---")

        acrl_centers, cluster_info = compute_acrl_cluster_centers(
            embeddings, coarse_labels, alpha=alpha, verbose=False
        )

        if use_cosine:
            distances = cosine_distances(embeddings, acrl_centers)
        else:
            distances = euclidean_distances(embeddings, acrl_centers)

        new_labels = np.argmin(distances, axis=1)

        changed = (new_labels != coarse_labels).sum()
        change_rate = changed / len(coarse_labels) * 100

        if verbose:
            print(f"    标签变化: {changed} 个样本 ({change_rate:.2f}%)")

        coarse_labels = new_labels

    actual_n_clusters = len(np.unique(coarse_labels))

    if verbose:
        print(f"\n  [3/3] 评估粗粒度标签质量...")

    # 使用归一化后的标签创建 confusion 矩阵
    confusion = np.zeros((actual_n_clusters, n_fine))

    for i in range(len(coarse_labels)):
        confusion[coarse_labels[i], fine_labels_normalized[i]] += 1

    cluster_purities = []
    for c in range(actual_n_clusters):
        if confusion[c].sum() > 0:
            purity = confusion[c].max() / confusion[c].sum()
            cluster_purities.append(purity)

    avg_purity = np.mean(cluster_purities) if cluster_purities else 0

    if verbose:
        print(f"    平均簇纯度: {avg_purity:.4f}")
        print(f"    最终簇数: {actual_n_clusters}")
        print(f"\n{'=' * 70}")
        print(f"ACRL混合策略完成!")
        print(f"{'=' * 70}\n")

    return coarse_labels, acrl_centers


def evaluate_clustering_comparison(embeddings, true_labels, n_clusters,
                                   methods=None, use_cosine=True, verbose=True):
    """
    对比多种聚类方法

    Args:
        embeddings: 嵌入向量 (n_samples, n_features)
        true_labels: 真实标签 (n_samples,)
        n_clusters: 簇数量
        methods: 聚类方法列表，可选：
                 ['hierarchical_gap', 'hierarchical_silhouette', 'kmeans', 'acrl_kmeans',
                  'acrl_hierarchical_gap', 'acrl_hierarchical_silhouette']
                 如果为None，默认使用前3种
        use_cosine: 是否使用余弦距离
        verbose: 是否打印详细信息

    Returns:
        dict: 包含所有方法结果的字典
            {
                'method_results': {方法名: 评估结果},
                'best_method': 最佳方法名,
                'overall_metrics': 总体平均指标
            }
    """
    if methods is None:
        methods = ['hierarchical_gap', 'hierarchical_silhouette', 'kmeans']

    results = {}
    best_nmi = 0
    best_method = None

    for method in methods:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"评估方法: {method}")
            print(f"{'=' * 60}")

        try:
            if method == 'acrl_kmeans':
                result = evaluate_clustering_acrl_kmeans(
                    embeddings, true_labels, n_clusters,
                    alpha=0.5,
                    max_iter=10,
                    use_cosine=use_cosine,
                    verbose=verbose
                )
            elif method == 'acrl_hierarchical_gap':
                result = evaluate_clustering_acrl_hierarchical_gap(
                    embeddings, true_labels,
                    n_clusters=n_clusters,
                    method='average' if use_cosine else 'ward',
                    alpha=0.5,
                    use_cosine=use_cosine,
                    verbose=verbose
                )
            elif method == 'acrl_hierarchical_silhouette':
                result = evaluate_clustering_acrl_hierarchical_silhouette(
                    embeddings, true_labels,
                    k_range=None,
                    method='average' if use_cosine else 'ward',
                    alpha=0.5,
                    use_cosine=use_cosine,
                    verbose=verbose
                )
            elif method == 'kmeans':
                result = evaluate_clustering_kmeans(
                    embeddings, true_labels, n_clusters,
                    use_cosine=use_cosine
                )
            elif method == 'hierarchical_gap':
                result = evaluate_clustering_hierarchical_auto(
                    embeddings, true_labels, n_clusters,
                    method='average' if use_cosine else 'ward',
                    auto_cut_method='gap',
                    use_cosine=use_cosine
                )
            elif method == 'hierarchical_silhouette':
                result = evaluate_clustering_hierarchical_auto(
                    embeddings, true_labels, n_clusters,
                    method='average' if use_cosine else 'ward',
                    auto_cut_method='silhouette',
                    use_cosine=use_cosine
                )
            else:
                if verbose:
                    print(f"未知方法: {method}，跳过")
                continue

            results[method] = result

            if verbose:
                print(f"  NMI: {result['NMI']:.4f}")
                print(f"  ACC: {result['ACC']:.4f}")
                print(f"  ARI: {result['ARI']:.4f}")
                if 'n_iterations' in result:
                    print(f"  迭代次数: {result['n_iterations']}")
                if 'optimal_k' in result:
                    print(f"  最优簇数: {result['optimal_k']}")

            if result['NMI'] > best_nmi:
                best_nmi = result['NMI']
                best_method = method

        except Exception as e:
            if verbose:
                print(f"方法 {method} 执行失败: {str(e)}")
            import traceback
            if verbose:
                traceback.print_exc()
            continue

    if results:
        overall_nmi = np.mean([r['NMI'] for r in results.values()])
        overall_ari = np.mean([r['ARI'] for r in results.values()])
        overall_acc = np.mean([r['ACC'] for r in results.values()])
    else:
        overall_nmi = overall_ari = overall_acc = 0.0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"最佳方法: {best_method} (NMI={best_nmi:.4f})")
        print(f"{'=' * 60}\n")

    return {
        'method_results': results,
        'best_method': best_method,
        'overall_metrics': {
            'NMI': overall_nmi,
            'ARI': overall_ari,
            'ACC': overall_acc
        }
    }


def compute_acrl_distance_matrix(
        embeddings: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.5,
        n_iterations: int = 2,
        use_cosine: bool = True,
        verbose: bool = False
) -> np.ndarray:
    """
    使用ACRL策略计算距离矩阵

    Args:
        embeddings: 特征向量 (n_samples, n_features)
        labels: 当前聚类标签 (n_samples,)
        alpha: ACRL平衡系数 [0,1]
        n_iterations: ACRL迭代次数
        use_cosine: 是否使用余弦距离
        verbose: 是否打印详细信息

    Returns:
        distance_matrix: ACRL优化后的距离矩阵 (n_samples, n_samples)
    """
    n_samples = len(embeddings)

    if use_cosine:
        D_base = cosine_distances(embeddings)
    else:
        D_base = squareform(pdist(embeddings, metric='euclidean'))

    D_acrl = D_base.copy()

    if verbose:
        print(f"   ACRL距离矩阵计算:")
        print(f"      初始距离: {'余弦' if use_cosine else '欧氏'}")
        print(f"      平衡系数α: {alpha:.2f}")
        print(f"      迭代次数: {n_iterations}")

    for iter_idx in range(n_iterations):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        centers = np.zeros((n_clusters, embeddings.shape[1]))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centers[i] = embeddings[mask].mean(axis=0)

        D_intra = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            center_i = centers[labels[i]]
            for j in range(n_samples):
                center_j = centers[labels[j]]
                if labels[i] == labels[j]:
                    # 类内：距离应该更小
                    if use_cosine:
                        d_i = 1 - np.dot(embeddings[i], center_i) / (
                                np.linalg.norm(embeddings[i]) * np.linalg.norm(center_i) + 1e-8
                        )
                    else:
                        d_i = np.linalg.norm(embeddings[i] - center_i)
                    D_intra[i, j] = d_i
                else:
                    # 类间：距离应该更大
                    if use_cosine:
                        d_ij = 1 - np.dot(center_i, center_j) / (
                                np.linalg.norm(center_i) * np.linalg.norm(center_j) + 1e-8
                        )
                    else:
                        d_ij = np.linalg.norm(center_i - center_j)
                    D_intra[i, j] = d_ij

        D_acrl = alpha * D_base + (1 - alpha) * D_intra

        if verbose:
            avg_intra = D_intra[labels[:, None] == labels].mean()
            avg_inter = D_intra[labels[:, None] != labels].mean()
            print(f"      Iter {iter_idx + 1}: 类内={avg_intra:.4f}, 类间={avg_inter:.4f}")

    return D_acrl


def gap_statistic_with_acrl(
        embeddings: np.ndarray,
        k_range: range,
        alpha: float = 0.5,
        n_iterations: int = 2,
        use_cosine: bool = True,
        n_refs: int = 10,
        verbose: bool = False
) -> int:
    """
    使用ACRL优化的Gap统计量确定最优簇数

    Args:
        embeddings: 特征向量
        k_range: 簇数搜索范围
        alpha: ACRL平衡系数
        n_iterations: ACRL迭代次数
        use_cosine: 是否使用余弦距离
        n_refs: 参考数据集数量
        verbose: 是否打印详细信息

    Returns:
        optimal_k: 最优簇数
    """
    gaps = []
    sks = []

    if verbose:
        print(f"\n   Gap统计量计算 (ACRL增强)...")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        D_acrl = compute_acrl_distance_matrix(
            embeddings, labels, alpha, n_iterations, use_cosine, verbose=False
        )

        Wk = 0
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                cluster_distances = D_acrl[mask][:, mask]
                Wk += cluster_distances.sum() / (2 * mask.sum())

        Wkbs = []
        for _ in range(n_refs):
            ref_data = np.random.uniform(
                embeddings.min(axis=0),
                embeddings.max(axis=0),
                embeddings.shape
            )

            ref_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            ref_labels = ref_kmeans.fit_predict(ref_data)

            D_ref = compute_acrl_distance_matrix(
                ref_data, ref_labels, alpha, n_iterations, use_cosine, verbose=False
            )

            Wkb = 0
            for i in range(k):
                mask = ref_labels == i
                if mask.sum() > 0:
                    cluster_distances = D_ref[mask][:, mask]
                    Wkb += cluster_distances.sum() / (2 * mask.sum())
            Wkbs.append(np.log(Wkb))

        gap = np.mean(Wkbs) - np.log(Wk)
        sk = np.std(Wkbs) * np.sqrt(1 + 1 / n_refs)

        gaps.append(gap)
        sks.append(sk)

        if verbose:
            print(f"      K={k}: Gap={gap:.4f}, sk={sk:.4f}")

    # 找到最优K（第一个满足 Gap(k) >= Gap(k+1) - sk(k+1) 的k）
    optimal_k = k_range[0]
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            optimal_k = k_range[i]
            break

    if verbose:
        print(f"   ACRL-Gap最优簇数: K={optimal_k}")

    return optimal_k


def evaluate_clustering_acrl_hierarchical_gap(
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        n_clusters: int = None,
        method: str = 'average',
        alpha: float = 0.5,
        use_cosine: bool = True,
        verbose: bool = False
) -> dict:
    """
    ACRL增强的层次聚类 + Gap统计量（先Gap后ACRL策略）

    优化策略：
    1. 先用普通层次聚类 + Gap统计量确定最优K
    2. 然后用ACRL在最优K上优化簇中心
    3. 基于ACRL簇中心重新分配标签

    这样避免了对每个K都计算ACRL，大幅降低计算量

    Args:
        embeddings: 特征向量 (n_samples, n_features)
        true_labels: 真实标签
        n_clusters: 指定簇数（如果为None则使用Gap统计量自动确定）
        method: 层次聚类方法 ('average', 'ward', 'complete')
        alpha: ACRL平衡系数
        use_cosine: 是否使用余弦距离
        verbose: 是否打印详细信息

    Returns:
        result_dict: 包含NMI, ARI, ACC等指标
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"ACRL层次聚类 + Gap统计量（先Gap后ACRL）")
        print(f"{'=' * 60}")
        print(f"   ACRL α: {alpha}")
        print(f"   距离度量: {'余弦' if use_cosine else '欧氏'}")

    if verbose:
        print(f"\n   [1/3] 使用Gap统计量确定最优K...")

    if n_clusters is None:
        if use_cosine:
            embeddings_for_gap = normalize_embeddings(embeddings)
            optimal_k = gap_statistic(embeddings_for_gap, max_k=20, metric='cosine')
        else:
            optimal_k = gap_statistic(embeddings, max_k=20, metric='euclidean')

        if verbose:
            print(f"      Gap统计量确定最优K = {optimal_k}")
    else:
        optimal_k = n_clusters
        if verbose:
            print(f"      使用指定K = {optimal_k}")

    if verbose:
        print(f"\n   [2/3] 对K={optimal_k}进行层次聚类...")

    if use_cosine:
        embeddings_norm = normalize_embeddings(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=optimal_k,
            linkage=method,
            metric='cosine'
        )
        initial_labels = clustering.fit_predict(embeddings_norm)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=optimal_k,
            linkage=method if method != 'ward' else 'ward',
            metric='euclidean'
        )
        initial_labels = clustering.fit_predict(embeddings)

    if verbose:
        print(f"      初始聚类完成")

    if verbose:
        print(f"\n   [3/3] 使用ACRL优化簇中心...")

    acrl_centers, cluster_info = compute_acrl_cluster_centers(
        embeddings, initial_labels, alpha=alpha, verbose=False
    )

    if verbose:
        avg_purity = np.mean([info['mean_cosine_sim'] for info in cluster_info.values()])
        print(f"      ACRL簇中心计算完成")
        print(f"      平均簇内相似度: {avg_purity:.4f}")

    if use_cosine:
        distances = cosine_distances(embeddings, acrl_centers)
    else:
        distances = euclidean_distances(embeddings, acrl_centers)

    final_labels = np.argmin(distances, axis=1)

    n_changed = (final_labels != initial_labels).sum()
    if verbose:
        print(f"      ACRL重分配: {n_changed}个样本改变标签 ({n_changed / len(embeddings) * 100:.2f}%)")

    metrics = compute_metrics(true_labels, final_labels, embeddings)

    if verbose:
        print(f"\n   最终结果:")
        print(f"      最优K: {optimal_k}")
        print(f"      NMI: {metrics['NMI']:.4f}")
        print(f"      ARI: {metrics['ARI']:.4f}")
        print(f"      ACC: {metrics['ACC']:.4f}")
        print(f"{'=' * 60}\n")

    return {
        'NMI': metrics['NMI'],
        'ARI': metrics['ARI'],
        'ACC': metrics['ACC'],
        'optimal_k': optimal_k,
        'pred_labels': final_labels,
        'n_changed': n_changed,
        'method': 'acrl_hierarchical_gap'
    }


def evaluate_clustering_acrl_hierarchical_silhouette(
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        k_range=None,
        method: str = 'average',
        alpha: float = 0.5,
        use_cosine: bool = True,
        verbose: bool = False
) -> dict:
    """
    ACRL增强的层次聚类 + Silhouette系数（先Silhouette后ACRL策略）

    优化策略：
    1. 先用普通层次聚类 + Silhouette系数确定最优K
    2. 然后用ACRL在最优K上优化簇中心
    3. 基于ACRL簇中心重新分配标签

    Args:
        embeddings: 特征向量
        true_labels: 真实标签
        k_range: 簇数搜索范围
        method: 层次聚类方法
        alpha: ACRL平衡系数
        use_cosine: 是否使用余弦距离
        verbose: 是否打印详细信息

    Returns:
        result_dict: 包含NMI, ARI, ACC等指标
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"ACRL层次聚类 + Silhouette系数（先Silhouette后ACRL）")
        print(f"{'=' * 60}")
        print(f"   ACRL α: {alpha}")
        print(f"   距离度量: {'余弦' if use_cosine else '欧氏'}")

    if verbose:
        print(f"\n   [1/3] 使用Silhouette系数确定最优K...")

    if k_range is None:
        n_true_clusters = len(np.unique(true_labels))
        k_range = range(max(2, n_true_clusters - 3), n_true_clusters + 4)

    best_score = -1
    optimal_k = k_range[0]

    if use_cosine:
        embeddings_for_sil = normalize_embeddings(embeddings)
    else:
        embeddings_for_sil = embeddings

    for k in k_range:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=k,
                linkage=method,
                metric='cosine' if use_cosine else 'euclidean'
            )
            labels_temp = clustering.fit_predict(embeddings_for_sil)

            score = silhouette_score(
                embeddings_for_sil,
                labels_temp,
                metric='euclidean'  # 归一化后用欧氏距离
            )

            if verbose:
                print(f"      K={k}: Silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                optimal_k = k

        except Exception as e:
            if verbose:
                print(f"      K={k}: 计算失败")
            continue

    if verbose:
        print(f"      Silhouette确定最优K = {optimal_k} (score={best_score:.4f})")

    if verbose:
        print(f"\n   [2/3] 对K={optimal_k}进行层次聚类...")

    clustering = AgglomerativeClustering(
        n_clusters=optimal_k,
        linkage=method,
        metric='cosine' if use_cosine else 'euclidean'
    )
    initial_labels = clustering.fit_predict(embeddings_for_sil)

    if verbose:
        print(f"      初始聚类完成")

    if verbose:
        print(f"\n   [3/3] 使用ACRL优化簇中心...")

    acrl_centers, cluster_info = compute_acrl_cluster_centers(
        embeddings, initial_labels, alpha=alpha, verbose=False
    )

    if verbose:
        avg_purity = np.mean([info['mean_cosine_sim'] for info in cluster_info.values()])
        print(f"      ACRL簇中心计算完成")
        print(f"      平均簇内相似度: {avg_purity:.4f}")

    if use_cosine:
        distances = cosine_distances(embeddings, acrl_centers)
    else:
        distances = euclidean_distances(embeddings, acrl_centers)

    final_labels = np.argmin(distances, axis=1)

    n_changed = (final_labels != initial_labels).sum()
    if verbose:
        print(f"      ACRL重分配: {n_changed}个样本改变标签 ({n_changed / len(embeddings) * 100:.2f}%)")

    metrics = compute_metrics(true_labels, final_labels, embeddings)

    if verbose:
        print(f"\n   最终结果:")
        print(f"      最优K: {optimal_k}")
        print(f"      NMI: {metrics['NMI']:.4f}")
        print(f"      ARI: {metrics['ARI']:.4f}")
        print(f"      ACC: {metrics['ACC']:.4f}")
        print(f"{'=' * 60}\n")

    return {
        'NMI': metrics['NMI'],
        'ARI': metrics['ARI'],
        'ACC': metrics['ACC'],
        'optimal_k': optimal_k,
        'pred_labels': final_labels,
        'n_changed': n_changed,
        'method': 'acrl_hierarchical_silhouette'
    }

def build_nested_pseudo_labels(embeddings, fine_percentile=90.0,
                               coarse_percentile=97.0):
    """Two cuts of one average-linkage cosine tree, without true labels.

    Percentiles refer to the N-1 merge heights, not all pairwise similarities.
    Higher height percentiles produce fewer clusters. Exact-count tree cuts
    only resolve tied/degenerate heights or cuts with fewer than two clusters.
    """
    from scipy.cluster.hierarchy import linkage, fcluster, cut_tree
    from scipy.spatial.distance import pdist

    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError('Pseudo labels require >=4 finite embedding rows')
    if not 0 < fine_percentile < coarse_percentile < 100:
        raise ValueError('Require 0 < fine_percentile < coarse_percentile < 100')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('Zero embedding: cosine pseudo labels are undefined')
    x = x / norms
    distances = np.clip(pdist(x, metric='cosine'), 0, 2)
    tree = linkage(distances, method='average')
    fine_height, coarse_height = np.percentile(
        tree[:, 2], [fine_percentile, coarse_percentile])
    fine = fcluster(tree, fine_height, criterion='distance') - 1
    coarse = fcluster(tree, coarse_height, criterion='distance') - 1
    fallback = not (2 <= len(np.unique(coarse)) < len(np.unique(fine)) < len(x))
    if fallback:
        k_fine = int(np.clip(round(len(x) * (1 - fine_percentile / 100)), 3, len(x)-1))
        k_coarse = int(np.clip(round(len(x) * (1 - coarse_percentile / 100)), 2, k_fine-1))
        labels = cut_tree(tree, n_clusters=[k_fine, k_coarse])
        fine, coarse = labels[:, 0], labels[:, 1]
    fine, coarse = fine.astype(np.int64), coarse.astype(np.int64)
    for label in np.unique(fine):
        if len(np.unique(coarse[fine == label])) != 1:
            raise RuntimeError('Fine partition is not nested in coarse partition')
    def stats(labels):
        sizes = np.unique(labels, return_counts=True)[1]
        return dict(clusters=len(sizes), min_size=int(sizes.min()),
                    max_size=int(sizes.max()), mean_size=float(sizes.mean()),
                    singleton_fraction=float((sizes == 1).sum() / len(sizes)))
    metadata = dict(fine=stats(fine), coarse=stats(coarse),
                    fine_percentile=float(fine_percentile),
                    coarse_percentile=float(coarse_percentile),
                    fine_height=float(fine_height), coarse_height=float(coarse_height),
                    exact_cut_fallback=fallback,
                    embedding_total_variance=float(x.var(axis=0).sum()))
    return coarse, fine, metadata


def pseudo_label_change(previous, current):
    """ARI and Hungarian-aligned change rate, invariant to label renumbering."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.cluster import contingency_matrix
    if previous is None:
        return dict(ari=None, aligned_change_rate=None)
    table = contingency_matrix(previous, current)
    rows, cols = linear_sum_assignment(-table)
    return dict(ari=float(adjusted_rand_score(previous, current)),
                aligned_change_rate=float(1 - table[rows, cols].sum() / len(current)))
