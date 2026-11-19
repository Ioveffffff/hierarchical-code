"""Propose complete upper-layer partitions from separate content and name views."""

import igraph as ig
import leidenalg
import numpy as np
from scipy.stats import rankdata


def _unit(vectors):
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all():
        raise ValueError("Leiden 输入须为至少两个有限向量")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Leiden 输入含零向量")
    return matrix / norms[:, None]


def _similarities(vectors):
    unit = _unit(vectors)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    upper = np.triu_indices(len(unit), 1)
    percentile = np.zeros_like(similarity)
    ranks = rankdata(similarity[upper], method="average") / len(upper[0])
    percentile[upper] = ranks
    percentile[(upper[1], upper[0])] = ranks
    return similarity, percentile


def _neighbors(similarity, k):
    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    k = min(k, len(scores) - 1)
    nearest = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    cutoff = np.take_along_axis(scores, nearest[:, -1:], axis=1)
    flags = scores >= cutoff - 1e-12
    np.fill_diagonal(flags, False)
    return flags


def leiden_candidates(content_vectors, name_vectors, neighbors,
                      resolutions, seeds):
    """Return complete partitions with caller-supplied dataset settings.

    调用方根据数据集提供邻居数、分辨率候选和随机种子。
    """
    content, content_pct = _similarities(content_vectors)
    names, name_pct = _similarities(name_vectors)
    if content.shape != names.shape or neighbors < 1 or not resolutions or not seeds:
        raise ValueError("双向量 Leiden 的节点数量或参数无效")
    if any(gamma < 0 for gamma in resolutions):
        raise ValueError("Leiden 分辨率不能为负")
    near_content = _neighbors(content, neighbors)
    near_names = _neighbors(names, neighbors)
    content_any = near_content | near_content.T
    name_any = near_names | near_names.T
    eligible = ((near_content & near_content.T) |
                (near_names & near_names.T) |
                (content_any & name_any))
    rows, cols = np.where(np.triu(eligible, 1))
    edges = list(zip(rows.tolist(), cols.tolist()))
    graph = ig.Graph(n=len(content), edges=edges, directed=False)
    weight = (content_pct + name_pct) / 2.0
    graph.es["weight"] = [float(weight[i, j]) for i, j in edges]
    partitions = []
    seen = set()
    for gamma in resolutions:
        for seed in seeds:
            if edges:
                result = leidenalg.find_partition(
                    graph, leidenalg.RBConfigurationVertexPartition,
                    weights="weight", resolution_parameter=float(gamma),
                    seed=int(seed), n_iterations=-1)
                membership = result.membership
            else:
                membership = list(range(len(content)))
            groups = {}
            for index, label in enumerate(membership):
                groups.setdefault(label, []).append(index)
            canonical = tuple(sorted(tuple(ids) for ids in groups.values()))
            if canonical in seen:
                continue
            seen.add(canonical)
            partitions.append({"resolution": float(gamma), "seed": int(seed),
                               "groups": [list(ids) for ids in canonical],
                               "community_count": len(canonical),
                               "singleton_count": sum(len(ids) == 1 for ids in canonical)})
    diagnostics = {"method": "dual_view_ranked_knn_leiden",
                   "neighbors": min(neighbors, len(content) - 1),
                   "edge_rule": "reciprocal_or_both",
                   "edge_weight": "mean_of_view_percentiles",
                   "edge_count": len(edges),
                   "component_count": len(graph.components()),
                   "isolated_node_count": sum(degree == 0 for degree in graph.degree()),
                   "resolutions": list(map(float, resolutions)),
                   "seeds": list(map(int, seeds)),
                   "candidate_count": len(partitions)}
    diagnostics["candidate_edges"] = edges
    return partitions, diagnostics
