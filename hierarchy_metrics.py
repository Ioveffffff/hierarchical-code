"""通用层次评价：读取 HCRL 输出的 ontology，计算树质量、稳定性与规模。

DP 需要平面参考类别；Dasgupta Cost 使用各方法共同的固定输入嵌入；
层次 FM 比较同一批术语在两棵树各对应层的划分，不需要参考树。
这些指标不能单独证明父子概念在领域知识中语义正确。
Dasgupta Cost: Dasgupta (2016), DOI 10.1145/2897518.2897527；
层次 FM: Fowlkes & Mallows (1983), DOI 10.1080/01621459.1983.10478008。
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import fowlkes_mallows_score


def load_ontology(path):
    """读取 HCRL 保存的 ontology_structure.json。"""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return document["ontology"]


def layer_partitions(ontology, n_samples):
    """将 L1、L2... 的 leaf_indices 转成逐样本簇编号，并检查覆盖与嵌套。"""
    names = sorted((name for name in ontology if name.startswith("L") and name != "L0"),
                   key=lambda name: int(name[1:]))
    if not names:
        raise ValueError("层次结构缺少 L1 及以上的概念层")
    partitions = []
    for name in names:
        labels = np.full(n_samples, -1, dtype=np.int64)
        groups = ontology[name].get("leaf_indices")
        if not groups:
            raise ValueError(f"{name} 缺少 leaf_indices")
        for number, leaves in enumerate(groups.values()):
            indices = np.asarray(leaves, dtype=np.int64)
            if len(indices) == 0 or (indices < 0).any() or (indices >= n_samples).any():
                raise ValueError(f"{name} 存在空簇或越界术语编号")
            if (labels[indices] != -1).any() or len(np.unique(indices)) != len(indices):
                raise ValueError(f"{name} 的簇有重复术语")
            labels[indices] = number
        if (labels < 0).any():
            raise ValueError(f"{name} 未覆盖所有术语")
        if partitions:
            previous = partitions[-1][1]
            for fine in np.unique(previous):
                if len(np.unique(labels[previous == fine])) != 1:
                    raise ValueError(f"{name} 与上一层不构成嵌套层次")
        partitions.append((name, labels))
    return partitions


def dendrogram_purity(partitions, reference_labels):
    """同类术语对在最近共同祖先内的类别纯度，越高越好。"""
    truth = np.asarray(reference_labels)
    n = len(truth)
    if any(len(labels) != n for _, labels in partitions):
        raise ValueError("参考标签与层次结构的术语数量不同")
    group_sizes = [np.bincount(labels) for _, labels in partitions]
    score = 0.0
    count = 0
    for category in np.unique(truth):
        indices = np.flatnonzero(truth == category)
        for pos, i in enumerate(indices[:-1]):
            others = indices[pos + 1:]
            ancestor_size = np.full(len(others), n, dtype=np.int64)
            same_category = np.full(len(others), len(indices), dtype=np.int64)
            # 从粗到细覆写，最后留下最近共同祖先。
            for (_, labels), sizes in zip(reversed(partitions), reversed(group_sizes)):
                mask = labels[others] == labels[i]
                if mask.any():
                    ancestor_size[mask] = sizes[labels[i]]
                    same_category[mask] = np.count_nonzero(
                        (labels == labels[i]) & (truth == category))
            score += float((same_category / ancestor_size).sum())
            count += len(others)
    if count == 0:
        raise ValueError("没有可用于 DP 的同类术语对")
    return score / count


def normalized_dasgupta_cost(partitions, fixed_embeddings):
    """余弦相似度非负部分加权的 Dasgupta Cost，除以 n*总权重，越低越好。

    没有共同祖先的森林分支视为在虚拟根节点合并。各方法必须共用同一份
    fixed_embeddings，否则指标不可直接比较。
    """
    x = np.asarray(fixed_embeddings, dtype=np.float64)
    if x.ndim != 2 or any(len(labels) != len(x) for _, labels in partitions):
        raise ValueError("嵌入与层次结构的术语数量不同")
    if not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) < 1e-12).any():
        raise ValueError("嵌入中存在非法值或零向量")
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sizes = [np.bincount(labels) for _, labels in partitions]
    weighted_cost = total_weight = 0.0
    n = len(x)
    for i in range(n - 1):
        weights = np.maximum(x[i + 1:] @ x[i], 0)
        ancestor_size = np.full(n - i - 1, n, dtype=np.int64)
        for (_, labels), group_sizes in zip(reversed(partitions), reversed(sizes)):
            match = labels[i + 1:] == labels[i]
            ancestor_size[match] = group_sizes[labels[i]]
        weighted_cost += float(weights @ ancestor_size)
        total_weight += float(weights.sum())
    if total_weight <= 0:
        raise ValueError("固定嵌入中没有正的术语相似度")
    return weighted_cost / (n * total_weight)


def hierarchical_fm(reference_partitions, compared_partitions):
    """按相同层号比较划分，返回每层 FM 与宏平均；并报告两树各层簇数。"""
    reference = dict(reference_partitions)
    compared = dict(compared_partitions)
    rows = []
    for name in sorted(reference.keys() & compared.keys(), key=lambda value: int(value[1:])):
        first, second = reference[name], compared[name]
        if len(first) != len(second):
            raise ValueError("两棵树的术语数量不同")
        first_k, second_k = len(np.unique(first)), len(np.unique(second))
        if not (2 <= first_k < len(first) and 2 <= second_k < len(second)):
            continue
        rows.append(dict(layer=name, reference_k=int(first_k),
                         compared_k=int(second_k),
                         fm=float(fowlkes_mallows_score(first, second))))
    if not rows:
        raise ValueError("两棵树没有共同的非平凡概念层")
    return dict(mean_fm=float(np.mean([row["fm"] for row in rows])), layers=rows)


def tree_size(partitions):
    """层数含 L0；概念节点数不含 L0，深度本身没有越高越好的方向。"""
    counts = {name: int(len(np.unique(labels))) for name, labels in partitions}
    depth = max(int(name[1:]) for name in counts) + 1
    return dict(max_depth=depth, concept_node_count=sum(counts.values()),
                nodes_by_layer=counts)
