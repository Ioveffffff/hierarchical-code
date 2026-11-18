"""通用层次 ECS：比较同一批术语的两棵完整概念树。

复现 Gates 等（Scientific Reports, 2019）的 CluSim element-centric similarity：
按层次节点构造术语-簇二部矩阵，将投影随机游走的个性化 PageRank
亲和矩阵用修正 L1 距离比较。CluSim 0.4 在当前 SciPy 中直接调用报错，
因此仅重写其公开数学步骤，不要求安装 CluSim。
按原始术语成员集合折叠逐层原样传递的重复节点。多棵根节点按森林处理，
不人为添加总根，不使用参考类别或 LLM 名称。固定 alpha=0.9、r=1.0。
"""

import networkx as nx
import numpy as np
from scipy.linalg import solve

from hierarchy_metrics import layer_partitions


ECS_ALPHA = 0.9
ECS_HIERARCHY_WEIGHT = 1.0


def ontology_cluster_graph(ontology, n_samples):
    """折叠传递节点后返回概念森林及各节点覆盖的术语。"""
    partitions = layer_partitions(ontology, n_samples)
    graph = nx.DiGraph()
    canonical = {}
    previous_ids = previous_leaves = None
    members_by_id = {}
    for layer, _ in partitions:
        current = ontology[layer]
        current_ids = {}
        current_leaves = {}
        for raw_id, raw_leaves in current["leaf_indices"].items():
            node_id = str(raw_id)
            leaves = tuple(sorted(int(i) for i in raw_leaves))
            if leaves not in canonical:
                canonical[leaves] = f"concept_{len(canonical)}"
            canonical_id = canonical[leaves]
            current_ids[node_id] = canonical_id
            current_leaves[node_id] = set(leaves)
            members_by_id[canonical_id] = leaves
            graph.add_node(canonical_id)

        if previous_ids is not None:
            members = current.get("member_indices", {})
            if set(map(str, members)) != set(current_ids):
                raise ValueError(f"{layer} 的 member_indices 与节点不一致")
            for raw_parent, raw_children in members.items():
                parent = str(raw_parent)
                children = [str(child) for child in raw_children]
                if not children or any(child not in previous_ids for child in children):
                    raise ValueError(f"{layer} 有无效的子簇编号")
                covered = set()
                for child in children:
                    if covered.intersection(previous_leaves[child]):
                        raise ValueError(f"{layer} 的子簇覆盖重复")
                    covered.update(previous_leaves[child])
                    if current_ids[parent] != previous_ids[child]:
                        graph.add_edge(current_ids[parent], previous_ids[child])
                if covered != current_leaves[parent]:
                    raise ValueError(f"{layer} 的父簇未完整覆盖子簇")
        previous_ids, previous_leaves = current_ids, current_leaves

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("折叠重复节点后层次图出现环")
    return graph, members_by_id


def element_affinity(ontology, n_samples, alpha=ECS_ALPHA,
                     hierarchy_weight=ECS_HIERARCHY_WEIGHT):
    """按 CluSim 二部投影与 PPR 定义求全部术语的亲和矩阵。"""
    if not 0 < alpha < 1:
        raise ValueError("ECS alpha 必须在 0 与 1 之间")
    graph, members = ontology_cluster_graph(ontology, n_samples)
    affinity = np.zeros((n_samples, n_samples), dtype=np.float64)
    covered = set()
    for component in nx.weakly_connected_components(graph):
        nodes = sorted(component)
        elements = sorted(set().union(*(members[node] for node in nodes)))
        if covered.intersection(elements):
            raise ValueError("概念森林的不同根覆盖了同一术语")
        covered.update(elements)
        index = {term: i for i, term in enumerate(elements)}
        subgraph = graph.subgraph(nodes)
        root_distance = {}
        for node in nx.topological_sort(subgraph):
            root_distance[node] = max(
                (root_distance[parent] + 1 for parent in subgraph.predecessors(node)),
                default=0)
        leaf_distance = {}
        for node in reversed(list(nx.topological_sort(subgraph))):
            leaf_distance[node] = max(
                (leaf_distance[child] + 1 for child in subgraph.successors(node)),
                default=0)
        bipartite = np.zeros((len(elements), len(nodes)), dtype=np.float64)
        for column, node in enumerate(nodes):
            distance = root_distance[node] + leaf_distance[node]
            height = root_distance[node] / distance if distance else 0.0
            weight = np.exp(hierarchy_weight * height)
            bipartite[[index[term] for term in members[node]], column] = weight
        row_sums = bipartite.sum(axis=1, keepdims=True)
        col_sums = bipartite.sum(axis=0, keepdims=True)
        if (row_sums == 0).any() or (col_sums == 0).any():
            raise ValueError("层次图含未分配术语或空概念")
        projected = (bipartite / row_sums) @ (bipartite / col_sums).T
        identity = np.eye(len(elements))
        block = (1.0 - alpha) * solve(identity - alpha * projected, identity,
                                       assume_a="gen", check_finite=False)
        affinity[np.ix_(elements, elements)] = block
    if covered != set(range(n_samples)) or not np.isfinite(affinity).all():
        raise ValueError("ECS 的亲和矩阵未覆盖所有术语或出现非法值")
    if not np.allclose(affinity.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("ECS 的 PPR 亲和向量未归一化")
    return affinity


def ecs_from_affinities(first, second, alpha=ECS_ALPHA, chunk_size=256):
    """按 CluSim 修正 L1 公式计算整树 ECS，越高越稳定。"""
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("ECS 需要相同术语顺序、相同尺寸的亲和矩阵")
    n = first.shape[0]
    difference = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        difference += float(np.abs(first[start:stop] - second[start:stop]).sum())
    score = 1.0 - difference / (2.0 * alpha * n)
    if not -1e-8 <= score <= 1 + 1e-8:
        raise ValueError(f"ECS 超出 [0,1]：{score}")
    return float(np.clip(score, 0.0, 1.0))
