"""Label-free hierarchy selection and conservative pseudo-label updates."""
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError('Need >=4 finite embedding rows')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError('Zero embedding is not a valid cosine representation')
    return x / norms


def choose_partition(embeddings, k_min=2, k_max=256, sample_size=1024, seed=42):
    """Select average-linkage cut by cosine silhouette, without true labels.

    Search a deterministic logarithmic grid then refine around its best cut.
    Every candidate is scored on the same fixed sample of entity indices.
    The returned value is an internal monitoring score, not external accuracy.
    """
    if k_min < 2 or k_max < k_min or sample_size < 4:
        raise ValueError('Invalid cluster search configuration')
    x = normalize_rows(embeddings)
    total_variance = float(x.var(0).sum())
    if total_variance < 1e-8:
        raise ValueError('Collapsed embeddings cannot yield meaningful clusters')
    n = len(x)
    lower = max(2, int(k_min))
    upper = min(int(k_max), n // 2)
    if lower > upper:
        raise ValueError('No valid cluster-count range')
    distances = np.clip(pdist(x, metric='cosine'), 0, 2)
    tree = linkage(distances, method='average')
    indices = np.sort(np.random.default_rng(seed).choice(n, min(n, sample_size), replace=False))
    sampled_distances = squareform(np.clip(pdist(x[indices], metric='cosine'), 0, 2))
    np.fill_diagonal(sampled_distances, 0)
    candidates = {}
    def evaluate(k):
        k = int(k)
        if k in candidates:
            return
        labels = cut_tree(tree, n_clusters=k).reshape(-1).astype(np.int64)
        actual = len(np.unique(labels))
        sampled = labels[indices]
        if not 2 <= len(np.unique(sampled)) < len(sampled) or actual < lower:
            return
        score = float(silhouette_score(sampled_distances, sampled, metric='precomputed'))
        candidates[k] = (score, labels)
    grid = np.unique(np.rint(np.geomspace(lower, upper, min(24, upper-lower+1))).astype(int))
    for k in grid:
        evaluate(k)
    if not candidates:
        raise ValueError('No nondegenerate hierarchy cut (possibly tied embeddings)')
    best = max(candidates, key=lambda k: (candidates[k][0], -k))
    below = [int(k) for k in grid if k < best]
    above = [int(k) for k in grid if k > best]
    lo, hi = (max(below) if below else lower), (min(above) if above else upper)
    for k in np.unique(np.rint(np.linspace(lo, hi, min(13, hi-lo+1))).astype(int)):
        evaluate(k)
    best = max(candidates, key=lambda k: (candidates[k][0], -k))
    score, labels = candidates[best]
    sizes = np.unique(labels, return_counts=True)[1]
    metadata = dict(selected_k=int(len(sizes)), requested_k=int(best), silhouette=score,
                    search_boundary=bool(best in (lower, upper)),
                    min_size=int(sizes.min()), max_size=int(sizes.max()),
                    singleton_fraction=float((sizes == 1).mean()),
                    embedding_total_variance=total_variance,
                    search_range=[lower, upper], sample_size=len(indices),
                    candidates=[dict(k=int(k), actual_k=int(len(np.unique(lab))), silhouette=s)
                                for k, (s, lab) in sorted(candidates.items())])
    return labels, metadata, tree


def coarse_cluster_count(n_fine, coarse_ratio):
    """Choose fewer coarse clusters while retaining at least two when possible."""
    if n_fine < 2 or not 0 < coarse_ratio < 1:
        raise ValueError('Need at least two fine clusters and a ratio in (0, 1)')
    if n_fine == 2:
        # There is no meaningful coarser cut with two fine clusters. Keep both
        # clusters; a single coarse label would turn every pair into a positive
        # and collapse Stage 2.
        return 2
    return max(2, min(n_fine - 1, round(n_fine * coarse_ratio)))


def pseudo_partition(embeddings, coarse_ratio=.3, k_min=2, k_max=256,
                     sample_size=1024, seed=42, confidence_margin=.02,
                     fixed_k=None):
    if not 0 < coarse_ratio < 1 or confidence_margin < 0:
        raise ValueError('Invalid coarse ratio/confidence margin')
    if fixed_k is None:
        fine, metadata, tree = choose_partition(embeddings, k_min, k_max, sample_size, seed)
        metadata['k_mode'] = 'silhouette_search'
    else:
        if int(fixed_k) < 2 or int(fixed_k) > len(embeddings) // 2:
            raise ValueError('fixed_k must be between 2 and half the sample count')
        fine, metadata, tree = choose_partition(
            embeddings, int(fixed_k), int(fixed_k), sample_size, seed)
        metadata['k_mode'] = 'prespecified_fixed_k'
    n_fine = len(np.unique(fine))
    k_coarse = coarse_cluster_count(n_fine, coarse_ratio)
    coarse = cut_tree(tree, n_clusters=k_coarse).reshape(-1).astype(np.int64)
    x = normalize_rows(embeddings)
    unique, inverse = np.unique(fine, return_inverse=True)
    centers = np.stack([x[fine == k].mean(0) for k in unique])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    similarities = x @ centers.T
    own = similarities[np.arange(len(x)), inverse].copy()
    similarities[np.arange(len(x)), inverse] = -np.inf
    margin = own - similarities.max(1)
    # Singletons cannot provide an inter-entity positive pair.
    sizes = np.bincount(inverse)
    reliable = (margin >= confidence_margin) & (sizes[inverse] > 1)
    coarse_sizes = np.unique(coarse, return_counts=True)[1]
    metadata.update(coarse_clusters=int(len(coarse_sizes)),
                    coarse_min_size=int(coarse_sizes.min()),
                    coarse_max_size=int(coarse_sizes.max()),
                    coarse_is_fine_fallback=bool(k_coarse == n_fine),
                    reliable_fraction=float(reliable.mean()),
                    confidence_margin=confidence_margin)
    for f in np.unique(fine):
        if len(np.unique(coarse[fine == f])) != 1:
            raise RuntimeError('Hierarchy is not nested')
    return coarse, fine, reliable, metadata


def inject_pseudo_noise(coarse, fine, reliable, rate=0.0, scope='fine', seed=42):
    """Corrupt reliable pseudo labels for a controlled robustness experiment.

    Fine noise moves an entity to another fine cluster under the same coarse
    parent. Coarse noise moves it to a fine cluster under another parent.
    At least one entity remains in each source fine cluster, preserving k and
    the nested hierarchy. The returned mask marks deliberately corrupted rows.
    """
    if not 0 <= rate <= 1 or scope not in ('fine', 'coarse'):
        raise ValueError('Invalid pseudo-label noise rate/scope')
    c, f, r = np.asarray(coarse).copy(), np.asarray(fine).copy(), np.asarray(reliable)
    if c.shape != f.shape or c.shape != r.shape or c.ndim != 1:
        raise ValueError('Pseudo labels and reliability must be aligned vectors')
    mask = np.zeros(len(f), dtype=bool)
    fine_by_coarse = {int(parent): np.unique(f[c == parent]).tolist()
                      for parent in np.unique(c)}
    counts = {int(label): int((f == label).sum()) for label in np.unique(f)}
    if scope == 'fine':
        eligible = [i for i in np.flatnonzero(r)
                    if counts[int(f[i])] > 1 and len(fine_by_coarse[int(c[i])]) > 1]
    else:
        eligible = [i for i in np.flatnonzero(r)
                    if counts[int(f[i])] > 1 and len(fine_by_coarse) > 1]
    target_count = round(rate * len(eligible))
    rng = np.random.default_rng(seed)
    for i in rng.permutation(eligible):
        if mask.sum() >= target_count:
            break
        old_fine, old_coarse = int(f[i]), int(c[i])
        if counts[old_fine] <= 1:
            continue
        if scope == 'fine':
            new_coarse = old_coarse
            choices = [k for k in fine_by_coarse[old_coarse] if k != old_fine]
        else:
            parents = [k for k in fine_by_coarse if k != old_coarse]
            new_coarse = int(rng.choice(parents))
            choices = fine_by_coarse[new_coarse]
        if not choices:
            continue
        new_fine = int(rng.choice(choices))
        c[i], f[i], mask[i] = new_coarse, new_fine, True
        counts[old_fine] -= 1
        counts[new_fine] += 1
    if any(len(np.unique(c[f == label])) != 1 for label in np.unique(f)):
        raise RuntimeError('Noise injection broke fine/coarse nesting')
    info = dict(noise_scope=scope, noise_rate_requested=float(rate),
                noise_eligible=len(eligible), noise_injected=int(mask.sum()),
                noise_rate_eligible=float(mask.sum() / len(eligible)) if eligible else 0.0,
                noise_rate_all=float(mask.mean()))
    return c, f, mask, info


def effective_training_labels(coarse, fine, reliable):
    """Uncertain entities retain only their own two-view positive pair."""
    c, f = np.array(coarse, copy=True), np.array(fine, copy=True)
    uncertain = np.flatnonzero(~reliable)
    c[uncertain] = c.max()+1+np.arange(len(uncertain))
    f[uncertain] = f.max()+1+np.arange(len(uncertain))
    return c, f


def smooth_teacher(current, previous=None, anchor=None, history_weight=.5, anchor_weight=.2):
    """Smooth features, never label IDs; allow both merges and splits on refresh."""
    if not 0 <= history_weight < 1 or not 0 <= anchor_weight < 1:
        raise ValueError('Smoothing weights must be in [0, 1)')
    x = normalize_rows(current)
    if previous is not None:
        x = (1-history_weight)*x + history_weight*normalize_rows(previous)
    if anchor is not None:
        x = (1-anchor_weight)*x + anchor_weight*normalize_rows(anchor)
    return normalize_rows(x)
