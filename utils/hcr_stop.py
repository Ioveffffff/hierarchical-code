"""HCR-Stop evaluation independent of the candidate clustering algorithm.

This module consumes complete, semantically reviewed candidate partitions. It
does not generate clusters or infer missing parent-child coverage scores.
"""

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    input_count: int
    parent_count: int
    merged_parent_count: int
    entropy_before: float
    entropy_after: float
    delta_h: float
    sc: float | None
    pcc: float
    score: float
    structural_semantic_quality: float

    def to_dict(self):
        return asdict(self)


class HCRStop:
    """Rank candidate layers and decide whether to accept another concept layer.

    Weights and the quality-decline tolerance must be fixed from a separate
    diagnostic run before constructing the final hierarchy. A negative SC is
    penalized in the score, but is not a hard rejection rule.
    """

    def __init__(self, weights: Mapping[str, float], decline_tolerance: float,
                 decline_patience: int = 1):
        expected = {"delta_h", "sc", "pcc"}
        if set(weights) != expected:
            raise ValueError(f"weights must contain exactly {sorted(expected)}")
        values = {key: float(value) for key, value in weights.items()}
        if (not all(np.isfinite(value) and value >= 0 for value in values.values())
                or not np.isclose(sum(values.values()), 1.0)
                or values["pcc"] <= 0):
            raise ValueError("weights must be nonnegative, sum to 1, and include PCC")
        if not np.isfinite(decline_tolerance) or not 0 <= decline_tolerance <= 1:
            raise ValueError("decline_tolerance must be in [0, 1]")
        if decline_patience < 1:
            raise ValueError("decline_patience must be positive")
        self.weights = values
        self.decline_tolerance = float(decline_tolerance)
        self.decline_patience = int(decline_patience)
        self.previous_quality = None
        self.decline_streak = 0
        self.history = []

    def evaluate(self, candidate_id: str, vectors: np.ndarray,
                 labels: Sequence[int], leaf_counts: Sequence[int],
                 child_scores_by_parent: Mapping[int, Sequence[int]]) -> CandidateMetrics:
        """Measure one complete candidate layer over the direct child nodes.

        A score is required for every edge of every newly formed parent. A
        single-child pass-through node has no scored edge and no PCC weight.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        labels = np.asarray(labels)
        leaf_counts = np.asarray(leaf_counts, dtype=np.float64)
        n = len(labels)
        if (vectors.ndim != 2 or len(vectors) != n or len(leaf_counts) != n
                or n == 0 or not np.isfinite(vectors).all()
                or np.any(np.linalg.norm(vectors, axis=1) <= 1e-12)
                or not np.isfinite(leaf_counts).all()
                or np.any(leaf_counts <= 0)):
            raise ValueError("vectors, labels and positive leaf_counts must align")
        if not candidate_id:
            raise ValueError("candidate_id must be nonempty")

        parents, inverse = np.unique(labels, return_inverse=True)
        parent_sizes = np.bincount(inverse)
        merged = [parent for parent, size in zip(parents, parent_sizes) if size >= 2]
        expected = set(merged)
        if set(child_scores_by_parent) != expected:
            raise ValueError("scores must be supplied exactly for newly formed parents")
        parent_scores = []
        for parent, size in zip(parents, parent_sizes):
            if size < 2:
                continue
            scores = np.asarray(child_scores_by_parent[parent], dtype=np.float64)
            if (len(scores) != size or not np.isfinite(scores).all()
                    or np.any(~np.isin(scores, [1, 2, 3]))):
                raise ValueError("each merged child edge needs an LLM score of 1, 2 or 3")
            parent_scores.append(float(np.mean(scores) / 3.0))

        probabilities_before = leaf_counts / leaf_counts.sum()
        parent_leaf_counts = np.bincount(inverse, weights=leaf_counts)
        probabilities_after = parent_leaf_counts / leaf_counts.sum()
        entropy_before = float(-np.sum(probabilities_before * np.log(probabilities_before)))
        entropy_after = float(-np.sum(probabilities_after * np.log(probabilities_after)))
        delta_h = ((entropy_before - entropy_after) / entropy_before
                   if entropy_before > 0 else 0.0)
        delta_h = float(np.clip(delta_h, 0.0, 1.0))

        # Silhouette is undefined for one parent or for all-singleton parents.
        sc = None
        if 1 < len(parents) < n:
            sc = float(silhouette_score(vectors, inverse, metric="cosine"))
        pcc = float(np.mean(parent_scores)) if parent_scores else 0.0
        w = self.weights
        score = w["delta_h"] * delta_h + w["sc"] * (sc or 0.0) + w["pcc"] * pcc
        # Cross-layer comparison excludes Delta H: its scale naturally shrinks
        # as the tree becomes coarser. An undefined SC contributes no evidence.
        quality_weight = w["pcc"] + (w["sc"] if sc is not None else 0.0)
        quality = (w["pcc"] * pcc + w["sc"] * sc if sc is not None
                   else w["pcc"] * pcc) / quality_weight
        return CandidateMetrics(
            candidate_id=str(candidate_id), input_count=n,
            parent_count=len(parents), merged_parent_count=len(merged),
            entropy_before=entropy_before, entropy_after=entropy_after,
            delta_h=delta_h, sc=sc, pcc=pcc, score=float(score),
            structural_semantic_quality=float(quality))

    def select(self, candidates: Sequence[CandidateMetrics]):
        """Select one reviewed candidate, or stop without changing the tree.

        Candidates are ordered by the three-metric score. If the best candidate
        breaches the quality-decline rule, a lower-ranked valid candidate can
        still be accepted. Rejected candidates never alter the saved baseline.
        """
        ordered = sorted(candidates, key=lambda item: (
            -item.score, -item.pcc, -item.delta_h, item.candidate_id))
        for item in ordered:
            if item.merged_parent_count == 0 or item.delta_h <= 0:
                continue
            decline = (self.previous_quality is not None and
                       item.structural_semantic_quality <
                       self.previous_quality - self.decline_tolerance)
            next_streak = self.decline_streak + 1 if decline else 0
            if next_streak >= self.decline_patience:
                continue
            self.decline_streak = next_streak
            self.previous_quality = item.structural_semantic_quality
            self.history.append({**item.to_dict(), "accepted": True,
                                 "decline_streak": self.decline_streak})
            return item, "accepted"

        reason = ("no_effective_merge" if not any(
            item.merged_parent_count and item.delta_h > 0 for item in ordered)
                  else "quality_decline")
        self.history.append({"accepted": False, "reason": reason,
                             "candidates": [item.to_dict() for item in ordered]})
        return None, reason
