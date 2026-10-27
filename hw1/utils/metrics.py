"""Top-K ranking metrics, beyond-accuracy metrics and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = [
    "bootstrap_ci",
    "catalog_coverage",
    "hits_matrix",
    "novelty",
    "paired_bootstrap",
    "ranking_metrics",
]


def hits_matrix(topk: np.ndarray, relevant: sp.csr_matrix) -> np.ndarray:
    """Boolean ``(n_users, k)``: is the item at each rank relevant for that user.

    ``relevant`` must have one row per row of ``topk`` (same order).
    """
    relevant = relevant.tocsr()
    relevant.sort_indices()
    n_rows, n_cols = relevant.shape
    rows = np.repeat(np.arange(n_rows), np.diff(relevant.indptr))
    rel_keys = rows * n_cols + relevant.indices  # sorted: rows ascending, indices sorted
    top_keys = np.arange(n_rows)[:, None] * n_cols + topk
    pos = np.searchsorted(rel_keys, top_keys)
    pos = np.minimum(pos, max(len(rel_keys) - 1, 0))
    if len(rel_keys) == 0:
        return np.zeros(topk.shape, dtype=bool)
    return rel_keys[pos] == top_keys


def ranking_metrics(topk: np.ndarray, relevant: sp.csr_matrix) -> Dict[str, np.ndarray]:
    """Per-user NDCG, Recall, HitRate, Precision and MRR at ``k = topk.shape[1]``.

    Binary relevance; ``IDCG`` uses ``min(|rel|, k)`` ideal positions, Recall divides
    by ``|rel|``. Users without relevant items must be filtered out beforehand.
    """
    k = topk.shape[1]
    hits = hits_matrix(topk, relevant)
    n_rel = np.diff(relevant.tocsr().indptr)
    if (n_rel == 0).any():
        raise ValueError("every evaluated user needs at least one relevant item")
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discounts
    idcg = np.cumsum(discounts)[np.minimum(n_rel, k) - 1]
    n_hits = hits.sum(axis=1)
    any_hit = hits.any(axis=1)
    mrr = np.zeros(len(hits))
    mrr[any_hit] = 1.0 / (hits[any_hit].argmax(axis=1) + 1.0)
    return {
        "ndcg": dcg / idcg,
        "recall": n_hits / n_rel,
        "hit_rate": (n_hits > 0).astype(np.float64),
        "precision": n_hits / k,
        "mrr": mrr,
    }


def catalog_coverage(topk: np.ndarray, n_candidates: int) -> float:
    """Share of the candidate catalogue that appears in at least one top-K list."""
    return float(np.unique(topk).size / n_candidates)


def novelty(topk: np.ndarray, item_counts: np.ndarray, n_users: int) -> float:
    """Mean self-information ``-log2(pop_i / n_users)`` of recommended items (bits).

    Higher means less popular (more novel) recommendations.
    """
    pop = np.maximum(item_counts[topk], 1) / n_users
    return float(-np.log2(pop).mean())


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap CI over users."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> Dict[str, float]:
    """Paired bootstrap of ``mean(a - b)`` over the same users.

    The two-sided p-value is the bootstrap share of resampled differences on the other
    side of zero, doubled (the usual percentile approximation).
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    p_value = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(diff.mean()), "lo": float(lo), "hi": float(hi), "p_value": float(min(p_value, 1.0))}
