"""Item metadata encoding and leakage-free temporal features.

Everything here is computed either from static catalogue metadata or from interactions
strictly before the moment at which the feature is evaluated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Keys ``item * _KEY_SCALE + day`` stay exact in float64: days since 1970 are < 1e5.
_KEY_SCALE = 1e5

__all__ = [
    "FeatureConfig",
    "ItemFeatures",
    "RecentCounter",
    "build_item_features",
    "item_first_day",
    "log2_bucket",
    "user_field_profile",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Vocabulary thresholds for categorical item metadata."""

    min_platform_items: int = 20
    min_type_items: int = 20
    min_store_items: int = 10
    n_price_bins: int = 10


@dataclass(frozen=True)
class ItemFeatures:
    """Static categorical item features: ``codes[field][item]`` indexes ``vocab[field]``."""

    codes: Dict[str, np.ndarray]
    vocab: Dict[str, List[str]]

    @property
    def fields(self) -> List[str]:
        return list(self.codes)

    def cardinality(self, field: str) -> int:
        return len(self.vocab[field])


def _encode_categorical(values: pd.Series, min_count: int) -> Tuple[np.ndarray, List[str]]:
    """Integer-encode a string column; values rarer than ``min_count`` become ``other``."""
    values = values.astype(str)
    counts = values.value_counts()
    frequent = set(counts[counts >= min_count].index)
    values = values.where(values.isin(frequent), "other")
    categories = pd.Categorical(values, categories=sorted(values.unique()))
    return categories.codes.astype(np.int64), list(categories.categories)


def _price_bins(price: np.ndarray, n_bins: int) -> Tuple[np.ndarray, List[str]]:
    """Quantile bins of the (log) price; items without a price get their own bin."""
    known = np.isfinite(price) & (price > 0)
    log_price = np.log(price[known])
    edges = np.quantile(log_price, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    codes = np.full(len(price), n_bins, dtype=np.int64)
    codes[known] = np.searchsorted(edges, log_price, side="right")
    bounds = np.exp(np.concatenate([[log_price.min()], edges, [log_price.max()]]))
    names = [f"${bounds[b]:.0f}-{bounds[b + 1]:.0f}" for b in range(n_bins)] + ["missing"]
    return codes, names


def build_item_features(items: pd.DataFrame, cfg: FeatureConfig) -> ItemFeatures:
    """Encode platform, item type, store, ESRB rating and price bucket of every item."""
    codes: Dict[str, np.ndarray] = {}
    vocab: Dict[str, List[str]] = {}
    for field, min_count in (
        ("platform", cfg.min_platform_items),
        ("item_type", cfg.min_type_items),
        ("store", cfg.min_store_items),
        ("esrb", 1),
    ):
        codes[field], vocab[field] = _encode_categorical(items[field], min_count)
    codes["price_bin"], vocab["price_bin"] = _price_bins(items["price"].to_numpy(np.float64), cfg.n_price_bins)
    for field in codes:
        logger.debug("item feature %s: %d values", field, len(vocab[field]))
    return ItemFeatures(codes=codes, vocab=vocab)


def item_first_day(items: np.ndarray, days: np.ndarray, n_items: int) -> np.ndarray:
    """Day of the first observed interaction of every item (``inf`` if never observed)."""
    first = np.full(n_items, np.inf)
    np.minimum.at(first, items, days)
    return first


class RecentCounter:
    """Number of interactions of an item inside ``[t - window, t)``, on any torch device.

    The window is half-open on the right, so the interaction being scored is never
    counted: a feature computed at time ``t`` only sees the past.
    """

    def __init__(self, items: np.ndarray, days: np.ndarray, device: torch.device) -> None:
        order = np.lexsort((days, items))
        self._keys = torch.from_numpy(items[order] * _KEY_SCALE + days[order]).to(device)  # float64

    def count(self, items: torch.Tensor, t: torch.Tensor, window: float) -> torch.Tensor:
        base = items.double() * _KEY_SCALE
        hi = torch.searchsorted(self._keys, base + t)
        lo = torch.searchsorted(self._keys, base + (t - window))
        return hi - lo


def log2_bucket(x: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """``floor(log2(1 + x))`` clipped to ``n_buckets - 1``; handy for counts and ages."""
    buckets = torch.floor(torch.log2(1.0 + x.double().clamp_min(0.0)))
    return buckets.clamp_max(n_buckets - 1).long()


def user_field_profile(
    users: np.ndarray,
    field_codes: np.ndarray,
    n_users: int,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised histogram of a categorical item field over each user's history.

    Returns ``(codes, weights)`` of shape ``(n_users, max_len)``; only the ``max_len``
    most frequent values are kept, padding has code ``-1`` and weight ``0``.
    """
    n_values = int(field_codes.max()) + 1
    pairs, counts = np.unique(users * n_values + field_codes, return_counts=True)
    owner, value = pairs // n_values, pairs % n_values
    order = np.lexsort((value, -counts, owner))
    owner, value, counts = owner[order], value[order], counts[order]
    group_start = np.searchsorted(owner, owner, side="left")
    rank = np.arange(len(owner)) - group_start
    keep = rank < max_len
    owner, value, counts, rank = owner[keep], value[keep], counts[keep], rank[keep]

    codes = np.full((n_users, max_len), -1, dtype=np.int64)
    weights = np.zeros((n_users, max_len), dtype=np.float32)
    codes[owner, rank] = value
    weights[owner, rank] = counts
    totals = weights.sum(axis=1, keepdims=True)
    np.divide(weights, totals, out=weights, where=totals > 0)
    return codes, weights
