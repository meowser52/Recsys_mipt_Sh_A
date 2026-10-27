"""Global temporal split and the data containers shared by all models.

Protocol: ``train < val_start <= validation < test_start <= test``. Models are tuned on
(train -> validation) and refitted with the chosen settings on (train + validation -> test).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.data import MS_PER_DAY
from utils.features import ItemFeatures

logger = logging.getLogger(__name__)

__all__ = [
    "EvalTarget",
    "FitData",
    "SplitConfig",
    "day_of",
    "make_fit_data",
    "make_target",
    "temporal_split",
]


@dataclass(frozen=True)
class SplitConfig:
    """Split boundaries (UTC dates) and the test-time definition of a relevant item."""

    val_start: str = "2021-07-01"
    test_start: str = "2022-07-01"
    relevance_threshold: float = 4.0


def day_of(date: str) -> float:
    """Days since the Unix epoch of a UTC date string, in the same unit as ``day``."""
    return pd.Timestamp(date, tz="UTC").value / 1e6 / MS_PER_DAY


@dataclass(frozen=True)
class FitData:
    """Everything a model may learn from: interactions strictly before ``t_rec``.

    ``t_rec`` is the moment at which recommendations are produced (the split boundary).
    User and item indices live in the global index space of the whole dataset; users or
    items without interactions here are cold and never evaluated or recommended.
    """

    n_users: int
    n_items: int
    users: np.ndarray  # int64, one entry per interaction, sorted by time
    items: np.ndarray  # int64
    ratings: np.ndarray  # float32, 1..5 stars
    days: np.ndarray  # float64, days since the Unix epoch
    t_rec: float
    item_features: ItemFeatures

    @property
    def n_interactions(self) -> int:
        return len(self.users)

    @cached_property
    def matrix(self) -> sp.csr_matrix:
        """Binary user x item matrix of observed interactions (float32)."""
        data = np.ones(self.n_interactions, dtype=np.float32)
        mat = sp.csr_matrix((data, (self.users, self.items)), shape=(self.n_users, self.n_items))
        mat.sum_duplicates()
        mat.data[:] = 1.0
        mat.sort_indices()
        return mat

    @cached_property
    def user_counts(self) -> np.ndarray:
        return np.bincount(self.users, minlength=self.n_users)

    @cached_property
    def item_counts(self) -> np.ndarray:
        return np.bincount(self.items, minlength=self.n_items)

    @property
    def warm_items(self) -> np.ndarray:
        return self.item_counts > 0

    @property
    def warm_users(self) -> np.ndarray:
        return self.user_counts > 0


@dataclass(frozen=True)
class EvalTarget:
    """Ground truth for one evaluation period.

    ``relevant`` marks, for every evaluated user, the items reviewed in the period with
    a rating above the threshold that are recommendable (warm and not seen before).
    """

    users: np.ndarray  # evaluated users, sorted
    relevant: sp.csr_matrix  # n_users x n_items, binary
    candidates: np.ndarray  # bool mask of recommendable items
    stats: Dict[str, int]


def temporal_split(interactions: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut the interaction log at two global timestamps."""
    t_val, t_test = day_of(cfg.val_start), day_of(cfg.test_start)
    day = interactions["day"]
    train = interactions[day < t_val]
    val = interactions[(day >= t_val) & (day < t_test)]
    test = interactions[day >= t_test]
    logger.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


def make_fit_data(
    history: pd.DataFrame,
    t_rec: float,
    n_users: int,
    n_items: int,
    item_features: ItemFeatures,
) -> FitData:
    """Wrap a history slice; fails loudly if it contains interactions after ``t_rec``."""
    if (history["day"] >= t_rec).any():
        raise ValueError("history contains interactions at or after the recommendation time")
    history = history.sort_values(["timestamp", "user", "item"], kind="stable")
    return FitData(  # copies: pandas may hand out read-only views (copy-on-write)
        n_users=n_users,
        n_items=n_items,
        users=history["user"].to_numpy(np.int64, copy=True),
        items=history["item"].to_numpy(np.int64, copy=True),
        ratings=history["rating"].to_numpy(np.float32, copy=True),
        days=history["day"].to_numpy(np.float64, copy=True),
        t_rec=float(t_rec),
        item_features=item_features,
    )


def make_target(fit: FitData, period: pd.DataFrame, threshold: float) -> EvalTarget:
    """Build the relevance matrix of a period and report what was filtered out."""
    users = period["user"].to_numpy(np.int64)
    items = period["item"].to_numpy(np.int64)
    warm_user = fit.user_counts[users] > 0
    warm_item = fit.item_counts[items] > 0
    seen = np.asarray(fit.matrix[users, items]).ravel() > 0
    positive = period["rating"].to_numpy() >= threshold
    keep = warm_user & warm_item & ~seen & positive

    relevant = sp.csr_matrix(
        (np.ones(keep.sum(), dtype=np.float32), (users[keep], items[keep])),
        shape=(fit.n_users, fit.n_items),
    )
    relevant.sum_duplicates()
    relevant.data[:] = 1.0
    relevant.sort_indices()
    eval_users = np.unique(users[keep])
    stats = {
        "period_interactions": len(period),
        "period_users": int(np.unique(users).size),
        "cold_user_interactions": int((~warm_user).sum()),
        "cold_item_interactions": int((warm_user & ~warm_item).sum()),
        "repeat_interactions": int((warm_user & warm_item & seen).sum()),
        "non_relevant_interactions": int((warm_user & warm_item & ~seen & ~positive).sum()),
        "relevant_interactions": int(relevant.nnz),
        "eval_users": int(eval_users.size),
        "candidate_items": int(fit.warm_items.sum()),
    }
    logger.info("Target: %s", stats)
    return EvalTarget(users=eval_users, relevant=relevant, candidates=fit.warm_items, stats=stats)
