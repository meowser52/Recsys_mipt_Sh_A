"""Non-personalised sanity baselines: random and (time-decayed) popularity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from models.base import BaseRecommender, EpochCallback, register_model
from utils.split import FitData

logger = logging.getLogger(__name__)

__all__ = ["Popular", "PopularConfig", "RandomConfig", "RandomRecommender"]


@dataclass(frozen=True)
class PopularConfig:
    """``half_life=None`` counts all interactions; otherwise each one is weighted by
    ``0.5 ** ((t_rec - t) / half_life)`` so that recent popularity dominates."""

    half_life: Optional[float] = None
    device: str = "cpu"  # runtime setting, not a hyperparameter


@register_model("Popular")
class Popular(BaseRecommender):
    """Ranks items by (optionally time-decayed) number of interactions."""

    config_cls = PopularConfig

    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "Popular":
        self.n_users, self.n_items = data.n_users, data.n_items
        if self.cfg.half_life is None:
            weights = np.ones(data.n_interactions)
        else:
            weights = 0.5 ** ((data.t_rec - data.days) / self.cfg.half_life)
        self.popularity = np.bincount(data.items, weights=weights, minlength=data.n_items)
        self.popularity = self.popularity.astype(np.float32)
        self._popularity_t = torch.from_numpy(self.popularity).to(torch.device(self.cfg.device))
        return self

    def score_tensor(self, users: np.ndarray) -> torch.Tensor:
        return self._popularity_t.expand(len(users), -1).clone()

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.popularity, (len(users), 1))


@dataclass(frozen=True)
class RandomConfig:
    seed: int = 42


@register_model("Random")
class RandomRecommender(BaseRecommender):
    """Uniformly random ranking; the stream of each user depends only on (seed, user)."""

    config_cls = RandomConfig

    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "RandomRecommender":
        self.n_users, self.n_items = data.n_users, data.n_items
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        out = np.empty((len(users), self.n_items), dtype=np.float32)
        for row, user in enumerate(users):
            out[row] = np.random.default_rng([self.cfg.seed, int(user)]).random(self.n_items)
        return out
