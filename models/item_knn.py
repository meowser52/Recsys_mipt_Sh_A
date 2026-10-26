"""Item-based nearest-neighbour collaborative filtering.

Similarity (asymmetric cosine with shrinkage, Aiolli 2013) between a candidate item
``i`` and an item ``j`` from the user's history::

    sim(i, j) = |U_i ∩ U_j| / (|U_i|^alpha * |U_j|^(1 - alpha) + shrink)

``alpha = 0.5`` gives the usual cosine. Only the ``n_neighbors`` most similar history
items are kept for every candidate. The score sums similarities over the history,
optionally weighting each history item by recency::

    score(u, i) = sum_{j in I(u)} w_uj * sim(i, j),   w_uj = 0.5 ** ((t_rec - t_uj) / half_life)

Co-occurrences are counted sparsely on the CPU; the normalised similarity of the warm
items is then kept as a dense matrix on the model's device (about 2 GB for 24k items),
where top-k pruning and scoring are single tensor operations.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from models.base import BaseRecommender, EpochCallback, dense_rows, register_model
from utils.split import FitData

logger = logging.getLogger(__name__)

__all__ = ["ItemKNN", "ItemKNNConfig"]


@dataclass(frozen=True)
class ItemKNNConfig:
    n_neighbors: int = 100
    alpha: float = 0.5
    shrink: float = 10.0
    half_life: Optional[float] = None  # days; None disables recency weighting
    device: str = "cpu"  # runtime setting, not a hyperparameter


@register_model("ItemKNN")
class ItemKNN(BaseRecommender):
    config_cls = ItemKNNConfig

    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "ItemKNN":
        cfg = self.cfg
        self.n_users, self.n_items = data.n_users, data.n_items
        self.device = torch.device(cfg.device)
        self.active = np.flatnonzero(data.warm_items)
        x = data.matrix[:, self.active].tocsr()
        co = (x.T @ x).tocsr().astype(np.float32)  # |U_i ∩ U_j|, exact integers
        co.setdiag(0.0)
        co.eliminate_zeros()
        n = np.asarray(x.sum(axis=0), dtype=np.float64).ravel()

        # Row = history item j, column = candidate item i (co is symmetric).
        history = np.repeat(np.arange(co.shape[0]), np.diff(co.indptr))
        candidate = co.indices
        co.data /= (n[candidate] ** cfg.alpha * n[history] ** (1.0 - cfg.alpha) + cfg.shrink).astype(np.float32)
        with warnings.catch_warnings():  # "sparse CSR support is in beta"
            warnings.simplefilter("ignore", UserWarning)
            sim = (
                torch.sparse_csr_tensor(
                    torch.from_numpy(co.indptr.astype(np.int64)),
                    torch.from_numpy(co.indices.astype(np.int64)),
                    torch.from_numpy(co.data),
                    size=co.shape,
                    check_invariants=False,
                )
                .to(self.device)
                .to_dense()
            )
        k = min(cfg.n_neighbors, sim.shape[0])
        values, rows = torch.topk(sim, k, dim=0)  # best history items of every candidate
        sim.zero_()
        sim.scatter_(0, rows, values)
        self.similarity = sim

        if cfg.half_life is None:
            weights = np.ones(data.n_interactions, dtype=np.float32)
        else:
            weights = (0.5 ** ((data.t_rec - data.days) / cfg.half_life)).astype(np.float32)
        active_pos = np.full(data.n_items, -1)
        active_pos[self.active] = np.arange(len(self.active))
        profile = sp.csr_matrix((weights, (data.users, active_pos[data.items])), shape=x.shape)
        profile.sum_duplicates()
        self.profile = profile
        return self

    def score_tensor(self, users: np.ndarray) -> torch.Tensor:
        rows = dense_rows(self.profile, users, self.device)
        out = torch.zeros((len(users), self.n_items), dtype=torch.float32, device=self.device)
        out[:, torch.from_numpy(self.active).to(self.device)] = rows @ self.similarity
        return out

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.score_tensor(users).cpu().numpy()
