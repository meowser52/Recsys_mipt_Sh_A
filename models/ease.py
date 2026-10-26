"""EASE: Embarrassingly Shallow Autoencoder (Steck, WWW 2019).

    min_B ||X - X B||_F^2 + lambda ||B||_F^2   s.t.  diag(B) = 0

has the closed form ``P = (X^T X + lambda I)^{-1}``, ``B_ij = -P_ij / P_jj`` (``i != j``),
``B_jj = 0``. Scores are ``s_u = x_u B``. The Gram matrix is built only over warm items.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from models.base import BaseRecommender, EpochCallback, dense_rows, register_model
from utils.split import FitData

logger = logging.getLogger(__name__)

__all__ = ["EASE", "EASEConfig"]


@dataclass(frozen=True)
class EASEConfig:
    l2: float = 500.0
    dtype: str = "float32"  # float32 halves memory for ~25k items; fine for ranking
    device: str = "cpu"  # runtime setting, not a hyperparameter


@register_model("EASE")
class EASE(BaseRecommender):
    config_cls = EASEConfig

    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "EASE":
        self.n_users, self.n_items = data.n_users, data.n_items
        self.active = np.flatnonzero(data.warm_items)
        self.device, self.dtype = torch.device(self.cfg.device), getattr(torch, self.cfg.dtype)
        x = data.matrix[:, self.active].tocsr()
        gram = torch.from_numpy((x.T @ x).toarray()).to(device=self.device, dtype=self.dtype)
        gram.diagonal().add_(self.cfg.l2)
        # X^T X + lambda I is SPD: Cholesky-based inversion is ~7x faster than LU here.
        chol = torch.linalg.cholesky(gram)
        del gram
        weights = torch.cholesky_inverse(chol)
        del chol
        weights /= -torch.diagonal(weights).clone()[None, :]  # B_ij = -P_ij / P_jj, in place
        weights.fill_diagonal_(0.0)
        self.weights = weights
        self.x = x
        logger.debug("EASE fitted on %d items", len(self.active))
        return self

    def score_tensor(self, users: np.ndarray) -> torch.Tensor:
        rows = dense_rows(self.x, users, self.device).to(self.dtype)
        out = torch.zeros((len(users), self.n_items), dtype=torch.float32, device=self.device)
        out[:, torch.from_numpy(self.active).to(self.device)] = (rows @ self.weights).float()
        return out

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.score_tensor(users).cpu().numpy()
