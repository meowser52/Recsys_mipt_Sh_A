"""Implicit ALS (Hu, Koren & Volinsky, ICDM 2008) with a conjugate-gradient solver.

Preference ``p_ui = 1`` for observed pairs and ``0`` otherwise, confidence
``c_ui = 1 + alpha * w_ui`` (``w_ui = 1`` for observed pairs, ``0`` otherwise)::

    min_{X,Y} sum_{u,i} c_ui (p_ui - x_u^T y_i)^2 + lambda (||X||^2 + ||Y||^2)

With ``Y`` fixed every user has the closed form

    x_u = (Y^T Y + lambda I + Y^T (C^u - I) Y)^{-1} Y^T C^u p(u),

and symmetrically for items. Instead of inverting a ``d x d`` matrix per user, the
system is solved approximately by a few warm-started conjugate-gradient steps
(Takács, Pilászy & Tikk, RecSys 2011), vectorised over all users at once. The
expensive part ``Y^T (C^u - I) Y v`` only touches the observed entries.

Training target and evaluation are kept separate: this is the "base" implicit model.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from models.base import BaseRecommender, EpochCallback, register_model
from utils.split import FitData

logger = logging.getLogger(__name__)

__all__ = ["IALS", "IALSConfig", "conjugate_gradient_solve"]


@dataclass(frozen=True)
class IALSConfig:
    factors: int = 64
    alpha: float = 10.0  # confidence of an observed pair is 1 + alpha
    l2: float = 1.0
    n_iters: int = 15
    cg_steps: int = 3
    init_std: float = 0.01
    seed: int = 42
    device: str = "cpu"  # runtime setting, not a hyperparameter


def conjugate_gradient_solve(
    conf: sp.csr_matrix,
    fixed: torch.Tensor,
    init: torch.Tensor,
    l2: float,
    steps: int,
) -> torch.Tensor:
    """Solve ``(F^T F + l2 I + F^T diag(conf_u) F) x_u = F^T (1 + conf_u)`` for every row.

    ``conf`` holds ``c_ui - 1`` on the observed entries of each row (CSR), ``fixed`` is
    the factor matrix of the other side and ``init`` the warm start.
    """
    device = fixed.device
    gram = fixed.T @ fixed + l2 * torch.eye(fixed.shape[1], dtype=fixed.dtype, device=device)
    crow = torch.from_numpy(conf.indptr.astype(np.int64)).to(device)
    cols = torch.from_numpy(conf.indices.astype(np.int64)).to(device)
    rows = torch.from_numpy(np.repeat(np.arange(conf.shape[0]), np.diff(conf.indptr))).to(device)
    extra = torch.from_numpy(conf.data.astype(np.float32)).to(device)
    gathered = fixed[cols]  # (nnz, d)

    def sparse_rows(values: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():  # "sparse CSR support is in beta" on every call
            warnings.simplefilter("ignore", UserWarning)
            return torch.sparse_csr_tensor(crow, cols, values, size=conf.shape, check_invariants=False)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        dots = (gathered * v[rows]).sum(dim=1) * extra
        return v @ gram + sparse_rows(dots) @ fixed

    rhs = sparse_rows(1.0 + extra) @ fixed
    x = init.clone()
    resid = rhs - matvec(x)
    direction = resid.clone()
    rs_old = (resid * resid).sum(dim=1)
    for _ in range(steps):
        a_dir = matvec(direction)
        step = rs_old / (direction * a_dir).sum(dim=1).clamp_min(1e-20)
        x += step[:, None] * direction
        resid -= step[:, None] * a_dir
        rs_new = (resid * resid).sum(dim=1)
        direction = resid + (rs_new / rs_old.clamp_min(1e-20))[:, None] * direction
        rs_old = rs_new
    return x


@register_model("iALS")
class IALS(BaseRecommender):
    config_cls = IALSConfig

    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "IALS":
        cfg = self.cfg
        self.n_users, self.n_items = data.n_users, data.n_items
        conf_users = data.matrix.copy()
        conf_users.data[:] = cfg.alpha  # c_ui - 1 on observed pairs
        conf_items = conf_users.T.tocsr()
        conf_items.sort_indices()

        gen = torch.Generator().manual_seed(cfg.seed)
        shape_u, shape_i = (self.n_users, cfg.factors), (self.n_items, cfg.factors)
        device = torch.device(cfg.device)
        self.user_factors = (torch.randn(shape_u, generator=gen) * cfg.init_std).to(device)
        self.item_factors = (torch.randn(shape_i, generator=gen) * cfg.init_std).to(device)
        for epoch in range(1, cfg.n_iters + 1):
            self.user_factors = conjugate_gradient_solve(
                conf_users, self.item_factors, self.user_factors, cfg.l2, cfg.cg_steps
            )
            self.item_factors = conjugate_gradient_solve(
                conf_items, self.user_factors, self.item_factors, cfg.l2, cfg.cg_steps
            )
            if callback is not None and callback(epoch, self):
                break
        return self

    def score_tensor(self, users: np.ndarray) -> torch.Tensor:
        index = torch.from_numpy(np.asarray(users, dtype=np.int64)).to(self.user_factors.device)
        return self.user_factors[index] @ self.item_factors.T

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.score_tensor(users).cpu().numpy()

    def loss(self, data: FitData) -> float:
        """Full objective (dense over items); only for tests on small data."""
        pred = (self.user_factors @ self.item_factors.T).cpu().numpy()
        pref = data.matrix.toarray()
        conf = 1.0 + self.cfg.alpha * pref
        reg = self.cfg.l2 * float(self.user_factors.square().sum() + self.item_factors.square().sum())
        return float((conf * (pref - pred) ** 2).sum() + reg)
