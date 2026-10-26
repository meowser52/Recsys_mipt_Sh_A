"""Shared evaluation loop: identical users, candidates and exclusion rules for every model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch

from utils.metrics import catalog_coverage, novelty, ranking_metrics
from utils.split import EvalTarget, FitData

if TYPE_CHECKING:
    from models.base import BaseRecommender

logger = logging.getLogger(__name__)

__all__ = ["EvalResult", "evaluate", "recommend"]


@dataclass(frozen=True)
class EvalResult:
    """Averaged metrics, per-user metric arrays and the produced top-K lists."""

    metrics: Dict[str, float]
    per_user: Dict[str, np.ndarray]
    topk: np.ndarray
    users: np.ndarray


def recommend(
    model: "BaseRecommender",
    fit: FitData,
    users: np.ndarray,
    candidates: np.ndarray,
    k: int = 10,
    batch_size: int = 2048,
    device: Optional[str] = None,
) -> np.ndarray:
    """Top-``k`` items for ``users``: non-candidate and already seen items are excluded.

    Ranking runs on ``device`` (default: wherever the model put its scores); only the
    top-K indices are copied back.
    """
    seen = fit.matrix
    out = np.empty((len(users), k), dtype=np.int64)
    blocked = None
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = model.score_tensor(batch).float()
        if device is not None:
            scores = scores.to(device)
        device_now = scores.device
        if blocked is None or blocked.device != device_now:
            blocked = torch.from_numpy(~candidates).to(device_now)
        n_nan = int(torch.isnan(scores).sum())
        if n_nan:
            logger.warning("%s produced %d NaN scores", type(model).__name__, n_nan)
            scores = torch.nan_to_num(scores, nan=-np.inf)
        scores.masked_fill_(blocked[None, :], -np.inf)
        sub = seen[batch]
        rows = torch.from_numpy(np.repeat(np.arange(len(batch)), np.diff(sub.indptr))).to(device_now)
        cols = torch.from_numpy(sub.indices.astype(np.int64)).to(device_now)
        scores[rows, cols] = -np.inf
        out[start : start + len(batch)] = torch.topk(scores, k, dim=1).indices.cpu().numpy()
    return out


def evaluate(
    model: "BaseRecommender",
    fit: FitData,
    target: EvalTarget,
    k: int = 10,
    batch_size: int = 2048,
    device: Optional[str] = None,
) -> EvalResult:
    """Recommend for every evaluated user and compute accuracy and diversity metrics."""
    topk = recommend(model, fit, target.users, target.candidates, k=k, batch_size=batch_size, device=device)
    per_user = ranking_metrics(topk, target.relevant[target.users])
    metrics = {f"{name}@{k}": float(values.mean()) for name, values in per_user.items()}
    metrics[f"coverage@{k}"] = catalog_coverage(topk, int(target.candidates.sum()))
    metrics[f"novelty@{k}"] = novelty(topk, fit.item_counts, int(fit.warm_users.sum()))
    return EvalResult(metrics=metrics, per_user=per_user, topk=topk, users=target.users)
