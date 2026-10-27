"""timeSVD++ (Koren, KDD 2009 "Collaborative filtering with temporal dynamics").

Prediction of the rating of user ``u`` for item ``i`` at day ``t``::

    r(u, i, t) = mu + b_u + alpha_u dev_u(t) + b_{u,t}                       (user bias drift)
               + (b_i + b_{i,Bin(t)}) (c_u + c_{u,t})                       (item bias drift, user scale)
               + q_i^T (p_u + alpha^p_u dev_u(t) + |N(u)|^{-1/2} sum_{j in N(u)} y_j)

    dev_u(t) = sign(t - t_u) |t - t_u|^beta / s,  t_u = mean rating day of u,  beta = 0.4

(``t`` in days; ``s`` is the standard deviation of the raw deviation over training
ratings, so that the drift coefficients live on the same scale as the other parameters.
Koren compensated the large raw scale with ~1e-5 learning rates for ``alpha``; with
Adam's per-parameter step sizes an unscaled ``dev`` lets the drift terms overfit first.)

``b_{u,t}`` and ``c_{u,t}`` are day-specific (one parameter per (user, day) pair seen in
training), ``Bin(t)`` splits the training period into ``n_bins`` equal time bins and
``N(u)`` is the set of items the user rated before ``t_rec``.

Classic training minimises the squared error on observed ratings. Such a model only
learns *how* a user rates an item they chose, not *which* items they choose, so it
ranks poorly on top-N tasks. With ``n_negatives > 0`` we add the missing-data term of
Steck (KDD 2010, "AllRank"): unobserved items, sampled uniformly (optionally among those existing at ``t``),
are imputed with a low rating ``r_m`` and weight ``w_m``::

    L = sum_{observed} (r - r(u,i,t))^2 + w_m sum_{sampled missing} (r_m - r(u,j,t))^2
        + l2_bias ||b||^2 + l2_factor (||P||^2 + ||Q||^2 + ||Y||^2) + l2_time ||time terms||^2

The predictor is unchanged: ``n_negatives = 0`` recovers the classic objective.
The penalty is MAP-style: every parameter is penalised once per epoch, not once per
rating it touches. With per-rating penalties a user with 5 ratings is shrunk exactly
as much (relatively) as one with 500, which overfits on data this sparse; here the
shrinkage ``n_u / (n_u + lambda)`` grows with the user's history, as in the classic
baseline predictor. Training uses mini-batches of shuffled ratings (as Koren's SGD visits
ratings); a rating of user ``u`` carries ``1 / n_u`` of the user's penalty (``1 / n_{u,t}``
for day terms) and a batch carries its share ``B / |R|`` of the item tables' penalty.

Predicting the future. Koren's test ratings lie inside each user's rating period, so
``dev_u(t)`` only interpolates. Here every prediction is made after ``t_rec``, often
years after the user's mean rating day, and ``|t - t_u|^beta`` keeps growing: the
drift terms ``alpha_u dev_u(t)`` and ``alpha^p_u dev_u(t)`` would extrapolate far
outside the range they were fitted on. By default (``extrapolate_drift=False``) every
time-dependent parameter is therefore frozen at its last observed value: the user
drift at the user's last training day, the item bias at the last time bin, and the
day-specific terms ``b_{u,t}``, ``c_{u,t}`` are zero for an unseen day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import BaseRecommender, EpochCallback, register_model
from models.sampling import NegativeSampler, concat_ranges
from utils.split import FitData

logger = logging.getLogger(__name__)

__all__ = ["TimeSVDpp", "TimeSVDppConfig"]


@dataclass(frozen=True)
class TimeSVDppConfig:
    factors: int = 64
    lr: float = 3e-3
    l2_bias: float = 10.0  # MAP-style: roughly "pseudo-ratings" of shrinkage
    l2_factor: float = 10.0
    l2_time: float = 10.0
    beta: float = 0.4
    n_bins: int = 30
    use_time: bool = True  # False -> plain SVD++
    n_negatives: int = 16  # sampled missing entries per observed rating (0 = classic)
    time_aware_negatives: bool = False  # sample only items that existed at t
    neg_target: float = 0.3  # imputed rating r_m
    neg_weight: float = 1.0  # weight w_m of the imputed entries
    batch_size: int = 4096
    n_epochs: int = 20
    init_std: float = 0.01
    seed: int = 42
    extrapolate_drift: bool = False  # future t: freeze user drift at the last training day
    device: str = "cpu"  # runtime setting, not a hyperparameter


def _embedding(n: int, dim: int, std: float = 0.0, fill: float = 0.0, padding: bool = False) -> nn.Embedding:
    emb = nn.Embedding(n + int(padding), dim, padding_idx=n if padding else None)
    with torch.no_grad():
        if std > 0:
            emb.weight.normal_(0.0, std)
        else:
            emb.weight.fill_(fill)
        if padding:
            emb.weight[n].zero_()
    return emb


class _TimeSVDppNet(nn.Module):
    mu: torch.Tensor  # registered buffer: global mean rating

    def __init__(self, n_users: int, n_items: int, n_user_days: int, mu: float, cfg: TimeSVDppConfig) -> None:
        super().__init__()
        k, std = cfg.factors, cfg.init_std
        self.cfg = cfg
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.b_u, self.b_i = _embedding(n_users, 1), _embedding(n_items, 1)
        self.p, self.q, self.y = _embedding(n_users, k, std), _embedding(n_items, k, std), _embedding(n_items, k, std)
        if cfg.use_time:
            self.a_u = _embedding(n_users, 1)  # alpha_u: linear drift of the user bias
            self.a_p = _embedding(n_users, k)  # alpha^p_u: linear drift of the user factors
            self.b_ibin = _embedding(n_items * cfg.n_bins, 1)
            self.c_u = _embedding(n_users, 1, fill=1.0)
            self.b_ut = _embedding(n_user_days, 1, padding=True)  # padding = unseen day
            self.c_ut = _embedding(n_user_days, 1, padding=True)

    def implicit(self, indices: torch.Tensor, offsets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``|N(u)|^{-1/2} sum_{j in N(u)} y_j`` for a block of users."""
        return F.embedding_bag(indices, self.y.weight, offsets, mode="sum", per_sample_weights=weights)

    def forward(self, s: Dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """Predicted ratings; ``z`` is the implicit term of each sample's user."""
        cfg = self.cfg
        u, i = s["user"], s["item"]
        b_u, b_i = self.b_u(u).squeeze(-1), self.b_i(i).squeeze(-1)
        p, q = self.p(u), self.q(i)
        if not cfg.use_time:
            return self.mu + b_u + b_i + (q * (p + z)).sum(-1)
        dev, ud = s["dev"], s["user_day"]
        b_ut, c_ut = self.b_ut(ud).squeeze(-1), self.c_ut(ud).squeeze(-1)
        b_ibin = self.b_ibin(i * cfg.n_bins + s["bin"]).squeeze(-1)
        user_bias = b_u + self.a_u(u).squeeze(-1) * dev + b_ut
        item_bias = (b_i + b_ibin) * (self.c_u(u).squeeze(-1) + c_ut)
        user_vec = p + self.a_p(u) * dev.unsqueeze(-1) + z
        return self.mu + user_bias + item_bias + (q * user_vec).sum(-1)

    def penalty(
        self, users: torch.Tensor, user_w: torch.Tensor, user_days: torch.Tensor, day_w: torch.Tensor, item_share: float
    ) -> torch.Tensor:
        """MAP-style L2 of one batch: weighted user and user-day rows, a share of the item tables."""
        cfg = self.cfg

        def rows(emb: nn.Embedding, idx: torch.Tensor, weight: torch.Tensor, center: float = 0.0) -> torch.Tensor:
            return ((emb(idx) - center).square().sum(-1) * weight).sum()

        reg = cfg.l2_bias * (rows(self.b_u, users, user_w) + item_share * self.b_i.weight.square().sum())
        reg = reg + cfg.l2_factor * (
            rows(self.p, users, user_w) + item_share * (self.q.weight.square().sum() + self.y.weight.square().sum())
        )
        if cfg.use_time:
            reg = reg + cfg.l2_time * (
                rows(self.a_u, users, user_w)
                + rows(self.a_p, users, user_w)
                + rows(self.c_u, users, user_w, center=1.0)
                + rows(self.b_ut, user_days, day_w)
                + rows(self.c_ut, user_days, day_w)
                + item_share * self.b_ibin.weight.square().sum()
            )
        return reg


@register_model("timeSVD++")
class TimeSVDpp(BaseRecommender):
    config_cls = TimeSVDppConfig

    def _t(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    def _prepare(self, data: FitData) -> None:
        """Training events grouped by user, moved to the device once."""
        order = np.lexsort((data.days, data.users))  # group by user, chronological inside
        users, items, days = data.users[order], data.items[order], data.days[order]
        counts = np.bincount(users, minlength=data.n_users)
        sums = np.bincount(users, weights=days, minlength=data.n_users)
        self.user_mean_day = np.divide(sums, counts, out=np.zeros(data.n_users), where=counts > 0)
        self.user_last_day = np.full(data.n_users, -np.inf)
        np.maximum.at(self.user_last_day, users, days)
        self.t0, self.t_rec = float(data.days.min()), data.t_rec
        day_keys = users * 100_000 + np.floor(days).astype(np.int64)
        self.user_day_keys, user_day = np.unique(day_keys, return_inverse=True)
        self.dev_scale = 1.0
        raw_dev = self._dev(users, days)
        self.dev_scale = max(float(raw_dev.std()), 1e-6)
        self.ev_user, self.ev_item = self._t(users), self._t(items)
        self.ev_rating = self._t(data.ratings[order].astype(np.float32))
        self.ev_day = self._t(days)
        self.ev_dev = self._t((raw_dev / self.dev_scale).astype(np.float32))
        self.ev_user_day = self._t(user_day.astype(np.int64))
        self.ev_bin = self._t(self._bin(days))
        # Per-rating shares of the MAP penalty: 1/n_u for user terms, 1/n_{u,day} for day terms.
        self.ev_user_w = self._t((1.0 / counts[users]).astype(np.float32))
        self.ev_day_w = self._t((1.0 / np.bincount(user_day)[user_day]).astype(np.float32))
        # N(u): items rated before t_rec, as CSR on the device.
        mat = data.matrix
        self.imp_ptr = self._t(mat.indptr.astype(np.int64))
        self.imp_len = self._t(np.diff(mat.indptr).astype(np.int64))
        self.imp_idx = self._t(mat.indices.astype(np.int64))
        self.imp_norm = self._t((1.0 / np.sqrt(np.maximum(np.diff(mat.indptr), 1))).astype(np.float32))
        logger.debug("timeSVD++: %d user-days, t0=%.0f", len(self.user_day_keys), self.t0)

    def _bin(self, days: np.ndarray) -> np.ndarray:
        frac = (days - self.t0) / max(self.t_rec - self.t0, 1e-9)
        return np.clip((frac * self.cfg.n_bins).astype(np.int64), 0, self.cfg.n_bins - 1)

    def _dev(self, users: np.ndarray, days: np.ndarray) -> np.ndarray:
        delta = days - self.user_mean_day[users]
        return np.sign(delta) * np.abs(delta) ** self.cfg.beta / self.dev_scale

    def _dev_future(self, users: np.ndarray, days: np.ndarray) -> np.ndarray:
        """``dev_u(t)`` for prediction times, frozen at the last training day unless extrapolating."""
        last = self.user_last_day[users]
        if not self.cfg.extrapolate_drift:
            days = np.minimum(days, last)
        return np.where(np.isfinite(last), self._dev(users, days), 0.0)

    def _implicit_block(self, users: torch.Tensor) -> torch.Tensor:
        """Implicit term ``|N(u)|^{-1/2} sum_{j in N(u)} y_j`` of every user in the block."""
        lengths = self.imp_len[users]
        idx = self.imp_idx[concat_ranges(self.imp_ptr[users], lengths)]
        offsets = torch.cumsum(lengths, 0) - lengths
        weights = self.imp_norm[users].repeat_interleave(lengths, output_size=len(idx))
        return self.net.implicit(idx, offsets, weights)

    def _samples(self, pos: torch.Tensor, items: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Feature tensors for training events at positions ``pos`` scored on ``items``."""
        return {
            "user": self.ev_user[pos],
            "item": items,
            "dev": self.ev_dev[pos],
            "user_day": self.ev_user_day[pos],
            "bin": self.ev_bin[pos],
        }

    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "TimeSVDpp":
        cfg = self.cfg
        self.n_users, self.n_items = data.n_users, data.n_items
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)  # parameter init happens on the CPU, then moves
        generator = torch.Generator().manual_seed(cfg.seed)  # sampling stream, same on any device
        self._prepare(data)
        mu = float(data.ratings.mean())
        self.net = _TimeSVDppNet(data.n_users, data.n_items, len(self.user_day_keys), mu, cfg).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        sampler = None
        if cfg.n_negatives > 0:
            sampler = NegativeSampler(data, generator, self.device, time_aware=cfg.time_aware_negatives)
        n_ratings = len(self.ev_user)

        for epoch in range(1, cfg.n_epochs + 1):
            self.net.train()
            perm = torch.randperm(n_ratings, generator=generator).to(self.device)
            sq_err = torch.zeros((), device=self.device)
            for start in range(0, n_ratings, cfg.batch_size):
                pos = perm[start : start + cfg.batch_size]
                users = self.ev_user[pos]
                block, local = torch.unique(users, return_inverse=True)
                z = self._implicit_block(block)
                pred = self.net(self._samples(pos, self.ev_item[pos]), z[local])
                err = self.ev_rating[pos] - pred
                data_loss = err.square().sum()
                if sampler is not None:
                    neg = sampler.sample(users, self.ev_day[pos], cfg.n_negatives).reshape(-1)
                    neg_samples = self._samples(pos.repeat_interleave(cfg.n_negatives), neg)
                    neg_pred = self.net(neg_samples, z[local.repeat_interleave(cfg.n_negatives)])
                    data_loss = data_loss + cfg.neg_weight * (cfg.neg_target - neg_pred).square().sum()
                penalty = self.net.penalty(
                    users, self.ev_user_w[pos], self.ev_user_day[pos], self.ev_day_w[pos], len(pos) / n_ratings
                )
                loss = (data_loss + penalty) / len(pos)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sq_err += err.detach().square().sum()
            logger.debug("timeSVD++ epoch %d: train RMSE=%.4f", epoch, (float(sq_err) / data.n_interactions) ** 0.5)
            if callback is not None and callback(epoch, self):
                break
        return self

    def _user_terms(self, users: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """User bias, item-bias scale and user vector at ``t_rec`` (unseen day)."""
        net, users_t = self.net, self._t(users)
        z = self._implicit_block(users_t)
        if not self.cfg.use_time:
            return net.b_u(users_t).squeeze(-1), torch.ones(len(users), device=self.device), net.p(users_t) + z
        dev = self._t(self._dev_future(users, np.full(len(users), self.t_rec)).astype(np.float32))
        bias = net.b_u(users_t).squeeze(-1) + net.a_u(users_t).squeeze(-1) * dev
        vec = net.p(users_t) + net.a_p(users_t) * dev.unsqueeze(-1) + z
        return bias, net.c_u(users_t).squeeze(-1), vec

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.score_tensor(users).cpu().numpy()

    def score_tensor(self, users: np.ndarray) -> torch.Tensor:
        """Predicted ratings of all items at ``t_rec`` (the last time bin)."""
        self.net.eval()
        users = np.asarray(users, dtype=np.int64)
        net = self.net
        with torch.no_grad():
            item_bias = net.b_i.weight.squeeze(-1)
            if self.cfg.use_time:
                last = torch.arange(self.n_items, device=self.device) * self.cfg.n_bins + self.cfg.n_bins - 1
                item_bias = item_bias + net.b_ibin(last).squeeze(-1)
            user_bias, scale, user_vec = self._user_terms(users)
            return net.mu + user_bias[:, None] + scale[:, None] * item_bias[None, :] + user_vec @ net.q.weight.T

    def predict(self, users: np.ndarray, items: np.ndarray, days: np.ndarray) -> np.ndarray:
        """Rating predictions for arbitrary (user, item, day) triples, e.g. for RMSE."""
        self.net.eval()
        users, items = np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)
        keys = users * 100_000 + np.floor(days).astype(np.int64)
        found = np.minimum(np.searchsorted(self.user_day_keys, keys), len(self.user_day_keys) - 1)
        user_day = np.where(self.user_day_keys[found] == keys, found, len(self.user_day_keys))
        uniq, local = np.unique(users, return_inverse=True)
        sample = {
            "user": self._t(users),
            "item": self._t(items),
            "dev": self._t(self._dev_future(users, days).astype(np.float32)),
            "user_day": self._t(user_day.astype(np.int64)),
            "bin": self._t(self._bin(days)),
        }
        with torch.no_grad():
            z = self._implicit_block(self._t(uniq))
            pred = self.net(sample, z[self._t(local)])
        return pred.cpu().numpy()
