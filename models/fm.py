"""Factorization Machine (Rendle, ICDM 2010) with metadata, trained with BPR.

An event ``(u, i, t)`` is a sparse vector ``x`` made of three groups of features:

* user side   -- user id; platform profile of the user's history (weighted multi-hot);
* item side   -- item id; platform, item type, store, price bucket, ESRB rating;
* context     -- item age (days since its first interaction) and item popularity over
                 the last ``recent_window`` days, both measured at ``t`` from interactions
                 strictly before ``t``. An item's very first event has no earlier history
                 and gets a dedicated age bucket; otherwise every first review would look
                 like "age 0 => positive" and the model would learn that artefact.

    y(x) = w0 + sum_f w_f x_f + 1/2 sum_k [(sum_f v_fk x_f)^2 - sum_f v_fk^2 x_f^2]

BPR (Rendle et al., UAI 2009) maximises ``sum log sigmoid(y(u, i, t) - y(u, j, t))`` over
observed events and negatives ``j`` not seen by ``u``, drawn uniformly from the warm
catalogue or, with ``time_aware_negatives``, only from items that existed at ``t``.

At recommendation time ``t_rec`` the user-only terms are a per-user constant, so

    score(u, i) = b_i + <s_u, s_i>,   s_u = sum_{f in user} v_f x_f,   s_i = sum_{f in item, ctx} v_f x_f,

where ``b_i`` collects the linear and pairwise terms inside the item/context side,
evaluated at ``t_rec``. Scoring all items is then a single matrix product.
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
from models.sampling import NegativeSampler
from utils.features import RecentCounter, item_first_day, log2_bucket, user_field_profile
from utils.split import FitData

logger = logging.getLogger(__name__)

__all__ = ["FM", "FMConfig"]


@dataclass(frozen=True)
class FMConfig:
    factors: int = 64
    lr: float = 2e-3
    l2: float = 1e-5  # on the weights of active features of every sample
    batch_size: int = 4096
    n_epochs: int = 20
    n_negatives: int = 1
    time_aware_negatives: bool = True  # sample only items that existed at t
    use_item_meta: bool = True
    use_user_profile: bool = True
    use_item_age: bool = True
    use_recent_pop: bool = True
    recent_window: float = 90.0  # days
    n_age_buckets: int = 14
    n_pop_buckets: int = 12
    max_profile: int = 8
    init_std: float = 0.01
    seed: int = 42
    device: str = "cpu"  # runtime setting, not a hyperparameter


class _FeatureSpace:
    """Maps events to padded feature indices on the model's device; index 0 is padding."""

    def __init__(self, data: FitData, cfg: FMConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.offsets: Dict[str, int] = {}
        self.size = 1
        self._add("user", data.n_users)
        self._add("item", data.n_items)
        meta = data.item_features
        meta_fields = meta.fields if cfg.use_item_meta else []
        for field in meta_fields:
            self._add(f"item_{field}", meta.cardinality(field))
        if cfg.use_user_profile:
            self._add("user_platform", meta.cardinality("platform"))
        if cfg.use_item_age:
            self._add("age", cfg.n_age_buckets + 1)  # +1: "no earlier interaction"
        if cfg.use_recent_pop:
            self._add("pop", cfg.n_pop_buckets)

        user_cols = [self.offsets["user"] + np.arange(data.n_users)[:, None]]
        user_vals = [np.ones((data.n_users, 1), dtype=np.float32)]
        if cfg.use_user_profile:
            codes, weights = user_field_profile(
                data.users, meta.codes["platform"][data.items], data.n_users, cfg.max_profile
            )
            user_cols.append(np.where(codes >= 0, self.offsets["user_platform"] + codes, 0))
            user_vals.append(weights)
        self.user_idx = torch.from_numpy(np.hstack(user_cols).astype(np.int64)).to(device)
        self.user_val = torch.from_numpy(np.hstack(user_vals).astype(np.float32)).to(device)

        item_cols = [self.offsets["item"] + np.arange(data.n_items)]
        item_cols += [self.offsets[f"item_{f}"] + meta.codes[f] for f in meta_fields]
        self.item_idx = torch.from_numpy(np.stack(item_cols, axis=1).astype(np.int64)).to(device)

        self.first_day = torch.from_numpy(item_first_day(data.items, data.days, data.n_items)).to(device)
        self.counter = RecentCounter(data.items, data.days, device)

    def _add(self, name: str, cardinality: int) -> None:
        self.offsets[name] = self.size
        self.size += cardinality

    def _context(self, items: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        cols = []
        if self.cfg.use_item_age:
            age = days - self.first_day[items]
            has_past = age > 0  # False for the item's first event (and never-seen items)
            bucket = 1 + log2_bucket(torch.where(has_past, age, 0.0), self.cfg.n_age_buckets)
            cols.append(self.offsets["age"] + torch.where(has_past, bucket, 0))
        if self.cfg.use_recent_pop:
            pop = self.counter.count(items, days, self.cfg.recent_window)
            cols.append(self.offsets["pop"] + log2_bucket(pop, self.cfg.n_pop_buckets))
        return torch.stack(cols, dim=1)

    def item_side(self, items: torch.Tensor, days: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.item_idx[items]
        if self.cfg.use_item_age or self.cfg.use_recent_pop:
            idx = torch.cat([idx, self._context(items, days)], dim=1)
        return idx, torch.ones(idx.shape, dtype=torch.float32, device=idx.device)

    def events(self, users: torch.Tensor, items: torch.Tensor, days: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        item_idx, item_val = self.item_side(items, days)
        idx = torch.cat([self.user_idx[users], item_idx], dim=1)
        val = torch.cat([self.user_val[users], item_val], dim=1)
        return idx, val


class _FMNet(nn.Module):
    def __init__(self, n_features: int, factors: int, init_std: float) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(n_features, 1, padding_idx=0)
        self.embed = nn.Embedding(n_features, factors, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embed.weight, std=init_std)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def parts(self, idx: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(sum_f v_f x_f, linear + pairwise terms, L2 of active weights)``."""
        emb = self.embed(idx)
        lin_w = self.linear(idx).squeeze(-1)
        weighted = emb * val.unsqueeze(-1)
        total = weighted.sum(dim=1)
        pair = 0.5 * (total.square() - weighted.square().sum(dim=1)).sum(dim=1)
        lin = (lin_w * val).sum(dim=1)
        active = (val != 0).float()
        reg = ((emb.square().sum(dim=-1) + lin_w.square()) * active).sum(dim=1)
        return total, lin + pair, reg

    def forward(self, idx: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, terms, reg = self.parts(idx, val)
        return self.bias + terms, reg


@register_model("FM")
class FM(BaseRecommender):
    config_cls = FMConfig
    _cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # item side at t_rec

    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "FM":
        cfg = self.cfg
        self.n_users, self.n_items = data.n_users, data.n_items
        self.t_rec = data.t_rec
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)  # parameter init happens on the CPU, then moves
        generator = torch.Generator().manual_seed(cfg.seed)  # sampling stream, same on any device
        self.space = _FeatureSpace(data, cfg, self.device)
        self.net = _FMNet(self.space.size, cfg.factors, cfg.init_std).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        sampler = NegativeSampler(data, generator, self.device, time_aware=cfg.time_aware_negatives)
        users = torch.from_numpy(data.users).to(self.device)
        items = torch.from_numpy(data.items).to(self.device)
        days = torch.from_numpy(data.days).to(self.device)
        n_neg = cfg.n_negatives

        for epoch in range(1, cfg.n_epochs + 1):
            self.net.train()
            perm = torch.randperm(data.n_interactions, generator=generator).to(self.device)
            total_loss = torch.zeros((), device=self.device)
            for start in range(0, len(perm), cfg.batch_size):
                batch = perm[start : start + cfg.batch_size]
                u, i, t = users[batch], items[batch], days[batch]
                negatives = sampler.sample(u, t, n_neg)
                y_pos, reg_pos = self.net(*self.space.events(u, i, t))
                y_neg, reg_neg = self.net(
                    *self.space.events(u.repeat_interleave(n_neg), negatives.reshape(-1), t.repeat_interleave(n_neg))
                )
                margin = y_pos[:, None] - y_neg.view(-1, n_neg)
                loss = -F.logsigmoid(margin).mean() + cfg.l2 * (reg_pos.mean() + reg_neg.mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.detach() * len(batch)
            logger.debug("FM epoch %d: loss=%.4f", epoch, float(total_loss) / len(perm))
            self._cache = None
            if callback is not None and callback(epoch, self):
                break
        self._cache = None
        return self

    def _events(self, users: np.ndarray, items: np.ndarray, days: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feature tensors for numpy (user, item, day) triples, e.g. for inspection."""
        return self.space.events(
            torch.from_numpy(np.asarray(users, dtype=np.int64)).to(self.device),
            torch.from_numpy(np.asarray(items, dtype=np.int64)).to(self.device),
            torch.from_numpy(np.asarray(days, dtype=np.float64)).to(self.device),
        )

    def _item_side_at_rec(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is None:
            items = torch.arange(self.n_items, device=self.device)
            days = torch.full((self.n_items,), self.t_rec, dtype=torch.float64, device=self.device)
            idx, val = self.space.item_side(items, days)
            with torch.no_grad():
                total, terms, _ = self.net.parts(idx, val)
            self._cache = (total, terms)
        return self._cache

    def score_tensor(self, users: np.ndarray) -> torch.Tensor:
        self.net.eval()
        item_vecs, item_bias = self._item_side_at_rec()
        index = torch.from_numpy(np.asarray(users, dtype=np.int64)).to(self.device)
        idx, val = self.space.user_idx[index], self.space.user_val[index]
        with torch.no_grad():
            user_vecs, _, _ = self.net.parts(idx, val)
            return item_bias[None, :] + user_vecs @ item_vecs.T

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.score_tensor(users).cpu().numpy()
