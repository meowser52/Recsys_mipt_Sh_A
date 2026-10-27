"""Device-resident helpers shared by the SGD-trained models (FM and timeSVD++).

Random numbers are always drawn from a CPU ``torch.Generator`` and then moved to the
model's device, so a given seed produces the same samples on CPU and GPU.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from utils.features import item_first_day
from utils.split import FitData

logger = logging.getLogger(__name__)

__all__ = ["NegativeSampler", "concat_ranges"]


def concat_ranges(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """``cat([arange(s, s + n) for s, n in zip(starts, lengths)])`` without a Python loop."""
    total = int(lengths.sum())
    shift = torch.repeat_interleave(starts - torch.cumsum(lengths, 0) + lengths, lengths, output_size=total)
    return shift + torch.arange(total, device=starts.device)


class NegativeSampler:
    """Uniform negatives, either over the whole warm catalogue or time-aware.

    Time-aware mode samples only items that already existed at the interaction time
    (first interaction in the fit data ``<= t``): a not-yet-released item cannot have
    been rejected by the user. The price is that a new item is rarely sampled (it has
    existed for a short time), so its few positives are barely counter-balanced and
    noisy "rates" of brand-new items get inflated. Catalogue-wide mode penalises every
    item at the same rate. Which one ranks better is decided on validation.

    Items the user interacted with anywhere in the fit data are always rejected.
    """

    def __init__(
        self,
        data: FitData,
        generator: torch.Generator,
        device: torch.device,
        time_aware: bool = True,
        max_rounds: int = 10,
    ) -> None:
        self.n_items, self.generator, self.device = data.n_items, generator, device
        self.time_aware, self.max_rounds = time_aware, max_rounds
        first = item_first_day(data.items, data.days, data.n_items)
        order = np.argsort(first, kind="stable")
        self.order = torch.from_numpy(order).to(device)
        self.sorted_first = torch.from_numpy(first[order]).to(device)  # float64, inf = never seen
        self.n_warm = int(np.isfinite(first).sum())
        self.seen_keys = torch.from_numpy(np.unique(data.users * data.n_items + data.items)).to(device)

    def _draw(self, n_exist: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand(len(n_exist), generator=self.generator, dtype=torch.float64).to(self.device)
        index = torch.minimum((uniform * n_exist).long(), n_exist - 1)
        return self.order[index]

    def _is_seen(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        keys = users * self.n_items + items
        pos = torch.searchsorted(self.seen_keys, keys).clamp_max(len(self.seen_keys) - 1)
        return self.seen_keys[pos] == keys

    def sample(self, users: torch.Tensor, days: torch.Tensor, n_negatives: int) -> torch.Tensor:
        """Tensor ``(len(users), n_negatives)`` of negative items for each (user, day)."""
        users = users.repeat_interleave(n_negatives)
        if self.time_aware:
            days = days.repeat_interleave(n_negatives)
            n_exist = torch.searchsorted(self.sorted_first, days, right=True).clamp_min(1)
        else:
            n_exist = torch.full_like(users, self.n_warm)
        items = self._draw(n_exist)
        bad = self._is_seen(users, items)
        for _ in range(self.max_rounds):
            redo = bad.nonzero().squeeze(1)
            if redo.numel() == 0:
                break
            items[redo] = self._draw(n_exist[redo])
            bad[redo] = self._is_seen(users[redo], items[redo])
        return items.view(-1, n_negatives)
