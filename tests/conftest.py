"""Small synthetic datasets for fast unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.features import ItemFeatures  # noqa: E402
from utils.split import FitData  # noqa: E402


def make_fit(n_users: int = 60, n_items: int = 40, n_events: int = 700, seed: int = 0) -> FitData:
    rng = np.random.default_rng(seed)
    # Two latent "platforms" so that there is structure to learn.
    user_group = rng.integers(0, 2, n_users)
    item_group = np.arange(n_items) % 2
    users = rng.integers(0, n_users, n_events)
    same = rng.random(n_events) < 0.85
    items = np.where(
        same,
        2 * rng.integers(0, n_items // 2, n_events) + user_group[users],
        rng.integers(0, n_items, n_events),
    )
    keys = np.unique(users * n_items + items)
    users, items = keys // n_items, keys % n_items
    days = rng.uniform(15000.0, 16000.0, len(users))
    order = np.argsort(days)
    ratings = rng.integers(1, 6, len(users)).astype(np.float32)
    codes = {
        "platform": item_group.astype(np.int64),
        "item_type": rng.integers(0, 3, n_items),
        "store": rng.integers(0, 4, n_items),
        "esrb": rng.integers(0, 2, n_items),
        "price_bin": rng.integers(0, 3, n_items),
    }
    vocab = {name: [f"{name}{v}" for v in range(int(c.max()) + 1)] for name, c in codes.items()}
    return FitData(
        n_users=n_users,
        n_items=n_items,
        users=users[order],
        items=items[order],
        ratings=ratings[order],
        days=days[order],
        t_rec=16000.0,
        item_features=ItemFeatures(codes=codes, vocab=vocab),
    )


@pytest.fixture
def fit() -> FitData:
    return make_fit()
