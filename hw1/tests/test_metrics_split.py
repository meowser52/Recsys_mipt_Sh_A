"""Metrics, protocol and helper correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from models import build_model
from models.sampling import NegativeSampler, concat_ranges
from utils.evaluation import recommend
from utils.features import RecentCounter, item_first_day, log2_bucket, user_field_profile
from utils.metrics import paired_bootstrap, ranking_metrics
from utils.split import make_fit_data, make_target


def test_ranking_metrics_hand_computed() -> None:
    topk = np.array([[3, 1, 2], [0, 4, 5]])
    relevant = sp.csr_matrix(np.array([[0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0]], dtype=np.float32))
    m = ranking_metrics(topk, relevant)
    d2 = 1 / np.log2(3)  # discount of rank 2
    assert m["ndcg"][0] == pytest.approx(d2 / (1 + d2))  # hit at rank 2, two relevant items
    assert m["ndcg"][1] == pytest.approx(1.0)  # both relevant items at ranks 1-2
    assert m["recall"].tolist() == [0.5, 1.0]
    assert m["hit_rate"].tolist() == [1.0, 1.0]
    assert m["mrr"].tolist() == [0.5, 1.0]
    assert m["precision"][0] == pytest.approx(1 / 3)


def test_topk_and_exclusion(fit) -> None:
    model = build_model("Popular").fit(fit)
    order = np.argsort(-model.popularity, kind="stable")
    top = recommend(model, fit, np.array([fit.n_users - 1]), np.ones(fit.n_items, dtype=bool), k=3)
    unseen = [i for i in order if fit.matrix[fit.n_users - 1, i] == 0]
    assert model.popularity[top[0]].min() >= model.popularity[unseen[2]]  # ties may reorder
    users = np.flatnonzero(fit.warm_users)[:20]
    candidates = fit.warm_items.copy()
    candidates[:5] = False
    top = recommend(model, fit, users, candidates, k=5)
    seen = fit.matrix[users].toarray() > 0
    assert not seen[np.arange(len(users))[:, None], top].any()
    assert candidates[top].all()


def test_paired_bootstrap_detects_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(500)
    res = paired_bootstrap(a + 0.1, a, n_boot=200)
    assert res["lo"] > 0 and res["p_value"] < 0.01


def test_concat_ranges() -> None:
    out = concat_ranges(torch.tensor([5, 0, 10]), torch.tensor([2, 0, 3]))
    assert out.tolist() == [5, 6, 10, 11, 12]


def test_recent_counter_is_strictly_past() -> None:
    items = np.array([0, 0, 0, 1])
    days = np.array([10.0, 20.0, 30.0, 20.0])
    counter = RecentCounter(items, days, torch.device("cpu"))
    count = counter.count(torch.tensor([0, 0, 1]), torch.tensor([30.0, 31.0, 20.0], dtype=torch.float64), 15.0)
    assert count.tolist() == [1, 2, 0]
    assert item_first_day(items, days, 3).tolist() == [10.0, 20.0, np.inf]
    assert log2_bucket(torch.tensor([0.0, 1.0, 3.0, 1e9]), 5).tolist() == [0, 1, 2, 4]


def test_user_field_profile_normalised() -> None:
    codes, weights = user_field_profile(np.array([0, 0, 0, 1]), np.array([2, 2, 1, 0]), 3, 2)
    assert codes[0].tolist() == [2, 1] and weights[0].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert codes[2].tolist() == [-1, -1] and weights[2].sum() == 0


def test_negative_sampler_respects_time_and_history(fit) -> None:
    sampler = NegativeSampler(fit, torch.Generator().manual_seed(0), torch.device("cpu"), time_aware=True)
    neg = sampler.sample(torch.from_numpy(fit.users), torch.from_numpy(fit.days), 3).numpy()
    first = item_first_day(fit.items, fit.days, fit.n_items)
    assert (first[neg] <= fit.days[:, None]).all()
    seen = fit.matrix.toarray() > 0
    assert seen[fit.users[:, None], neg].mean() < 0.01


def test_make_target_filters_cold_seen_and_negative(fit) -> None:
    history = pd.DataFrame(
        {
            "user": [0, 0, 1],
            "item": [0, 1, 1],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": [1, 2, 3],
            "day": [1.0, 2.0, 3.0],
        }
    )
    small = make_fit_data(history, t_rec=5.0, n_users=4, n_items=4, item_features=fit.item_features)
    period = pd.DataFrame(
        {
            "user": [0, 0, 1, 2, 1, 1],
            "item": [1, 2, 0, 0, 0, 1],
            "rating": [5.0, 5.0, 2.0, 5.0, 5.0, 5.0],
            "timestamp": [6] * 6,
            "day": [6.0] * 6,
        }
    )
    target = make_target(small, period, threshold=4.0)
    # (0,1) repeat, (0,2) cold item, (1,0) low rating, (2,0) cold user, (1,0) relevant, (1,1) repeat
    assert target.users.tolist() == [1]
    assert target.relevant[1].toarray().ravel().tolist() == [1, 0, 0, 0]
    assert target.stats["cold_user_interactions"] == 1
    assert target.stats["cold_item_interactions"] == 1
    assert target.stats["repeat_interactions"] == 2
    with pytest.raises(ValueError):
        make_fit_data(history, t_rec=2.0, n_users=4, n_items=4, item_features=fit.item_features)
