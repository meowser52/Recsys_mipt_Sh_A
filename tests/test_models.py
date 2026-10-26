"""Each model is checked against a direct (slow) computation of its definition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models import build_model
from models.ials import conjugate_gradient_solve


def test_ease_matches_constrained_ridge(fit) -> None:
    model = build_model("EASE", l2=5.0, dtype="float64").fit(fit)
    weights = model.weights.cpu().numpy()
    x = fit.matrix[:, model.active].toarray()
    n = x.shape[1]
    for j in range(0, n, 7):
        others = np.delete(np.arange(n), j)
        a = x[:, others]
        b = np.linalg.solve(a.T @ a + 5.0 * np.eye(n - 1), a.T @ x[:, j])
        assert np.allclose(weights[others, j], b, atol=1e-8)
        assert weights[j, j] == 0.0


def test_ials_cg_converges_to_exact_solution(fit) -> None:
    conf = fit.matrix.copy()
    conf.data[:] = 4.0
    gen = torch.Generator().manual_seed(0)
    fixed = torch.randn(fit.n_items, 8, generator=gen, dtype=torch.float64).float()
    init = torch.zeros(fit.n_users, 8)
    approx = conjugate_gradient_solve(conf, fixed, init, l2=0.5, steps=30)
    y = fixed.double().numpy()
    for u in range(0, fit.n_users, 5):
        cols = conf[u].indices
        a = y.T @ y + 0.5 * np.eye(8) + 4.0 * y[cols].T @ y[cols]
        b = 5.0 * y[cols].sum(axis=0)
        assert np.allclose(approx[u].numpy(), np.linalg.solve(a, b), atol=1e-4)


def test_ials_objective_decreases(fit) -> None:
    losses = []

    def record(epoch, model):
        losses.append(model.loss(fit))
        return False

    build_model("iALS", factors=8, alpha=5.0, l2=0.1, n_iters=6, cg_steps=4).fit(fit, callback=record)
    assert all(b <= a + 1e-3 for a, b in zip(losses[:-1], losses[1:], strict=True))


def test_item_knn_matches_bruteforce(fit) -> None:
    model = build_model("ItemKNN", n_neighbors=fit.n_items, alpha=0.3, shrink=2.0, half_life=100.0).fit(fit)
    x = fit.matrix.toarray()
    co = x.T @ x
    np.fill_diagonal(co, 0.0)
    n = x.sum(axis=0)
    sim = co / (n[None, :] ** 0.3 * n[:, None] ** 0.7 + 2.0)  # [history j, candidate i]
    w = np.zeros_like(x)
    w[fit.users, fit.items] = 0.5 ** ((fit.t_rec - fit.days) / 100.0)
    users = np.arange(10)
    assert np.allclose(model.score(users), (w @ sim)[users], atol=1e-5)


def test_item_knn_keeps_top_neighbours(fit) -> None:
    full = build_model("ItemKNN", n_neighbors=fit.n_items).fit(fit).similarity.cpu().numpy()
    kept = build_model("ItemKNN", n_neighbors=3).fit(fit).similarity.cpu().numpy()
    assert ((kept > 0).sum(axis=0) <= 3).all()
    for col in range(full.shape[1]):
        dropped = full[:, col][(kept[:, col] == 0) & (full[:, col] > 0)]
        if dropped.size and (kept[:, col] > 0).any():
            assert kept[:, col][kept[:, col] > 0].min() >= dropped.max() - 1e-7


def _full_fm_score(model, user: int, items: np.ndarray) -> np.ndarray:
    idx, val = model._events(np.full(len(items), user), items, np.full(len(items), model.t_rec))
    with torch.no_grad():
        y, _ = model.net(idx, val)
    return y.cpu().numpy()


@pytest.mark.parametrize("flags", [(True, True, True), (False, False, False), (True, False, True)])
def test_fm_fast_score_equals_full_model(fit, flags) -> None:
    meta, profile, context = flags
    model = build_model(
        "FM",
        factors=8,
        n_epochs=2,
        batch_size=128,
        use_item_meta=meta,
        use_user_profile=profile,
        use_item_age=context,
        use_recent_pop=context,
        lr=0.01,
    ).fit(fit)
    items = np.arange(fit.n_items)
    for user in (0, 7, 33):
        fast = model.score(np.array([user]))[0]
        full = _full_fm_score(model, user, items)
        assert np.allclose(fast - fast[0], full - full[0], atol=1e-4)


def test_fm_is_reproducible(fit) -> None:
    a = build_model("FM", factors=8, n_epochs=2, batch_size=128).fit(fit).score(np.arange(5))
    b = build_model("FM", factors=8, n_epochs=2, batch_size=128).fit(fit).score(np.arange(5))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("use_time", [True, False])
def test_time_svdpp_score_equals_predict(fit, use_time) -> None:
    model = build_model("timeSVD++", factors=8, n_epochs=2, batch_size=64, use_time=use_time).fit(fit)
    users = np.array([0, 5, 9])
    scores = model.score(users)
    for row, user in enumerate(users):
        items = np.arange(fit.n_items)
        pred = model.predict(np.full(fit.n_items, user), items, np.full(fit.n_items, fit.t_rec))
        assert np.allclose(scores[row], pred, atol=1e-4)


def test_time_svdpp_fits_ratings(fit) -> None:
    model = build_model(
        "timeSVD++",
        factors=8,
        n_epochs=40,
        batch_size=64,
        n_negatives=0,
        lr=0.02,
        l2_bias=1e-4,
        l2_factor=1e-4,
        l2_time=1e-4,
    ).fit(fit)
    pred = model.predict(fit.users, fit.items, fit.days)
    rmse = np.sqrt(np.mean((pred - fit.ratings) ** 2))
    baseline = np.sqrt(np.mean((fit.ratings.mean() - fit.ratings) ** 2))
    assert rmse < 0.7 * baseline


def test_time_svdpp_day_terms_used_in_training(fit) -> None:
    model = build_model("timeSVD++", factors=4, n_epochs=3, batch_size=64, n_negatives=0, lr=0.05).fit(fit)
    assert model.net.b_ut.weight[:-1].abs().sum() > 0
    assert model.net.b_ut.weight[-1].abs().sum() == 0  # padding row = unseen day


CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize(
    "name,params",
    [
        ("EASE", {"l2": 5.0}),
        ("ItemKNN", {"n_neighbors": 1000, "half_life": 50.0}),  # no pruning: top-k ties may differ by device
        ("iALS", {"factors": 8, "n_iters": 3}),
        ("FM", {"factors": 8, "n_epochs": 2, "batch_size": 128}),
        ("timeSVD++", {"factors": 8, "n_epochs": 2, "batch_size": 64}),
    ],
)
def test_cuda_matches_cpu_and_is_deterministic(fit, name, params) -> None:
    users = np.arange(10)
    cpu = build_model(name, **params).fit(fit).score(users)
    gpu_a = build_model(name, device="cuda", **params).fit(fit).score(users)
    gpu_b = build_model(name, device="cuda", **params).fit(fit).score(users)
    assert np.array_equal(gpu_a, gpu_b)
    assert np.allclose(cpu, gpu_a, atol=1e-3, rtol=1e-3)
