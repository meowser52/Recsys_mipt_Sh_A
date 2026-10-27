"""Hyperparameter search on the validation period (Optuna) and final refits."""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from models import BaseRecommender, build_model
from utils.evaluation import EvalResult, evaluate
from utils.split import EvalTarget, FitData

logger = logging.getLogger(__name__)

SearchSpace = Callable[[optuna.trial.BaseTrial], Dict[str, Any]]
ScoreFn = Callable[[BaseRecommender], float]  # higher is better

__all__ = [
    "EpochMonitor",
    "ScoreFn",
    "SearchSpace",
    "TuningConfig",
    "best_params",
    "fit_evaluate",
    "load_study",
    "release_memory",
    "save_study",
    "topn_score",
    "trials_frame",
    "tune",
]


@dataclass(frozen=True)
class TuningConfig:
    """The main metric is fixed before any search; ``patience`` counts evaluations."""

    metric: str = "ndcg@10"
    k: int = 10
    seed: int = 42
    patience: int = 2
    batch_size: int = 2048
    device: Optional[str] = None  # where top-K ranking runs (runtime setting)


def release_memory() -> None:
    """Collect unreachable models so their GPU blocks return to PyTorch's cache.

    The cache is deliberately not emptied: on a shared GPU memory handed back to the
    driver is taken by other jobs within minutes, and the next model would then OOM.
    """
    gc.collect()


def topn_score(fit: FitData, target: EvalTarget, cfg: TuningConfig) -> ScoreFn:
    """Validation objective: ``cfg.metric`` of the top-K lists on ``target``."""

    def score(model: BaseRecommender) -> float:
        return evaluate(model, fit, target, k=cfg.k, batch_size=cfg.batch_size, device=cfg.device).metrics[cfg.metric]

    return score


class EpochMonitor:
    """Validation callback for iterative models.

    Evaluates every ``eval_every`` epochs, remembers the best epoch, reports to Optuna
    (enabling pruning) and asks the model to stop after ``patience`` evaluations
    without improvement.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        patience: int,
        trial: Optional[optuna.Trial] = None,
        eval_every: int = 1,
    ) -> None:
        self.score_fn, self.patience, self.trial = score_fn, patience, trial
        self.eval_every = eval_every
        self.best_value = -np.inf
        self.best_epoch = 0
        self.history: List[Tuple[int, float]] = []
        self._bad = 0

    def __call__(self, epoch: int, model: BaseRecommender) -> bool:
        if epoch % self.eval_every:
            return False
        value = self.score_fn(model)
        self.history.append((epoch, value))
        logger.debug("epoch %d: %.5f", epoch, value)
        if value > self.best_value:
            self.best_value, self.best_epoch, self._bad = value, epoch, 0
        else:
            self._bad += 1
        if self.trial is not None:
            self.trial.report(value, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return self._bad >= self.patience


def tune(
    name: str,
    space: SearchSpace,
    fit: FitData,
    n_trials: int,
    cfg: TuningConfig,
    score_fn: ScoreFn,
    fixed: Optional[Dict[str, Any]] = None,
    iterative: bool = False,
    eval_every: int = 1,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    enqueue: Optional[Dict[str, Any]] = None,
) -> optuna.Study:
    """Maximise ``score_fn`` over ``n_trials`` configurations of model ``name``.

    ``enqueue`` (search-space values, e.g. the model's default configuration) is
    evaluated as the first trial, so the search starts from a sensible point. For
    iterative models the number of epochs is selected by early stopping and stored as
    the ``best_epoch`` user attribute of the trial.
    """
    sampler = sampler or optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = pruner or optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=name)
    if enqueue is not None:
        study.enqueue_trial(enqueue)

    def run_trial(trial: optuna.Trial, params: Dict[str, Any]) -> float:
        model = build_model(name, **params)
        start = time.perf_counter()
        if not iterative:
            model.fit(fit)
            value = score_fn(model)
            trial.set_user_attr("seconds", time.perf_counter() - start)
            return value
        monitor = EpochMonitor(score_fn, cfg.patience, trial=trial, eval_every=eval_every)
        try:
            model.fit(fit, callback=monitor)
        finally:
            trial.set_user_attr("best_epoch", monitor.best_epoch)
            trial.set_user_attr("history", monitor.history)
            trial.set_user_attr("seconds", time.perf_counter() - start)
        return monitor.best_value

    def objective(trial: optuna.Trial) -> float:
        params = {**(fixed or {}), **space(trial)}
        try:
            value = run_trial(trial, params)
        finally:
            release_memory()  # also after pruned or failed trials: the model is out of scope here
        logger.info("%s #%d: %.5f %s", name, trial.number, value, params)
        return value

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    logger.info("%s best %.5f with %s", name, study.best_value, study.best_params)
    return study


def best_params(
    study: optuna.Study,
    space: SearchSpace,
    fixed: Optional[Dict[str, Any]] = None,
    epochs_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Model parameters of the best trial (conditional spaces are replayed via ``FixedTrial``).

    For iterative models the epoch budget becomes the best validation epoch.
    """
    params = {**(fixed or {}), **space(optuna.trial.FixedTrial(study.best_trial.params))}
    if epochs_key is not None:
        params[epochs_key] = int(study.best_trial.user_attrs["best_epoch"])
    return params


def save_study(study: optuna.Study, params: Dict[str, Any], path: Path) -> None:
    """Persist the selected parameters and the full trial log as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_value": study.best_value,
        "params": params,
        "trials": json.loads(trials_frame(study).to_json(orient="records")),
    }
    path.write_text(json.dumps(payload, indent=1, default=str))


def load_study(path: Path) -> Dict[str, Any]:
    """Read what :func:`save_study` wrote."""
    return json.loads(path.read_text())


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: parameters, value, state, selected epoch and wall time."""
    rows = []
    for trial in study.trials:
        row = {"trial": trial.number, "value": trial.value, "state": trial.state.name}
        row.update(trial.params)
        row["best_epoch"] = trial.user_attrs.get("best_epoch")
        row["seconds"] = trial.user_attrs.get("seconds")
        rows.append(row)
    return pd.DataFrame(rows)


def fit_evaluate(
    name: str, params: Dict[str, Any], fit: FitData, target: EvalTarget, cfg: TuningConfig
) -> Tuple[BaseRecommender, EvalResult, float]:
    """Fit a model with fixed parameters and evaluate it once; returns fit time too."""
    model = build_model(name, **params)
    start = time.perf_counter()
    model.fit(fit)
    seconds = time.perf_counter() - start
    result = evaluate(model, fit, target, k=cfg.k, batch_size=cfg.batch_size, device=cfg.device)
    logger.info("%s: %s (fit %.1fs)", name, {k: round(v, 5) for k, v in result.metrics.items()}, seconds)
    return model, result, seconds
