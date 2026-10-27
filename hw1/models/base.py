"""Common recommender interface and the model registry."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, TypeVar

import numpy as np
import scipy.sparse as sp
import torch

from utils.split import FitData

logger = logging.getLogger(__name__)

# Called after every epoch with (epoch, model); returning True stops training early.
EpochCallback = Callable[[int, "BaseRecommender"], bool]

MODEL_REGISTRY: Dict[str, Type["BaseRecommender"]] = {}

ModelT = TypeVar("ModelT", bound=Type["BaseRecommender"])

__all__ = ["MODEL_REGISTRY", "BaseRecommender", "EpochCallback", "build_model", "dense_rows", "register_model"]


def dense_rows(mat: sp.csr_matrix, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    """Rows of a CSR matrix as a dense float32 tensor, densified on ``device``.

    Only the non-zeros cross the host-device boundary, which matters when the host
    CPU is the bottleneck.
    """
    sub = mat[rows]
    with warnings.catch_warnings():  # "sparse CSR support is in beta"
        warnings.simplefilter("ignore", UserWarning)
        sparse = torch.sparse_csr_tensor(
            torch.from_numpy(sub.indptr.astype(np.int64)),
            torch.from_numpy(sub.indices.astype(np.int64)),
            torch.from_numpy(sub.data.astype(np.float32)),
            size=sub.shape,
            check_invariants=False,
        )
        return sparse.to(device).to_dense()


def register_model(name: str) -> Callable[[ModelT], ModelT]:
    """Class decorator that makes a model constructible by name."""

    def decorator(cls: ModelT) -> ModelT:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        MODEL_REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return decorator


def build_model(name: str, **params: Any) -> "BaseRecommender":
    """Instantiate a registered model from keyword hyperparameters."""
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as err:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}") from err
    return cls(cls.config_cls(**params))


class BaseRecommender(ABC):
    """A model is configured by a frozen dataclass, fitted on ``FitData`` and scores items.

    ``score`` returns ranking scores of all items for the given users at the
    recommendation time ``data.t_rec``. Scores only need to be comparable within a user;
    candidate filtering and exclusion of seen items are done by the shared evaluator.
    """

    config_cls: Type[Any]
    registry_name: str = ""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.n_users = 0
        self.n_items = 0

    @abstractmethod
    def fit(self, data: FitData, callback: Optional[EpochCallback] = None) -> "BaseRecommender":
        """Learn parameters from interactions strictly before ``data.t_rec``."""

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Float32 array ``(len(users), n_items)``; higher means more relevant."""

    def score_tensor(self, users: np.ndarray) -> torch.Tensor:
        """Same scores as a torch tensor on the model's device.

        The evaluator ranks on this tensor, so GPU models never copy the full
        ``users x items`` score matrix to the host; only top-K indices are transferred.
        """
        return torch.from_numpy(np.ascontiguousarray(self.score(users), dtype=np.float32))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.cfg})"
