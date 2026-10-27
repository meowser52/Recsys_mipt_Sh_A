"""Recommender models implemented from scratch; importing registers them by name."""

from models.base import MODEL_REGISTRY, BaseRecommender, EpochCallback, build_model, register_model
from models.baselines import Popular, PopularConfig, RandomConfig, RandomRecommender
from models.ease import EASE, EASEConfig
from models.fm import FM, FMConfig
from models.ials import IALS, IALSConfig
from models.item_knn import ItemKNN, ItemKNNConfig
from models.time_svdpp import TimeSVDpp, TimeSVDppConfig

__all__ = [
    "EASE",
    "FM",
    "IALS",
    "MODEL_REGISTRY",
    "BaseRecommender",
    "EASEConfig",
    "EpochCallback",
    "FMConfig",
    "IALSConfig",
    "ItemKNN",
    "ItemKNNConfig",
    "Popular",
    "PopularConfig",
    "RandomConfig",
    "RandomRecommender",
    "TimeSVDpp",
    "TimeSVDppConfig",
    "build_model",
    "register_model",
]
