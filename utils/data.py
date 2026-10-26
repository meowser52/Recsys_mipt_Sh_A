"""Download, parse and encode the Amazon Reviews 2023 *Video Games* dataset.

Source: McAuley Lab, "Amazon Reviews 2023" (https://amazon-reviews-2023.github.io/).
We use two official files of the ``Video_Games`` category:

* ``benchmark/5core/rating_only/Video_Games.csv.gz`` -- user, item, rating, timestamp
  (the official 5-core: every user and every item has at least 5 reviews);
* ``raw/meta_categories/meta_Video_Games.jsonl.gz`` -- item metadata
  (title, category path, store, price, details, ...).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AMAZON_2023_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
MS_PER_DAY = 86_400_000
UNKNOWN = "unknown"

__all__ = [
    "AMAZON_2023_URL",
    "DataConfig",
    "EncodedData",
    "download_dataset",
    "encode",
    "file_sha256",
    "load_item_meta",
    "load_ratings",
    "parse_category_path",
]


@dataclass(frozen=True)
class DataConfig:
    """Where the raw files come from and where they are cached locally."""

    data_dir: str = "data/raw"
    category: str = "Video_Games"
    base_url: str = AMAZON_2023_URL

    @property
    def ratings_url(self) -> str:
        return f"{self.base_url}/benchmark/5core/rating_only/{self.category}.csv.gz"

    @property
    def meta_url(self) -> str:
        return f"{self.base_url}/raw/meta_categories/meta_{self.category}.jsonl.gz"

    @property
    def ratings_path(self) -> Path:
        return Path(self.data_dir) / f"{self.category}.csv.gz"

    @property
    def meta_path(self) -> Path:
        return Path(self.data_dir) / f"meta_{self.category}.jsonl.gz"


@dataclass(frozen=True)
class EncodedData:
    """Interactions with integer ids plus item metadata aligned to the item index."""

    interactions: pd.DataFrame  # user, item, rating, timestamp, day
    user_ids: np.ndarray  # raw user id of every user index
    item_ids: np.ndarray  # raw parent_asin of every item index
    items: pd.DataFrame  # row ``i`` describes item index ``i``

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def _download(url: str, path: Path, chunk_size: int = 1 << 20) -> Path:
    """Download ``url`` into ``path`` unless the file is already there."""
    if path.exists():
        logger.info("Found %s (%.1f MB), skipping download", path, path.stat().st_size / 2**20)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, chunk_size)
    tmp.rename(path)
    logger.info("Saved %s (%.1f MB)", path, path.stat().st_size / 2**20)
    return path


def download_dataset(cfg: DataConfig) -> Tuple[Path, Path]:
    """Fetch the ratings and metadata files; returns their local paths."""
    return _download(cfg.ratings_url, cfg.ratings_path), _download(cfg.meta_url, cfg.meta_path)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, used to pin the exact data version in the notebook."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ratings(path: Path) -> pd.DataFrame:
    """Read the 5-core rating file (user_id, parent_asin, rating, timestamp in ms)."""
    dtypes = {"user_id": str, "parent_asin": str, "rating": np.float32, "timestamp": np.int64}
    return pd.read_csv(path, dtype=dtypes)


def _parse_price(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_category_path(path: Sequence[str]) -> Tuple[str, str]:
    """Map an Amazon category path to ``(platform, item type)``.

    ``Video Games > Legacy Systems > PlayStation Systems > PlayStation 2 > Games``
    becomes ``("PlayStation 2", "Games")`` and
    ``Video Games > PC > Accessories > Gaming Mice`` becomes ``("PC", "Gaming Mice")``.
    """
    if not path or path[0] != "Video Games":
        return UNKNOWN, UNKNOWN
    tokens = list(path[1:])
    if tokens and tokens[0] == "Legacy Systems":
        tokens = tokens[1:]
    while tokens and tokens[0].endswith(" Systems"):  # "Nintendo Systems", "Game Boy Systems"
        tokens = tokens[1:]
    if not tokens:
        return UNKNOWN, UNKNOWN
    item_type = tokens[-1] if len(tokens) > 1 else UNKNOWN
    return tokens[0], item_type


def _iter_meta(path: Path, keep: Optional[Set[str]]) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if keep is None or record["parent_asin"] in keep:
                yield record


def load_item_meta(path: Path, keep: Optional[Set[str]] = None) -> pd.DataFrame:
    """Parse the metadata JSONL, keeping the fields used in EDA and in the FM.

    ``average_rating``, ``rating_number`` and the sales rank are aggregates crawled in
    2023, i.e. after our test period; they are kept for EDA only and never used as features.
    """
    rows = []
    for record in _iter_meta(path, keep):
        details = record.get("details") or {}
        categories = record.get("categories") or []
        platform, item_type = parse_category_path(categories)
        description = " ".join(record.get("description") or [])
        rows.append(
            {
                "parent_asin": record["parent_asin"],
                "title": record.get("title") or "",
                "main_category": record.get("main_category") or UNKNOWN,
                "category_path": " > ".join(categories),
                "platform": platform,
                "item_type": item_type,
                "store": record.get("store") or UNKNOWN,
                "price": _parse_price(record.get("price")),
                "esrb": details.get("Rated") or UNKNOWN,
                "date_first_available": details.get("Date First Available"),
                "description_len": len(description),
                "n_feature_bullets": len(record.get("features") or []),
                "average_rating": record.get("average_rating"),
                "rating_number": record.get("rating_number"),
            }
        )
    meta = pd.DataFrame(rows)
    logger.info("Parsed metadata for %d items", len(meta))
    return meta


def encode(ratings: pd.DataFrame, meta: pd.DataFrame) -> EncodedData:
    """Assign contiguous integer ids and align metadata with the item index.

    Ids are assigned in lexicographic order of the raw ids, so the encoding does not
    depend on the row order of the input file.
    """
    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(ratings["parent_asin"].unique())
    interactions = pd.DataFrame(
        {
            "user": np.searchsorted(user_ids, ratings["user_id"].to_numpy()).astype(np.int64),
            "item": np.searchsorted(item_ids, ratings["parent_asin"].to_numpy()).astype(np.int64),
            "rating": ratings["rating"].to_numpy(np.float32),
            "timestamp": ratings["timestamp"].to_numpy(np.int64),
        }
    )
    interactions["day"] = interactions["timestamp"] / MS_PER_DAY
    interactions = interactions.sort_values(["timestamp", "user", "item"], kind="stable")
    interactions = interactions.reset_index(drop=True)

    items = pd.DataFrame({"parent_asin": item_ids})
    items = items.merge(meta, on="parent_asin", how="left", validate="one_to_one")
    for col in ("platform", "item_type", "store", "esrb", "main_category"):
        items[col] = items[col].fillna(UNKNOWN)
    missing = items["title"].isna().sum()
    if missing:
        logger.warning("%d items have no metadata record", missing)
    return EncodedData(interactions=interactions, user_ids=user_ids, item_ids=item_ids, items=items)
