"""Dataset preparation and leakage-aware temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Literal

import pandas as pd


REQUIRED_COLUMNS = {"UserId", "ProductId", "Score", "Time"}


@dataclass(frozen=True)
class DataConfig:
    """Rules shared by every model in an experiment."""

    min_user_interactions: int = 5
    min_item_interactions: int = 5
    positive_threshold: int = 4
    drop_cross_asin_duplicate_reviews: bool = True


@dataclass(frozen=True)
class PreparationReport:
    raw_rows: int
    after_required_values: int
    after_cross_asin_dedup: int
    after_pair_aggregation: int
    after_k_core: int
    users: int
    items: int
    positive_fraction: float
    k_core_rounds: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SplitData:
    """A split with an immutable protocol name and train/validation/test frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    protocol: str


def _normalise_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ReviewDatasetBuilder:
    """Prepare Fine Food Reviews without allowing duplicate review copies to leak.

    Amazon Fine Food Reviews includes the same review event under several ASINs.
    Keeping all copies lets a random or temporal holdout see the target review in
    another product's history.  The default policy retains one deterministic copy
    per ``(user, rating, normalised review)`` before collapsing user-item pairs.
    Timestamp is intentionally not part of this key: the legacy dataset contains
    copied ASIN records whose timestamps can differ even though the review body and
    rating are identical.
    """

    def __init__(self, config: DataConfig = DataConfig()) -> None:
        self.config = config
        self.report: PreparationReport | None = None

    def load_csv(self, path: str) -> pd.DataFrame:
        columns = ["UserId", "ProductId", "Score", "Time", "Summary", "Text"]
        preview = pd.read_csv(path, nrows=0)
        available = [column for column in columns if column in preview.columns]
        if not REQUIRED_COLUMNS.issubset(available):
            missing = REQUIRED_COLUMNS.difference(available)
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        return pd.read_csv(path, usecols=available)

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not REQUIRED_COLUMNS.issubset(raw.columns):
            missing = REQUIRED_COLUMNS.difference(raw.columns)
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        raw_rows = len(raw)
        frame = raw.copy()
        for column in ("Summary", "Text"):
            if column not in frame:
                frame[column] = ""
        frame = frame.dropna(subset=["UserId", "ProductId", "Score", "Time"])
        after_required_values = len(frame)

        frame["_row_order"] = range(len(frame))
        frame["_review_key"] = (
            frame["Summary"].map(_normalise_text) + " " + frame["Text"].map(_normalise_text)
        ).str.strip()
        frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
        frame = frame.dropna(subset=["Time"])

        if self.config.drop_cross_asin_duplicate_reviews:
            frame = frame.sort_values(["UserId", "Score", "_review_key", "Time", "ProductId", "_row_order"])
            frame = frame.drop_duplicates(["UserId", "Score", "_review_key"], keep="first")
        after_cross_asin_dedup = len(frame)

        # A user can revise a rating.  Keeping only the latest event prevents the
        # exact user-item pair from being present in both train and evaluation.
        frame = frame.sort_values(["UserId", "ProductId", "Time", "_row_order"])
        frame = frame.drop_duplicates(["UserId", "ProductId"], keep="last")
        after_pair_aggregation = len(frame)

        rounds = 0
        previous_rows = -1
        while previous_rows != len(frame):
            previous_rows = len(frame)
            rounds += 1
            user_counts = frame["UserId"].value_counts()
            frame = frame[frame["UserId"].isin(user_counts[user_counts >= self.config.min_user_interactions].index)]
            item_counts = frame["ProductId"].value_counts()
            frame = frame[frame["ProductId"].isin(item_counts[item_counts >= self.config.min_item_interactions].index)]

        frame = frame.rename(
            columns={"UserId": "user_id", "ProductId": "item_id", "Score": "rating", "Time": "timestamp"}
        )
        frame["rating"] = frame["rating"].astype(float)
        frame["is_positive"] = (frame["rating"] >= self.config.positive_threshold).astype("int8")
        frame = frame.sort_values(["user_id", "timestamp", "_row_order"]).reset_index(drop=True)
        frame = frame[["user_id", "item_id", "rating", "timestamp", "Summary", "Text", "is_positive"]]

        self.report = PreparationReport(
            raw_rows=raw_rows,
            after_required_values=after_required_values,
            after_cross_asin_dedup=after_cross_asin_dedup,
            after_pair_aggregation=after_pair_aggregation,
            after_k_core=len(frame),
            users=frame["user_id"].nunique(),
            items=frame["item_id"].nunique(),
            positive_fraction=float(frame["is_positive"].mean()) if len(frame) else 0.0,
            k_core_rounds=rounds,
        )
        return frame


def per_user_temporal_split(data: pd.DataFrame, min_distinct_timestamps: int = 3) -> SplitData:
    """Hold out each user's final timestamp for test and penultimate for validation.

    Events sharing a timestamp remain together. Users with fewer than three distinct
    timestamps cannot form a chronology without arbitrary tie breaking and are
    excluded from this warm-start protocol.
    """

    required = {"user_id", "timestamp"}
    if not required.issubset(data.columns):
        raise ValueError(f"Expected columns {sorted(required)}")
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in data.groupby("user_id", sort=False):
        ordered = group.sort_values("timestamp")
        times = ordered["timestamp"].drop_duplicates().tolist()
        if len(times) < min_distinct_timestamps:
            continue
        validation_time, test_time = times[-2], times[-1]
        train_parts.append(ordered[ordered["timestamp"] < validation_time])
        validation_parts.append(ordered[ordered["timestamp"] == validation_time])
        test_parts.append(ordered[ordered["timestamp"] == test_time])
    if not train_parts:
        raise ValueError("No users have enough distinct timestamps for temporal splitting")
    return SplitData(
        train=pd.concat(train_parts, ignore_index=True),
        validation=pd.concat(validation_parts, ignore_index=True),
        test=pd.concat(test_parts, ignore_index=True),
        protocol="per_user_last_timestamp",
    )


def global_temporal_split(
    data: pd.DataFrame, train_quantile: float = 0.8, validation_quantile: float = 0.9
) -> SplitData:
    """Strict global-time split; cold users/items remain in validation/test."""

    if not 0 < train_quantile < validation_quantile < 1:
        raise ValueError("Expected 0 < train_quantile < validation_quantile < 1")
    train_cutoff = data["timestamp"].quantile(train_quantile)
    validation_cutoff = data["timestamp"].quantile(validation_quantile)
    train = data[data["timestamp"] <= train_cutoff].copy()
    validation = data[(data["timestamp"] > train_cutoff) & (data["timestamp"] <= validation_cutoff)].copy()
    test = data[data["timestamp"] > validation_cutoff].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Global split produced an empty partition; choose different quantiles")
    return SplitData(train, validation, test, "global_timestamp_quantiles")
