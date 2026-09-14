"""Shared evaluation code for explicit-rating and Top-N recommendation tasks."""

from __future__ import annotations

from collections import defaultdict
from hashlib import blake2b
from math import log2
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


class RankingModel(Protocol):
    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        """Return one score for every candidate item, in the same order."""


class RatingModel(Protocol):
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Return an estimated 1--5 rating."""


def _tie_breakers(seed: int, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
    """A deterministic pseudo-random order independent of positive-item placement."""

    values = []
    for item_id in item_ids:
        token = f"{seed}|{user_id}|{item_id}".encode("utf-8")
        values.append(int.from_bytes(blake2b(token, digest_size=8).digest(), "big"))
    return np.asarray(values, dtype=np.uint64)


def _rank_items(scores: np.ndarray, user_id: str, item_ids: Sequence[str], seed: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(item_ids),):
        raise ValueError("score_items must return exactly one score per candidate")
    scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.finfo(float).max)
    # lexsort's final key is primary. Equal scores are never resolved by candidate
    # insertion order, which fixes the positive-first tie artifact in the legacy code.
    return np.lexsort((_tie_breakers(seed, user_id, item_ids), -scores))


def evaluate_ratings(model: RatingModel, test: pd.DataFrame) -> dict[str, float]:
    """Evaluate all 1--5 ratings and guard against high-score-dominated summaries."""

    required = {"user_id", "item_id", "rating"}
    if not required.issubset(test.columns):
        raise ValueError(f"Expected columns {sorted(required)}")
    actual = test["rating"].to_numpy(dtype=float)
    predicted = np.asarray(
        [model.predict_rating(user_id, item_id) for user_id, item_id in test[["user_id", "item_id"]].itertuples(index=False)],
        dtype=float,
    )
    errors = np.abs(predicted - actual)
    result: dict[str, float] = {
        "n": float(len(test)),
        "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))) if len(test) else 0.0,
        "mae": float(np.mean(errors)) if len(test) else 0.0,
    }
    per_rating_mae = []
    for rating in range(1, 6):
        mask = actual == rating
        value = float(np.mean(errors[mask])) if mask.any() else float("nan")
        result[f"mae_rating_{rating}"] = value
        if not np.isnan(value):
            per_rating_mae.append(value)
    result["macro_mae"] = float(np.mean(per_rating_mae)) if per_rating_mae else 0.0
    return result


def evaluate_ranking(
    model: RankingModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    k_values: Sequence[int] = (5, 10, 20),
    positive_threshold: float = 4.0,
    seed: int = 42,
    max_users: int | None = None,
) -> dict[str, float]:
    """Full-catalog Top-N evaluation with correct precision/recall semantics.

    Only ratings at or above ``positive_threshold`` are relevant for the Top-N task.
    This is intentionally separate from :func:`evaluate_ratings`, where every score
    remains a prediction target.  The candidate catalog is exactly the train catalog,
    seen train items are masked, and cold targets are disclosed rather than skipped.
    """

    required = {"user_id", "item_id", "rating"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError(f"Expected columns {sorted(required)}")
    k_values = tuple(sorted(set(int(k) for k in k_values if k > 0)))
    if not k_values:
        raise ValueError("k_values must contain at least one positive K")

    catalog = sorted(train["item_id"].astype(str).unique().tolist())
    catalog_set = set(catalog)
    train_seen = train.groupby("user_id")["item_id"].agg(lambda values: set(values.astype(str))).to_dict()
    train_users = set(train["user_id"].astype(str))
    positives = test[test["rating"] >= positive_threshold].copy()
    positive_by_user = positives.groupby("user_id")["item_id"].agg(lambda values: set(values.astype(str))).to_dict()
    users = sorted(positive_by_user)
    if max_users is not None:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(users, size=min(max_users, len(users)), replace=False).tolist())

    aggregates: dict[int, dict[str, list[float]]] = {
        k: defaultdict(list) for k in k_values
    }
    recommended: dict[int, set[str]] = {k: set() for k in k_values}
    item_popularity = train[train["rating"] >= positive_threshold]["item_id"].value_counts().to_dict()
    eligible_users = 0
    cold_user_targets = 0
    cold_item_targets = 0
    no_candidate_users = 0

    for user_id in users:
        target_items = positive_by_user[user_id]
        if str(user_id) not in train_users:
            cold_user_targets += len(target_items)
            continue
        warm_targets = target_items.intersection(catalog_set)
        cold_item_targets += len(target_items.difference(catalog_set))
        seen = train_seen.get(user_id, set())
        candidates = [item_id for item_id in catalog if item_id not in seen]
        targets = warm_targets.difference(seen)
        if not candidates or not targets:
            no_candidate_users += 1
            continue
        scores = model.score_items(user_id, candidates)
        order = _rank_items(scores, user_id, candidates, seed)
        ranked = [candidates[index] for index in order]
        eligible_users += 1

        for k in k_values:
            top_k = ranked[:k]
            hits = len(set(top_k).intersection(targets))
            aggregates[k]["precision"].append(hits / k)
            aggregates[k]["recall"].append(hits / len(targets))
            aggregates[k]["hit_rate"].append(float(hits > 0))
            dcg = sum(1.0 / log2(rank + 2) for rank, item_id in enumerate(top_k) if item_id in targets)
            ideal = sum(1.0 / log2(rank + 2) for rank in range(min(k, len(targets))))
            aggregates[k]["ndcg"].append(dcg / ideal if ideal else 0.0)
            first_rank = next((rank + 1 for rank, item_id in enumerate(ranked) if item_id in targets), None)
            aggregates[k]["mrr"].append(1.0 / first_rank if first_rank and first_rank <= k else 0.0)
            recommended[k].update(top_k)

    result: dict[str, float] = {
        "positive_test_events": float(len(positives)),
        "positive_test_users": float(len(users)),
        "eligible_users": float(eligible_users),
        "cold_user_targets": float(cold_user_targets),
        "cold_item_targets": float(cold_item_targets),
        "no_candidate_users": float(no_candidate_users),
        "catalog_items": float(len(catalog)),
    }
    for k in k_values:
        for metric in ("precision", "recall", "hit_rate", "ndcg", "mrr"):
            values = aggregates[k][metric]
            result[f"{metric}@{k}"] = float(np.mean(values)) if values else 0.0
        result[f"coverage@{k}"] = len(recommended[k]) / len(catalog) if catalog else 0.0
        if recommended[k]:
            result[f"average_positive_popularity@{k}"] = float(
                np.mean([item_popularity.get(item_id, 0) for item_id in recommended[k]])
            )
        else:
            result[f"average_positive_popularity@{k}"] = 0.0
    return result

