"""Validation-tuned score fusion without notebook-global state."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from foodrec.evaluation import evaluate_ranking


class WeightedHybridRanker:
    """Fuse independently fitted retrievers through per-candidate rank percentiles."""

    def __init__(self, models: Mapping[str, object], weights: Mapping[str, float]) -> None:
        if set(weights) != set(models):
            raise ValueError("Hybrid weights must name exactly the supplied models")
        if any(weight < 0 for weight in weights.values()) or not any(weights.values()):
            raise ValueError("Hybrid weights must be non-negative and contain a positive value")
        self.models = dict(models)
        total = float(sum(weights.values()))
        self.weights = {name: float(weight) / total for name, weight in weights.items()}

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if not item_ids:
            return np.asarray([], dtype=float)
        fused = np.zeros(len(item_ids), dtype=float)
        for name, model in self.models.items():
            raw = np.asarray(model.score_items(user_id, item_ids), dtype=float)
            # Rank normalisation avoids incomparable raw scales from TF-IDF, MF and CF.
            ranks = rankdata(raw, method="average")
            percentile = (ranks - 1) / max(1, len(raw) - 1)
            fused += self.weights[name] * percentile
        return fused


def select_hybrid_weights(
    models: Mapping[str, object],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    k: int = 10,
    positive_threshold: float = 4.0,
    seed: int = 42,
    max_users: int = 200,
) -> tuple[WeightedHybridRanker, dict[str, float], dict[str, float]]:
    """Choose a bounded set of fusion weights only on validation, never on test.

    Exhaustively searching a five-point grid across six retrievers would require
    15,625 full-catalog passes. We instead compare transparent candidates: equal
    weights, each single retriever, and a small set of domain-motivated pairs. This
    keeps model selection tractable and reproducible while avoiding any access to
    test.
    """

    names = list(models)
    best: tuple[WeightedHybridRanker, dict[str, float], dict[str, float]] | None = None
    candidates: list[dict[str, float]] = [{name: 1.0 for name in names}]
    candidates.extend({key: float(key == name) for key in names} for name in names)
    for first, second in (("item_knn", "biased_mf"), ("item_knn", "content_tfidf"), ("two_tower", "item_knn")):
        if first in models and second in models:
            candidates.append({key: 0.5 if key in (first, second) else 0.0 for key in names})
    for weights in candidates:
        hybrid = WeightedHybridRanker(models, weights)
        metrics = evaluate_ranking(
            hybrid,
            train,
            validation,
            k_values=(k,),
            positive_threshold=positive_threshold,
            seed=seed,
            max_users=max_users,
        )
        if best is None or metrics[f"ndcg@{k}"] > best[2][f"ndcg@{k}"]:
            best = (hybrid, hybrid.weights, metrics)
    assert best is not None
    return best
