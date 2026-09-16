"""Reproducible classical baselines used by the benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from foodrec.evaluation import evaluate_ranking


def _positive_events(train: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return train[train["rating"] >= threshold]


class PopularityRanker:
    """Training-only popularity baseline with explicit cold-user support."""

    def __init__(self, positive_threshold: float = 4.0) -> None:
        self.positive_threshold = positive_threshold
        self.scores: dict[str, float] = {}
        self.global_rating = 3.0

    def fit(self, train: pd.DataFrame) -> "PopularityRanker":
        positives = _positive_events(train, self.positive_threshold)
        self.scores = np.log1p(positives["item_id"].value_counts()).to_dict()
        self.global_rating = float(train["rating"].mean())
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        return np.asarray([self.scores.get(item_id, 0.0) for item_id in item_ids], dtype=float)

    def predict_rating(self, user_id: str, item_id: str) -> float:
        return self.global_rating


class BiasedMF:
    """Biased matrix factorisation for rating prediction and ranking.

    It has a validation hook but deliberately never accepts a test frame.  Callers
    may use a validation split for early stopping and reserve test for one final run.
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 40,
        learning_rate: float = 0.01,
        regularization: float = 0.04,
        seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.global_mean = 3.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.best_validation_rmse: float | None = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> "BiasedMF":
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["item_id"].astype(str).unique())
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        user_idx = train["user_id"].map(self.user_to_index).to_numpy(dtype=int)
        item_idx = train["item_id"].map(self.item_to_index).to_numpy(dtype=int)
        ratings = train["rating"].to_numpy(dtype=float)
        self.global_mean = float(ratings.mean())
        rng = np.random.default_rng(self.seed)
        self.user_bias = np.zeros(len(users), dtype=float)
        self.item_bias = np.zeros(len(items), dtype=float)
        self.user_factors = rng.normal(0, 0.05, size=(len(users), self.n_factors))
        self.item_factors = rng.normal(0, 0.05, size=(len(items), self.n_factors))
        best_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        best_rmse = float("inf")

        for _ in range(self.n_epochs):
            for index in rng.permutation(len(ratings)):
                user = user_idx[index]
                item = item_idx[index]
                prediction = self._predict_indices(user, item)
                error = ratings[index] - prediction
                old_user = self.user_factors[user].copy()
                old_item = self.item_factors[item].copy()
                self.user_bias[user] += self.learning_rate * (error - self.regularization * self.user_bias[user])
                self.item_bias[item] += self.learning_rate * (error - self.regularization * self.item_bias[item])
                self.user_factors[user] += self.learning_rate * (error * old_item - self.regularization * old_user)
                self.item_factors[item] += self.learning_rate * (error * old_user - self.regularization * old_item)
            if validation is not None and len(validation):
                actual = validation["rating"].to_numpy(dtype=float)
                predicted = np.asarray(
                    [self.predict_rating(user, item) for user, item in validation[["user_id", "item_id"]].itertuples(index=False)],
                    dtype=float,
                )
                rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_state = (
                        self.user_bias.copy(), self.item_bias.copy(), self.user_factors.copy(), self.item_factors.copy()
                    )
        if best_state is not None:
            self.user_bias, self.item_bias, self.user_factors, self.item_factors = best_state
            self.best_validation_rmse = best_rmse
        return self

    def _predict_indices(self, user_index: int, item_index: int) -> float:
        assert self.user_bias is not None and self.item_bias is not None
        assert self.user_factors is not None and self.item_factors is not None
        return float(
            self.global_mean
            + self.user_bias[user_index]
            + self.item_bias[item_index]
            + np.dot(self.user_factors[user_index], self.item_factors[item_index])
        )

    def predict_rating(self, user_id: str, item_id: str) -> float:
        user_index = self.user_to_index.get(str(user_id))
        item_index = self.item_to_index.get(str(item_id))
        if user_index is None and item_index is None:
            return self.global_mean
        if user_index is None:
            assert self.item_bias is not None
            return float(self.global_mean + self.item_bias[item_index])
        if item_index is None:
            assert self.user_bias is not None
            return float(self.global_mean + self.user_bias[user_index])
        return float(np.clip(self._predict_indices(user_index, item_index), 1.0, 5.0))

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        user_index = self.user_to_index.get(str(user_id))
        item_indices = np.asarray([self.item_to_index.get(item_id, -1) for item_id in item_ids], dtype=int)
        scores = np.full(len(item_ids), self.global_mean, dtype=float)
        if self.item_bias is not None:
            known_items = item_indices >= 0
            scores[known_items] += self.item_bias[item_indices[known_items]]
        if user_index is not None:
            assert self.user_bias is not None and self.user_factors is not None and self.item_factors is not None
            scores += self.user_bias[user_index]
            known_items = item_indices >= 0
            scores[known_items] += self.item_factors[item_indices[known_items]] @ self.user_factors[user_index]
        return np.clip(scores, 1.0, 5.0)


class EASERanker:
    """Embarrassingly Shallow Autoencoder (EASE) for implicit Top-N retrieval.

    It learns a regularised item-item linear reconstruction on positive training
    events only. The implementation keeps the learned coefficient matrix dense
    because this dataset's item catalogue is small; it must not be used unchanged
    for production-scale catalogues.
    """

    def __init__(self, regularization: float = 300.0, positive_threshold: float = 4.0) -> None:
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.regularization = regularization
        self.positive_threshold = positive_threshold
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.interactions: csr_matrix | None = None
        self.coefficients: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "EASERanker":
        events = _positive_events(train, self.positive_threshold)
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["item_id"].astype(str).unique())
        if events.empty or not items:
            raise ValueError("EASERanker needs at least one positive training interaction")
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        self.interactions = csr_matrix(
            (
                np.ones(len(events), dtype=np.float64),
                (events["user_id"].map(self.user_to_index), events["item_id"].map(self.item_to_index)),
            ),
            shape=(len(users), len(items)),
        )
        gram = (self.interactions.T @ self.interactions).toarray()
        gram.flat[:: len(items) + 1] += self.regularization
        precision = np.linalg.inv(gram)
        coefficients = -precision / np.diag(precision)
        np.fill_diagonal(coefficients, 0.0)
        self.coefficients = coefficients
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if self.interactions is None or self.coefficients is None:
            raise RuntimeError("Call fit before scoring")
        user_index = self.user_to_index.get(str(user_id))
        if user_index is None:
            return np.zeros(len(item_ids), dtype=float)
        all_scores = np.asarray(self.interactions.getrow(user_index) @ self.coefficients).ravel()
        return np.asarray([all_scores[self.item_to_index[item]] if item in self.item_to_index else 0.0 for item in item_ids])


def select_ease_regularization(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    candidates: Sequence[float] = (10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0),
    k: int = 10,
    positive_threshold: float = 4.0,
    seed: int = 42,
    max_users: int = 200,
) -> tuple[EASERanker, float, dict[str, float]]:
    """Select EASE regularisation using validation only, never the test split."""
    best: tuple[EASERanker, float, dict[str, float]] | None = None
    for regularization in candidates:
        model = EASERanker(regularization=regularization, positive_threshold=positive_threshold).fit(train)
        metrics = evaluate_ranking(
            model, train, validation, k_values=(k,), positive_threshold=positive_threshold, seed=seed, max_users=max_users
        )
        if best is None or metrics[f"ndcg@{k}"] > best[2][f"ndcg@{k}"]:
            best = (model, float(regularization), metrics)
    assert best is not None
    return best


class ItemKNNRanker:
    """Sparse item-item CF storing only top-K neighbours, never a dense similarity matrix."""

    def __init__(self, n_neighbors: int = 100, shrinkage: float = 0.0, positive_threshold: float = 4.0) -> None:
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.positive_threshold = positive_threshold
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.interactions: csr_matrix | None = None
        self.item_similarity: csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> "ItemKNNRanker":
        events = _positive_events(train, self.positive_threshold)
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["item_id"].astype(str).unique())
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        rows = events["user_id"].map(self.user_to_index).to_numpy(dtype=int)
        cols = events["item_id"].map(self.item_to_index).to_numpy(dtype=int)
        values = np.ones(len(events), dtype=np.float32)
        self.interactions = csr_matrix((values, (rows, cols)), shape=(len(users), len(items)), dtype=np.float32)
        if not len(items) or not len(events):
            self.item_similarity = csr_matrix((len(items), len(items)), dtype=np.float32)
            return self
        item_user = normalize(self.interactions.T, norm="l2", axis=1)
        co_counts = (self.interactions.T @ self.interactions).tocsr()
        neighbor_count = min(self.n_neighbors + 1, len(items))
        nearest = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=neighbor_count, n_jobs=-1)
        nearest.fit(item_user)
        distances, indices = nearest.kneighbors(item_user, return_distance=True)
        row_indices: list[int] = []
        col_indices: list[int] = []
        similarities: list[float] = []
        for item_index, (item_neighbors, item_distances) in enumerate(zip(indices, distances)):
            for neighbor, distance in zip(item_neighbors, item_distances):
                if neighbor == item_index:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if self.shrinkage:
                    similarity *= float(co_counts[item_index, int(neighbor)]) / (
                        float(co_counts[item_index, int(neighbor)]) + self.shrinkage
                    )
                if similarity > 0:
                    row_indices.append(item_index)
                    col_indices.append(int(neighbor))
                    similarities.append(similarity)
        self.item_similarity = csr_matrix(
            (similarities, (row_indices, col_indices)), shape=(len(items), len(items)), dtype=np.float32
        )
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if self.interactions is None or self.item_similarity is None:
            raise RuntimeError("Call fit before scoring")
        user_index = self.user_to_index.get(str(user_id))
        if user_index is None:
            return np.zeros(len(item_ids), dtype=float)
        all_scores = (self.interactions.getrow(user_index) @ self.item_similarity).toarray().ravel()
        return np.asarray([all_scores[self.item_to_index[item]] if item in self.item_to_index else 0.0 for item in item_ids])


def select_item_knn_parameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    candidates: Sequence[tuple[int, float]] = ((20, 0.0), (50, 0.0), (100, 0.0), (200, 0.0), (50, 10.0), (100, 10.0), (100, 50.0)),
    k: int = 10,
    positive_threshold: float = 4.0,
    seed: int = 42,
    max_users: int = 200,
) -> tuple[ItemKNNRanker, dict[str, float], dict[str, float]]:
    """Select ItemKNN neighbourhood/shrinkage on validation only."""
    best: tuple[ItemKNNRanker, dict[str, float], dict[str, float]] | None = None
    for neighbors, shrinkage in candidates:
        model = ItemKNNRanker(neighbors, shrinkage, positive_threshold).fit(train)
        metrics = evaluate_ranking(
            model, train, validation, k_values=(k,), positive_threshold=positive_threshold, seed=seed, max_users=max_users
        )
        parameters = {"n_neighbors": float(neighbors), "shrinkage": float(shrinkage)}
        if best is None or metrics[f"ndcg@{k}"] > best[2][f"ndcg@{k}"]:
            best = (model, parameters, metrics)
    assert best is not None
    return best


class ContentRanker:
    """Training-only TF-IDF item profiles; test review text is never fitted or queried."""

    def __init__(self, positive_threshold: float = 4.0, max_features: int = 20_000) -> None:
        self.positive_threshold = positive_threshold
        self.max_features = max_features
        self.item_to_index: dict[str, int] = {}
        self.user_profiles: dict[str, np.ndarray] = {}
        self.item_matrix = None

    def fit(self, train: pd.DataFrame) -> "ContentRanker":
        items = sorted(train["item_id"].astype(str).unique())
        self.item_to_index = {value: index for index, value in enumerate(items)}
        docs = (
            train.assign(_doc=train.get("Summary", "").fillna("").astype(str) + " " + train.get("Text", "").fillna("").astype(str))
            .groupby("item_id")["_doc"]
            .agg(" ".join)
            .reindex(items, fill_value="")
        )
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features, ngram_range=(1, 2))
            self.item_matrix = vectorizer.fit_transform(docs)
        except ValueError:  # Empty vocabulary in a tiny test fixture.
            self.item_matrix = csr_matrix((len(items), 1), dtype=np.float32)
        positive = _positive_events(train, self.positive_threshold)
        histories = positive.groupby("user_id")["item_id"].agg(list)
        self.user_profiles = {}
        for user_id, item_ids in histories.items():
            indices = [self.item_to_index[item_id] for item_id in item_ids if item_id in self.item_to_index]
            if indices:
                profile = self.item_matrix[indices].mean(axis=0)
                self.user_profiles[str(user_id)] = normalize(csr_matrix(profile), norm="l2", axis=1).toarray().ravel()
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if self.item_matrix is None:
            raise RuntimeError("Call fit before scoring")
        profile = self.user_profiles.get(str(user_id))
        if profile is None:
            return np.zeros(len(item_ids), dtype=float)
        indices = [self.item_to_index.get(item_id, -1) for item_id in item_ids]
        scores = np.zeros(len(item_ids), dtype=float)
        valid = [position for position, index in enumerate(indices) if index >= 0]
        if valid:
            matrix = self.item_matrix[[indices[position] for position in valid]]
            scores[valid] = np.asarray(matrix @ profile).ravel()
        return scores


class NMFImplicitRanker:
    """NMF implicit-feedback baseline; it is not presented as an explicit-rating model."""

    def __init__(self, n_components: int = 32, positive_threshold: float = 4.0, seed: int = 42) -> None:
        self.n_components = n_components
        self.positive_threshold = positive_threshold
        self.seed = seed
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "NMFImplicitRanker":
        events = _positive_events(train, self.positive_threshold)
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["item_id"].astype(str).unique())
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        matrix = csr_matrix(
            (
                np.ones(len(events), dtype=float),
                (events["user_id"].map(self.user_to_index), events["item_id"].map(self.item_to_index)),
            ),
            shape=(len(users), len(items)),
        )
        components = max(1, min(self.n_components, min(matrix.shape)))
        model = NMF(n_components=components, init="nndsvda", max_iter=300, random_state=self.seed)
        self.user_factors = model.fit_transform(matrix)
        self.item_factors = model.components_
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Call fit before scoring")
        user_index = self.user_to_index.get(str(user_id))
        if user_index is None:
            return np.zeros(len(item_ids), dtype=float)
        item_indices = np.asarray([self.item_to_index.get(item, -1) for item in item_ids], dtype=int)
        scores = np.zeros(len(item_ids), dtype=float)
        valid = item_indices >= 0
        scores[valid] = self.user_factors[user_index] @ self.item_factors[:, item_indices[valid]]
        return scores
