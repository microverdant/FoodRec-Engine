"""A compact two-tower retrieval model with train-only text features."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover - project dependency normally supplies torch.
    raise ImportError("TwoTowerRetriever requires the optional torch dependency") from error


class _TwoTower(nn.Module):
    def __init__(self, users: int, items: int, embedding_dim: int, item_text: torch.Tensor) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(users, embedding_dim)
        self.item_embedding = nn.Embedding(items, embedding_dim)
        self.register_buffer("item_text", item_text)
        self.text_projection = nn.Linear(item_text.shape[1], embedding_dim, bias=False)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_embedding(user_indices), dim=-1)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        vectors = self.item_embedding(item_indices) + self.text_projection(self.item_text[item_indices])
        return F.normalize(vectors, dim=-1)


class TwoTowerRetriever:
    """In-batch contrastive retrieval, avoiding duplicate-item false negatives.

    Every text vector is learned from training reviews only.  In a batch, duplicate
    copies of the same positive item are treated as *multiple positives* in the
    contrastive denominator rather than as negatives of one another.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        text_dim: int = 64,
        epochs: int = 10,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        temperature: float = 0.1,
        positive_threshold: float = 4.0,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.text_dim = text_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.positive_threshold = positive_threshold
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.model: _TwoTower | None = None
        self._user_vectors: np.ndarray | None = None
        self._item_vectors: np.ndarray | None = None

    def _item_text_features(self, train: pd.DataFrame, items: list[str]) -> np.ndarray:
        summary = train["Summary"].fillna("").astype(str) if "Summary" in train else pd.Series("", index=train.index)
        text = train["Text"].fillna("").astype(str) if "Text" in train else pd.Series("", index=train.index)
        docs = (
            pd.DataFrame({"item_id": train["item_id"].astype(str), "doc": summary + " " + text})
            .groupby("item_id")["doc"]
            .agg(" ".join)
            .reindex(items, fill_value="")
        )
        try:
            matrix = TfidfVectorizer(stop_words="english", max_features=20_000, ngram_range=(1, 2)).fit_transform(docs)
            components = min(self.text_dim, max(1, min(matrix.shape) - 1))
            if matrix.shape[1] <= 1 or matrix.shape[0] <= 1:
                return np.zeros((len(items), self.text_dim), dtype=np.float32)
            reduced = TruncatedSVD(n_components=components, random_state=self.seed).fit_transform(matrix)
        except ValueError:
            return np.zeros((len(items), self.text_dim), dtype=np.float32)
        result = np.zeros((len(items), self.text_dim), dtype=np.float32)
        result[:, : reduced.shape[1]] = reduced.astype(np.float32)
        return result

    def fit(self, train: pd.DataFrame) -> "TwoTowerRetriever":
        positives = train[train["rating"] >= self.positive_threshold][["user_id", "item_id"]].drop_duplicates()
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["item_id"].astype(str).unique())
        if positives.empty:
            raise ValueError("TwoTowerRetriever needs at least one positive training interaction")
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        pairs = np.column_stack(
            [
                positives["user_id"].astype(str).map(self.user_to_index).to_numpy(dtype=np.int64),
                positives["item_id"].astype(str).map(self.item_to_index).to_numpy(dtype=np.int64),
            ]
        )
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        item_text = torch.tensor(self._item_text_features(train, items), dtype=torch.float32)
        self.model = _TwoTower(len(users), len(items), self.embedding_dim, item_text).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.model.train()
        for _ in range(self.epochs):
            for batch_indices in np.array_split(rng.permutation(len(pairs)), max(1, int(np.ceil(len(pairs) / self.batch_size)))):
                batch = pairs[batch_indices]
                user_indices = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
                item_indices = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)
                user_vectors = self.model.encode_users(user_indices)
                item_vectors = self.model.encode_items(item_indices)
                logits = user_vectors @ item_vectors.T / self.temperature
                same_item = item_indices[:, None].eq(item_indices[None, :])
                numerator = torch.logsumexp(logits.masked_fill(~same_item, float("-inf")), dim=1)
                denominator = torch.logsumexp(logits, dim=1)
                loss = -(numerator - denominator).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()
        with torch.no_grad():
            user_indices = torch.arange(len(users), device=self.device)
            item_indices = torch.arange(len(items), device=self.device)
            self._user_vectors = self.model.encode_users(user_indices).cpu().numpy()
            self._item_vectors = self.model.encode_items(item_indices).cpu().numpy()
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if self._user_vectors is None or self._item_vectors is None:
            raise RuntimeError("Call fit before scoring")
        user_index = self.user_to_index.get(str(user_id))
        if user_index is None:
            return np.zeros(len(item_ids), dtype=float)
        scores = np.zeros(len(item_ids), dtype=float)
        valid_positions = [position for position, item_id in enumerate(item_ids) if item_id in self.item_to_index]
        if valid_positions:
            item_indices = [self.item_to_index[item_ids[position]] for position in valid_positions]
            scores[valid_positions] = self._item_vectors[item_indices] @ self._user_vectors[user_index]
        return scores

