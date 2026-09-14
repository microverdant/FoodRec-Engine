"""A causal-history two-tower retriever with train-only text features."""

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
        # Index ``items`` is a zero-vector padding token for an empty/short history.
        self.item_embedding = nn.Embedding(items + 1, embedding_dim, padding_idx=items)
        self.register_buffer("item_text", item_text)
        self.text_projection = nn.Linear(item_text.shape[1], embedding_dim, bias=False)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight[:-1], std=0.02)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        vectors = self.item_embedding(item_indices) + self.text_projection(self.item_text[item_indices])
        return F.normalize(vectors, dim=-1)

    def encode_users(self, user_indices: torch.Tensor, history_indices: torch.Tensor) -> torch.Tensor:
        history_mask = history_indices.ne(self.item_embedding.padding_idx)
        safe_history = history_indices.clamp(max=self.item_embedding.padding_idx - 1)
        history_vectors = self.encode_items(safe_history)
        history_vectors = history_vectors * history_mask.unsqueeze(-1)
        denominator = history_mask.sum(dim=1, keepdim=True).clamp(min=1)
        history_mean = history_vectors.sum(dim=1) / denominator
        return F.normalize(self.user_embedding(user_indices) + history_mean, dim=-1)


class TwoTowerRetriever:
    """Causal history-aware retrieval with a multi-positive contrastive objective.

    Each training example predicts a positive item using only positive interactions
    with a *strictly earlier* timestamp. Events sharing a timestamp do not enter
    one another's histories. This preserves the merged notebook's two-tower idea
    while removing its random-split/history leakage.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        text_dim: int = 64,
        history_length: int = 20,
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
        self.history_length = history_length
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

    def _causal_examples(self, train: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[int]]]:
        """Return (user, item, history) positives and final training histories."""
        required = {"user_id", "item_id", "rating", "timestamp"}
        missing = required.difference(train.columns)
        if missing:
            raise ValueError(f"TwoTowerRetriever needs columns: {sorted(missing)}")
        positives = train[train["rating"] >= self.positive_threshold].copy()
        positives["user_id"] = positives["user_id"].astype(str)
        positives["item_id"] = positives["item_id"].astype(str)
        positives = positives.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
        rows: list[list[int]] = []
        final_histories: dict[str, list[int]] = {}
        for user_id, events in positives.groupby("user_id", sort=False):
            history: list[int] = []
            # A timestamp group is scored against the history preceding the group,
            # then the whole group becomes available to later timestamps.
            for _, group in events.groupby("timestamp", sort=False):
                encoded_history = ([len(self.item_to_index)] * self.history_length + history[-self.history_length :])[-self.history_length :]
                user_index = self.user_to_index[user_id]
                for item_id in group["item_id"].drop_duplicates():
                    rows.append([user_index, self.item_to_index[item_id], *encoded_history])
                history.extend(self.item_to_index[item_id] for item_id in group["item_id"].drop_duplicates())
            final_histories[user_id] = history[-self.history_length :]
        if not rows:
            raise ValueError("TwoTowerRetriever needs at least one positive training interaction")
        return np.asarray(rows, dtype=np.int64), final_histories

    def fit(self, train: pd.DataFrame) -> "TwoTowerRetriever":
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["item_id"].astype(str).unique())
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        examples, final_histories = self._causal_examples(train)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        item_text = torch.tensor(self._item_text_features(train, items), dtype=torch.float32)
        self.model = _TwoTower(len(users), len(items), self.embedding_dim, item_text).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.model.train()
        for _ in range(self.epochs):
            batches = max(1, int(np.ceil(len(examples) / self.batch_size)))
            for batch_indices in np.array_split(rng.permutation(len(examples)), batches):
                batch = examples[batch_indices]
                user_indices = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
                item_indices = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)
                histories = torch.tensor(batch[:, 2:], dtype=torch.long, device=self.device)
                user_vectors = self.model.encode_users(user_indices, histories)
                item_vectors = self.model.encode_items(item_indices)
                logits = user_vectors @ item_vectors.T / self.temperature
                same_item = item_indices[:, None].eq(item_indices[None, :])
                numerator = torch.logsumexp(logits.masked_fill(~same_item, float("-inf")), dim=1)
                loss = -(numerator - torch.logsumexp(logits, dim=1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()
        padding = len(items)
        final_history_matrix = np.full((len(users), self.history_length), padding, dtype=np.int64)
        for user_id, history in final_histories.items():
            final_history_matrix[self.user_to_index[user_id], -len(history) :] = history
        with torch.no_grad():
            user_indices = torch.arange(len(users), device=self.device)
            histories = torch.tensor(final_history_matrix, dtype=torch.long, device=self.device)
            item_indices = torch.arange(len(items), device=self.device)
            self._user_vectors = self.model.encode_users(user_indices, histories).cpu().numpy()
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
