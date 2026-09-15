"""Small causal SASRec-style next-item retriever for timestamped review histories."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover
    raise ImportError("SASRecRanker requires the optional torch dependency") from error


class _SASRec(nn.Module):
    def __init__(self, items: int, embedding_dim: int, max_history: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.padding_index = items
        self.item_embedding = nn.Embedding(items + 1, embedding_dim, padding_idx=items)
        self.position_embedding = nn.Embedding(max_history, embedding_dim)
        block = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=heads, dim_feedforward=embedding_dim * 2, dropout=dropout, batch_first=True, activation="gelu"
        )
        # Padded fixed-length histories do not benefit from PyTorch's experimental
        # nested-tensor conversion and keeping the dense layout is more portable.
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        nn.init.normal_(self.item_embedding.weight[:-1], std=0.02)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0)
        padding = history.eq(self.padding_index)
        vectors = self.item_embedding(history) + self.position_embedding(positions)
        encoded = self.encoder(vectors, src_key_padding_mask=padding)
        lengths = (~padding).sum(dim=1).clamp(min=1) - 1
        return F.normalize(encoded[torch.arange(len(history), device=history.device), lengths], dim=-1)

    def item_vectors(self, item_indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_embedding(item_indices), dim=-1)


class SASRecRanker:
    """Causal self-attentive next-item retrieval with timestamp-group-safe history.

    This compact implementation is a research comparison for short review histories,
    not a claim of industrial sequence scale. It excludes validation/test events from
    both target construction and negative sampling.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_history: int = 20,
        layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        regularization: float = 1e-5,
        positive_threshold: float = 4.0,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.embedding_dim = embedding_dim
        self.max_history = max_history
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.positive_threshold = positive_threshold
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.item_to_index: dict[str, int] = {}
        self.model: _SASRec | None = None
        self._user_vectors: dict[str, np.ndarray] = {}
        self._item_vectors: np.ndarray | None = None

    def _examples(
        self, train: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], dict[str, set[int]]]:
        required = {"user_id", "item_id", "rating", "timestamp"}
        missing = required.difference(train.columns)
        if missing:
            raise ValueError(f"SASRecRanker needs columns: {sorted(missing)}")
        positive = train[train["rating"] >= self.positive_threshold].copy()
        positive["user_id"] = positive["user_id"].astype(str)
        positive["item_id"] = positive["item_id"].astype(str)
        positive = positive.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
        histories: list[list[int]] = []
        targets: list[int] = []
        example_users: list[str] = []
        final_histories: dict[str, list[int]] = {}
        positives_by_user: dict[str, set[int]] = {}
        padding = len(self.item_to_index)
        for user_id, events in positive.groupby("user_id", sort=False):
            history: list[int] = []
            known: set[int] = set()
            for _, group in events.groupby("timestamp", sort=False):
                group_items = [self.item_to_index[item_id] for item_id in group["item_id"].drop_duplicates()]
                if history:
                    encoded = ([padding] * self.max_history + history[-self.max_history :])[-self.max_history :]
                    for target in group_items:
                        histories.append(encoded)
                        targets.append(target)
                        example_users.append(user_id)
                history.extend(group_items)
                known.update(group_items)
            final_histories[user_id] = history[-self.max_history :]
            positives_by_user[user_id] = known
        return (
            np.asarray(histories, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
            np.asarray(example_users, dtype=object),
            final_histories,
            positives_by_user,
        )

    def fit(self, train: pd.DataFrame) -> "SASRecRanker":
        items = sorted(train["item_id"].astype(str).unique())
        if not items:
            raise ValueError("SASRecRanker needs at least one training item")
        self.item_to_index = {item_id: index for index, item_id in enumerate(items)}
        histories, targets, example_users, final_histories, positives_by_user = self._examples(train)
        if not len(targets):
            raise ValueError("SASRecRanker needs a user with positive events at two distinct timestamps")
        # A sampled negative may not be any train-period positive for that user.
        # This prevents false-negative supervision without touching validation/test.
        eligible = np.asarray([len(positives_by_user[user_id]) < len(items) for user_id in example_users])
        histories, targets, example_users = histories[eligible], targets[eligible], example_users[eligible]
        if not len(targets):
            raise ValueError("SASRecRanker needs a target whose user has an unseen training item")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.model = _SASRec(len(items), self.embedding_dim, self.max_history, self.layers, self.heads, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
        self.model.train()
        for _ in range(self.epochs):
            for batch_indices in np.array_split(rng.permutation(len(targets)), max(1, int(np.ceil(len(targets) / self.batch_size)))):
                batch_history = torch.tensor(histories[batch_indices], dtype=torch.long, device=self.device)
                positive = torch.tensor(targets[batch_indices], dtype=torch.long, device=self.device)
                negative_indices = np.empty(len(batch_indices), dtype=np.int64)
                for batch_position, example_index in enumerate(batch_indices):
                    user_positives = positives_by_user[example_users[example_index]]
                    candidate = int(rng.integers(len(items)))
                    while candidate in user_positives:
                        candidate = int(rng.integers(len(items)))
                    negative_indices[batch_position] = candidate
                negative = torch.tensor(negative_indices, dtype=torch.long, device=self.device)
                user_vectors = self.model.encode_history(batch_history)
                positive_scores = (user_vectors * self.model.item_vectors(positive)).sum(dim=1)
                negative_scores = (user_vectors * self.model.item_vectors(negative)).sum(dim=1)
                loss = -F.logsigmoid(positive_scores - negative_scores).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()
        self._user_vectors = {}
        with torch.no_grad():
            for user_id, history in final_histories.items():
                if not history:
                    continue
                encoded = ([len(items)] * self.max_history + history)[-self.max_history :]
                vector = self.model.encode_history(torch.tensor([encoded], dtype=torch.long, device=self.device))
                self._user_vectors[user_id] = vector.cpu().numpy()[0]
            indices = torch.arange(len(items), dtype=torch.long, device=self.device)
            self._item_vectors = self.model.item_vectors(indices).cpu().numpy()
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if self._item_vectors is None:
            raise RuntimeError("Call fit before scoring")
        user_vector = self._user_vectors.get(str(user_id))
        if user_vector is None:
            return np.zeros(len(item_ids), dtype=float)
        indices = np.asarray([self.item_to_index.get(str(item_id), -1) for item_id in item_ids], dtype=int)
        scores = np.zeros(len(item_ids), dtype=float)
        valid = indices >= 0
        scores[valid] = self._item_vectors[indices[valid]] @ user_vector
        return scores
