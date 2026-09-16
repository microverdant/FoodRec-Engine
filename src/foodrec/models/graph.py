"""Graph collaborative filtering models trained on positive train interactions only."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover
    raise ImportError("LightGCNRanker requires the optional torch dependency") from error


class _LightGCN(nn.Module):
    def __init__(self, users: int, items: int, embedding_dim: int, layers: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(users, embedding_dim)
        self.item_embedding = nn.Embedding(items, embedding_dim)
        self.layers = layers
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, normalized_adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        outputs = [embeddings]
        for _ in range(self.layers):
            embeddings = torch.sparse.mm(normalized_adjacency, embeddings)
            outputs.append(embeddings)
        final = torch.stack(outputs, dim=0).mean(dim=0)
        return final[: self.user_embedding.num_embeddings], final[self.user_embedding.num_embeddings :]


class LightGCNRanker:
    """LightGCN with BPR loss and a train-positive bipartite graph.

    This is deliberately a retrieval model: ratings below the positive threshold do
    not become graph edges, and validation/test interactions never enter either the
    graph or negative sampler.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        layers: int = 2,
        epochs: int = 40,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        regularization: float = 1e-4,
        positive_threshold: float = 4.0,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.positive_threshold = positive_threshold
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.model: _LightGCN | None = None
        self._user_vectors: np.ndarray | None = None
        self._item_vectors: np.ndarray | None = None

    def _normalised_adjacency(self, pairs: np.ndarray, users: int, items: int) -> torch.Tensor:
        user_nodes = pairs[:, 0]
        item_nodes = pairs[:, 1] + users
        rows = np.concatenate((user_nodes, item_nodes))
        cols = np.concatenate((item_nodes, user_nodes))
        degree = np.bincount(rows, minlength=users + items).astype(np.float32)
        values = 1.0 / np.sqrt(degree[rows] * degree[cols])
        indices = torch.tensor(np.vstack((rows, cols)), dtype=torch.long, device=self.device)
        return torch.sparse_coo_tensor(
            indices,
            torch.tensor(values, dtype=torch.float32, device=self.device),
            (users + items, users + items),
            check_invariants=False,
        ).coalesce()

    def fit(self, train: pd.DataFrame) -> "LightGCNRanker":
        events = train[train["rating"] >= self.positive_threshold][["user_id", "item_id"]].drop_duplicates().copy()
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["item_id"].astype(str).unique())
        if events.empty or not users or not items:
            raise ValueError("LightGCNRanker needs at least one positive training interaction")
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        pairs = np.column_stack((
            events["user_id"].astype(str).map(self.user_to_index).to_numpy(dtype=np.int64),
            events["item_id"].astype(str).map(self.item_to_index).to_numpy(dtype=np.int64),
        ))
        positives_by_user: dict[int, set[int]] = {}
        for user_index, item_index in pairs:
            positives_by_user.setdefault(int(user_index), set()).add(int(item_index))
        eligible_pairs = np.asarray(
            [(user, item) for user, item in pairs if len(positives_by_user[int(user)]) < len(items)], dtype=np.int64
        )
        if not len(eligible_pairs):
            raise ValueError("LightGCNRanker needs at least one candidate negative item")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        adjacency = self._normalised_adjacency(pairs, len(users), len(items))
        self.model = _LightGCN(len(users), len(items), self.embedding_dim, self.layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        for _ in range(self.epochs):
            for batch_indices in np.array_split(rng.permutation(len(eligible_pairs)), max(1, int(np.ceil(len(eligible_pairs) / self.batch_size)))):
                batch = eligible_pairs[batch_indices]
                negative = np.empty(len(batch), dtype=np.int64)
                for index, user_index in enumerate(batch[:, 0]):
                    candidate = int(rng.integers(len(items)))
                    while candidate in positives_by_user[int(user_index)]:
                        candidate = int(rng.integers(len(items)))
                    negative[index] = candidate
                user_vectors, item_vectors = self.model.propagate(adjacency)
                user_index = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
                positive_index = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)
                negative_index = torch.tensor(negative, dtype=torch.long, device=self.device)
                positive_scores = (user_vectors[user_index] * item_vectors[positive_index]).sum(dim=1)
                negative_scores = (user_vectors[user_index] * item_vectors[negative_index]).sum(dim=1)
                bpr = -F.logsigmoid(positive_scores - negative_scores).mean()
                l2 = (user_vectors[user_index].square().sum() + item_vectors[positive_index].square().sum() + item_vectors[negative_index].square().sum()) / len(batch)
                loss = bpr + self.regularization * l2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()
        with torch.no_grad():
            user_vectors, item_vectors = self.model.propagate(adjacency)
            self._user_vectors = user_vectors.cpu().numpy()
            self._item_vectors = item_vectors.cpu().numpy()
        return self

    def score_items(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        if self._user_vectors is None or self._item_vectors is None:
            raise RuntimeError("Call fit before scoring")
        user_index = self.user_to_index.get(str(user_id))
        if user_index is None:
            return np.zeros(len(item_ids), dtype=float)
        item_indices = np.asarray([self.item_to_index.get(str(item_id), -1) for item_id in item_ids], dtype=int)
        scores = np.zeros(len(item_ids), dtype=float)
        valid = item_indices >= 0
        scores[valid] = self._item_vectors[item_indices[valid]] @ self._user_vectors[user_index]
        return scores
