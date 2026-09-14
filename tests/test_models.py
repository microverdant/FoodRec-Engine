import numpy as np
import pytest

from foodrec.models import (
    BiasedMF,
    ContentRanker,
    ItemKNNRanker,
    NMFImplicitRanker,
    PopularityRanker,
    TwoTowerRetriever,
    select_hybrid_weights,
)


@pytest.mark.parametrize(
    "model",
    [
        PopularityRanker(),
        ItemKNNRanker(n_neighbors=2),
        BiasedMF(n_factors=4, n_epochs=2),
        NMFImplicitRanker(n_components=2),
        ContentRanker(max_features=30),
    ],
)
def test_classical_models_return_candidate_aligned_scores(model, train_frame):
    fitted = model.fit(train_frame)
    scores = fitted.score_items("u1", ["c", "d", "missing"])
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_two_tower_returns_candidate_aligned_scores(train_frame):
    model = TwoTowerRetriever(embedding_dim=8, text_dim=4, epochs=1, batch_size=4, seed=7, device="cpu").fit(train_frame)
    scores = model.score_items("u1", ["c", "d", "missing"])
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_hybrid_weight_selection_uses_complete_weight_mappings(train_frame):
    validation = train_frame[train_frame.user_id.isin(["u1", "u2"])].copy()
    models = {"popularity": PopularityRanker().fit(train_frame), "item_knn": ItemKNNRanker(n_neighbors=2).fit(train_frame)}
    _, weights, _ = select_hybrid_weights(models, train_frame, validation, max_users=2)
    assert set(weights) == set(models)
    assert sum(weights.values()) == 1.0
