import numpy as np
import pandas as pd
import pytest

from foodrec.models import (
    BiasedMF,
    ContentRanker,
    ItemKNNRanker,
    NMFImplicitRanker,
    PopularityRanker,
    SVDNMFBlend,
    TwoTowerRetriever,
    select_hybrid_weights,
    select_svd_nmf_blend,
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


def test_two_tower_history_excludes_same_timestamp_events():
    frame = pd.DataFrame(
        [
            ("u1", "a", 5, 1, "a", "a"),
            ("u1", "b", 5, 1, "b", "b"),
            ("u1", "c", 5, 2, "c", "c"),
        ],
        columns=["user_id", "item_id", "rating", "timestamp", "Summary", "Text"],
    )
    model = TwoTowerRetriever(history_length=3, embedding_dim=4, text_dim=2, epochs=1, device="cpu")
    model.user_to_index = {"u1": 0}
    model.item_to_index = {"a": 0, "b": 1, "c": 2}
    examples, _ = model._causal_examples(frame)
    # Rows a/b share timestamp 1, so neither may contain either item as history.
    assert examples[0, 2:].tolist() == [3, 3, 3]
    assert examples[1, 2:].tolist() == [3, 3, 3]
    # c at timestamp 2 may use both earlier events, ordered deterministically.
    assert examples[2, 2:].tolist() == [3, 0, 1]


def test_hybrid_weight_selection_uses_complete_weight_mappings(train_frame):
    validation = train_frame[train_frame.user_id.isin(["u1", "u2"])].copy()
    models = {"popularity": PopularityRanker().fit(train_frame), "item_knn": ItemKNNRanker(n_neighbors=2).fit(train_frame)}
    _, weights, _ = select_hybrid_weights(models, train_frame, validation, max_users=2)
    assert set(weights) == set(models)
    assert sum(weights.values()) == 1.0


def test_svd_nmf_blend_selects_validation_weight(train_frame):
    svd = BiasedMF(n_factors=4, n_epochs=2).fit(train_frame)
    nmf = NMFImplicitRanker(n_components=2).fit(train_frame)
    validation = train_frame[train_frame.user_id.isin(["u1", "u2"])].copy()
    blend, weight, _ = select_svd_nmf_blend(svd, nmf, train_frame, validation, max_users=2)
    assert isinstance(blend, SVDNMFBlend)
    assert 0.0 <= weight <= 1.0
