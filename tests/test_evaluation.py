import numpy as np
import pandas as pd

from foodrec.evaluation import evaluate_ranking, evaluate_ratings


class ConstantRanker:
    def score_items(self, user_id, item_ids):
        return np.zeros(len(item_ids))


class GlobalRatingModel:
    def predict_rating(self, user_id, item_id):
        return 4.0


def test_single_positive_precision_is_hit_rate_divided_by_k():
    train = pd.DataFrame(
        [("u1", "a", 5), ("u2", "b", 5), ("u3", "c", 5), ("u4", "d", 5)],
        columns=["user_id", "item_id", "rating"],
    )
    test = pd.DataFrame([("u1", "d", 5)], columns=["user_id", "item_id", "rating"])
    first = evaluate_ranking(ConstantRanker(), train, test, k_values=(2,), seed=42)
    second = evaluate_ranking(ConstantRanker(), train.sample(frac=1, random_state=99), test, k_values=(2,), seed=42)
    assert first["precision@2"] == first["hit_rate@2"] / 2
    assert first == second


def test_rating_metrics_include_macro_mae():
    test = pd.DataFrame(
        [("u1", "a", 1), ("u2", "b", 5)], columns=["user_id", "item_id", "rating"]
    )
    metrics = evaluate_ratings(GlobalRatingModel(), test)
    assert metrics["mae"] == 2.0
    assert metrics["macro_mae"] == 2.0
    assert metrics["mae_rating_1"] == 3.0
    assert metrics["mae_rating_5"] == 1.0


def test_ranking_bootstrap_is_deterministic_and_user_level():
    train = pd.DataFrame(
        [("u1", "a", 5), ("u2", "b", 5), ("u3", "c", 5), ("u4", "d", 5)],
        columns=["user_id", "item_id", "rating"],
    )
    test = pd.DataFrame(
        [("u1", "d", 5), ("u2", "c", 5), ("u3", "a", 5)],
        columns=["user_id", "item_id", "rating"],
    )
    first = evaluate_ranking(ConstantRanker(), train, test, k_values=(2,), seed=7, bootstrap_samples=100)
    second = evaluate_ranking(ConstantRanker(), train, test, k_values=(2,), seed=7, bootstrap_samples=100)
    assert first == second
    assert first["recall@2_ci95_low"] <= first["recall@2_ci95_high"]
    assert first["ndcg@2_ci95_low"] <= first["ndcg@2_ci95_high"]
