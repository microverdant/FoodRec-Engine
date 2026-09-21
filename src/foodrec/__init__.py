"""FoodRec Engine: reproducible and leakage-aware recommendation experiments."""

from .data import DataConfig, ReviewDatasetBuilder, SplitData
from .evaluation import evaluate_ranking, evaluate_ratings

__all__ = [
    "DataConfig",
    "ReviewDatasetBuilder",
    "SplitData",
    "evaluate_ranking",
    "evaluate_ratings",
]

