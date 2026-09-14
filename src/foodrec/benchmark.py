"""Command-line benchmark using one data protocol for every retriever."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import DataConfig, ReviewDatasetBuilder, global_temporal_split, per_user_temporal_split
from .evaluation import evaluate_ranking, evaluate_ratings
from .models import (
    BiasedMF,
    ContentRanker,
    ItemKNNRanker,
    NMFImplicitRanker,
    PopularityRanker,
    TwoTowerRetriever,
    select_hybrid_weights,
)


def _fit_models(train, validation, include_two_tower: bool):
    models = {
        "popularity": PopularityRanker().fit(train),
        "item_knn": ItemKNNRanker().fit(train),
        "biased_mf": BiasedMF().fit(train, validation),
        "implicit_nmf": NMFImplicitRanker().fit(train),
        "content_tfidf": ContentRanker().fit(train),
    }
    if include_two_tower:
        models["two_tower"] = TwoTowerRetriever().fit(train)
    return models


def run(csv_path: str, output_path: str, protocol: str, include_two_tower: bool) -> dict:
    builder = ReviewDatasetBuilder(DataConfig())
    data = builder.prepare(builder.load_csv(csv_path))
    split = per_user_temporal_split(data) if protocol == "per-user" else global_temporal_split(data)

    # Validation is used for model selection and hybrid weights. Test is untouched
    # until models and weights have been fixed.
    validation_models = _fit_models(split.train, split.validation, include_two_tower)
    validation_metrics = {
        name: evaluate_ranking(model, split.train, split.validation)
        for name, model in validation_models.items()
    }
    hybrid, weights, hybrid_selection_metrics = select_hybrid_weights(
        validation_models, split.train, split.validation
    )
    # The bounded validation subset above is only for weight selection. Publish the
    # hybrid's validation result on the same full validation cohort as every model.
    validation_metrics["hybrid"] = evaluate_ranking(hybrid, split.train, split.validation)

    combined_train = pd.concat([split.train, split.validation], ignore_index=True)
    final_models = _fit_models(combined_train, None, include_two_tower)
    from .models import WeightedHybridRanker

    final_models["hybrid"] = WeightedHybridRanker(final_models, weights)
    test_metrics = {
        name: evaluate_ranking(model, combined_train, split.test)
        for name, model in final_models.items()
    }
    rating_metrics = evaluate_ratings(final_models["biased_mf"], split.test)
    report = {
        "preparation": builder.report.to_dict() if builder.report else {},
        "protocol": split.protocol,
        "split_rows": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "hybrid_weights_selected_on_validation": weights,
        "hybrid_weight_selection_validation_sample": hybrid_selection_metrics,
        "validation_ranking": validation_metrics,
        "test_ranking": test_metrics,
        "test_rating_biased_mf": rating_metrics,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to Kaggle Reviews.csv")
    parser.add_argument("--output", default="artifacts/benchmark.json")
    parser.add_argument("--protocol", choices=("per-user", "global"), default="per-user")
    parser.add_argument("--skip-two-tower", action="store_true")
    args = parser.parse_args()
    report = run(args.csv, args.output, args.protocol, include_two_tower=not args.skip_two_tower)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
