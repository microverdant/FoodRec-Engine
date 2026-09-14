# FoodRec Engine

Leakage-aware, reproducible recommendation experiments for the Amazon Fine Food
Reviews dataset. The repository replaces notebook-global variables and incomparable
metrics with one shared data contract and evaluation layer.

## What this fixes

- Cross-ASIN copies of the same user/rating/review are removed before splitting,
  then each `(user, item)` pair is reduced to its final rating. This prevents a
  copied target review from appearing in training under another product ID.
- User and item filtering is an iterative k-core, not one pass of stale counts.
- Rating prediction and Top-N recommendation are separate tasks. Rating reports
  RMSE, MAE, per-star MAE and macro-MAE over all 1--5 ratings. Top-N treats only
  ratings `>= 4` as relevant and reports Precision, Recall, Hit Rate, NDCG, MRR,
  Coverage, cold-target counts, and popularity.
- Top-N evaluates the complete training catalog with train-seen items masked. Ties
  are broken by a deterministic hash independent of target insertion order.
- All features, including TF-IDF documents, are fit on training data only. Hybrid
  weights are selected on validation only; the test split is evaluated once.

## Models

| Model | Role | Fixed failure from the original notebooks |
|---|---|---|
| Popularity | cold-start baseline | training-only positive counts |
| Sparse ItemKNN | collaborative retrieval | retained item-based model; top-K sparse similarity, no dense item-item matrix or positive-first tie |
| LightGCN | graph collaborative retrieval | BPR on the positive train graph only; validation/test events never become graph edges |
| BiasedMF | explicit SVD-style baseline | validation-only early stopping; shared candidate and seen-item policy |
| Implicit NMF | factorisation retrieval baseline | clearly labelled implicit, rather than a misleading rating precision metric |
| TF-IDF content | content retrieval | product documents built only from training reviews |
| Causal-history two tower | neural retrieval | train-only text and strictly earlier positive histories; multi-positive loss prevents duplicate-item false negatives |
| SASRec | sequential retrieval experiment | compact self-attention over strictly earlier timestamp groups; suitable for testing whether order adds signal |
| SVD/NMF blend | retained merged fusion | rank-normalised blend with its SVD coefficient selected on validation, never test |
| Weighted hybrid | score fusion | defined dependencies and validation-selected, rank-normalised weights |

## Setup

Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
without committing it, then install and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m foodrec.benchmark --csv D:\path\to\Reviews.csv --output artifacts\benchmark.json
```

The default primary protocol is per-user temporal splitting: the last distinct
timestamp is test and the penultimate timestamp is validation. Users without three
distinct timestamps are excluded rather than being split with an arbitrary order.
Run `--protocol global` as a stricter production-style robustness check and report
warm/cold cohorts separately.

Use the benchmark output only to compare models under the same protocol. Do not
compare it with legacy notebook numbers based on different catalogs or sampled
negative sets.

## Legacy notebook migration

The original ItemCF is retained as `ItemKNNRanker`, and the useful merged-notebook
ideas are retained as named source models: causal-history `TwoTowerRetriever`,
train-only `ContentRanker`, explicit `BiasedMF`, implicit `NMFImplicitRanker`, and
the validation-tuned `SVDNMFBlend`/`WeightedHybridRanker`. See
[the migration notes](docs/legacy-model-migration.md) for exactly what changed and
why the notebook evaluation cells are not used as benchmark results.

LightGCN and SASRec are additional research comparisons. The benchmark includes
both by default; use `--skip-neural` for a CPU-only classical/graph smoke run.
The repository does not present DLRM/DCNv2 as CTR models because this dataset has
no impression or non-click logs; adding a genuine industrial reranker requires
those labels and request/context features.

## Development

```powershell
pytest -q
```

Raw data, model checkpoints and generated artifacts are deliberately ignored. The
implementation is developed through pull requests; see the included PR template.
