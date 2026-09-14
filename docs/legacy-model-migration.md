# Legacy model migration

This repository retains the useful model families from `Itemcf_and_svd.ipynb` and
`merged.ipynb`, but replaces notebook-state-dependent code with importable models
that use the shared data split and evaluator.

| Legacy work | Retained implementation | Required correction |
|---|---|---|
| Item-based CF | `ItemKNNRanker` | Sparse top-neighbour similarity, train-seen masking, and deterministic target-independent ties. |
| SVD rating model | `BiasedMF` | Explicit 1--5 rating task is reported with RMSE/MAE; it can also be ranked under the common Top-N protocol. |
| NMF model | `NMFImplicitRanker` | Explicitly an implicit retriever, not a rating-precision measure. |
| SVD + NMF fusion | `SVDNMFBlend` | Rank-normalised and its SVD coefficient is selected only on validation. |
| TF-IDF content model | `ContentRanker` | Product documents and vectorizer fit on the training partition only. |
| Two-tower model | `TwoTowerRetriever` | Training targets see only strictly earlier positive history; same-timestamp events cannot leak into history. Text derives only from training reviews and duplicate positive item IDs are not false negatives. |
| Generic hybrid | `WeightedHybridRanker` | Named dependencies and validation-only weights replace notebook globals and test-set tuning. |
| New graph comparison | `LightGCNRanker` | Positive train interactions form a normalized bipartite graph; BPR samples never access validation/test. |
| New sequential comparison | `SASRecRanker` | Next-item examples only use strictly earlier timestamp groups; this tests sequence value without random-order leakage. |

The historical notebook outputs are not comparable benchmark results: they used
different sampling, tie, split, and/or metric conventions. The retained models are
therefore evaluated only through `foodrec.benchmark`, which uses full training
catalog ranking, masks train-seen items, and evaluates the untouched temporal test
partition once.
