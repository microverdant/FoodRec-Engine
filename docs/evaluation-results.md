# Checked evaluation results

These results use the strict preprocessing contract, full train-catalog ranking,
train-seen masking, and rating `>= 4` as Top-N relevance. Generated JSON artifacts
and raw data are intentionally not committed.

## Per-user temporal holdout

The final timestamp group per user is test and the penultimate group is validation.
ItemKNN and EASE hyperparameters are selected on validation before refitting.

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| ItemKNN (`100` neighbours, shrinkage `50`) | **0.1747** | **0.0974** |
| EASE (regularisation `300`) | 0.1494 | 0.0869 |
| SASRec | 0.1429 | 0.0740 |
| Implicit NMF | 0.1210 | 0.0740 |
| LightGCN | 0.0877 | 0.0448 |

This is a warm-user protocol. It answers whether a model can recommend a later
item for a user with sufficient earlier history; it does not measure deployment
cold start.

## Global calendar-time holdout

The global 80%/10%/10% timestamp split produced 20,649 train, 2,662 validation,
and 2,488 test rows. Validation selected EASE regularisation `10` and ItemKNN
`200` neighbours with no shrinkage.

| Model | Recall@10 (95% user bootstrap CI) | NDCG@10 (95% user bootstrap CI) |
|---|---:|---:|
| EASE | **0.0926** (0.0767–0.1119) | **0.0582** (0.0472–0.0710) |
| Content TF-IDF | 0.0841 (0.0672–0.0996) | 0.0431 (0.0338–0.0522) |
| ItemKNN | 0.0831 (0.0655–0.0983) | 0.0472 (0.0377–0.0571) |
| Implicit NMF | 0.0738 (0.0572–0.0903) | 0.0525 (0.0397–0.0657) |
| LightGCN | 0.0430 (0.0322–0.0552) | 0.0323 (0.0232–0.0431) |

The test period contains 1,661 positive events from 937 users. Metrics are
conditional on the 822 eligible warm users; the evaluator separately discloses
30 cold-user target events, 232 cold-item target events, and 111 users without an
evaluable warm target/candidate set. Individual confidence intervals overlap, so
the point-estimate lead of EASE is not presented as a statistically significant
pairwise win.

## Reproduction

```powershell
python -m foodrec.benchmark --csv D:\path\to\Reviews.csv --output artifacts\per_user.json --skip-neural
python -m foodrec.benchmark --csv D:\path\to\Reviews.csv --output artifacts\global.json --protocol global --skip-neural
```
