# V3 Ranking Pipeline Experiments

This document tracks all v3 experiments, including configuration, results, and learnings.

## Experiment Log

### exp001 - Baseline (v3.0.0)

**Date:** TBD  
**Config:** `configs/v3_0_0_baseline.yaml`  
**Branch:** `feature/v3-ranking-pipeline`

**Objective:**
Validate that ranking labels + lambdarank objective resolve v2's binary classification mismatch.
Success = ICIR ≥ 0.5 and positive top5 excess returns.

**Configuration:**
- Features: 626 features from v2 (excluding label/return columns)
- Labels: Cross-sectional percentile ranks (0-1 scale per day)
- Model: LightGBM lambdarank with NDCG optimization
- Training: 2001-2021 (≤2021-12-31)
- Validation: 2022-2025 (2022-01-01 to 2025-07-31)
- Horizons: 10d, 30d

**Hyperparameters:**
```yaml
objective: lambdarank
metric: ndcg
ndcg_eval_at: [5, 10, 20]
num_leaves: 31
learning_rate: 0.05
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 5
min_data_in_leaf: 100
num_boost_round: 1000
early_stopping_rounds: 50
```

**Results:**

| Horizon | Rank IC | Rank ICIR | Top5 Excess | Top10 Excess | NDCG@5 | NDCG@10 | Tradable? |
|---------|---------|-----------|-------------|--------------|--------|---------|-----------|
| 10d     | -       | -         | -           | -            | -      | -       | -         |
| 30d     | -       | -         | -           | -            | -      | -       | -         |

**Analysis:**
_To be filled after experiment execution._

**Next Steps:**
_To be determined based on results._

---

## Experiment Template

### expXXX - [Brief Description]

**Date:** YYYY-MM-DD  
**Config:** `configs/[config_file].yaml`  
**Branch:** `feature/[branch-name]`

**Objective:**
[What hypothesis are we testing? What problem are we solving?]

**Configuration:**
- Features: [Feature set description]
- Labels: [Label configuration]
- Model: [Model type and key settings]
- Training: [Date range]
- Validation: [Date range]
- Horizons: [List of horizons]

**Hyperparameters:**
```yaml
[Key hyperparameters]
```

**Results:**

| Horizon | Rank IC | Rank ICIR | Top5 Excess | Top10 Excess | NDCG@5 | NDCG@10 | Tradable? |
|---------|---------|-----------|-------------|--------------|--------|---------|-----------|
| [horizon] | [value] | [value] | [value] | [value] | [value] | [value] | [yes/no] |

**Analysis:**
[Key observations, patterns, unexpected behaviors]

**Next Steps:**
[What to try next based on these results]
