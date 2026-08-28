# V3 Ranking Pipeline

## Purpose

V3 is a complete rewrite of the prediction pipeline to address the fundamental mismatch between binary classification labels and ranking-based evaluation.

**The Problem:** V2 uses binary labels (future_return > 1%) optimized with binary cross-entropy, but evaluation requires ranking stocks by expected returns.
This mismatch produces models with 45% hit rate but negative excess returns (ICIR 0.162, 9× below tradable threshold).

**The Solution:** V3 uses cross-sectional ranking labels (0-1 percentile ranks per day) optimized with LightGBM's lambdarank objective.
The model learns to rank stocks directly, aligning training objective with evaluation metrics (Rank IC, Rank ICIR, top-N excess returns).

## Key Differences from V2

| Aspect | V2 (Binary Classification) | V3 (Ranking) |
|--------|---------------------------|--------------|
| **Labels** | Binary: future_return > 1% (0/1) | Ranking: cross-sectional percentile (0-1) |
| **Objective** | binary cross-entropy | lambdarank with NDCG |
| **Output** | Probability of >1% gain | Relative score for ranking |
| **Evaluation** | Aligned for classification, misaligned for ranking | Directly aligned with ranking metrics |
| **Feature Space** | Reuses v2's 626 features | Same features (isolates label improvement) |

## Directory Structure

```
v3_pipeline/
├── src/                    # Core modules
│   ├── ranking_labels.py   # Compute cross-sectional percentile ranks
│   ├── build_feature_cache.py  # Transform v2 cache → v3 cache with ranking labels
│   ├── train_ranking.py    # LightGBM lambdarank training
│   └── evaluate_ranking.py # Rank IC, ICIR, excess returns
├── scripts/                # Executable scripts
│   └── run_baseline_experiment.sh  # End-to-end exp001 execution
├── models/                 # Trained model artifacts
│   └── .gitkeep
├── configs/                # YAML experiment configurations
│   └── v3_0_0_baseline.yaml
├── results/                # Experiment outputs (reports, plots)
│   └── .gitkeep
├── docs/                   # Experiment logs and documentation
│   └── EXPERIMENTS.md      # Experiment journal
└── README.md
```

## Quick Start

### 1. Build V3 Feature Cache
Transform v2's feature cache to add ranking labels:
```bash
python v3_pipeline/src/build_feature_cache.py \
  --input feature_cache_all.parquet \
  --output v3_pipeline/models/v3_0_0_feature_cache.parquet \
  --horizons 10d 30d
```

### 2. Train Baseline Model
```bash
python v3_pipeline/src/train_ranking.py \
  --config v3_pipeline/configs/v3_0_0_baseline.yaml \
  --horizon 10d \
  --output-dir v3_pipeline/models/exp001_10d
```

### 3. Evaluate on Validation Set
```bash
python v3_pipeline/src/evaluate_ranking.py \
  --model v3_pipeline/models/exp001_10d/model.txt \
  --cache v3_pipeline/models/v3_0_0_feature_cache.parquet \
  --horizon 10d \
  --val-start 2022-01-01 \
  --val-end 2025-07-31 \
  --output v3_pipeline/results/exp001_10d_evaluation.json
```

### 4. Run Full Baseline Experiment (Recommended)
```bash
bash v3_pipeline/scripts/run_baseline_experiment.sh
```

## Success Criteria

A model is considered **tradable** if validation set metrics meet:
- **Rank ICIR ≥ 0.5** (3× improvement over v2's 0.162)
- **Top5 Excess Return > 0%** (positive alpha)

If baseline (exp001) meets these thresholds, v3 validates the ranking approach and we proceed to feature engineering iterations (v3.1.0+).

## Experiment Tracking

All experiments are documented in `docs/EXPERIMENTS.md` with:
- Configuration snapshot
- Validation metrics
- Analysis and insights
- Next iteration plan

See `docs/EXPERIMENTS.md` for full experiment log.
