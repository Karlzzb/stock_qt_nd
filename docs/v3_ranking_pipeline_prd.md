# V3 Ranking-Based Prediction Pipeline

## Problem Statement

The current stock prediction system (v2) uses binary classification to predict whether a stock will rise above a threshold (e.g., >1% return), but evaluates performance using ranking metrics (Rank IC, ICIR, top5 excess return). This fundamental mismatch causes the model to achieve ~45% binary hit rates while producing **negative excess returns** on top-ranked stocks.

Specifically:
- Best v2 model: ICIR = 0.162, which is **9× below** the tradable threshold of 1.5
- Top-5 stock picks show negative excess returns (-1.32% for 15d label)
- The model correctly identifies "winning stocks" but picks the **wrong stocks within winners**
- Binary labels (return > threshold) discard magnitude information critical for ranking

This makes the system unusable for actual trading, as selecting top-N stocks from binary predictions optimizes the wrong objective.

## Solution

Build a new v3 pipeline that directly optimizes for ranking objectives using continuous return targets and learning-to-rank models. The v3 pipeline will be **completely independent** from v2 to enable parallel experimentation without breaking existing systems.

Key changes:
1. **Ranking targets**: Replace binary labels with cross-sectional rank labels (0-1 per day)
2. **Ranking models**: Use LightGBM's lambdarank objective optimized for NDCG
3. **Direct alignment**: Training objective matches evaluation metrics (Rank IC/ICIR)
4. **Clean architecture**: Separate codebase (v3_pipeline/) with semantic versioning

## User Stories

1. As a quant researcher, I want to train models on ranking objectives, so that the training aligns with how I'll actually use the predictions (selecting top-N stocks)

2. As a quant researcher, I want v3 to be completely independent from v2, so that I can experiment with new approaches without risk of breaking production systems

3. As a quant researcher, I want to evaluate v3 against v2 on identical test data, so that I can quantify the improvement from ranking-based approaches

4. As a quant researcher, I want v3 to use the same 7 return horizons [3d, 5d, 10d, 15d, 20d, 25d, 30d], so that I can validate the ranking approach works across all timeframes where binary classification failed

5. As a quant researcher, I want v3 to reuse the existing 626 features initially, so that I can isolate whether the improvement comes from the ranking objective rather than different features

6. As a quant researcher, I want cross-sectional ranking labels computed per-day, so that the model learns to rank stocks relative to each other on the same date

7. As a quant researcher, I want the v3 feature cache to be separate from v2, so that future v3 feature changes don't break v2 experiments

8. As a quant researcher, I want v3 to use the same train/validation split (train ≤2021-12-31, validate 2022-01-01 to 2025-07-31), so that results are directly comparable to v2 baseline

9. As a quant researcher, I want a single LightGBM lambdarank model for v3.0.0, so that I can prove the ranking concept works before adding architectural complexity

10. As a quant researcher, I want v3 evaluation to compute the same metrics as v2 (Rank IC, ICIR, top5_excess_ret), so that I can directly compare performance

11. As a quant researcher, I want v3 to target ICIR > 0.5 as the success criterion, so that I validate a meaningful improvement (3× over v2's 0.162) without expecting perfection immediately

12. As a quant researcher, I want v3 experiments to be numbered (exp001, exp002...), so that I can track iterations and compare results over time

13. As a quant researcher, I want an EXPERIMENTS.md log in v3_pipeline/, so that each experiment run is documented with config and results

14. As a quant researcher, I want v3 to save models with horizon-specific names (model_10d.txt, model_30d.txt), so that I can inspect and deploy individual horizon models

15. As a quant researcher, I want v3 evaluation reports to clearly show ICIR and top5_excess for each horizon, so that I can identify which timeframes benefit most from ranking

16. As a quant researcher, I want v3 to fail fast with clear error messages if ranking labels are misconfigured, so that I catch bugs early rather than training bad models

17. As a quant researcher, I want v3 to preserve the original continuous returns in the cache, so that I can experiment with different ranking transformations without regenerating features

18. As a quant researcher, I want v3 scripts to be simple and readable (~200-300 lines), so that future researchers can understand and modify them easily

19. As a quant researcher, I want v3 to avoid premature optimization, so that initial iterations focus on proving the concept rather than squeezing marginal gains

20. As a quant researcher, I want v3 documentation to explain why lambdarank was chosen over alternatives, so that future work can make informed decisions about model changes

## Implementation Decisions

### Architecture

**Separate v3 pipeline structure:**
```
v3_pipeline/
├── src/
│   ├── feature_pipeline.py       # Copied from v2, generates same 626 features
│   └── ranking_labels.py         # NEW: Compute cross-sectional ranks
├── scripts/
│   ├── build_feature_cache.py    # Transform v2 cache → v3 cache with ranks
│   ├── train_ranking.py          # NEW: Single LGBM lambdarank training
│   └── evaluate_ranking.py       # Copied from scripts/run_rank_metrics.py
├── models/                        # Model outputs
├── configs/
│   └── v3_0_0_baseline.yaml      # Hyperparameters, horizons, paths
├── results/                       # JSON outputs from experiments
├── docs/
│   └── EXPERIMENTS.md            # Experiment log
└── README.md                      # V3 overview and quick start
```

**Version numbering:**
- Use semantic versioning: v3.0.0 for baseline
- v3.1.0 for new features, v3.0.1 for bugfixes
- Clearly separates major paradigm shift (v3) from iterations

### Data Pipeline

**Feature cache transformation:**
- Read `feature_cache_all.parquet` (1.4M rows, 232 cols including future_return_* continuous columns)
- For each horizon [3d, 5d, 10d, 15d, 20d, 25d, 30d]:
  - Group by timestamp
  - Compute cross-sectional rank [0-1] within each date
  - Store as `rank_future_return_10d`, etc.
- Save to `v3_pipeline/feature_cache_v3.parquet`
- Preserve original continuous returns for future experiments

**Ranking label computation:**
```python
# Per-day cross-sectional ranking
df['rank_future_return_10d'] = (
    df.groupby('timestamp')['future_return_10d']
    .rank(pct=True, method='average')  # 0-1 scale
)
```

**Data split:**
- Training: timestamp ≤ 2021-12-31
- Validation: 2022-01-01 ≤ timestamp ≤ 2025-07-31
- Matches v2 for direct comparison

### Model Architecture

**Single LightGBM lambdarank for v3.0.0:**
- Objective: `lambdarank` (optimizes NDCG)
- Features: All 626 from v2 (OPTIMIZED_FEATURE_COLS from model_config)
- Target: `rank_future_return_10d` (and other horizons)
- Query groups: Group by `timestamp` (stocks ranked within each day)

**Why lambdarank over alternatives:**
- Purpose-built for learning-to-rank problems
- Directly optimizes ranking metrics (NDCG)
- Handles the list-wise ranking structure naturally
- LightGBM implementation is production-tested

**Why single model (not two-layer stack):**
- Two-layer (LGBM → LR) in v2 was for binary probability calibration
- Ranking objectives don't need calibration - only relative order matters
- Simpler = faster iteration for initial validation
- Can add stacking in v3.1.0 if single model has systematic biases

**Hyperparameters (starting point):**
```yaml
lgbm_params:
  objective: lambdarank
  metric: ndcg
  ndcg_eval_at: [5, 10]
  num_leaves: 31
  learning_rate: 0.05
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  min_data_in_leaf: 100
  num_boost_round: 500
  early_stopping_rounds: 50
```

### Training Script Structure

**train_ranking.py (~250 lines):**
1. Load config from YAML
2. Load feature_cache_v3.parquet with needed columns
3. Split train/validation by timestamp
4. For each horizon in [10d, 30d] (initial scope):
   - Prepare ranking dataset (features, rank_labels, query_groups)
   - Train LGBM with lambdarank objective
   - Save model to `models/v3_0_0_exp001_model_10d.txt`
   - Generate predictions on validation set
   - Save predictions to `results/v3_0_0_exp001_pred_10d.parquet`
5. Log experiment to EXPERIMENTS.md

**Key implementation details:**
- Use `lgb.Dataset` with `group=` parameter for query grouping
- Handle missing values in ranking labels (stocks with NaN returns)
- Validate prediction ordering: higher rank_label → higher prediction
- Save both raw predictions and re-ranked predictions (0-1)

### Evaluation

**evaluate_ranking.py:**
- Copy from existing `scripts/run_rank_metrics.py`
- Input: prediction parquet (timestamp, symbol, prediction, actual_return)
- Output: JSON with per-horizon metrics
  ```json
  {
    "future_return_10d": {
      "daily_rank_ic": 0.XXX,
      "daily_rank_icir": 0.XXX,
      "daily_top5_excess_ret": 0.XXX,
      "top5_turnover": 0.XXX,
      ...
    }
  }
  ```
- Compare against v2 baseline stored in `experiments/rank_metrics_2021-12-31_2022-01-01_2025-07-31.json`

### Configuration Management

**v3_0_0_baseline.yaml:**
```yaml
version: v3.0.0
experiment_id: exp001
description: "Baseline ranking model - same features as v2, lambdarank objective"

data:
  feature_cache: "v3_pipeline/feature_cache_v3.parquet"
  train_end: "2021-12-31"
  val_start: "2022-01-01"
  val_end: "2025-07-31"

horizons:
  - 10d
  - 30d  # Start with best v2 performers

features:
  source: "v2_626_features"  # Reuse exactly
  count: 626

model:
  type: "lgbm_lambdarank"
  params:
    objective: lambdarank
    metric: ndcg
    # ... (full params)

evaluation:
  metrics:
    - rank_ic
    - rank_icir
    - top5_excess_ret
    - top5_turnover
  success_threshold:
    icir: 0.5
```

### Experiment Tracking

**EXPERIMENTS.md format:**
```markdown
# V3 Ranking Pipeline Experiments

## exp001 - 2026-08-28 - v3.0.0 Baseline

**Config:** v3_0_0_baseline.yaml

**Objective:** Prove lambdarank improves over v2 binary classification

**Results:**
- future_return_10d: ICIR = 0.XXX (v2: 0.162)
- future_return_30d: ICIR = 0.XXX (v2: 0.139)

**Analysis:** 
- [Key findings]
- [What worked / didn't work]

**Next steps:**
- [Follow-up experiments]

---

## exp002 - 2026-08-29 - Feature screening

...
```

## Testing Decisions

### Testing Philosophy

**Good tests validate external behavior, not implementation:**
- Test that ranking labels are correctly ordered (higher return → higher rank)
- Test that predictions preserve relative ordering
- Test that evaluation metrics match mathematical definitions
- Do NOT test internal LightGBM mechanics (that's LightGBM's job)

### Modules to Test

**1. ranking_labels.py**
- Test: Cross-sectional ranking produces [0-1] range per timestamp
- Test: Stocks with NaN returns get NaN ranks
- Test: Ranking is stable (no randomness without ties)
- Test: Edge case - single stock on a date gets rank 0.5
- Prior art: None in v2, write new tests in `v3_pipeline/tests/test_ranking_labels.py`

**2. train_ranking.py**
- Test: Training completes without errors on sample data (10 days)
- Test: Model saves to expected path
- Test: Predictions have correct shape (n_validation_samples,)
- Test: Predictions are in reasonable range [0-1] after normalization
- Prior art: Similar to v2's training script tests (if they exist), otherwise smoke tests only

**3. evaluate_ranking.py**
- Test: Rank IC calculation matches reference implementation
- Test: ICIR = IC_mean / IC_std
- Test: top5_excess_ret correctly selects top 5 per day
- Test: Handle edge cases (days with <5 stocks, all predictions same)
- Prior art: Check existing `scripts/run_rank_metrics.py` tests, copy pattern

**4. Integration test**
- Test: End-to-end pipeline (build cache → train → evaluate) runs successfully
- Test: v3 produces different results from v2 (sanity check they're not identical)
- Test: Results JSON has expected structure
- Prior art: Similar to v2 integration tests if they exist

### Testing Infrastructure

**Test data:**
- Create `v3_pipeline/tests/fixtures/sample_cache.parquet` (100 stocks × 10 days)
- Small enough to run quickly, large enough to test group operations

**Test runner:**
- Use pytest
- Tests should run in <10 seconds total
- Mock expensive operations (actual LGBM training uses sample data)

**What NOT to test:**
- LightGBM's lambdarank correctness (trust the library)
- Whether ICIR > 0.5 is achieved (that's an experiment outcome, not a test)
- Feature generation quality (v2's responsibility)

## Out of Scope

**For v3.0.0 baseline:**
1. New feature engineering - use v2's 626 features as-is
2. Two-layer model stacking - single LGBM only
3. Hyperparameter tuning - use reasonable defaults, tune in v3.1.0
4. Additional horizons beyond 10d/30d - expand after proving concept
5. Ensemble methods - defer to v3.1.0
6. Walk-forward training - use single split for direct v2 comparison
7. Production deployment - v3.0.0 is research/validation only
8. Trading strategy layer (position sizing, risk management) - separate concern
9. Alternative ranking objectives (pairwise, listwise) - lambdarank first
10. Feature screening/selection - do after baseline proves ranking works

**Explicitly deferred to future versions:**
- v3.1.0: Feature screening, hyperparameter tuning
- v3.2.0: Model ensembles, two-layer stacking if needed
- v3.3.0: New feature categories based on v3 baseline insights
- v4.0.0: Production deployment with trading strategy integration

## Further Notes

### Why Ranking is the Right Approach

The v2 analysis revealed a fundamental issue: **binary labels optimize for "which stocks win" but ranking optimizes for "which stocks win most"**.
Example from v2:

- Stock A: +2% return → binary label 1 (win)
- Stock B: +10% return → binary label 1 (win)
- Model learns both are "winners" but can't distinguish magnitude

In a ranking formulation:
- Stock A: rank 0.3
- Stock B: rank 0.95
- Model learns B is a better pick than A

This is why v2 achieved 45% hit rate (correctly identifies winners) but negative excess returns (picks the wrong winners).

### Success Criteria

**v3.0.0 is successful if:**
- ICIR > 0.5 on at least one horizon (3× improvement over v2's 0.162)
- top5_excess_ret > 0.5% (positive alpha, even if below 2% threshold)
- Rank IC is positive and stable (no wild swings day-to-day)

**This validates the approach justifies further investment.**

**v3.0.0 does NOT need to:**
- Reach tradable threshold (ICIR > 1.5) - that requires further optimization
- Beat all 7 horizons immediately - even 2/7 working proves the concept
- Have production-ready code - this is research infrastructure

### Migration Path

If v3 proves superior:
1. v3.1.0+: Iterate to reach tradable thresholds
2. v4.0.0: Productionize with proper trading strategy
3. Deprecate v2 binary classification approach
4. Keep v2 codebase as historical reference

If v3 fails to improve over v2:
- Investigate why: model capacity, data quality, or different issue entirely
- Re-evaluate hypothesis that binary labels are the problem
- v2 remains the baseline, explore other directions

### Key Risks

1. **Lambdarank complexity**: If query grouping or NDCG optimization causes issues, fall back to regression on rank labels
2. **Label noise**: If ranking labels amplify noise vs binary labels, may need smoothing or robust ranking methods
3. **Overfitting**: Ranking on small daily cross-sections (50-100 stocks) may overfit - monitor train/val gap
4. **Computational cost**: Lambdarank is slower than regression - if training takes >hours, may need optimization

### Related Issues

- Issue #16: Label × Feature joint iteration framework - v3 implements the "label correction" priority
- Issue #18: Divergence detector rollback - v3 inherits the fixed v2 features
- Memory note: "排名口径揭穿二元label错位" - v3 directly addresses this insight

### References

- v2 baseline: `experiments/rank_metrics_2021-12-31_2022-01-01_2025-07-31.json`
- Feature documentation: `docs/new_features_v2.3.0.md`
- Evaluation protocol: `docs/evaluation-protocol-v2.md`
- Training baseline: `docs/training-baseline-v2.md`
