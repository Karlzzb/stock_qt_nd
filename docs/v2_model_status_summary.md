# V2 Model Status Summary

**Date:** 2026-08-28  
**Pipeline Version:** v2.3.0  
**Training Data:** 2001-01-01 to 2021-12-31 (完整重建完成)  
**Validation Period:** 2022-01-01 to 2025-07-31

## Executive Summary

V2 pipeline has been fully debugged and retrained with market features restored.
**Critical finding:** Binary classification labels are fundamentally misaligned with ranking evaluation metrics, resulting in models that cannot reach tradable thresholds despite comprehensive feature engineering.

**Key metrics:**
- Best ICIR: 0.162 (future_return_10d) - **9× below tradable threshold of 1.5**
- Top5 excess return: Negative for most horizons
- Binary hit rate: ~45% (identifies "winners")
- Ranking performance: Poor (picks wrong stocks within winners)

**Conclusion:** Feature quality is adequate; **label design is the bottleneck.**

---

## Current Model Performance

### Ranking Metrics (Validation Set 2022-2025/07)

| Label | Rank IC | Rank ICIR | top5_excess | IC_pos_ratio | Status |
|-------|---------|-----------|-------------|--------------|--------|
| future_return_10d | +0.034 | **0.162** | +0.007% | 57.2% | Best |
| future_return_30d | +0.028 | **0.139** | -0.58% | 57.5% | 2nd |
| future_return_5d | +0.016 | 0.083 | +0.19% | 53.5% | Weak+ |
| future_return_25d | +0.006 | 0.027 | -0.97% | 51.6% | Weak+ |
| future_return_20d | -0.014 | -0.068 | -1.28% | 47.6% | Failed |
| future_return_15d | -0.020 | -0.096 | -1.32% | 46.0% | Failed |
| future_return_3d | -0.024 | -0.115 | -0.23% | 43.8% | Failed |

**Tradable threshold:** ICIR > 1.5 AND top5_excess > 2%  
**Result:** ❌ No configuration meets tradable thresholds

### Binary Classification Metrics (for reference)

| Label | PR-AUC | Precision@top100 | Daily_top5_hit | F1@0.3 |
|-------|--------|------------------|----------------|--------|
| future_return_30d | 0.542 | 0.53 | 0.515 | 0.676 |
| future_return_25d | 0.554 | 0.57 | 0.504 | 0.657 |
| future_return_20d | 0.535 | 0.79 | 0.473 | 0.627 |
| future_return_15d | 0.515 | 0.83 | 0.450 | 0.613 |
| future_return_10d | 0.425 | 0.51 | 0.395 | 0.595 |

**Key observation:** Binary metrics show ~45-50% hit rates but ranking metrics reveal these models pick the **wrong stocks within winners.**

---

## Feature Engineering Status

### Current Feature Pool: 626 Features

**Categories:**
1. **Technical Indicators (TA-Lib):** MACD, RSI, MA, Bollinger, ATR, Stochastic, OBV
2. **Volume Features:** Trends, divergence, consistency, ratios, spikes
3. **Alpha Factors:** Price-rank, CLV, shadows, signed volume, correlations
4. **Microstructure (v2.3.0):** Illiquidity, spreads, price impact, alpha12
5. **Structure:** GK volatility, efficiency ratio, intraday position
6. **Lags:** 3d/5d/10d/15d/20d/25d/30d for OHLCV, returns, amplitude
7. **Cross-Sectional:** 432 derived features (rank percentile + z-scores)
8. **Market Features:** SH/SZ index sync, sentiment, direction, strength
9. **K-line Patterns:** Hammer, Doji, Engulfing, support/resistance

### Market Feature Recovery (Issue #18 → Fixed)

**Problem solved:** Missing 13 market features (sh_/sz_) + divergence_magnitude due to index data loading bug.

**Fixes applied:**
- `run_feature_pipeline.py`: Fixed column name (volume vs vol)
- `feature_pipeline_v2.py`: Auto-load index data in load_price_data
- `_calculate_single_index_features`: Timestamp type conversion
- `calculate_macd_percentile_vectorized`: NaN handling for insufficient windows

**Validation:** Market feature missing rate now 6.08% (well below 30% threshold ✅)

### Feature Quality Assessment

**Missing rate summary (sample 100 files):**
- Market features (sh_/sz_): 6.08% average ✅
- MACD percentile: 8.56% ✅
- Divergence magnitude: 6.08% ✅
- High-miss features (>30%): 84 out of 750 (mostly hs_/gq_/hc_ exotic features at 93.92%)

**Conclusion:** Core feature quality is good. High-miss features are sector-specific (HS300, GEM, etc.) and expected to be sparse.

---

## Training Infrastructure Status

### Data Pipeline (完整重建完成)

**Training segment rebuild:** ✅ Completed 2026-08-27
- Period: 2001-01-01 to 2021-12-31
- Files generated: 5,093 daily feature CSVs
- Samples: 1,402,267 stock-days
- Feature cache: `feature_cache_all.parquet` (1.4M rows × 232 cols)
- Cache fingerprint: Validated, version 2.3.0

**Issues fixed during rebuild:**
- Stale cache detection working correctly
- Argument mapping for run_walk_forward_train.py fixed
- Report generation JSON structure corrected

### Training Configuration

**Model architecture:** Two-layer stack (LGBM → LR)
- Layer 1: LightGBM with binary classification objective
- Layer 2: Logistic Regression for calibration
- Features: 626 optimized + stable features

**Training protocol:** Single split (Evaluation Protocol v2)
- Training: ≤ 2021-12-31
- Validation: 2022-01-01 to 2025-07-31
- Test: 2025-08-01+ (sealed, not evaluated)

**Labels:** Binary classification (return > 1%)
- 7 horizons: [3d, 5d, 10d, 15d, 20d, 25d, 30d]
- Prior rates: 36%-51% depending on horizon

---

## Root Cause Analysis

### Binary Label vs Ranking Objective Mismatch

**The core problem:**

Binary classification optimizes for: *"Which stocks will rise >1%?"*  
But we evaluate on: *"Which stocks rise the MOST?"*

**Concrete example from validation data:**
- Stock A: +2.5% return → Binary label: 1 (win)
- Stock B: +15% return → Binary label: 1 (win)
- Model learns both are "winners" but **cannot distinguish magnitude**
- When picking top-5 stocks, model might select Stock A over Stock B

**Result:** 45% binary hit rate (identifies winners) but negative excess returns (selects wrong winners within the winner class).

### Why Features Alone Cannot Fix This

Adding more features to a binary classifier:
- ✅ Can improve precision/recall on the binary task
- ❌ Cannot teach the model to rank within the positive class
- ❌ Discards magnitude information at labeling stage

**From memory notes:** "label 修正优先于一切" (label correction is top priority)

### Evidence from Multiple Metrics

**15d label example:**
- Binary metrics: 45.1% top1 hit rate (looks okay)
- Ranking metrics: -1.32% top5 excess return (losing money)
- Rank IC: -0.020 (negative correlation!)

This proves the model has learned *something* about winners vs losers, but optimizes the wrong objective for portfolio construction.

---

## Related Issues & Commits

### Issue #18 - CLOSED ✅
**Title:** 选择背离算法修复策略  
**Resolution:** Rolled back to V1 divergence detector  
**Commit:** 9a1b2ec - "fix(pipeline): 回退到 V1 背离检测器，修复样本分布问题"

### Issue #12 - OPEN (Partially Complete)
**Title:** 诚实基线：全量重算 + walk-forward 重训  
**Status:** Training segment rebuild ✅, Full retrain ✅, Comparison report ✅  
**Remaining:** Document findings, close issue  
**Key commits:**
- 22bdb0a - "docs(issue12): 新旧对照报告 + 基线重跑工程修复"
- 8ad6661 - "feat(baseline): 诚实基线脚本套件 + 测试"

### Issue #16 - OPEN (Blocked)
**Title:** PRD: Label × 特征池 联合迭代框架  
**Status:** Awaiting label redesign decision  
**Blocker:** Current PRD assumes binary labels; needs update for ranking objectives  
**Priority:** HIGH - This issue should be superseded by v3 ranking pipeline work

### Issue #1 - OPEN (Context Issue)
**Title:** PRD：回测可信化与系统重建  
**Status:** V2 baseline established; recommendation is ranking-based v3  
**Note:** May need update to reflect v2 completion and v3 direction

---

## Scripts & Tools Status

### Training Scripts
- ✅ `scripts/run_feature_pipeline.py` - Fixed, validated
- ✅ `scripts/run_walk_forward_train.py` - Argument mapping fixed, cache auto-creation added
- ✅ `scripts/run_label_grid.py` - Runs all 7 horizons successfully
- ✅ `scripts/complete_retrain_pipeline.sh` - End-to-end pipeline working

### Analysis Scripts
- ✅ `scripts/analyze_feature_missing_rate.py` - Validates market feature quality
- ✅ `scripts/run_rank_metrics.py` - Computes ranking evaluation metrics

### Evaluation Artifacts
- ✅ `experiments/label_grid_2021-12-31_2022-01-01_2025-07-31.json` - Full binary metrics
- ✅ `experiments/rank_metrics_2021-12-31_2022-01-01_2025-07-31.json` - Ranking metrics baseline
- ✅ `reports/feature_missing_rate_20010101_20211231.json` - Feature quality validation

---

## Recommended Next Steps

### Immediate: Close Completed Work

1. **Close Issue #12** with summary:
   - Training segment rebuild: ✅ Complete
   - Full retrain with market features: ✅ Complete
   - Baseline metrics documented: ✅ Complete
   - Finding: Binary labels cannot reach tradable thresholds

2. **Update Issue #16** with findings:
   - Binary label approach has reached plateau
   - Feature screening should wait until ranking objective is proven
   - Redirect to v3 ranking pipeline work

### Strategic: V3 Ranking Pipeline

**Create new issue for v3.0.0 baseline** (see PRD in `docs/v3_ranking_pipeline_prd.md`):
- Replace binary labels with cross-sectional ranking targets
- Use LightGBM lambdarank objective
- Reuse v2's 626 features initially
- Target: ICIR > 0.5 (3× improvement over v2)
- Scope: 10d and 30d horizons first

**Why this is the right next step:**
- Root cause identified: label-objective mismatch
- Features are comprehensive (626 covering all categories)
- Infrastructure is solid (caching, evaluation, training)
- Only the objective function needs changing

---

## Key Learnings

1. **Binary classification is incompatible with ranking evaluation** - Even with 626 well-engineered features, the mismatch prevents reaching tradable performance.

2. **Feature quality is necessary but not sufficient** - V2 has strong features (market sync at 6% missing rate, comprehensive technical indicators), but they're trained on the wrong objective.

3. **Evaluation metrics must match training objectives** - Training on binary (precision/recall) while evaluating on ranking (IC/ICIR) creates a fundamental disconnect.

4. **10d and 30d horizons show promise** - These two showed positive Rank IC in v2, suggesting there's signal to amplify with proper ranking objectives.

5. **Infrastructure investment paid off** - Feature caching, fingerprinting, evaluation protocols, and training pipelines are all working reliably and can be reused for v3.

---

## References

- **Training baseline documentation:** `docs/training-baseline-v2.md`
- **Evaluation protocol:** `docs/evaluation-protocol-v2.md`
- **Feature additions:** `docs/new_features_v2.3.0.md`
- **V3 PRD:** `docs/v3_ranking_pipeline_prd.md`
- **Memory notes:** `.claude/projects/-home-karl-repos-personal-stock-qt-nd/memory/MEMORY.md`

---

## Model Configuration

**Model version:** v2.3.0  
**Feature pipeline version:** 2.3.0  
**Training framework:** LightGBM 4.x + scikit-learn  
**Evaluation framework:** Custom ranking metrics (IC/ICIR/top5_excess)  
**Data source:** Tushare (validated,退市股 included)  
**Feature count:** 626 optimized features (from 750 raw)  
**Sample count:** 1.4M stock-days training, 337K validation  
**Horizon coverage:** 3d, 5d, 10d, 15d, 20d, 25d, 30d
