# Stage 4: Test Set Final Validation - 3d

**Generated:** 2026-08-28 18:52:00

## Status

✅ **Test Set Evaluation COMPLETED**

Test set backtest executed on 100,227 samples from 2025-08-01 to 2026-08-14 (252 trading days).

## Best Configuration (from Validation)

| Parameter | Value |
|-----------|-------|
| Label Horizon | 3d |
| Positions | 10 |
| Rebalance Threshold | 0.7 |
| Commission Rate | 0.03% |
| Slippage Rate | 0.10% |

## Test Set Results

| Metric | Value |
|--------|-------|
| Total Return | -71.05% |
| Annual Return | -71.05% |
| Sharpe Ratio | -2.556 |
| Max Drawdown | -46.49% |
| Win Rate | 49.40% |
| Trading Days | 252 |

## V2 vs V3 Performance Comparison

| Version | Type | Annual Return | Sharpe | Max DD | Win Rate | Period/Notes |
|---------|------|---------------|--------|--------|----------|--------------|
| V2 | Production | 12.00% | N/A | N/A | N/A | 2026-01-26 to 2026-05-07 (real money) |
| V2 | Backtest | 41.00% | N/A | N/A | N/A | Overfitted (6400 params) |
| V2 | **Gap** | **29.00%** | - | - | - | Selection effect |
| V3 | Validation | -57.55% | -1.220 | -77.53% | 44.10% | 2022-01 to 2025-07 |
| V3 | **Test** | **-71.05%** | **-2.556** | **-46.49%** | **49.40%** | 2025-08 to 2026-08 |

## Validation-Test Performance Gap

| Metric | Validation | Test | Gap | Warning |
|--------|------------|------|-----|---------|
| Annual Return | -57.55% | -71.05% | 13.5% | ✓ (< 20%) |
| Sharpe Ratio | -1.220 | -2.556 | 1.336 | ⚠️ (> 0.5) |
| Max Drawdown | -77.53% | -46.49% | 31.0% | - |
| Win Rate | 44.10% | 49.40% | 5.3% | - |

**Gap Analysis:** Annual return gap (13.5%) is within 20% threshold BUT performance degraded further on test set.
Model consistency is poor but consistently bad across both periods.

## Key Observations

1. **V2 Overfitting Gap:** V2 shows a 29% gap between backtest (+41%) and production (+12%), indicating severe selection bias from testing 6400 parameter combinations.

2. **V3 Catastrophic Failure:** V3 shows **-71.05% annual loss** on test set, even worse than validation (-57.55%).
Model is not just unprofitable but actively destroys capital.

3. **Risk-Adjusted Returns:** Sharpe ratio of **-2.556** indicates severe risk-adjusted losses.
This is worse than validation's -1.220.

4. **Drawdown:** Maximum drawdown of -46.49% is better than validation (-77.53%) but still far exceeds acceptable limits (target < 30%).

5. **Model Direction:** Win rate of 49.40% suggests the model has almost no directional edge.
Losses are driven by poor signal quality, not just transaction costs.

## Go/No-Go Decision

**Status:** ❌ **NO-GO**

**Reason:** Test set shows catastrophic negative returns (-71.05% annual) and deeply negative Sharpe ratio (-2.556).
Model fails all production criteria.

**Criteria Check:**
- ❌ Validation-test gap < 15%: Gap is 13.5% (PASS) but irrelevant given negative returns
- ❌ Test set annual return > 0%: **FAIL (-71.05%)**
- ❌ Test set Sharpe ratio > 0.5: **FAIL (-2.556)**
- ❌ Test set max drawdown < -30%: **FAIL (-46.49%)**

**Recommendation:** **Reject for production. Fundamental model redesign required.**

## Root Cause Analysis

V3 performs worse than V2 in every aspect:
- V2 production delivered +12% (positive)
- V3 validation delivered -57.55% (deeply negative)
- V3 test delivered -71.05% (catastrophic)

**Identified Issues:**

1. **Signal Quality:** Model predictions have near-zero edge (49.40% win rate vs 50% random)

2. **Label Definition:** 3-day return labels may not capture tradeable patterns in current market regime (2022-2026)

3. **Feature Set:** 100 selected features may not contain predictive information for ranking stocks

4. **Market Regime:** Both validation (2022-2025) and test (2025-2026) periods show consistent negative performance, suggesting model doesn't work in recent market conditions

5. **Ranking vs Classification:** Model trained for binary classification but used for ranking may lose critical information

## Next Steps

**DO NOT proceed to production.**
Fundamental redesign required:

1. **Re-examine label definition:** Consider alternative return horizons (5d, 10d, 15d) and risk-adjusted labels

2. **Feature engineering overhaul:** Current feature set shows no predictive power.
Need to identify which features (if any) have actual signal

3. **Model architecture:** Consider pure ranking objectives (LambdaRank) instead of classification-then-rank

4. **Market regime analysis:** Understand why model fails in 2022-2026 period.
May need regime-specific models

5. **Simpler baseline:** Test if simple momentum/reversal strategies outperform before adding complexity

6. **Data quality audit:** Verify feature calculations, survivorship bias handling, and label generation are correct

## Reference

- **V2 Baseline:** docs/evaluation-protocol-v2.md (lines 63-67)
- **Validation Results:** v3_pipeline/results/strategy_search_3d.json
- **Validation Report:** v3_pipeline/reports/backtest_validation_3d.md
- **Test Results:** v3_pipeline/results/test_set_evaluation_3d.json
- **Config Hash:** cc86f424
