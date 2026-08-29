#!/usr/bin/env python3
"""
在测试集上生成V3模型预测（2025-08-01 to 2026-08-14）。

使用已训练的模型，在真实的hold-out测试集上生成预测，
用于验证模型的泛化能力和策略表现。
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Paths
CACHE_PATH = REPO_ROOT / "v3_pipeline/feature_cache_v3.parquet"
MODEL_DIR = REPO_ROOT / "v3_pipeline/models/v3_0_1_label_selection"

# Test period
TEST_START = "2025-08-01"
TEST_END = "2026-08-14"

TIMESTAMP_COL = "timestamp"
SYMBOL_COL = "symbol"

print("=" * 80)
print("GENERATING TEST SET PREDICTIONS")
print("=" * 80)

# Load cache
print(f"\nLoading feature cache from {CACHE_PATH}...")
cache = pd.read_parquet(CACHE_PATH)
print(f"Total rows: {len(cache):,}")

# Filter test set
test_mask = (cache[TIMESTAMP_COL] >= TEST_START) & (cache[TIMESTAMP_COL] <= TEST_END)
test_df = cache[test_mask].copy()
print(f"\nTest set: {len(test_df):,} rows")
print(f"Date range: {test_df[TIMESTAMP_COL].min()} to {test_df[TIMESTAMP_COL].max()}")
print(f"Unique dates: {test_df[TIMESTAMP_COL].nunique()}")

# Load features
features_path = MODEL_DIR / "features.txt"
features = features_path.read_text().strip().split("\n")
print(f"\nLoaded {len(features)} features")

# Process each horizon
horizons = ['3d', '5d', '10d', '15d', '20d', '25d', '30d']

for horizon in horizons:
    print(f"\n{'=' * 80}")
    print(f"Processing horizon: {horizon}")
    print(f"{'=' * 80}")

    # Load model and imputer
    model_path = MODEL_DIR / f"model_{horizon}.txt"
    imputer_path = MODEL_DIR / f"imputer_{horizon}.pkl"

    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        continue

    print(f"Loading model from {model_path}...")
    booster = lgb.Booster(model_file=str(model_path))

    print(f"Loading imputer from {imputer_path}...")
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)

    # Prepare test data
    rank_col = f"rank_future_return_{horizon}"
    return_col = f"future_return_{horizon}"

    # Filter valid rows (non-NaN labels)
    test_clean = test_df[test_df[rank_col].notna()].copy()

    # Remove duplicates - keep last
    before = len(test_clean)
    test_clean = test_clean.drop_duplicates(subset=[TIMESTAMP_COL, SYMBOL_COL], keep="last")
    after = len(test_clean)
    if before > after:
        print(f"Removed {before - after} duplicates")

    print(f"Valid test rows: {len(test_clean):,}")

    # Prepare features
    X_test = test_clean[features].values.astype(float)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X_test = imputer.transform(X_test)

    # Generate predictions
    print("Generating predictions...")
    predictions = booster.predict(X_test)

    # Save predictions
    pred_df = pd.DataFrame({
        TIMESTAMP_COL: test_clean[TIMESTAMP_COL].values,
        SYMBOL_COL: test_clean[SYMBOL_COL].values,
        "prediction": predictions,
        "actual_rank": test_clean[rank_col].values,
        "actual_return": test_clean[return_col].values,
    })

    output_path = MODEL_DIR / f"test_pred_{horizon}.parquet"
    pred_df.to_parquet(output_path, index=False)
    print(f"✓ Saved {len(pred_df):,} predictions to {output_path}")

    # Quick stats
    print(f"\nPrediction stats:")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Std: {predictions.std():.6f}")

print(f"\n{'=' * 80}")
print("TEST SET PREDICTION GENERATION COMPLETE")
print(f"{'=' * 80}")
print(f"\nAll predictions saved to {MODEL_DIR}/test_pred_*.parquet")
print("\nNext: Run strategy backtest on test set to validate performance.")
