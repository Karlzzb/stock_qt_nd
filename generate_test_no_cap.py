#!/usr/bin/env python3
"""在测试集上验证无截断模型"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Paths
CACHE_PATH = REPO_ROOT / "v3_pipeline/feature_cache_v3_no_cap.parquet"
MODEL_DIR = REPO_ROOT / "v3_pipeline/models/v3_0_3_no_cap"

TEST_START = "2025-08-01"
TEST_END = "2026-08-14"

print("=" * 80)
print("NO-CAP MODEL: TEST SET PREDICTION")
print("=" * 80)

# Load cache
print(f"\nLoading cache from {CACHE_PATH}...")
cache = pd.read_parquet(CACHE_PATH)

# Filter test set
test_mask = (cache['timestamp'] >= TEST_START) & (cache['timestamp'] <= TEST_END)
test_df = cache[test_mask].copy()

print(f"Test set: {len(test_df):,} rows")
print(f"Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

# Load features
features_path = MODEL_DIR / "features.txt"
features = features_path.read_text().strip().split("\n")
print(f"Features: {len(features)}")

# Load model and imputer for 3d
model_path = MODEL_DIR / "model_3d.txt"
imputer_path = MODEL_DIR / "imputer_3d.pkl"

print(f"\nLoading model from {model_path}...")
booster = lgb.Booster(model_file=str(model_path))

print(f"Loading imputer from {imputer_path}...")
with open(imputer_path, "rb") as f:
    imputer = pickle.load(f)

# Prepare test data
rank_col = "rank_future_return_3d"
return_col = "future_return_3d"

test_clean = test_df[test_df[rank_col].notna()].copy()
test_clean = test_clean.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

print(f"Valid test rows: {len(test_clean):,}")

# Prepare features
X_test = test_clean[features].values.astype(float)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=np.nan, neginf=np.nan)
X_test = imputer.transform(X_test)

# Generate predictions
print("Generating predictions...")
predictions = booster.predict(X_test)

# Save
pred_df = pd.DataFrame({
    'timestamp': test_clean['timestamp'].values,
    'symbol': test_clean['symbol'].values,
    'prediction': predictions,
    'actual_rank': test_clean[rank_col].values,
    'actual_return': test_clean[return_col].values,
})

output_path = MODEL_DIR / "test_pred_3d.parquet"
pred_df.to_parquet(output_path, index=False)

print(f"\n✓ Saved {len(pred_df):,} predictions to {output_path}")
print(f"\nPrediction stats:")
print(f"  Min: {predictions.min():.6f}")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Max: {predictions.max():.6f}")
