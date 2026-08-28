#!/usr/bin/env python3
"""Train LightGBM lambdarank models for V3 ranking pipeline.

Loads v3 feature cache with ranking labels, trains lambdarank models
for specified horizons, and generates out-of-sample predictions.

Usage:
    python v3_pipeline/scripts/train_ranking.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Column name constants
TIMESTAMP_COL = "timestamp"
SYMBOL_COL = "symbol"
DATE_COL = "date"


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _deduplicate_and_log(
    df: pd.DataFrame,
    subset: list[str],
    keep: str = "last"
) -> pd.DataFrame:
    """Remove duplicates and log the operation.

    Args:
        df: DataFrame to deduplicate
        subset: Column names to identify duplicates
        keep: Which duplicates to keep ('first', 'last', or False)

    Returns:
        Deduplicated DataFrame
    """
    before_dedup = len(df)
    df_deduped = df.drop_duplicates(subset=subset, keep=keep)
    after_dedup = len(df_deduped)

    if before_dedup > after_dedup:
        removed = before_dedup - after_dedup
        pct = removed / before_dedup
        logger.warning(
            f"Removed {removed} duplicate {tuple(subset)} pairs ({pct:.1%})"
        )

    return df_deduped


def prepare_features(df: pd.DataFrame, exclude_patterns: list[str]) -> list[str]:
    """Select features by excluding patterns from config."""
    import re

    all_cols = df.columns.tolist()
    features = []

    # Additional exclusion patterns for non-feature columns
    system_patterns = [
        r".*_date$",  # Any date columns (prev_time, detection_date, etc.)
        r".*_signal$",  # Signal columns (volume_signal, etc.)
        r"^prev_",  # Previous values that are metadata
        r"^rank_future_return_",  # Ranking labels (future information)
        r"^future_return_",  # Raw future returns (future information)
    ]

    for col in all_cols:
        # Skip non-feature columns
        if col in [TIMESTAMP_COL, SYMBOL_COL, DATE_COL]:
            continue

        # Check system exclusion patterns
        excluded = False
        for pattern in system_patterns:
            if re.match(pattern, col):
                excluded = True
                logger.debug(f"Dropped {col}: matches system pattern {pattern}")
                break

        if excluded:
            continue

        # Check config exclusion patterns
        for pattern in exclude_patterns:
            if re.match(pattern, col):
                excluded = True
                break

        if not excluded:
            features.append(col)

    # Filter out features with too much missing data and non-numeric columns
    valid_features = []
    for feat in features:
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(df[feat]):
            logger.debug(f"Dropped {feat}: non-numeric type {df[feat].dtype}")
            continue

        missing_rate = df[feat].isna().mean()
        if missing_rate < 0.30:
            valid_features.append(feat)
        else:
            logger.debug(f"Dropped {feat}: {missing_rate:.2%} missing")

    logger.info(f"Selected {len(valid_features)} features from {len(all_cols)} columns")
    return valid_features


def prepare_ranking_dataset(
    df: pd.DataFrame,
    features: list[str],
    rank_col: str,
    imputer: Optional[SimpleImputer] = None
) -> tuple[lgb.Dataset, np.ndarray, SimpleImputer]:
    """Prepare LightGBM Dataset with group information for ranking.

    Args:
        df: DataFrame with features and rank labels
        features: List of feature column names
        rank_col: Name of rank label column
        imputer: Optional pre-fitted imputer. If None, creates and fits new imputer.

    Returns:
        lgb.Dataset: Training dataset with group sizes
        np.ndarray: Group sizes (number of stocks per day)
        SimpleImputer: The imputer used (fitted if new, passed-through if provided)
    """
    # Sort by timestamp to ensure proper grouping
    df = df.sort_values(TIMESTAMP_COL).copy()

    # Remove duplicates first - keep last occurrence per (timestamp, symbol)
    df = _deduplicate_and_log(df, subset=[TIMESTAMP_COL, SYMBOL_COL], keep="last")

    # Filter out rows with missing rank labels
    valid = df[rank_col].notna()
    df = df[valid].copy()
    logger.info(f"After filtering NaN ranks: {len(df)} rows ({valid.mean():.2%} coverage)")

    # Get group sizes (stocks per day)
    group_sizes = df.groupby(TIMESTAMP_COL).size().values
    logger.info(f"Groups: {len(group_sizes)} days, avg {group_sizes.mean():.1f} stocks/day")

    # Prepare feature matrix - select only numeric columns and handle inf/nan
    X = df[features].values.astype(float)
    y_ranks = df[rank_col].values

    # Replace inf with nan
    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

    # Handle missing values in features
    if imputer is None:
        # Create and fit new imputer (training mode)
        imputer = SimpleImputer(strategy="median", copy=False)
        X = imputer.fit_transform(X)
    else:
        # Use pre-fitted imputer (validation/test mode)
        X = imputer.transform(X)

    # Convert ranks to relevance labels for LambdaRank
    # LightGBM ranking expects integer labels representing relevance levels
    # Convert [0, 1] ranks to integer relevance: 0-4 (5 levels)
    y = np.digitize(y_ranks, bins=[0.2, 0.4, 0.6, 0.8], right=False)

    # Create LightGBM dataset with group information
    dataset = lgb.Dataset(X, label=y, group=group_sizes, free_raw_data=False)

    return dataset, group_sizes, imputer


def train_horizon(
    config: dict,
    cache_df: pd.DataFrame,
    features: list[str],
    horizon: str,
    output_dir: Path,
) -> dict:
    """Train lambdarank model for one horizon.

    Returns:
        dict: Training metadata including paths and metrics
    """
    logger.info(f"=" * 60)
    logger.info(f"Training horizon: {horizon}")
    logger.info(f"=" * 60)

    train_cfg = config["training"]
    model_cfg = config["model"]

    # Prepare train/val split
    train_mask = cache_df[TIMESTAMP_COL] <= train_cfg["train_end"]
    val_mask = (cache_df[TIMESTAMP_COL] >= train_cfg["val_start"]) & (
        cache_df[TIMESTAMP_COL] <= train_cfg["val_end"]
    )

    train_df = cache_df[train_mask].copy()
    val_df = cache_df[val_mask].copy()

    logger.info(f"Train: {len(train_df)} rows, {train_df[TIMESTAMP_COL].min()} to {train_df[TIMESTAMP_COL].max()}")
    logger.info(f"Val: {len(val_df)} rows, {val_df[TIMESTAMP_COL].min()} to {val_df[TIMESTAMP_COL].max()}")

    rank_col = f"rank_future_return_{horizon}"

    # Prepare datasets
    logger.info("Preparing training dataset...")
    train_data, train_groups, imputer = prepare_ranking_dataset(train_df, features, rank_col)

    logger.info("Preparing validation dataset...")
    val_data, val_groups, _ = prepare_ranking_dataset(val_df, features, rank_col, imputer=imputer)

    # Configure LightGBM parameters
    params = {
        "objective": model_cfg["objective"],
        "metric": model_cfg["metric"],
        "ndcg_eval_at": model_cfg["ndcg_eval_at"],
        "num_leaves": model_cfg["num_leaves"],
        "max_depth": model_cfg["max_depth"],
        "learning_rate": model_cfg["learning_rate"],
        "feature_fraction": model_cfg["feature_fraction"],
        "bagging_fraction": model_cfg["bagging_fraction"],
        "bagging_freq": model_cfg["bagging_freq"],
        "min_data_in_leaf": model_cfg["min_data_in_leaf"],
        "lambda_l1": model_cfg["lambda_l1"],
        "lambda_l2": model_cfg["lambda_l2"],
        "verbose": -1,
    }

    logger.info(f"LightGBM params: {params}")

    # Train model
    logger.info("Training lambdarank model...")
    evals_result = {}

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=model_cfg["num_boost_round"],
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(model_cfg["early_stopping_rounds"]),
            lgb.log_evaluation(model_cfg["verbose"]),
            lgb.record_evaluation(evals_result),
        ],
    )

    logger.info(f"Best iteration: {booster.best_iteration}")
    logger.info(f"Best score: {booster.best_score}")

    # Save model
    model_path = output_dir / f"model_{horizon}.txt"
    booster.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Generate predictions on validation set
    logger.info("Generating validation predictions...")
    val_clean = val_df[val_df[rank_col].notna()].copy()

    # Remove duplicates - keep last occurrence (most recent data)
    val_clean = _deduplicate_and_log(val_clean, subset=[TIMESTAMP_COL, SYMBOL_COL], keep="last")

    X_val = val_clean[features].values
    X_val = imputer.transform(X_val)

    predictions = booster.predict(X_val, num_iteration=booster.best_iteration)

    # Save predictions with metadata
    pred_df = pd.DataFrame({
        TIMESTAMP_COL: val_clean[TIMESTAMP_COL].values,
        SYMBOL_COL: val_clean[SYMBOL_COL].values,
        "prediction": predictions,
        "actual_rank": val_clean[rank_col].values,
        "actual_return": val_clean[f"future_return_{horizon}"].values,
    })

    pred_path = output_dir / f"pred_{horizon}.parquet"
    pred_df.to_parquet(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path} ({len(pred_df)} rows)")

    # Save imputer
    import pickle
    imputer_path = output_dir / f"imputer_{horizon}.pkl"
    with open(imputer_path, "wb") as f:
        pickle.dump(imputer, f)

    return {
        "horizon": horizon,
        "model_path": str(model_path),
        "pred_path": str(pred_path),
        "best_iteration": booster.best_iteration,
        "best_score": booster.best_score,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "val_pred_rows": len(pred_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3 ranking models")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/configs/v3_0_0_baseline.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/feature_cache_v3.parquet",
        help="Path to v3 feature cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/models/v3_0_0_baseline",
        help="Output directory for models and predictions",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output}")

    # Load feature cache
    logger.info(f"Loading feature cache from {args.cache}")
    cache_df = pd.read_parquet(args.cache)
    logger.info(f"Loaded {len(cache_df)} rows × {len(cache_df.columns)} columns")
    logger.info(f"Date range: {cache_df[TIMESTAMP_COL].min()} to {cache_df[TIMESTAMP_COL].max()}")

    # Prepare features
    exclude_patterns = config["features"]["exclude_patterns"]
    features = prepare_features(cache_df, exclude_patterns)

    # Save feature list
    feature_list_path = args.output / "features.txt"
    feature_list_path.write_text("\n".join(features))
    logger.info(f"Feature list saved to {feature_list_path}")

    # Train each horizon
    horizons = config["training"]["horizons"]
    results = []

    for horizon in horizons:
        try:
            result = train_horizon(config, cache_df, features, horizon, args.output)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {horizon}: {e}", exc_info=True)
            continue

    # Save training summary
    import json
    summary_path = args.output / "training_summary.json"
    summary = {
        "config": str(args.config),
        "cache": str(args.cache),
        "n_features": len(features),
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training complete! Summary saved to {summary_path}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Trained {len(results)} horizons: {', '.join(r['horizon'] for r in results)}")


if __name__ == "__main__":
    main()
