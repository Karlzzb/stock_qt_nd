#!/usr/bin/env python3 -u
"""
Stage 2: Feature Screening with Permutation Importance
Reduces feature set while maintaining 95%+ of baseline ICIR.
"""

import sys
import json
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rank_metrics import daily_rank_metrics


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(cache_path, train_start, train_end, val_start, val_end):
    """Load and split data into train/validation sets."""
    print(f"Loading feature cache from {cache_path}...")
    df = pd.read_parquet(cache_path)

    # Convert timestamp column to datetime and rename to date for consistency
    df['date'] = pd.to_datetime(df['timestamp'])

    # Rename symbol to stock_code for consistency
    if 'symbol' in df.columns:
        df['stock_code'] = df['symbol']

    # Split into train and validation
    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()

    print(f"Train set: {len(train_df)} rows, {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Val set: {len(val_df)} rows, {val_df['date'].min()} to {val_df['date'].max()}")

    return train_df, val_df


def get_feature_columns(df, label_horizon):
    """Extract feature columns (exclude identifiers and labels)."""
    exclude_patterns = ['date', 'stock_code', 'timestamp', 'symbol',
                       'future_return', 'rank_future_return', 'future_sell',
                       'stop_loss', 'detection_date', 'prev_time']

    # Get candidate feature columns
    candidate_cols = [col for col in df.columns if not any(pat in col for pat in exclude_patterns)]

    # Filter to only numeric columns
    feature_cols = [col for col in candidate_cols if df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

    return feature_cols


def remove_collinear_features(X, feature_names, vif_threshold, corr_threshold):
    """Remove highly collinear features using VIF and correlation analysis."""
    print("\n=== Collinearity Analysis ===")
    print(f"Working with {len(X)} samples and {len(feature_names)} features")

    # Step 1: Remove features with correlation > threshold
    # Use sampling for large datasets to speed up correlation computation
    print(f"Step 1: Removing features with |correlation| > {corr_threshold}")

    sample_size = min(50000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    print(f"  - Using {sample_size} samples for correlation analysis")

    corr_matrix = X_sample.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop_corr = set()
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > corr_threshold):
            to_drop_corr.add(column)

    print(f"  - Found {len(to_drop_corr)} highly correlated features")

    X_filtered = X.drop(columns=list(to_drop_corr))
    remaining_features = [f for f in feature_names if f not in to_drop_corr]

    # Step 2: Remove features with VIF > threshold
    print(f"Step 2: Removing features with VIF > {vif_threshold}")

    # Calculate VIF for remaining features (sample for speed)
    sample_size = min(10000, len(X_filtered))
    X_sample = X_filtered.sample(n=sample_size, random_state=42)
    print(f"  - Using {sample_size} samples for VIF analysis")

    to_drop_vif = set()
    max_iterations = 10

    for iteration in range(max_iterations):
        if len(X_sample.columns) == 0:
            break

        vif_data = pd.DataFrame()
        vif_data["feature"] = X_sample.columns

        # Calculate VIF with error handling for numerical instability
        vif_values = []
        for i in range(len(X_sample.columns)):
            try:
                vif = variance_inflation_factor(X_sample.values, i)
                # Handle inf/nan values
                if np.isnan(vif) or np.isinf(vif):
                    vif = 999999  # Very high VIF for problematic features
            except (np.linalg.LinAlgError, ValueError) as e:
                # Numerical instability - mark for removal
                vif = 999999
            vif_values.append(vif)

        vif_data["VIF"] = vif_values

        max_vif = vif_data["VIF"].max()
        if max_vif <= vif_threshold:
            break

        # Remove feature with highest VIF
        worst_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        to_drop_vif.add(worst_feature)
        X_sample = X_sample.drop(columns=[worst_feature])

        if max_vif < 999999:
            print(f"  - Iteration {iteration + 1}: Removed {worst_feature} (VIF={max_vif:.2f})")
        else:
            print(f"  - Iteration {iteration + 1}: Removed {worst_feature} (VIF=unstable)")

    print(f"  - Total features removed by VIF: {len(to_drop_vif)}")

    # Apply VIF filtering to full dataset
    X_filtered = X_filtered.drop(columns=list(to_drop_vif))
    remaining_features = [f for f in remaining_features if f not in to_drop_vif]

    total_removed = len(to_drop_corr) + len(to_drop_vif)
    print(f"\nCollinearity cleanup: {len(feature_names)} → {len(remaining_features)} features ({total_removed} removed)")

    return X_filtered, remaining_features, {
        'removed_by_correlation': list(to_drop_corr),
        'removed_by_vif': list(to_drop_vif),
        'remaining_count': len(remaining_features)
    }


def prepare_lgb_dataset(X, y, feature_names, group_sizes, max_group_size=10000):
    """Prepare LightGBM dataset with group information.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: Feature names
        group_sizes: Group sizes (stocks per date)
        max_group_size: Maximum allowed group size for LightGBM ranking (default 10000)

    Returns:
        lgb.Dataset with capped group sizes
    """
    # Convert [0, 1] ranks to integer relevance labels for LambdaRank
    # LightGBM ranking expects integer labels representing relevance levels
    # Convert [0, 1] ranks to integer relevance: 0-4 (5 levels)
    y_relevance = np.digitize(y, bins=[0.2, 0.4, 0.6, 0.8], right=False)

    # Cap group sizes if any exceed max_group_size
    if any(g > max_group_size for g in group_sizes):
        print(f"  Warning: {sum(g > max_group_size for g in group_sizes)} groups exceed {max_group_size}, sampling...")

        # Sample from large groups
        new_X = []
        new_y = []
        new_groups = []

        idx = 0
        for group_size in group_sizes:
            group_X = X[idx:idx+group_size]
            group_y = y_relevance[idx:idx+group_size]

            if group_size > max_group_size:
                # Random sample within this group
                sample_indices = np.random.choice(group_size, max_group_size, replace=False)
                group_X = group_X[sample_indices]
                group_y = group_y[sample_indices]
                new_groups.append(max_group_size)
            else:
                new_groups.append(group_size)

            new_X.append(group_X)
            new_y.append(group_y)
            idx += group_size

        X = np.vstack(new_X)
        y_relevance = np.concatenate(new_y)
        group_sizes = new_groups

    return lgb.Dataset(
        X,
        label=y_relevance,
        feature_name=feature_names,
        group=group_sizes,
        free_raw_data=False
    )


def train_model(train_X, train_y, train_groups, val_X, val_y, val_groups,
                feature_names, lgbm_params, num_boost_round, early_stopping_rounds, verbose_eval):
    """Train a LightGBM lambdarank model."""

    train_data = prepare_lgb_dataset(train_X, train_y, feature_names, train_groups)
    val_data = prepare_lgb_dataset(val_X, val_y, feature_names, val_groups)

    model = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=verbose_eval)
        ]
    )

    return model


def compute_feature_importance(model, feature_names):
    """Compute feature importance using LightGBM's built-in gain importance."""
    print("\n=== Computing Feature Importance ===")
    print("Using LightGBM gain-based importance (much faster than permutation)...")

    # Get feature importance from model
    importance_scores = model.feature_importance(importance_type='gain')

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importance_scores,
        'importance_std': 0.0  # No std for gain-based importance
    }).sort_values('importance_mean', ascending=False)

    # Normalize to [0, 1] for interpretability
    if importance_df['importance_mean'].max() > 0:
        importance_df['importance_mean'] = importance_df['importance_mean'] / importance_df['importance_mean'].max()

    print(f"\nTop 20 most important features:")
    for idx, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance_mean']:.4f}")

    return importance_df


def evaluate_model_icir(model, val_X, val_df, label_col):
    """Evaluate model ICIR on validation set."""
    preds = model.predict(val_X)

    # Get the corresponding future return column
    # Extract horizon from label_col (e.g., "rank_future_return_3d" -> "3d")
    horizon = label_col.replace('rank_future_return_', '')
    return_col = f'future_return_{horizon}'

    # Create evaluation DataFrame
    eval_df = val_df[['date', 'stock_code', return_col]].copy()
    eval_df['pred'] = preds

    # Compute daily Rank IC (Spearman correlation between predictions and actual returns)
    ic_values = []
    top5_values = []

    for date, group in eval_df.groupby('date'):
        # Filter out missing returns
        group = group[group[return_col].notna()]

        if len(group) >= 5:
            # Compute Rank IC (predictions vs actual returns)
            ic, _ = spearmanr(group['pred'], group[return_col])
            if not np.isnan(ic):
                ic_values.append(ic)

            # Compute top 5% excess return (using actual returns)
            top5_threshold = group['pred'].quantile(0.95)
            top5_mask = group['pred'] >= top5_threshold
            if top5_mask.sum() > 0:
                top5_avg = group.loc[top5_mask, return_col].mean()
                all_avg = group[return_col].mean()
                top5_values.append(top5_avg - all_avg)

    overall = {
        'rank_ic': np.mean(ic_values) if ic_values else 0.0,
        'icir': (np.mean(ic_values) / (np.std(ic_values) + 1e-10)) if ic_values else 0.0,
        'ic_positive_rate': np.mean([ic > 0 for ic in ic_values]) if ic_values else 0.0,
        'top5_excess_annual': np.mean(top5_values) * 252 if top5_values else 0.0
    }

    return overall


def test_feature_subsets(train_df, val_df, feature_names, importance_df,
                        subset_sizes, label_col, config):
    """Test different feature subset sizes and evaluate performance."""
    print("\n=== Testing Feature Subsets ===")

    results = []

    for n_features in subset_sizes:
        print(f"\nTesting top {n_features} features...")

        # Select top N features
        selected_features = importance_df.head(n_features)['feature'].tolist()

        # Prepare data
        train_X = train_df[selected_features].values
        train_y = train_df[label_col].values
        train_groups = train_df.groupby('date').size().tolist()

        val_X = val_df[selected_features].values
        val_y = val_df[label_col].values
        val_groups = val_df.groupby('date').size().tolist()

        # Train model
        model = train_model(
            train_X, train_y, train_groups,
            val_X, val_y, val_groups,
            selected_features,
            config['lgbm_params'],
            config['training']['num_boost_round'],
            config['training']['early_stopping_rounds'],
            config['training']['verbose_eval']
        )

        # Evaluate
        metrics = evaluate_model_icir(model, val_X, val_df, label_col)

        result = {
            'n_features': n_features,
            'features': selected_features,
            'metrics': metrics,
            'model': model
        }
        results.append(result)

        print(f"  ICIR: {metrics['icir']:.3f}")
        print(f"  Rank IC: {metrics['rank_ic']:.4f}")
        print(f"  IC Positive Rate: {metrics['ic_positive_rate']:.1%}")

    return results


def select_best_subset(results, baseline_icir, min_retention):
    """Select the smallest feature subset that meets performance target."""
    target_icir = baseline_icir * min_retention

    print(f"\n=== Selecting Best Subset ===")
    print(f"Baseline ICIR: {baseline_icir:.3f}")
    print(f"Target ICIR (≥{min_retention:.1%} of baseline): {target_icir:.3f}")

    # Find subsets that meet target
    valid_subsets = [r for r in results if r['metrics']['icir'] >= target_icir]

    if not valid_subsets:
        print("WARNING: No subset meets the target ICIR. Selecting best performing subset.")
        best = max(results, key=lambda r: r['metrics']['icir'])
    else:
        # Select smallest subset that meets target
        best = min(valid_subsets, key=lambda r: r['n_features'])
        print(f"✓ Found {len(valid_subsets)} subsets meeting target")

    print(f"\nSelected: {best['n_features']} features")
    print(f"  ICIR: {best['metrics']['icir']:.3f} ({best['metrics']['icir']/baseline_icir:.1%} of baseline)")
    print(f"  Rank IC: {best['metrics']['rank_ic']:.4f}")

    return best


def save_results(best_subset, all_results, collinearity_stats, importance_df,
                baseline_icir, config, output_dir):
    """Save feature screening results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = config['label_horizon']

    # Save feature importance
    importance_path = output_dir / f"feature_importance_{label}.json"
    importance_data = {
        'baseline_features': collinearity_stats['remaining_count'] +
                           len(collinearity_stats['removed_by_correlation']) +
                           len(collinearity_stats['removed_by_vif']),
        'after_collinearity_cleanup': collinearity_stats['remaining_count'],
        'collinearity_removed': {
            'by_correlation': collinearity_stats['removed_by_correlation'],
            'by_vif': collinearity_stats['removed_by_vif']
        },
        'importance_ranking': importance_df.to_dict(orient='records'),
        'selected_subset': {
            'n_features': best_subset['n_features'],
            'features': best_subset['features']
        }
    }

    with open(importance_path, 'w') as f:
        json.dump(importance_data, f, indent=2)
    print(f"\nSaved feature importance to {importance_path}")

    # Save comparison results
    comparison_path = output_dir / f"feature_screening_comparison_{label}.json"
    comparison_data = {
        'baseline': {
            'n_features': collinearity_stats['remaining_count'] +
                         len(collinearity_stats['removed_by_correlation']) +
                         len(collinearity_stats['removed_by_vif']),
            'icir': baseline_icir,
            'note': 'Before feature screening'
        },
        'after_screening': {
            'n_features': best_subset['n_features'],
            'metrics': best_subset['metrics'],
            'retention_rate': best_subset['metrics']['icir'] / baseline_icir
        },
        'all_tested_subsets': [
            {
                'n_features': r['n_features'],
                'metrics': r['metrics']
            }
            for r in all_results
        ],
        'reduction': {
            'features_removed': (collinearity_stats['remaining_count'] +
                                len(collinearity_stats['removed_by_correlation']) +
                                len(collinearity_stats['removed_by_vif'])) - best_subset['n_features'],
            'reduction_rate': 1 - (best_subset['n_features'] /
                                  (collinearity_stats['remaining_count'] +
                                   len(collinearity_stats['removed_by_correlation']) +
                                   len(collinearity_stats['removed_by_vif']))),
            'icir_change': best_subset['metrics']['icir'] - baseline_icir,
            'icir_change_pct': (best_subset['metrics']['icir'] / baseline_icir - 1) * 100
        }
    }

    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Saved comparison to {comparison_path}")

    return importance_path, comparison_path


def save_optimized_model(best_subset, train_df, imputer, config):
    """Train and save the final optimized model with selected features."""
    print("\n=== Training Final Optimized Model ===")

    label_col = f"rank_future_return_{config['label_horizon']}"
    selected_features = best_subset['features']

    # Prepare full training data with selected features
    train_X = train_df[selected_features].values
    train_y = train_df[label_col].values
    train_groups = train_df.groupby('date').size().tolist()

    # Train on full training set (no validation split for final model)
    train_data = prepare_lgb_dataset(train_X, train_y, selected_features, train_groups)

    final_model = lgb.train(
        config['lgbm_params'],
        train_data,
        num_boost_round=config['training']['num_boost_round']
    )

    # Save model
    output_dir = Path(config['output_model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"model_{config['label_horizon']}.txt"
    final_model.save_model(str(model_path))
    print(f"Saved model to {model_path}")

    # Save feature list
    features_path = output_dir / "features.txt"
    with open(features_path, 'w') as f:
        f.write('\n'.join(selected_features))
    print(f"Saved feature list to {features_path}")

    # Save imputer
    imputer_path = output_dir / f"imputer_{config['label_horizon']}.pkl"
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    print(f"Saved imputer to {imputer_path}")

    # Save metadata
    metadata = {
        'label_horizon': config['label_horizon'],
        'n_features': len(selected_features),
        'features': selected_features,
        'train_period': f"{config['data']['train_start']} to {config['data']['train_end']}",
        'val_period': f"{config['data']['val_start']} to {config['data']['val_end']}",
        'baseline_icir': config['performance_target']['baseline_icir'],
        'achieved_icir': best_subset['metrics']['icir'],
        'retention_rate': best_subset['metrics']['icir'] / config['performance_target']['baseline_icir'],
        'created_at': datetime.now().isoformat()
    }

    metadata_path = output_dir / "training_summary.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    return model_path


def main():
    # Load configuration
    config_path = project_root / "v3_pipeline/configs/v3_0_2_feature_screening.yaml"
    config = load_config(config_path)

    print("=" * 80)
    print("Stage 2: Feature Screening with Permutation Importance")
    print("=" * 80)
    print(f"Label horizon: {config['label_horizon']}")
    print(f"Baseline ICIR: {config['performance_target']['baseline_icir']:.3f}")
    print(f"Target: ≥ {config['performance_target']['min_retention']:.1%} of baseline")

    # Load data
    cache_path = project_root / "v3_pipeline/feature_cache_v3.parquet"
    train_df, val_df = load_data(
        cache_path,
        config['data']['train_start'],
        config['data']['train_end'],
        config['data']['val_start'],
        config['data']['val_end']
    )

    # Get feature columns
    label_col = f"rank_future_return_{config['label_horizon']}"
    feature_cols = get_feature_columns(train_df, config['label_horizon'])

    print(f"\nInitial features: {len(feature_cols)}")

    # Remove collinear features
    train_X_full = train_df[feature_cols]
    train_X_clean, remaining_features, collinearity_stats = remove_collinear_features(
        train_X_full,
        feature_cols,
        config['collinearity']['vif_threshold'],
        config['collinearity']['correlation_threshold']
    )

    # Impute missing values
    print("\nImputing missing values...")

    # Replace inf values with NaN before imputation
    train_df_clean = train_df.copy()
    val_df_clean = val_df.copy()

    for col in remaining_features:
        train_df_clean[col] = train_df_clean[col].replace([np.inf, -np.inf], np.nan)
        val_df_clean[col] = val_df_clean[col].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    train_X_imputed = imputer.fit_transform(train_df_clean[remaining_features])
    val_X_imputed = imputer.transform(val_df_clean[remaining_features])

    # Update DataFrames with imputed values
    train_df_clean[remaining_features] = train_X_imputed
    val_df_clean[remaining_features] = val_X_imputed

    # Train baseline model with cleaned features (for permutation importance)
    print("\n=== Training Baseline Model (After Collinearity Cleanup) ===")
    train_X = train_df_clean[remaining_features].values
    train_y = train_df_clean[label_col].values
    train_groups = train_df_clean.groupby('date').size().tolist()

    val_X = val_df_clean[remaining_features].values
    val_y = val_df_clean[label_col].values
    val_groups = val_df_clean.groupby('date').size().tolist()
    val_dates = val_df_clean['date'].values

    baseline_model = train_model(
        train_X, train_y, train_groups,
        val_X, val_y, val_groups,
        remaining_features,
        config['lgbm_params'],
        config['training']['num_boost_round'],
        config['training']['early_stopping_rounds'],
        config['training']['verbose_eval']
    )

    # Compute feature importance (using LightGBM gain, not permutation)
    importance_df = compute_feature_importance(baseline_model, remaining_features)

    # Test feature subsets
    subset_results = test_feature_subsets(
        train_df_clean,
        val_df_clean,
        remaining_features,
        importance_df,
        config['feature_subsets'],
        label_col,
        config
    )

    # Select best subset
    best_subset = select_best_subset(
        subset_results,
        config['performance_target']['baseline_icir'],
        config['performance_target']['min_retention']
    )

    # Save results
    save_results(
        best_subset,
        subset_results,
        collinearity_stats,
        importance_df,
        config['performance_target']['baseline_icir'],
        config,
        config['output_results_dir']
    )

    # Train and save optimized model
    save_optimized_model(best_subset, train_df_clean, imputer, config)

    print("\n" + "=" * 80)
    print("✓ Stage 2 Complete")
    print("=" * 80)
    print(f"Features: {len(feature_cols)} → {best_subset['n_features']}")
    print(f"ICIR: {config['performance_target']['baseline_icir']:.3f} → {best_subset['metrics']['icir']:.3f}")
    print(f"Retention: {best_subset['metrics']['icir']/config['performance_target']['baseline_icir']:.1%}")
    print(f"\nOutputs:")
    print(f"  - Model: {config['output_model_dir']}/")
    print(f"  - Results: {config['output_results_dir']}/")


if __name__ == "__main__":
    main()