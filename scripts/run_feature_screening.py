#!/usr/bin/env python3
"""特征筛选（评估协议 v2 / Issue #16 User Story 10 的单切分实现）。

在单切分全量池产物上做 OOS 置换重要性筛选：

1. 训练"标签打乱"对照模型 → 其置换重要性分布 = 噪声零分布；
2. 真实模型在验证集上按年度分段计算置换重要性（ΔPR-AUC）；
3. 选择规则：某特征在 ≥3/4 个年度分段中 ΔPR-AUC > 噪声 95 分位；
4. 可选 --retrain-selected：用筛后特征重训（LR 层稳态特征取重要性 top15），
   报告筛选前后验证集 PR-AUC 对照。

不继承任何历史特征清单；全部判定只发生在验证集，测试集不接触。

用法
----
    uv run python scripts/run_feature_screening.py \
        --split-dir models/single_2021-12-31_2022-01-01_2025-07-31_poolfull \
        --retrain-selected
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_walk_forward_train import (  # noqa: E402
    _release_memory,
    _scan_to_pandas_f32,
    _validate_cache_fingerprint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _ensemble_proba(models, X: pd.DataFrame, feats: list[str], weights) -> np.ndarray:
    preds = [m.predict_proba(X[feats])[:, 1] for m in models]
    return np.average(preds, axis=0, weights=weights)


def _load_split_data(cache_path: Path, train_end: str, score_start: str,
                     score_end: str, label_col: str, feats: list[str]):
    """按切分日期从缓存重扫训练/验证帧，丢标签缺失行并生成 label。"""
    from comm_fun import get_return_threshold

    dataset = ds.dataset(cache_path, format="parquet")
    needed = list(set(feats + ["timestamp", "symbol", label_col]))
    feat_cols = [c for c in needed if c not in ("timestamp", "symbol", label_col)]

    train_filter = ds.field("timestamp") <= pa.scalar(str(train_end))
    score_filter = (ds.field("timestamp") >= pa.scalar(str(score_start))) & (
        ds.field("timestamp") <= pa.scalar(str(score_end))
    )
    train_df = _scan_to_pandas_f32(dataset, needed, train_filter, feat_cols)
    score_df = _scan_to_pandas_f32(dataset, needed, score_filter, feat_cols)

    for name, df in (("训练", train_df), ("验证", score_df)):
        n0 = len(df)
        df.dropna(subset=[label_col], inplace=True)
        logger.info(f"{name}集：{n0} 行 → 丢标签缺失后 {len(df)} 行")
        df["label"] = (df[label_col] > get_return_threshold(df)).astype(int)
    return train_df, score_df


def main() -> None:
    parser = argparse.ArgumentParser(description="特征筛选：OOS 置换重要性 + 零分布卡线")
    parser.add_argument("--split-dir", required=True, help="单切分全量池模型目录")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--score-start", default="2022-01-01")
    parser.add_argument("--score-end", default="2025-07-31")
    parser.add_argument("--max-per-year", type=int, default=15000,
                        help="每年度分段置换重要性时的最大抽样行数")
    parser.add_argument("--null-feats", type=int, default=100,
                        help="零分布估计用的随机特征数（对照模型只算这些特征的置换重要性）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain-selected", action="store_true",
                        help="用筛后特征重训并报告前后对照")
    args = parser.parse_args()

    from config.settings import DAILY_FEATURE_DIR, MODEL_DIR
    from comm_fun import model_config

    split_dir = Path(args.split_dir)
    cache_path = DAILY_FEATURE_DIR.parent / "feature_cache_all.parquet"
    _validate_cache_fingerprint(cache_path, DAILY_FEATURE_DIR)

    lgbm_feats: list[str] = json.load(open(split_dir / "lgbm_features.json"))
    lgb_models = joblib.load(split_dir / "lgb_models.pkl")
    imputer = joblib.load(split_dir / "imputer.pkl")
    fold_scores = joblib.load(split_dir / "fold_scores.pkl")
    weights = np.array(fold_scores) / np.sum(fold_scores)
    medians = pd.Series(imputer.statistics_, index=lgbm_feats)
    logger.info(f"加载切分模型：{len(lgbm_feats)} 个特征，{len(lgb_models)} 折")

    train_df, score_df = _load_split_data(
        cache_path, args.train_end, args.score_start, args.score_end,
        model_config.LABEL_COL, lgbm_feats,
    )

    # ------------------------------------------------------------------
    # 1. 标签打乱对照模型（噪声零分布）
    # ------------------------------------------------------------------
    null_path = split_dir / "lgb_models_null.pkl"
    if null_path.exists():
        null_models = joblib.load(null_path)
        logger.info("加载已有对照模型")
    else:
        logger.info("训练标签打乱对照模型（与真实模型同训练集同配置）…")
        from stock_model_Tflod_v2 import train_lgb_models

        X_tr = train_df[lgbm_feats].astype(np.float32, copy=False)
        X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(medians)
        y_shuf = (
            train_df["label"].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        )
        _, null_models, _, _ = train_lgb_models(X_tr[lgbm_feats], y_shuf, model_config)
        joblib.dump(null_models, null_path)
        del X_tr
        _release_memory()
        logger.info("对照模型已保存")
    null_weights = np.ones(len(null_models)) / len(null_models)

    # ------------------------------------------------------------------
    # 2. 分年度 OOS 置换重要性
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    score_df["year"] = pd.to_datetime(score_df["timestamp"]).dt.year
    years = sorted(score_df["year"].unique())
    logger.info(f"验证集年度分段：{years}")

    imp = pd.DataFrame(index=lgbm_feats)   # 真实模型 ΔPR-AUC
    null_deltas: list[float] = []          # 对照模型 ΔPR-AUC（随机子集特征）
    null_subset = list(
        rng.choice(lgbm_feats, size=min(args.null_feats, len(lgbm_feats)), replace=False)
    )

    for yr in years:
        seg = score_df[score_df["year"] == yr]
        if len(seg) > args.max_per_year:
            seg = seg.sample(args.max_per_year, random_state=args.seed)
        X_seg = (
            seg[lgbm_feats]
            .astype(np.float32, copy=False)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(medians)
        )
        y_seg = seg["label"].to_numpy()

        base_auc = average_precision_score(
            y_seg, _ensemble_proba(lgb_models, X_seg, lgbm_feats, weights)
        )
        null_base_auc = average_precision_score(
            y_seg, _ensemble_proba(null_models, X_seg, lgbm_feats, null_weights)
        )
        logger.info(
            f"{yr}：{len(seg)} 行，基线 PR-AUC={base_auc:.4f}"
            f"（先验 {y_seg.mean():.4f}，对照模型 {null_base_auc:.4f}）"
        )

        deltas = {}
        for f in lgbm_feats:
            Xp = X_seg.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            deltas[f] = base_auc - average_precision_score(
                y_seg, _ensemble_proba(lgb_models, Xp, lgbm_feats, weights)
            )
            if f in null_subset:
                null_deltas.append(
                    null_base_auc
                    - average_precision_score(
                        y_seg, _ensemble_proba(null_models, Xp, lgbm_feats, null_weights)
                    )
                )
        imp[yr] = pd.Series(deltas)
        logger.info(f"{yr} 置换重要性完成")
        del X_seg
        _release_memory()

    # ------------------------------------------------------------------
    # 3. 零分布卡线 + 选择
    # ------------------------------------------------------------------
    null95 = float(np.percentile(null_deltas, 95))
    n_years = len(years)
    min_years = max(3, n_years - 1)
    imp["mean_delta"] = imp[years].mean(axis=1)
    imp["years_above_null95"] = (imp[years] > null95).sum(axis=1)
    selected = imp[imp["years_above_null95"] >= min_years].index.tolist()

    logger.info(
        f"噪声 95 分位：{null95:+.5f}（{len(null_deltas)} 个对照观测）；"
        f"选中 {len(selected)}/{len(lgbm_feats)} 个特征（≥{min_years}/{n_years} 年度超线）"
    )

    # ------------------------------------------------------------------
    # 4. 落盘
    # ------------------------------------------------------------------
    exp_dir = REPO_ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)
    tag = f"{args.train_end}_{args.score_start}_{args.score_end}"
    imp.sort_values("mean_delta", ascending=False).to_csv(
        exp_dir / f"screening_importance_{tag}.csv"
    )
    summary = {
        "split_dir": str(split_dir),
        "n_features_in": len(lgbm_feats),
        "n_selected": len(selected),
        "null95": null95,
        "years": [int(y) for y in years],
        "min_years": min_years,
        "selected": selected,
    }
    (exp_dir / f"screening_summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    logger.info(f"重要性表与摘要已写入 {exp_dir}")

    # ------------------------------------------------------------------
    # 5. 可选：筛后特征重训对照
    # ------------------------------------------------------------------
    if args.retrain_selected and selected:
        from run_walk_forward_train import train_one_split

        model_config.OPTIMIZED_FEATURE_COLS = selected
        # LR 层稳态特征：重要性 top15（不再使用历史 STABLE 清单）
        model_config.STABLE_FEATURES = (
            imp.loc[selected].sort_values("mean_delta", ascending=False)
            .head(15).index.tolist()
        )
        sel_dir = MODEL_DIR / f"single_{tag}_selected"
        logger.info(
            f"筛后重训：{len(selected)} 个 LGBM 特征，"
            f"LR 稳态 {len(model_config.STABLE_FEATURES)} 个 → {sel_dir}"
        )
        score_df.drop(columns=["year"], inplace=True)
        scored, pr_auc = train_one_split(train_df, score_df, 99, sel_dir)
        logger.info(f"筛后模型验证集 PR-AUC：{pr_auc:.4f}（产物 {sel_dir}）")

    logger.info("=== 特征筛选完成 ===")


if __name__ == "__main__":
    main()
