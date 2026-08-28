#!/usr/bin/env python3
"""两层架构消融（评估协议 v2）。

在同一个单切分（默认全量池）上对照三种架构：

- A 纯 LGBM：fold 加权集成直接打分，无 LR meta 层（从已存产物预测，不重训）；
- B LGBM + LR(仅 pred + pred²)：现状 poolfull 产物，直接评其 oos_predictions；
- C LGBM + LR(pred + pred² + 筛选 top-N 稳态特征)：重训（LGBM 部分 ~80s）。

每个配置在验证集上输出完整指标块（enhanced_evaluate 口径）：
precision/recall/f1 @PROBA_THRESHOLD(0.7)、f1@{0.3,0.5,0.7}、pr_auc、
precision_top{1%n, 5%n, 100}。

用法
----
    MALLOC_ARENA_MAX=4 uv run python scripts/run_two_layer_ablation.py \
        --split-dir models/single_2021-12-31_2022-01-01_2025-07-31_poolfull
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_walk_forward_train import (  # noqa: E402
    _release_memory,
    _validate_cache_fingerprint,
    train_one_split,
)
from run_feature_screening import _ensemble_proba, _load_split_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _full_metrics(y_true: pd.Series, proba: np.ndarray, title: str) -> dict:
    from stock_model_Tflod_v2 import enhanced_evaluate

    return enhanced_evaluate(y_true.reset_index(drop=True), proba, title=title)


def main() -> None:
    parser = argparse.ArgumentParser(description="两层架构消融：纯LGBM vs LR(pred) vs LR(+稳态)")
    parser.add_argument(
        "--split-dir",
        default="models/single_2021-12-31_2022-01-01_2025-07-31_poolfull",
    )
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--score-start", default="2022-01-01")
    parser.add_argument("--score-end", default="2025-07-31")
    parser.add_argument("--top-n-stable", type=int, default=15,
                        help="配置 C 的 LR 稳态特征数（按筛选 mean_delta 取 top-N）")
    args = parser.parse_args()

    from config.settings import DAILY_FEATURE_DIR, MODEL_DIR
    from comm_fun import model_config

    split_dir = Path(args.split_dir)
    tag = f"{args.train_end}_{args.score_start}_{args.score_end}"
    cache_path = DAILY_FEATURE_DIR.parent / "feature_cache_all.parquet"
    _validate_cache_fingerprint(cache_path, DAILY_FEATURE_DIR)

    lgbm_feats: list[str] = json.load(open(split_dir / "lgbm_features.json"))
    lgb_models = joblib.load(split_dir / "lgb_models.pkl")
    imputer = joblib.load(split_dir / "imputer.pkl")
    fold_scores = joblib.load(split_dir / "fold_scores.pkl")
    weights = np.array(fold_scores) / np.sum(fold_scores)
    medians = pd.Series(imputer.statistics_, index=lgbm_feats)
    logger.info(f"加载切分产物：{len(lgbm_feats)} 个特征，{len(lgb_models)} 折")

    train_df, score_df = _load_split_data(
        cache_path, args.train_end, args.score_start, args.score_end,
        model_config.LABEL_COL, lgbm_feats,
    )
    y_score = score_df["label"]

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # A. 纯 LGBM（无 LR 层）
    # ------------------------------------------------------------------
    X_score = (
        score_df[lgbm_feats]
        .astype(np.float32, copy=False)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )
    proba_a = _ensemble_proba(lgb_models, X_score, lgbm_feats, weights)
    results["A_pure_lgbm"] = _full_metrics(y_score, proba_a, "A 纯LGBM（无LR层）")
    del X_score, proba_a
    _release_memory()

    # ------------------------------------------------------------------
    # B. 现状：LGBM + LR(仅 pred + pred²)，直接评已存 OOS 预测
    # ------------------------------------------------------------------
    oos = pd.read_parquet(split_dir / "oos_predictions.parquet")
    if len(oos) != len(y_score):
        logger.warning(
            f"B 配置行数不一致：oos={len(oos)} vs 重扫={len(y_score)}，各评各的"
        )
    results["B_lr_pred_only"] = _full_metrics(
        oos["label"], oos["y_pred_proba"].to_numpy(), "B LGBM+LR(仅pred+pred²)"
    )
    del oos
    _release_memory()

    # ------------------------------------------------------------------
    # C. LGBM + LR(pred + pred² + 筛选 top-N 稳态特征)，重训
    # ------------------------------------------------------------------
    imp_csv = REPO_ROOT / "experiments" / f"screening_importance_{tag}.csv"
    imp = pd.read_csv(imp_csv, index_col=0)
    stable_topn = (
        imp.sort_values("mean_delta", ascending=False).head(args.top_n_stable).index.tolist()
    )
    logger.info(f"C 配置稳态特征 top{args.top_n_stable}：{stable_topn}")

    model_config.OPTIMIZED_FEATURE_COLS = list(lgbm_feats)
    model_config.STABLE_FEATURES = stable_topn
    c_dir = MODEL_DIR / f"single_{tag}_abl_stable{args.top_n_stable}"
    scored_c, _ = train_one_split(train_df, score_df, 98, c_dir)
    results[f"C_lr_stable{args.top_n_stable}"] = _full_metrics(
        scored_c["label"], scored_c["y_pred_proba"].to_numpy(),
        f"C LGBM+LR(pred+稳态top{args.top_n_stable})",
    )
    del scored_c, train_df, score_df
    _release_memory()

    # ------------------------------------------------------------------
    # 汇总落盘
    # ------------------------------------------------------------------
    exp_dir = REPO_ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)
    out = exp_dir / f"ablation_{tag}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    keys = list(next(iter(results.values())).keys())
    table = pd.DataFrame(results).loc[keys]
    logger.info(f"\n=== 消融对照表 ===\n{table.to_string(float_format=lambda v: f'{v:.4f}')}")
    logger.info(f"结果已写入 {out}")
    logger.info("=== 两层消融完成 ===")


if __name__ == "__main__":
    main()
