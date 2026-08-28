#!/usr/bin/env python3
"""Label 周期网格（评估协议 v2 / Issue #16 Stage 1 的单切分实现）。

固定特征集（全量池）与架构，只换 label 周期，比较哪个预测目标信号最强：

    label = 1{ future_return_{d}d > 0.01 }，d ∈ --horizons（默认 3,5,10,15,20,25,30）

每格：重扫训练/验证（按该 label dropna）→ 全量池重训 → 验证集完整指标块
（precision/recall/f1@0.7、f1@0.3/0.5/0.7、pr_auc、precision_top{1%n,5%n,100}）
+ 分年度 pr_auc 与 top1% 精度。

注意：不同 label 的先验不同，PR-AUC 跨格不可直接比——汇总表同时给出
先验与 lift（PR-AUC/先验），横向比较以 lift 与 top-k 精度为准。

用法
----
    MALLOC_ARENA_MAX=4 uv run python scripts/run_label_grid.py \
        --horizons 3,5,10,15,20,25,30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_walk_forward_train import (  # noqa: E402
    _release_memory,
    _validate_cache_fingerprint,
    build_full_feature_pool,
    train_one_split,
)
from run_feature_screening import _load_split_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _per_day_metrics(scored: pd.DataFrame, top_ns=(1, 3, 5)) -> dict[str, float]:
    """逐日截面 top-N 命中率（与实盘"每天买当天分最高"对齐）。

    每天按 proba 取 top N，命中=该日 top N 中 label=1 的比例；
    跨交易日取平均。当日信号数 < N 的日子跳过。
    """
    out: dict[str, float] = {}
    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["timestamp"]).dt.date
    for n in top_ns:
        hits = []
        for _, day in scored.groupby("date"):
            if len(day) < n:
                continue
            top = day.nlargest(n, "y_pred_proba")
            hits.append(float(top["label"].mean()))
        out[f"daily_top{n}_hit"] = float(np.mean(hits)) if hits else float("nan")
        out[f"daily_top{n}_ndays"] = len(hits)
    return out


def _per_year_metrics(scored: pd.DataFrame) -> dict[int, dict]:
    scored = scored.copy()
    scored["year"] = pd.to_datetime(scored["timestamp"]).dt.year
    out = {}
    for yr, seg in scored.groupby("year"):
        y = seg["label"]
        k = max(1, int(0.01 * len(seg)))
        top_idx = seg["y_pred_proba"].nlargest(k).index
        out[int(yr)] = {
            "n": int(len(seg)),
            "prior": float(y.mean()),
            "pr_auc": float(average_precision_score(y, seg["y_pred_proba"])),
            "precision_top1pct": float(y.loc[top_idx].mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Label 周期网格：固定特征与架构，只换预测周期")
    parser.add_argument("--horizons", default="3,5,10,15,20,25,30",
                        help="逗号分隔的周期天数列表")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--score-start", default="2022-01-01")
    parser.add_argument("--score-end", default="2025-07-31")
    parser.add_argument("--stable-features-json", default=None,
                        help="可选：LR 稳态特征清单 JSON（默认空=LR 仅 pred+pred²）")
    args = parser.parse_args()

    from config.settings import DAILY_FEATURE_DIR, MODEL_DIR
    from comm_fun import model_config
    from stock_model_Tflod_v2 import enhanced_evaluate

    horizons = [int(x) for x in args.horizons.split(",")]
    tag = f"{args.train_end}_{args.score_start}_{args.score_end}"
    cache_path = DAILY_FEATURE_DIR.parent / "feature_cache_all.parquet"
    _validate_cache_fingerprint(cache_path, DAILY_FEATURE_DIR)

    stable_feats: list[str] = []
    if args.stable_features_json:
        stable_feats = json.load(open(args.stable_features_json))
        logger.info(f"LR 稳态特征 {len(stable_feats)} 个（来自 {args.stable_features_json}）")

    all_results: dict[str, dict] = {}

    for d in horizons:
        label_col = f"future_return_{d}d"
        logger.info(f"\n{'='*60}\n=== Label: {label_col} ===\n{'='*60}")

        pool = build_full_feature_pool(cache_path, label_col)
        model_config.LABEL_COL = label_col
        model_config.OPTIMIZED_FEATURE_COLS = pool
        model_config.STABLE_FEATURES = stable_feats

        train_df, score_df = _load_split_data(
            cache_path, args.train_end, args.score_start, args.score_end,
            label_col, pool,
        )
        prior = float(score_df["label"].mean())

        cell_dir = MODEL_DIR / f"single_{tag}_label{d}d"
        scored, pr_auc = train_one_split(train_df, score_df, 90 + d, cell_dir)

        metrics = enhanced_evaluate(
            scored["label"].reset_index(drop=True),
            scored["y_pred_proba"].to_numpy(),
            title=f"Label {label_col}（先验 {prior:.4f}）",
        )
        metrics["prior"] = prior
        metrics["lift"] = metrics["pr_auc"] / prior if prior > 0 else float("nan")
        metrics["n_features"] = len(json.load(open(cell_dir / "lgbm_features.json")))
        metrics["per_day"] = _per_day_metrics(scored)
        metrics["per_year"] = _per_year_metrics(scored)
        all_results[label_col] = metrics

        del train_df, score_df, scored
        _release_memory()
        logger.info(f"{label_col} 完成：PR-AUC={pr_auc:.4f}，lift={metrics['lift']:.4f}")

    # ------------------------------------------------------------------
    # 汇总表
    # ------------------------------------------------------------------
    exp_dir = REPO_ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / f"label_grid_{tag}.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2)
    )

    rows = {}
    for label_col, m in all_results.items():
        row = {k: v for k, v in m.items() if k not in ("per_year", "per_day")}
        row.update(m["per_day"])
        rows[label_col] = row
    table = pd.DataFrame(rows)
    logger.info(f"\n=== Label 网格对照表 ===\n{table.to_string(float_format=lambda v: f'{v:.4f}')}")
    logger.info(f"完整结果（含分年度）已写入 {exp_dir}/label_grid_{tag}.json")
    logger.info("=== Label 网格完成 ===")


if __name__ == "__main__":
    main()
