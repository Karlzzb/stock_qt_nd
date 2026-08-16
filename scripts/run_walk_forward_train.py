#!/usr/bin/env python3
"""
Walk-forward 季度滚动重训脚本（Issue #12 验收第 2 条）。

功能
----
1. 读取 DAILY_FEATURE_DIR 中全部 realistic_features_*.csv（特征管线产物）。
2. 使用 WalkForwardProtocol 构造季度级滚动切分（无菌终审段 = 最近 12 个月）。
3. 按每个切分训练 LightGBM 5 折 TimeSeriesSplit + L1 LR stacking，
   在打分窗口产出 OOF 预测概率，拼成全段 OOS 预测 CSV。
4. 用全量非终审数据（train_end = sterile_start - 1d）训练最终模型，
   覆写 MODEL_DIR，供后续回测使用。
5. 将 OOS 预测集保存到 DATASET_DIR/wf_oos_predictions.csv。
6. 打印每个切分的 PR-AUC，以及 OOS 整体 PR-AUC。

用法
----
    uv run python scripts/run_walk_forward_train.py [选项]

选项
----
    --data-start YYYY-MM-DD   特征数据起始（默认：自动从文件名推断）
    --data-end   YYYY-MM-DD   特征数据截止（默认：今天）
    --sterile-months N        无菌终审段月数（默认：12）
    --min-train-quarters N    最少训练季度数（默认：4）
    --dry-run                 仅打印切分计划，不执行训练

注意：本脚本需要先完成 run_feature_pipeline.py，预计数小时。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_feature_csvs(feature_dir: Path) -> pd.DataFrame:
    """加载 DAILY_FEATURE_DIR 中全部 realistic_features_*.csv，拼接返回。"""
    csv_files = sorted(feature_dir.glob("realistic_features_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"DAILY_FEATURE_DIR ({feature_dir}) 中无 realistic_features_*.csv，"
            "请先运行 scripts/run_feature_pipeline.py。"
        )
    logger.info(f"发现 {len(csv_files)} 个特征 CSV，开始加载…")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as exc:
            logger.warning(f"[{f.name}] 加载失败：{exc}")
    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"原始合并行数：{len(data)}")
    return data


def build_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """应用 prepare_real_daily_features + 标签编码，返回清洁数据集。"""
    from data_process import prepare_real_daily_features
    from comm_fun import model_config, get_return_threshold

    data = prepare_real_daily_features(data)
    data["label"] = (data[model_config.LABEL_COL] > get_return_threshold(data)).astype(int)
    logger.info(
        f"清洗后行数：{len(data)}，正样本比例：{data['label'].mean():.4f}"
    )
    return data


def train_one_split(
    train_data: pd.DataFrame,
    score_data: pd.DataFrame,
    split_idx: int,
    model_dir: Path,
) -> tuple[pd.DataFrame, float]:
    """训练单切分：LightGBM + LR stacking，对打分窗口打分。"""
    from comm_fun import model_config
    from stock_model_Tflod_v2 import train_lgb_models, train_meta_model_reduce, meta_features_process

    lgbm_feats = model_config.OPTIMIZED_FEATURE_COLS

    X_train_raw = train_data[lgbm_feats].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    y_train = train_data["label"].reset_index(drop=True)
    X_train_raw = X_train_raw.reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=lgbm_feats)

    oof_preds, lgb_models, optimal_threshold, fold_scores = train_lgb_models(
        X_train[lgbm_feats], y_train, model_config
    )
    lr_meta, scaler, dim_reducer, lr_feature_names, _ = train_meta_model_reduce(
        X_train, y_train, oof_preds, lgb_models, model_config, lgbm_feats
    )

    # 打分窗口预测
    X_score_raw = score_data[lgbm_feats].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    X_score = pd.DataFrame(imputer.transform(X_score_raw), columns=lgbm_feats)

    weights = np.array(fold_scores) / np.sum(fold_scores)
    lgb_preds = np.average(
        [m.predict_proba(X_score[lgbm_feats])[:, 1] for m in lgb_models],
        axis=0, weights=weights,
    )
    X_meta_scaled, *_ = meta_features_process(
        X_score, lgb_preds, lgb_models, model_config, lgbm_feats, scaler, dim_reducer
    )
    final_proba = lr_meta.predict_proba(X_meta_scaled)[:, 1]

    result_df = score_data.copy().reset_index(drop=True)
    result_df["y_pred_proba"] = final_proba
    result_df["wf_split_idx"] = split_idx

    y_score = score_data["label"].reset_index(drop=True)
    pr_auc = (
        average_precision_score(y_score, final_proba)
        if len(set(y_score)) > 1 else float("nan")
    )

    # 保存本切分模型
    split_dir = model_dir / f"wf_split_{split_idx:02d}"
    split_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgb_models, split_dir / "lgb_models.pkl")
    joblib.dump(lr_meta, split_dir / "lr_meta.pkl")
    joblib.dump(imputer, split_dir / "imputer.pkl")
    joblib.dump(scaler, split_dir / "stack_scaler.pkl")
    joblib.dump(fold_scores, split_dir / "fold_scores.pkl")
    with open(split_dir / "lgbm_features.json", "w", encoding="utf-8") as f:
        json.dump(lgbm_feats, f, ensure_ascii=False)
    with open(split_dir / "lr_features.json", "w", encoding="utf-8") as f:
        json.dump(lr_feature_names, f, ensure_ascii=False)
    logger.info(f"  切分 {split_idx:02d} 模型已保存至 {split_dir}，PR-AUC={pr_auc:.4f}")
    return result_df, pr_auc


def save_final_model(train_data: pd.DataFrame, model_dir: Path) -> None:
    """用全量非终审数据训练最终模型，覆写 MODEL_DIR。"""
    from comm_fun import model_config
    from stock_model_Tflod_v2 import train_lgb_models, train_meta_model_reduce

    lgbm_feats = model_config.OPTIMIZED_FEATURE_COLS
    X_raw = train_data[lgbm_feats].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    y = train_data["label"].reset_index(drop=True)
    X_raw = X_raw.reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=lgbm_feats)

    oof_preds, lgb_models, optimal_threshold, fold_scores = train_lgb_models(
        X[lgbm_feats], y, model_config
    )
    lr_meta, scaler, dim_reducer, lr_feature_names, _ = train_meta_model_reduce(
        X, y, oof_preds, lgb_models, model_config, lgbm_feats
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgb_models, model_dir / "lgb_models.pkl")
    joblib.dump(lr_meta, model_dir / "lr_meta.pkl")
    joblib.dump(imputer, model_dir / "imputer.pkl")
    joblib.dump(scaler, model_dir / "stack_scaler.pkl")
    joblib.dump(fold_scores, model_dir / "fold_scores.pkl")
    joblib.dump(optimal_threshold, model_dir / "optimal_threshold.pkl")
    with open(model_dir / "lgbm_features.json", "w", encoding="utf-8") as f:
        json.dump(lgbm_feats, f, ensure_ascii=False)
    with open(model_dir / "lr_features.json", "w", encoding="utf-8") as f:
        json.dump(lr_feature_names, f, ensure_ascii=False)
    if dim_reducer is not None:
        joblib.dump(dim_reducer, model_dir / "dim_reducer.joblib")
    logger.info(f"最终模型已覆写至 {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward 季度滚动重训")
    parser.add_argument("--data-start", default=None, help="特征数据起始 YYYY-MM-DD")
    parser.add_argument("--data-end", default=None, help="特征数据截止 YYYY-MM-DD")
    parser.add_argument("--sterile-months", type=int, default=12)
    parser.add_argument("--min-train-quarters", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="仅打印切分计划")
    args = parser.parse_args()

    from config.settings import DAILY_FEATURE_DIR, DATASET_DIR, MODEL_DIR
    from eval_protocol import WalkForwardProtocol

    logger.info("=== run_walk_forward_train.py 启动 ===")

    # 1. 加载全量特征数据
    raw = load_feature_csvs(DAILY_FEATURE_DIR)
    data = build_dataset(raw)

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    # 2. 推断日期范围
    data_start_dt = (
        pd.to_datetime(args.data_start).date()
        if args.data_start
        else data["timestamp"].min().date()
    )
    data_end_dt = (
        pd.to_datetime(args.data_end).date()
        if args.data_end
        else date.today()
    )

    proto = WalkForwardProtocol(
        data_start=data_start_dt,
        data_end=data_end_dt,
        sterile_months=args.sterile_months,
        min_train_quarters=args.min_train_quarters,
    )
    proto.validate()

    logger.info(repr(proto))
    logger.info(f"无菌终审段起点：{proto.sterile_start}")
    logger.info(f"有效切分数量：{len(proto.splits)}")
    for i, s in enumerate(proto.splits):
        logger.info(
            f"  切分 {i:02d}：train_end={s.train_end}  "
            f"score=[{s.score_start}, {s.score_end}]"
        )

    if args.dry_run:
        logger.info("--dry-run 模式，退出。")
        return

    # 3. 逐切分训练 + OOS 打分
    oos_parts: list[pd.DataFrame] = []
    pr_aucs: list[float] = []

    for i, split in enumerate(proto.splits):
        train_mask = data["timestamp"].dt.date <= split.train_end
        score_mask = (
            (data["timestamp"].dt.date >= split.score_start)
            & (data["timestamp"].dt.date <= split.score_end)
        )
        train_df = data[train_mask].copy()
        score_df = data[score_mask].copy()

        if len(train_df) == 0 or len(score_df) == 0:
            logger.warning(f"切分 {i:02d} 数据不足，跳过。")
            continue

        logger.info(
            f"\n切分 {i:02d}：训练 {len(train_df)} 行，打分 {len(score_df)} 行"
        )
        scored_df, pr_auc = train_one_split(train_df, score_df, i, MODEL_DIR)
        oos_parts.append(scored_df)
        pr_aucs.append(pr_auc)

    # 4. 合并 OOS 预测并保存
    if oos_parts:
        oos_df = pd.concat(oos_parts, ignore_index=True)
        oos_path = DATASET_DIR / "wf_oos_predictions.csv"
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        oos_df.to_csv(oos_path, index=False)
        logger.info(f"OOS 预测已保存至 {oos_path}（{len(oos_df)} 行）")

        valid_aucs = [a for a in pr_aucs if not np.isnan(a)]
        logger.info(f"各切分 PR-AUC：{[f'{a:.4f}' for a in pr_aucs]}")
        if valid_aucs:
            logger.info(f"平均 OOS PR-AUC：{np.mean(valid_aucs):.4f}")

        # 整体 OOS PR-AUC
        if "label" in oos_df.columns and "y_pred_proba" in oos_df.columns:
            overall_auc = average_precision_score(
                oos_df["label"], oos_df["y_pred_proba"]
            )
            logger.info(f"整体 OOS PR-AUC：{overall_auc:.4f}")

    # 5. 训练最终模型（全量非终审数据）
    sterile_start = proto.sterile_start
    final_mask = data["timestamp"].dt.date < sterile_start
    final_train = data[final_mask].copy()

    # 断言：无菌终审段未参与训练
    sterile_mask = data["timestamp"].dt.date >= sterile_start
    sterile_count = sterile_mask.sum()
    if sterile_count > 0:
        logger.warning(
            f"数据集包含 {sterile_count} 行终审段数据（>= {sterile_start}），"
            "但最终模型训练已排除。"
        )
    assert len(final_train) > 0, "最终训练集为空，请检查 sterile_start 与数据日期范围"
    assert (final_train["timestamp"].dt.date >= sterile_start).sum() == 0, (
        f"最终训练集中仍有 >= {sterile_start} 的行！终审段泄露检测失败。"
    )

    logger.info(
        f"\n最终模型训练数据：{len(final_train)} 行（截止 {sterile_start} 之前）"
    )
    save_final_model(final_train, MODEL_DIR)

    logger.info("=== walk-forward 重训完成 ===")


if __name__ == "__main__":
    main()
