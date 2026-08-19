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
import ctypes
import gc
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
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _release_memory() -> None:
    """gc + malloc_trim：把已释放内存真正还给 OS。

    glibc 默认每线程独立 arena（28 核机器可达 224 个），释放的块滞留在
    各 arena 中导致 RSS 只增不减、后续切分加载时叠加 OOM。
    配合启动环境变量 MALLOC_ARENA_MAX=4 使用。
    """
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()  # pyarrow jemalloc/mimalloc 池归还
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _scan_to_pandas_f32(dataset, columns, filt, f32_cols) -> pd.DataFrame:
    """分批扫描 parquet 并即时转 float32，直接产出 float32 DataFrame。

    一次性 to_table().to_pandas() 会让全量 float64 Arrow 表、pandas 副本、
    astype 结果同时驻留（7.8M 行实测 RSS 53.8GB）；分批 cast 后
    峰值只有单批次 transient + 最终 float32 pandas 帧。
    """
    f32_set = set(f32_cols)
    fields = []
    for name in columns:
        f = dataset.schema.field(name)
        if name in f32_set and pa.types.is_floating(f.type):
            f = f.with_type(pa.float32())
        fields.append(f)
    target_schema = pa.schema(fields)
    scanner = dataset.scanner(columns=columns, filter=filt, batch_size=100_000)
    batches = [b.cast(target_schema) for b in scanner.to_batches()]
    if not batches:
        return pd.DataFrame(columns=columns)
    df = pa.Table.from_batches(batches).to_pandas()
    del batches
    _release_memory()  # 归还 arrow 解码/cast 缓冲（jemalloc 池不自动归还）
    return df


def load_feature_csvs(feature_dir: Path, cache_path: Path = None) -> pd.DataFrame:
    """加载 DAILY_FEATURE_DIR 中全部 realistic_features_*.csv，拼接返回。

    使用 Parquet 缓存避免重复加载大量 CSV。
    缓存命中时只加载 OPTIMIZED_FEATURE_COLS + timestamp + symbol + LABEL_COL，
    避免全列 OOM。
    """
    if cache_path is None:
        cache_path = feature_dir.parent / "feature_cache_all.parquet"

    # 如果缓存存在，只加载需要的列
    if cache_path.exists():
        logger.info(f"发现缓存文件 {cache_path}，按需列加载…")
        from comm_fun import model_config
        import pyarrow.parquet as _pq
        # 先读 schema，过滤出缓存中实际存在的列
        available_cols = set(_pq.ParquetFile(cache_path).schema_arrow.names)
        needed_cols = [
            c for c in
            set(model_config.OPTIMIZED_FEATURE_COLS + ["timestamp", "symbol", model_config.LABEL_COL])
            if c in available_cols
        ]
        data = pd.read_parquet(cache_path, columns=needed_cols)
        logger.info(f"从缓存加载完成：{len(data)} 行，{len(data.columns)} 列")
        return data

    # 否则分批加载并创建缓存
    csv_files = sorted(feature_dir.glob("realistic_features_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"DAILY_FEATURE_DIR ({feature_dir}) 中无 realistic_features_*.csv，"
            "请先运行 scripts/run_feature_pipeline.py。"
        )

    logger.info(f"发现 {len(csv_files)} 个特征 CSV，分批加载并创建缓存…")

    # 使用临时 Parquet 文件避免内存累积
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    batch_size = 100
    total_rows = 0

    try:
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(csv_files)-1)//batch_size + 1
            logger.info(f"  批次 {batch_num}/{total_batches}：加载 {len(batch_files)} 个文件")

            batch_dfs = []
            for f in batch_files:
                try:
                    df = pd.read_csv(f, low_memory=False)
                    batch_dfs.append(df)
                except Exception as exc:
                    logger.warning(f"[{f.name}] 加载失败：{exc}")

            if batch_dfs:
                batch_data = pd.concat(batch_dfs, ignore_index=True)
                batch_rows = len(batch_data)
                total_rows += batch_rows

                # 写入临时 Parquet
                temp_parquet = temp_dir / f"batch_{batch_num:03d}.parquet"
                batch_data.to_parquet(temp_parquet, index=False, compression="snappy")
                logger.info(f"  批次完成：{batch_rows} 行，已写入临时文件")

                # 立即释放内存
                del batch_data, batch_dfs

        # 合并所有临时 Parquet 文件（流式追加写入）
        logger.info(f"合并 {total_batches} 个临时文件到最终缓存…")
        temp_parquets = sorted(temp_dir.glob("batch_*.parquet"))

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # 流式合并：统一 schema 后逐个追加写入
        writer = None
        unified_schema = None

        for i, temp_file in enumerate(temp_parquets, 1):
            table = pq.read_table(str(temp_file))

            # 第一个表：建立统一 schema（将所有 string → large_string）
            if unified_schema is None:
                fields = []
                for field in table.schema:
                    if field.type == pa.string():
                        fields.append(pa.field(field.name, pa.large_string()))
                    else:
                        fields.append(field)
                unified_schema = pa.schema(fields)
                writer = pq.ParquetWriter(str(cache_path), unified_schema, compression="snappy")

            # 统一当前表的 schema
            if table.schema != unified_schema:
                table = table.cast(unified_schema)

            writer.write_table(table)
            del table
            if i % 10 == 0:
                logger.info(f"  已合并 {i}/{len(temp_parquets)} 个文件")

        if writer:
            writer.close()

        logger.info(f"缓存保存完成：{total_rows} 行")

        # 加载最终缓存
        data = pd.read_parquet(cache_path)

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

    return data


def build_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """应用标签编码，返回清洁数据集。

    注意：feature_cache_all.parquet 已经包含完整特征，
    无需再调用 prepare_real_daily_features（会破坏数据）。
    """
    from comm_fun import model_config, get_return_threshold

    # 只需要添加 label
    if model_config.LABEL_COL not in data.columns:
        logger.error(f"缺少标签列 {model_config.LABEL_COL}，无法生成 label")
        raise ValueError(f"数据中缺少 {model_config.LABEL_COL}")

    data["label"] = (data[model_config.LABEL_COL] > get_return_threshold(data)).astype(int)

    # 收窄 OPTIMIZED_FEATURE_COLS 和 STABLE_FEATURES 为实际存在的列
    existing = [c for c in model_config.OPTIMIZED_FEATURE_COLS if c in data.columns]
    dropped = [c for c in model_config.OPTIMIZED_FEATURE_COLS if c not in data.columns]
    if dropped:
        logger.warning(f"OPTIMIZED_FEATURE_COLS 以下特征不在数据中，将跳过：{dropped}")
    model_config.OPTIMIZED_FEATURE_COLS = existing

    existing_stable = [c for c in model_config.STABLE_FEATURES if c in data.columns]
    dropped_stable = [c for c in model_config.STABLE_FEATURES if c not in data.columns]
    if dropped_stable:
        logger.warning(f"STABLE_FEATURES 以下特征不在数据中，将跳过：{dropped_stable}")
    model_config.STABLE_FEATURES = existing_stable

    logger.info(
        f"数据行数：{len(data)}，正样本比例：{data['label'].mean():.4f}"
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

    # float32：LightGBM 按 ≤255 bin 离散化，精度损失可忽略，内存减半（62GB 机器必需）
    # copy=False：流式读取已产出 float32，避免无谓整帧复制
    X_train_raw = train_data[lgbm_feats].astype(np.float32, copy=False).replace([np.inf, -np.inf], np.nan)
    y_train = train_data["label"].reset_index(drop=True)
    X_train_raw = X_train_raw.reset_index(drop=True)

    # 去掉全 NaN 列，避免 SimpleImputer 列数不一致
    all_nan_cols = X_train_raw.columns[X_train_raw.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(f"切分 {split_idx:02d} 训练集全 NaN 列，将剔除：{all_nan_cols}")
        lgbm_feats = [c for c in lgbm_feats if c not in all_nan_cols]
        X_train_raw = X_train_raw[lgbm_feats]

    # 逐列精确中位数（统计量与 SimpleImputer(strategy='median') 完全一致），
    # 避免 sklearn 批量 fit 的 masked-array 中间副本（14M 行切分实测 OOM）
    medians = np.array(
        [float(np.nanmedian(X_train_raw[c].to_numpy())) for c in lgbm_feats],
        dtype=np.float64,
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_raw.head(2))  # 建立 fit 内部属性（_fit_dtype 等）
    imputer.statistics_ = medians     # 统计量覆盖为全量精确中位数
    X_train = X_train_raw.fillna(pd.Series(medians, index=lgbm_feats))
    del X_train_raw, train_data  # 训练前释放原始副本与基础帧，降低峰值内存
    gc.collect()

    oof_preds, lgb_models, optimal_threshold, fold_scores = train_lgb_models(
        X_train[lgbm_feats], y_train, model_config
    )
    lr_meta, scaler, dim_reducer, lr_feature_names, _ = train_meta_model_reduce(
        X_train, y_train, oof_preds, lgb_models, model_config, lgbm_feats
    )
    del X_train, oof_preds  # 打分前释放训练矩阵
    gc.collect()

    # 打分窗口预测
    X_score_raw = score_data[lgbm_feats].astype(np.float32, copy=False).replace([np.inf, -np.inf], np.nan)
    X_score = pd.DataFrame(imputer.transform(X_score_raw), columns=lgbm_feats)
    del X_score_raw

    weights = np.array(fold_scores) / np.sum(fold_scores)
    lgb_preds = np.average(
        [m.predict_proba(X_score[lgbm_feats])[:, 1] for m in lgb_models],
        axis=0, weights=weights,
    )
    X_meta_scaled, *_ = meta_features_process(
        X_score, lgb_preds, lgb_models, model_config, lgbm_feats, scaler, dim_reducer
    )
    final_proba = lr_meta.predict_proba(X_meta_scaled)[:, 1]

    # 只保留必要列，避免 OOS 结果过大
    keep_cols = ["timestamp", "symbol", "label"]
    result_df = score_data[keep_cols].copy().reset_index(drop=True)
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
    # 即时保存本切分 OOS 预测，支持断点续跑
    result_df.to_parquet(split_dir / "oos_predictions.parquet", index=False)
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
    X_raw = train_data[lgbm_feats].astype(np.float32, copy=False).replace([np.inf, -np.inf], np.nan)
    y = train_data["label"].reset_index(drop=True)
    X_raw = X_raw.reset_index(drop=True)

    # 去掉全 NaN 列
    all_nan = X_raw.columns[X_raw.isna().all()].tolist()
    if all_nan:
        logger.warning(f"最终模型全 NaN 列，将剔除：{all_nan}")
        lgbm_feats = [c for c in lgbm_feats if c not in all_nan]
        X_raw = X_raw[lgbm_feats]

    # 逐列精确中位数（同 train_one_split，避免 sklearn 批量 fit 的内存峰值）
    medians = np.array(
        [float(np.nanmedian(X_raw[c].to_numpy())) for c in lgbm_feats],
        dtype=np.float64,
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_raw.head(2))
    imputer.statistics_ = medians
    X = X_raw.fillna(pd.Series(medians, index=lgbm_feats))
    del X_raw, train_data  # 释放原始副本与基础帧（~2000 万行，fold 训练前必须释放）
    gc.collect()

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
    parser.add_argument(
        "--splits",
        default=None,
        help="只跑指定切分，如 '63' 或 '63-70' 或 '63,65,67'。"
        "指定后跳过 OOS 合并与最终模型（用于单切分独立进程模式，防内存滞留 OOM）",
    )
    args = parser.parse_args()

    from config.settings import DAILY_FEATURE_DIR, DATASET_DIR, MODEL_DIR
    from eval_protocol import WalkForwardProtocol

    logger.info("=== run_walk_forward_train.py 启动 ===")

    from comm_fun import model_config, get_return_threshold
    import pyarrow.parquet as _pq
    import pyarrow as _pa

    cache_path = DAILY_FEATURE_DIR.parent / "feature_cache_all.parquet"
    available_cols = set(_pq.ParquetFile(cache_path).schema_arrow.names)

    # 收窄 OPTIMIZED_FEATURE_COLS 和 STABLE_FEATURES 为缓存中存在的列
    existing_opt = [c for c in model_config.OPTIMIZED_FEATURE_COLS if c in available_cols]
    dropped_opt = [c for c in model_config.OPTIMIZED_FEATURE_COLS if c not in available_cols]
    if dropped_opt:
        logger.warning(f"OPTIMIZED_FEATURE_COLS 跳过缺失列：{dropped_opt}")
    model_config.OPTIMIZED_FEATURE_COLS = existing_opt

    existing_stable = [c for c in model_config.STABLE_FEATURES if c in available_cols]
    dropped_stable = [c for c in model_config.STABLE_FEATURES if c not in available_cols]
    if dropped_stable:
        logger.warning(f"STABLE_FEATURES 跳过缺失列：{dropped_stable}")
    model_config.STABLE_FEATURES = existing_stable

    needed_cols = list(set(
        model_config.OPTIMIZED_FEATURE_COLS
        + model_config.STABLE_FEATURES
        + ["timestamp", "symbol", model_config.LABEL_COL]
    ))
    logger.info(f"所需列数：{len(needed_cols)}")

    # 用 parquet 行组统计信息推断日期范围（避免全列扫描 2200 万行及其内存滞留）
    data_start_dt = pd.to_datetime(args.data_start).date() if args.data_start else None
    data_end_dt = pd.to_datetime(args.data_end).date() if args.data_end else None
    if data_start_dt is None or data_end_dt is None:
        md = _pq.read_metadata(cache_path)
        ts_idx = md.schema.names.index("timestamp")
        ts_mins, ts_maxs = [], []
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(ts_idx).statistics
            if st is not None and st.has_min_max:
                ts_mins.append(st.min)
                ts_maxs.append(st.max)
        if ts_mins:
            # timestamp 为 ISO 字符串，字典序 min/max 即日期 min/max
            if data_start_dt is None:
                data_start_dt = pd.to_datetime(min(ts_mins)).date()
            if data_end_dt is None:
                data_end_dt = pd.to_datetime(max(ts_maxs)).date()
        else:
            # 统计信息缺失时回退为全列扫描
            ts_col = pd.to_datetime(
                pd.read_parquet(cache_path, columns=["timestamp"])["timestamp"]
            )
            if data_start_dt is None:
                data_start_dt = ts_col.min().date()
            if data_end_dt is None:
                data_end_dt = ts_col.max().date()
            del ts_col
            _release_memory()

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
    #   每个切分单独从 parquet 按日期范围读取，避免全量加载 2200 万行
    #   --splits 模式下只跑指定切分（配合外层 shell 循环实现单切分独立进程）
    if args.splits:
        selected: list[int] = []
        for part in args.splits.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                selected.extend(range(int(a), int(b) + 1))
            else:
                selected.append(int(part))
        selected = [i for i in selected if 0 <= i < len(proto.splits)]
        logger.info(f"--splits 模式：仅处理切分 {selected}")
    else:
        selected = list(range(len(proto.splits)))

    oos_parts: list[pd.DataFrame] = []
    pr_aucs: list[float] = []

    for i in selected:
        split = proto.splits[i]
        # 断点续跑：已完成的切分直接加载已保存的预测
        split_dir = MODEL_DIR / f"wf_split_{i:02d}"
        cached_pred = split_dir / "oos_predictions.parquet"
        if cached_pred.exists():
            scored_df = pd.read_parquet(cached_pred)
            pr_auc = average_precision_score(scored_df["label"], scored_df["y_pred_proba"])
            logger.info(
                f"切分 {i:02d} 已完成（缓存命中），PR-AUC={pr_auc:.4f}，跳过训练"
            )
            oos_parts.append(scored_df)
            pr_aucs.append(pr_auc)
            continue

        # 用 pyarrow filter 按日期范围读取
        import pyarrow.dataset as ds
        dataset = ds.dataset(cache_path, format="parquet")

        train_filter = (
            (ds.field("timestamp") >= _pa.scalar(str(data_start_dt)))
            & (ds.field("timestamp") <= _pa.scalar(str(split.train_end)))
        )
        score_filter = (
            (ds.field("timestamp") >= _pa.scalar(str(split.score_start)))
            & (ds.field("timestamp") <= _pa.scalar(str(split.score_end)))
        )

        # 特征列 float32（流式分批读取时即时转换，避免 float64 中间表驻留）
        feat_cols = [
            c for c in needed_cols
            if c not in ("timestamp", "symbol", model_config.LABEL_COL)
        ]
        train_df = _scan_to_pandas_f32(dataset, needed_cols, train_filter, feat_cols)
        score_df = _scan_to_pandas_f32(dataset, needed_cols, score_filter, feat_cols)

        if len(train_df) == 0 or len(score_df) == 0:
            logger.warning(f"切分 {i:02d} 数据不足，跳过。")
            continue

        # 加 label
        train_df["label"] = (
            train_df[model_config.LABEL_COL] > get_return_threshold(train_df)
        ).astype(int)
        score_df["label"] = (
            score_df[model_config.LABEL_COL] > get_return_threshold(score_df)
        ).astype(int)

        logger.info(
            f"\n切分 {i:02d}：训练 {len(train_df)} 行，打分 {len(score_df)} 行"
        )
        # 唯一引用移交：holder.pop() 后调用方不再持有引用，
        # train_one_split 内部 del train_data 才能真正释放基础帧
        holder = [(train_df, score_df)]
        del train_df, score_df
        scored_df, pr_auc = train_one_split(*holder.pop(), i, MODEL_DIR)
        oos_parts.append(scored_df)
        pr_aucs.append(pr_auc)
        _release_memory()

    # 4. 合并 OOS 预测并保存
    #    --splits 单切分模式下跳过合并与最终模型（由最后的全量调用统一执行）
    if args.splits:
        logger.info("--splits 模式：跳过 OOS 合并与最终模型训练。")
        return
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
    # 释放 OOS 中间结果，为 ~2000 万行全量训练腾内存
    if oos_parts:
        del oos_parts, oos_df
        _release_memory()
    sterile_start = proto.sterile_start

    # 从 parquet 读取 sterile_start 之前的数据
    import pyarrow.dataset as ds
    dataset = ds.dataset(cache_path, format="parquet")
    final_filter = ds.field("timestamp") < _pa.scalar(str(sterile_start))
    # 特征列 float32 流式分批读取（~2000 万行全量数据，float64 一次性加载必 OOM）
    feat_cols = [
        c for c in needed_cols
        if c not in ("timestamp", "symbol", model_config.LABEL_COL)
    ]
    final_train = _scan_to_pandas_f32(dataset, needed_cols, final_filter, feat_cols)

    # 与切分循环一致：由 LABEL_COL（收益率）按阈值生成二分类 label
    final_train["label"] = (
        final_train[model_config.LABEL_COL] > get_return_threshold(final_train)
    ).astype(int)

    # 断言：无菌终审段未参与训练
    final_train["timestamp"] = pd.to_datetime(final_train["timestamp"])
    sterile_mask = final_train["timestamp"].dt.date >= sterile_start
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
    # 唯一引用移交：holder.pop() 后此处不再持有引用，
    # save_final_model 内部 del train_data 才能真正释放 ~17GB 基础帧
    holder = [final_train]
    del final_train
    _release_memory()
    save_final_model(holder.pop(), MODEL_DIR)

    logger.info("=== walk-forward 重训完成 ===")


if __name__ == "__main__":
    main()
