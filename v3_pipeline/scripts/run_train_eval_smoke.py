#!/usr/bin/env python3
"""T5 训练评测管线冒烟驱动（issue #25）：主池单标签端到端。

流程:
  1. 载主表 master_{pool}.parquet（issue #24 产物），泄漏排除断言（AC 守卫）;
  2. 段界与 30 个交易日隔离带无串段断言（AC2，日历 = 上证指数交易日）;
  3. 合并狙击标签 hit_N20_k2.0（m_scan 锁定事件表的 div 组标签）;
  4. 训练段五折时间序列折外概率 -> 校准层 [p, p²] 拟合 -> 全训练段重训终模;
  5. 复现性断言（AC3）: OOF 全程复跑一遍，逐位一致才继续;
  6. 指标表: train（折外口径）/ val（终模+校准口径）两段，
     平均精确率 + 头部五名精确率（日加权/事件加权）;
     test 段只断言在场、不出任何数字（#20 终审前每候选只碰一次）。

输出:
  v3_pipeline/reports/train_eval_smoke/
    metrics_{pool}_{label}.csv / metrics_*.md   指标表
    oof_{pool}_{label}.parquet                  折外概率产物（复现性凭证）
    val_predictions_*.parquet                   val 段校准后概率
    smoke_results.json                          口径/参数/断言台账
    progress.log                                阶段心跳

用法: python v3_pipeline/scripts/run_train_eval_smoke.py [--pool main]
      [--label-col hit_N20_k2.0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import train_eval_pipeline as tep  # noqa: E402

MASTER_DIR = REPO / "v3_pipeline" / "reports" / "feature_master"
LAB_DIR = REPO / "v3_pipeline" / "reports" / "divergence_lab" / "m_scan"
POOL_LAB = {"main": "m_fractal15_full", "backup": "m_zigzag05_nofilter"}
CALENDAR_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "train_eval_smoke"
PROGRESS = OUT_DIR / "progress.log"


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_calendar() -> np.ndarray:
    d = pd.read_parquet(CALENDAR_PATH, columns=["trade_date"])
    cal = np.sort(pd.to_datetime(d["trade_date"].astype(str)).unique())
    return cal


def md5_of_frame(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
                       + df.to_numpy().tobytes()).hexdigest()


def md_table(df: pd.DataFrame) -> str:
    """无 tabulate 依赖的 GitHub 风格 markdown 表。"""
    def _fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="main", choices=["main", "backup"])
    ap.add_argument("--label-col", default="hit_N20_k2.0")
    args = ap.parse_args()
    t0 = time.time()
    tag = f"{args.pool}_{args.label_col}"
    log(f"[T5 冒烟] pool={args.pool} label={args.label_col}")

    # ---------------------------------------------------------- 1. 主表与泄漏守卫
    master = pd.read_parquet(MASTER_DIR / f"master_{args.pool}.parquet")
    feat_cols = tep.model_feature_columns(master)
    log(f"[阶段1] 主表 {len(master)} 行 x {len(master.columns)} 列, 特征 {len(feat_cols)} 列, "
        f"泄漏排除断言通过")

    # ---------------------------------------------------------- 2. 段界与隔离带断言（AC2）
    cal = load_calendar()
    tep.assert_segment_integrity(master[["date", "seg"]], cal)
    seg_counts = master["seg"].value_counts().to_dict()
    log(f"[阶段2] 段界与隔离带断言通过: {seg_counts}")

    # ---------------------------------------------------------- 3. 标签合并
    labdir = LAB_DIR / POOL_LAB[args.pool]
    lab = tep.load_div_labels(labdir / "events.parquet", labdir / "labels.parquet",
                              args.label_col)
    df = master.merge(lab, on=["ts_code", "date"], how="left", validate="one_to_one")
    match = df[args.label_col].notna().mean()
    log(f"[阶段3] 标签合并匹配率 {match:.4%}")

    # ---------------------------------------------------------- 4. 建模型行集
    model_df = df[df["seg"].isin(["train", "val", "test"])].copy()
    # 只统计 train/val 的标签缺失；test 段不做任何逐行统计（终审前零触碰纪律）
    n_nan = model_df[model_df["seg"].isin(["train", "val"])].groupby("seg")[
        args.label_col].apply(lambda s: int(s.isna().sum()))
    log(f"[阶段4] train/val 标签 NaN 计数（截断/停牌顺延）: {n_nan.to_dict()}; 剔除后建模")
    train = model_df[(model_df["seg"] == "train") & model_df[args.label_col].notna()]
    val = model_df[(model_df["seg"] == "val") & model_df[args.label_col].notna()]
    test = model_df[model_df["seg"] == "test"]
    assert len(test) > 0, "test 段应在场（本脚本不评测）"
    log(f"  train {len(train)} 行 / val {len(val)} 行 / test {len(test)} 行(不触碰)")

    # ---------------------------------------------------------- 5. 五折 OOF + 校准层 + 终模
    train = train.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    X_tr = train[feat_cols]
    y_tr = train[args.label_col].to_numpy(dtype=np.float64)
    dates_tr = pd.to_datetime(train["date"]).to_numpy()

    oof, best_iters = tep.time_series_oof(X_tr, y_tr, dates_tr)
    oof_mask = np.isfinite(oof)
    log(f"[阶段5] 五折 OOF 完成: 覆盖 {oof_mask.sum()}/{len(y_tr)} 行（首块无折外）, "
        f"best_iters={best_iters}")

    oof2, best_iters2 = tep.time_series_oof(X_tr, y_tr, dates_tr)
    assert np.array_equal(oof, oof2, equal_nan=True) and best_iters == best_iters2, \
        "OOF 复跑不逐位一致（AC3 失败）"
    log("[阶段5b] 复现性断言通过: OOF 复跑逐位一致")

    calibrator = tep.SquaredLogitCalibrator().fit(oof[oof_mask], y_tr[oof_mask])
    log(f"  校准层系数 [p, p²] = {calibrator.coef_.tolist()}, 截距 {calibrator.intercept_:.6f}")

    n_round = tep.final_num_boost_round(best_iters)
    booster = tep.fit_final_model(X_tr, y_tr, num_boost_round=n_round)
    log(f"  终模重训完成: num_boost_round={n_round}（五折均值）")

    # ---------------------------------------------------------- 6. 指标表（train 折外 / val 终模）
    train_ev = train.loc[oof_mask, ["date", "ts_code", "event_id", "seg"]].copy()
    train_ev["y"] = y_tr[oof_mask]
    train_ev["prob"] = calibrator.predict(oof[oof_mask])
    row_train = {"segment": "train_oof"} | tep.evaluate_segment(train_ev)

    val_ev = val[["date", "ts_code", "event_id", "seg"]].copy()
    val_ev["y"] = val[args.label_col].to_numpy(dtype=np.float64)
    val_ev["prob"] = calibrator.predict(booster.predict(val[feat_cols]))
    row_val = {"segment": "val"} | tep.evaluate_segment(val_ev)

    table = pd.DataFrame([row_train, row_val])
    log(f"[阶段6] 指标表:\n{table.to_string(index=False)}")

    # ---------------------------------------------------------- 7. 落盘
    csv_path = OUT_DIR / f"metrics_{tag}.csv"
    md_path = OUT_DIR / f"metrics_{tag}.md"
    oof_path = OUT_DIR / f"oof_{tag}.parquet"
    val_path = OUT_DIR / f"val_predictions_{tag}.parquet"
    table.to_csv(csv_path, index=False)
    md_path.write_text(md_table(table) + "\n", encoding="utf-8")
    oof_df = train[["event_id", "ts_code", "date"]].copy()
    oof_df["oof_prob"] = oof
    oof_df["y"] = y_tr
    oof_df.to_parquet(oof_path, index=False)
    val_ev.to_parquet(val_path, index=False)

    results = {
        "issue": 25, "pool": args.pool, "label_col": args.label_col,
        "label_def": "hit_N20_k2.0: T+1 开盘入场, 20 交易日内 +2*ATR(14)（狙击标签, 冒烟用单标签）",
        "n_features": len(feat_cols),
        "seg_counts": {k: int(v) for k, v in seg_counts.items()},
        "label_nan_dropped": {k: int(v) for k, v in n_nan.items()},
        "lgbm_params": tep.DEFAULT_LGBM_PARAMS,
        "n_splits": tep.N_SPLITS, "best_iters": best_iters,
        "final_num_boost_round": n_round,
        "calibrator": {"form": "logistic_regression(p, p^2)",
                       "C": tep.CALIBRATOR_C, "coef": calibrator.coef_.tolist(),
                       "intercept": calibrator.intercept_},
        "metrics": table.to_dict(orient="records"),
        "assertions": {
            "ac2_segment_embargo": "PASS (assert_segment_integrity: seg 逐行一致/隔离带无串段/两段各>=30交易日)",
            "ac3_oof_reproducible": "PASS (OOF 复跑 np.array_equal equal_nan 逐位一致)",
            "leakage_exclusion": "PASS (model_feature_columns = feature_master 口径 + 排除模式零命中)",
            "test_untouched": f"PASS (test {len(test)} 行在场, 未出任何指标)",
        },
        "oof_md5": md5_of_frame(oof_df),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with (OUT_DIR / "smoke_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[阶段7] 落盘完成 -> {OUT_DIR} (耗时 {results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
