#!/usr/bin/env python3
"""T6 标签赛全量裁决驱动（issue #26）：十九候选 x 三十六配置，指标表以配置为行。

预登记正文见 issue #26 置顶评论（先于本赛任何结果落盘），要点:
  - 主池训练段 2838 < 3000 触发 #20 兜底：主备合并池（--pool merged）为正赛，
    主池（--pool main）为对照；裁决只在正赛。
  - 网格 36 组全因子（label_race.GRID），每配置管线与 #25 完全一致
    （五折 OOF -> 校准层 [p, p²] -> 终模均值轮数 -> train_oof/val 指标）。
  - 每候选选配置：val 头部五名精确率（日加权）-> val 平均精确率 -> 网格序；
    总裁决加平均精确率中位数约束。
  - 十九候选当选配置 OOF 与跑赛落盘产物逐位一致断言（不可旁路）；test 段零触碰。

并行结构: 进程池初始化时每 worker 载主表与标签表各一次（_globals），
任务粒度 = (候选, 配置)，负载均衡且避免重复 pickle 大帧。
断点续跑: 已落盘的 metrics_{pool}_{candidate}.csv 候选直接跳过。

输出（v3_pipeline/reports/label_race/）:
  master_merged.parquet                  合并池主表（复核复用凭证）
  metrics_{pool}_{candidate}.csv         每候选指标表（配置为行，36 行）
  summary_{pool}.csv                     十九候选当选配置汇总（裁决输入）
  adjudication_{pool}.json               裁决结果（正赛为正式，对照为参考）
  race_results_{pool}.json               口径/断言/台账
  progress.log                           阶段心跳

用法: python v3_pipeline/scripts/run_label_race.py --pool merged --workers 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import label_race as lr  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402

MASTER_DIR = REPO / "v3_pipeline" / "reports" / "feature_master"
CALENDAR_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "label_race"
PROGRESS = OUT_DIR / "progress.log"

_globals: dict = {}


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_calendar() -> np.ndarray:
    d = pd.read_parquet(CALENDAR_PATH, columns=["trade_date"])
    return np.sort(pd.to_datetime(d["trade_date"].astype(str)).unique())


def load_labels(pool: str) -> pd.DataFrame:
    """标签表: merged = 两池标签表带 pool 列纵向拼接; 否则单池。"""
    if pool == "merged":
        parts = []
        for p in ("main", "backup"):
            t = pd.read_parquet(OUT_DIR / f"labels_race_{p}.parquet")
            t["pool"] = p
            parts.append(t)
        return pd.concat(parts, ignore_index=True)
    return pd.read_parquet(OUT_DIR / f"labels_race_{pool}.parquet")


def load_master(pool: str) -> pd.DataFrame:
    """主表: merged 走 label_race.build_merged_master（首跑后落盘复用）。"""
    if pool == "merged":
        cache = OUT_DIR / "master_merged.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        main = pd.read_parquet(MASTER_DIR / "master_main.parquet")
        backup = pd.read_parquet(MASTER_DIR / "master_backup.parquet")
        merged = lr.build_merged_master(main, backup)
        merged.to_parquet(cache, index=False)
        return merged
    return pd.read_parquet(MASTER_DIR / f"master_{pool}.parquet")


# ------------------------------------------------------------- worker 侧
def _worker_init(pool: str) -> None:
    """每 worker 载一次主表+标签，合并出建模型行集（特征列与段断言主进程已过）。"""
    sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
    import label_race as _lr
    import train_eval_pipeline as _tep
    master = load_master(pool)
    feat_cols = _tep.model_feature_columns(master)
    labels = load_labels(pool)
    on = ["pool", "ts_code", "date"] if pool == "merged" else ["ts_code", "date"]
    df = master.merge(labels, on=on, how="left",
                      validate="one_to_one")
    model_df = df[df["seg"].isin(["train", "val", "test"])]
    _globals.clear()
    _globals.update({"lr": _lr, "tep": _tep, "model_df": model_df,
                     "feat_cols": feat_cols})


def _run_one(candidate: str, config_id: int) -> tuple:
    """单候选单配置跑训（worker 内）。

    返回 (candidate, config_id, 指标行, event_ids, oof)；oof 与 event_ids
    供主进程按候选落盘，当选配置复现性断言以该落盘产物为对照基准。
    """
    _lr, model_df = _globals["lr"], _globals["model_df"]
    feat_cols = _globals["feat_cols"]
    cfg = _lr.GRID[config_id]
    sub = model_df[model_df["seg"].isin(["train", "val"])]
    sub = sub[sub[candidate].notna()]
    train = sub[sub["seg"] == "train"]
    val = sub[sub["seg"] == "val"]
    row, art = _lr.run_single_config(train, val, feat_cols, candidate,
                                     _lr.grid_params(cfg))
    row = {"candidate": candidate, "config_id": config_id, **cfg, **row}
    train_sorted = train.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    return (candidate, config_id, row,
            train_sorted["event_id"].to_numpy(), art["oof"])


def _rerun_oof(candidate: str, config_id: int, pool: str) -> tuple[str, bool, str]:
    """当选配置复现性断言（不可旁路）。

    两模式:
      - stored_vs_rerun: 跑赛已落盘折外概率 -> 重算一遍与之逐位对比（首选）;
      - self_rerun_x2: 折外概率未落盘（首批跑赛产物早于落盘加固）->
        全程自复跑两遍逐位一致后补档单配置列（即 #25 口径，强度等价登记）。
    返回 (candidate, 是否逐位一致, 模式)。
    """
    _lr, model_df = _globals["lr"], _globals["model_df"]
    feat_cols = _globals["feat_cols"]
    cfg = _lr.GRID[config_id]
    sub = model_df[(model_df["seg"] == "train") & model_df[candidate].notna()]
    sub = sub.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    X = sub[feat_cols]
    y = sub[candidate].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(sub["date"]).to_numpy()
    tep_ = _globals["tep"]
    params = _lr.grid_params(cfg)
    oof_path = OUT_DIR / f"oof_{pool}_{candidate}.parquet"
    if oof_path.exists():
        oof_new, _ = tep_.time_series_oof(X, y, dates, params=params)
        stored = pd.read_parquet(oof_path)
        same_keys = (stored["event_id"].to_numpy() == sub["event_id"].to_numpy()).all()
        ok = bool(same_keys) and np.array_equal(
            oof_new, stored[f"config_{config_id}"].to_numpy(), equal_nan=True)
        return candidate, ok, "stored_vs_rerun"
    oof1, it1 = tep_.time_series_oof(X, y, dates, params=params)
    oof2, it2 = tep_.time_series_oof(X, y, dates, params=params)
    ok = bool(np.array_equal(oof1, oof2, equal_nan=True)) and it1 == it2
    if ok:
        pd.DataFrame({"event_id": sub["event_id"].to_numpy(),
                      f"config_{config_id}": oof1}).to_parquet(oof_path, index=False)
    return candidate, ok, "self_rerun_x2"


# ------------------------------------------------------------- 主进程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=["merged", "main"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    pool = args.pool
    cands = lr.candidate_labels()
    log(f"[T6 标签赛] pool={pool} 开工: {len(cands)} 候选 x {len(lr.GRID)} 配置, "
        f"workers={args.workers}")

    # ---------------------------------------------------------- 1. 主表/标签/段断言
    master = load_master(pool)
    feat_cols = tep.model_feature_columns(master)
    log(f"[阶段1] 主表 {len(master)} 行 x {len(master.columns)} 列, 特征 {len(feat_cols)} 列")
    cal = load_calendar()
    tep.assert_segment_integrity(master[["date", "seg"]], cal)
    seg_counts = {k: int(v) for k, v in master["seg"].value_counts().items()}
    log(f"[阶段1] 段界与隔离带断言通过: {seg_counts}")
    n_train_events = int((master["seg"] == "train").sum())
    if pool == "merged":
        # 兜底触发条件实测（非硬编码）：主池 train < 3000 才允许合并池升正赛
        main_train = int((pd.read_parquet(MASTER_DIR / "master_main.parquet",
                                          columns=["seg"])["seg"] == "train").sum())
        assert main_train < lr.MIN_TRAIN_EVENTS, \
            f"主池 train {main_train} >= {lr.MIN_TRAIN_EVENTS}，兜底未触发，合并池不得升正赛"
        assert n_train_events >= lr.MIN_TRAIN_EVENTS, \
            f"合并池 train {n_train_events} < {lr.MIN_TRAIN_EVENTS}（兜底后仍不足）"
        log(f"[阶段1] 兜底规则实测触发: 主池 train {main_train} < {lr.MIN_TRAIN_EVENTS}, "
            f"合并池 train {n_train_events} 升正赛")

    labels = load_labels(pool)
    on = ["pool", "ts_code", "date"] if pool == "merged" else ["ts_code", "date"]
    df = master.merge(labels, on=on, how="left", validate="one_to_one")
    test_present = int((df["seg"] == "test").sum())
    assert test_present > 0, "test 段应在场（本赛零触碰）"
    assert df[lr.candidate_labels()[1:]].notna().any().all(), "存在全 NaN 收益标签列"

    # ---------------------------------------------------------- 2. 网格跑训（断点续跑）
    pending: dict[str, list[int]] = {}
    for cand in cands:
        csv = OUT_DIR / f"metrics_{pool}_{cand}.csv"
        if csv.exists():
            log(f"[阶段2] {cand}: 已有落盘指标表, 跳过（断点续跑）")
            continue
        pending[cand] = list(range(len(lr.GRID)))
    rows_buf: dict[str, dict[int, dict]] = {c: {} for c in pending}
    oof_buf: dict[str, dict[int, np.ndarray]] = {c: {} for c in pending}
    eid_buf: dict[str, np.ndarray] = {}
    n_tasks = sum(len(v) for v in pending.values())
    log(f"[阶段2] 待跑 {len(pending)} 候选 / {n_tasks} 配置任务")
    done = 0
    if n_tasks:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init, initargs=(pool,)) as ex:
            futs = {ex.submit(_run_one, c, i): (c, i)
                    for c, ids in pending.items() for i in ids}
            for fut in as_completed(futs):
                c, i = futs[fut]
                try:
                    _, _, row, eids, oof = fut.result()
                except Exception:
                    log(f"[失败] {c} config_id={i}\n{traceback.format_exc()}")
                    raise
                rows_buf[c][i] = row
                oof_buf[c][i] = oof
                if c in eid_buf:
                    assert (eid_buf[c] == eids).all(), f"{c} 配置间训练行序不一致"
                else:
                    eid_buf[c] = eids
                done += 1
                if done % 20 == 0 or done == n_tasks:
                    log(f"[心跳] 配置任务 {done}/{n_tasks} "
                        f"(耗时 {round(time.time() - t0)}s)")
                if len(rows_buf[c]) == len(lr.GRID):
                    table = pd.DataFrame([rows_buf[c][i] for i in range(len(lr.GRID))])
                    table.to_csv(OUT_DIR / f"metrics_{pool}_{c}.csv", index=False)
                    oof_df = pd.DataFrame(
                        {f"config_{i}": oof_buf[c][i] for i in range(len(lr.GRID))})
                    oof_df.insert(0, "event_id", eid_buf[c])
                    oof_df.to_parquet(OUT_DIR / f"oof_{pool}_{c}.parquet", index=False)
                    del oof_buf[c]
                    log(f"[阶段2] {c}: {len(lr.GRID)} 配置指标表与折外概率落盘")

    # ---------------------------------------------------------- 3. 选配置 + 汇总
    summary_rows = []
    for cand in cands:
        table = pd.read_csv(OUT_DIR / f"metrics_{pool}_{cand}.csv")
        assert len(table) == len(lr.GRID), f"{cand} 指标表行数 {len(table)} != {len(lr.GRID)}"
        best = table.iloc[lr.select_best_config(table)]
        summary_rows.append({"candidate": cand, "label_cn": lr.candidate_cn(cand),
                             **{k: best[k] for k in (
            "config_id", "num_leaves", "min_data_in_leaf", "learning_rate",
            "feature_fraction", "final_num_boost_round",
            "val_n_events", "val_base_rate", "val_average_precision",
            "val_precision_at_5_dayavg", "val_precision_at_5_eventavg",
            "val_n_days", "train_oof_average_precision",
            "train_oof_precision_at_5_dayavg")}})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / f"summary_{pool}.csv", index=False)
    log(f"[阶段3] 十九候选当选配置汇总落盘 summary_{pool}.csv")

    # ---------------------------------------------------------- 4. 当选配置复现性断言（正赛与对照同口径）
    repro_tasks = [(r["candidate"], int(r["config_id"])) for r in summary_rows]
    repro: dict[str, bool] = {}
    repro_mode: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init, initargs=(pool,)) as ex:
        futs = {ex.submit(_rerun_oof, c, i, pool): c for c, i in repro_tasks}
        for fut in as_completed(futs):
            c, ok, mode = fut.result()
            repro[c] = ok
            repro_mode[c] = mode
            log(f"[阶段4] {c} 当选配置 OOF 复现断言[{mode}]: "
                f"{'逐位一致' if ok else '不一致!'}")
    assert all(repro.values()), \
        f"复现性断言失败: {[c for c, ok in repro.items() if not ok]}"
    log("[阶段4] 十九候选当选配置 OOF 复现断言全部逐位一致")

    # ---------------------------------------------------------- 5. 裁决（独立复核前为待核状态）
    adj = lr.adjudicate(summary)
    adj["role"] = "正赛" if pool == "merged" else "对照（不参与正式裁决）"
    adj["status"] = "provisional_pending_independent_review"
    with (OUT_DIR / f"adjudication_{pool}.json").open("w", encoding="utf-8") as f:
        json.dump(adj, f, ensure_ascii=False, indent=2)
    log(f"[阶段5] 裁决（待独立复核）: winner={adj['winner']} "
        f"(头部五名精确率日加权={adj['winner_val_precision_at_5_dayavg']:.4f}, "
        f"平均精确率={adj['winner_val_average_precision']:.4f} vs 中位数 "
        f"{adj['median_val_average_precision']:.4f}, "
        f"约束 {'通过' if adj['ap_constraint_passed'] else '未过——无当选'})")

    # ---------------------------------------------------------- 6. 台账
    results = {
        "issue": 26, "pool": pool, "role": adj["role"],
        "n_candidates": len(cands), "grid_size": len(lr.GRID), "grid": lr.GRID,
        "n_features": len(feat_cols), "seg_counts": seg_counts,
        "fallback_rule": ("主池 train 2838 < 3000 -> 合并池升正赛（#20 预登记兜底, 实测触发）"
                          if pool == "merged" else "本池为对照"),
        "adjudication": adj,
        "assertions": {
            "segment_embargo": "PASS (assert_segment_integrity)",
            "leakage_exclusion": "PASS (model_feature_columns 排除模式零命中)",
            "winner_oof_reproducible": (
                f"PASS (19 候选当选配置 OOF 复现断言逐位一致; 模式: {repro_mode})"),
            "test_untouched": f"PASS (test {test_present} 行在场, 零指标零逐行统计)",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with (OUT_DIR / f"race_results_{pool}.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[完成] pool={pool} 耗时 {results['elapsed_sec']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
