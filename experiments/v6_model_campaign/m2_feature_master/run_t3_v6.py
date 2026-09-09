#!/usr/bin/env python3
"""来源 4:T3 新特征 v6 事件日快照构建驱动(战役 #47 M2,issue #49)。

机器来源:v3_pipeline/scripts/build_t3_features.py 改造(#44 §2.4 来源 4,
适配等级 B=改键+单池)。改造点:
  - 键源: v5 双池 -> v6 事件表 96,577 键(单池);
  - 覆盖率登记双池口径改单池("all");评审遗留 callouts 原样保留(_main 后缀改 _all);
  - 计算本体 build_ctx/build_panel/compute_all/snapshot/prefix_recompute_at 零改动。

验收口径(v5 原样):
  1. 前缀稳定性抽检: 抽样 (股, 事件日) 单元格,截断 [T-1600 自然日, T] 重算
     与全历史快照值逐位一致(rtol 1e-9),0 不一致才放行;
  2. 覆盖率登记: 全特征列 NaN 率 + 评审遗留约定口径;
  3. 快照键覆盖: 缺失计数落盘,主表左连接记 NaN。

输入:
  stock_data/{daily,daily_basic,stk_limit,index,meta,universe}     (原始数据,只读)
  事件表 experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet(只读)
输出:
  cache/s4_t3_snapshot.parquet
  cache/s4_results.json

用法: python run_t3_v6.py [--workers 24] [--spot 8] [--out cache/s4_t3_snapshot.parquet]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import t3_features as t3  # noqa: E402

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

SPOT_SEED = 20260902


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [s4] {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_event_keys():
    ev = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date"])
    ev = ev.rename(columns={"event_date": "date"})
    ev["date"] = pd.to_datetime(ev["date"])
    assert not ev.duplicated(["ts_code", "date"]).any(), "事件键不唯一"
    return ev[["ts_code", "date"]].reset_index(drop=True)


def coverage_report(snap):
    """单池覆盖率 + 评审遗留约定口径(v5 callouts 原样,后缀 _main 改 _all)。"""
    rates = {c: float(snap[c].isna().mean()) for c in t3.T3_COLUMNS}
    rep = {"pools": {"all": {"rows": int(len(snap)), "nan_rate": rates}},
           "callouts": {
               # #14 撬板触发率(事件池)
               "downlimit_unsealed_trigger_rate_all":
                   float((snap["DOWNLIMIT_UNSEALED"] == 1).mean()),
               # 封涨停口径来源分布: 2=精确 1=近似 0=缺失
               "sealed_src_dist_all": {
                   str(k): float(v) for k, v in
                   snap["LIMITUP_SEALED_SRC"].value_counts(normalize=True).items()},
               # #10 精确版 vs 主板规则
               "oneword_trigger_rate_all":
                   float((snap["ONEWORD_LIMITUP"] == 1).mean()),
               "touch_fail_trigger_rate_all":
                   float((snap["TOUCH_LIMITUP_FAIL"] == 1).mean()),
               # #28 风险警示状态分布
               "st_status_dist_all": {
                   str(k): float(v) for k, v in
                   snap["ST_STATUS"].value_counts(normalize=True).items()},
               # #25 市场收益口径拼接标记分布
               "resid_mksrc_dist_all": {
                   str(k): float(v) for k, v in
                   snap["RESID_MOM60_MKSRC"].value_counts(normalize=True).items()},
               "resid_indna_rate_all":
                   float((snap["RESID_MOM60_INDNA"] == 1).mean()),
               # #29 面值警戒
               "par_gap_active_rate_all":
                   float((snap["PAR_VALUE_GAP"] < np.log(3.0)).mean()),
               "consec_susp_gap_rate_all":
                   float((snap["CONSEC_SUSP_GAP"] == 1).mean()),
           }}
    return rep


def prefix_spot_check(feat_full, keys, n_spot):
    """前缀稳定性抽检: 截断重算 vs 全历史快照,逐位一致(v5 原口径)。"""
    rng = np.random.default_rng(SPOT_SEED)
    picks = keys.iloc[rng.choice(len(keys), size=min(n_spot, len(keys)),
                                 replace=False)]
    ctx = t3.build_ctx()
    n_checked, mismatches = 0, []
    for row in picks.itertuples():
        T = pd.Timestamp(row.date)
        ref = t3.prefix_recompute_at(t3.STOCK_DATA, ctx, row.ts_code, T,
                                     workers=0)
        if ref is None:
            continue
        full = feat_full[(feat_full["ts_code"] == row.ts_code)
                         & (feat_full["date"] == T)]
        if len(full) == 0:
            continue
        a = full[t3.T3_COLUMNS].iloc[0].to_numpy(np.float64)
        b = ref[t3.T3_COLUMNS].iloc[0].to_numpy(np.float64)
        n_checked += 1
        if not np.allclose(a, b, rtol=1e-9, atol=0, equal_nan=True):
            bad = [t3.T3_COLUMNS[k] for k in np.where(
                ~np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True))[0]]
            mismatches.append({"ts_code": row.ts_code,
                               "date": str(T.date()), "cols": bad[:10]})
        log(f"  抽检 {row.ts_code} {T.date()} 完成")
    return {"n_checked": n_checked, "mismatches": mismatches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--spot", type=int, default=8, help="前缀稳定性抽检单元数")
    ap.add_argument("--skip-spot", action="store_true")
    ap.add_argument("--out", type=str, default=str(CACHE_DIR / "s4_t3_snapshot.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log("构建静态上下文 ctx ...")
    ctx = t3.build_ctx()
    log(f"交易日历 {len(ctx['calendar'])} 天 "
        f"[{pd.Timestamp(ctx['calendar'][0]).date()} .. "
        f"{pd.Timestamp(ctx['calendar'][-1]).date()}] ({time.time() - t0:.0f}s)")

    keys = load_event_keys()
    log(f"事件键(单池) {len(keys)}")

    log(f"装配全市场面板 (workers={args.workers}) ...")
    panel = t3.build_panel(workers=args.workers)
    log(f"面板 {panel.shape} ({time.time() - t0:.0f}s)")

    log("计算 T3 全量特征 ...")
    feat = t3.compute_all(panel, ctx)
    del panel
    log(f"特征面板 {feat.shape} ({time.time() - t0:.0f}s)")

    log("事件日快照 ...")
    snap = t3.snapshot(feat, keys)
    out_path = Path(args.out)
    snap.to_parquet(out_path, index=False)
    n_miss = int(keys.merge(
        feat[["ts_code", "date"]].drop_duplicates(),
        on=["ts_code", "date"], how="left", indicator=True)
        ["_merge"].eq("left_only").sum())
    log(f"快照 {snap.shape} -> {out_path.name} (键缺失 {n_miss}) ({time.time() - t0:.0f}s)")

    rep = coverage_report(snap)
    rep["pools"]["all"]["key_coverage"] = {"keys": int(len(keys)), "missing": n_miss}
    log("覆盖率报告已生成")

    if args.skip_spot:
        stab = {"n_checked": 0, "mismatches": [], "skipped": True}
    else:
        log(f"前缀稳定性抽检 {args.spot} 个单元格 ...")
        stab = prefix_spot_check(feat, keys, args.spot)
        log(f"抽检 {stab['n_checked']} 单元,不一致 {len(stab['mismatches'])} "
            f"({time.time() - t0:.0f}s)")
        assert stab["n_checked"] >= args.spot - 2, "前缀稳定性抽检覆盖不足"
        assert not stab["mismatches"], \
            f"前缀稳定性不一致: {stab['mismatches'][:3]}"

    results = {"prefix_stability": stab, "coverage": rep,
               "columns": t3.T3_COLUMNS, "cn": t3.T3_CN,
               "elapsed_sec": time.time() - t0}
    rpath = CACHE_DIR / "s4_results.json" \
        if out_path.resolve() == (CACHE_DIR / "s4_t3_snapshot.parquet").resolve() \
        else out_path.with_suffix(".results.json")
    rpath.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                default=str))
    log(f"结果 -> {rpath}; 总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
