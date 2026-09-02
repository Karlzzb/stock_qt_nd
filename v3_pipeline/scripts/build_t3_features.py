#!/usr/bin/env python3
"""v5-T4 新特征全池事件日快照构建驱动（issue #24，特征主表来源 4）。

输入:
  stock_data/{daily,daily_basic,stk_limit,index,meta,universe}     (原始数据)
  v3_pipeline/reports/feature_matrix/{main,backup}_pool_features.parquet  (事件键)
输出:
  v3_pipeline/reports/feature_master/cache/t3_snapshot_{main,backup}.parquet
  v3_pipeline/reports/feature_master/cache/t3_results.json
  v3_pipeline/reports/feature_master/cache/t3_progress.log  (阶段日志/心跳)

验收口径:
  1. 前缀稳定性抽检：抽样 (股, 事件日) 单元格，截断 [T-1600 自然日, T] 重算
     与全历史快照值逐位一致（rtol 1e-9），0 不一致才放行。
  2. 覆盖率登记：全特征列两池 NaN 率 + 评审遗留约定口径（#7/#8 NaN 覆盖、
     #14 触发率、#3 次新 NaN、#26/#27 早年 NaN、口径来源分布）。
  3. 快照键覆盖：两池事件键 100% 命中（缺失计数落盘，主表左连接记 NaN）。

用法: python v3_pipeline/scripts/build_t3_features.py [--workers 24] [--spot 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import t3_features as t3  # noqa: E402

FM_DIR = REPO / "v3_pipeline" / "reports" / "feature_matrix"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "feature_master" / "cache"
POOLS = ("main", "backup")
PROGRESS = OUT_DIR / "t3_progress.log"

SPOT_SEED = 20260902


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_pool_keys():
    keys = {}
    for pool in POOLS:
        fm = pd.read_parquet(FM_DIR / f"{pool}_pool_features.parquet",
                             columns=["ts_code", "date"])
        fm["date"] = pd.to_datetime(fm["date"])
        assert not fm.duplicated(["ts_code", "date"]).any()
        keys[pool] = fm[["ts_code", "date"]].drop_duplicates().reset_index(
            drop=True)
    return keys


def coverage_report(snap):
    """两池覆盖率 + 评审遗留约定口径。"""
    rep = {"pools": {}, "callouts": {}}
    for pool, df in snap.items():
        rates = {c: float(df[c].isna().mean()) for c in t3.T3_COLUMNS}
        rep["pools"][pool] = {"rows": int(len(df)), "nan_rate": rates}
    main = snap["main"]
    rep["callouts"] = {
        # #14 撬板触发率（事件池）
        "downlimit_unsealed_trigger_rate_main":
            float((main["DOWNLIMIT_UNSEALED"] == 1).mean()),
        "downlimit_unsealed_trigger_rate_backup":
            float((snap["backup"]["DOWNLIMIT_UNSEALED"] == 1).mean()),
        # 封涨停口径来源分布（主池）：2=精确 1=近似 0=缺失
        "sealed_src_dist_main": {
            str(k): float(v) for k, v in
            main["LIMITUP_SEALED_SRC"].value_counts(normalize=True).items()},
        # #10 精确版 vs 主板规则：2007 后精确判定占比
        "oneword_trigger_rate_main":
            float((main["ONEWORD_LIMITUP"] == 1).mean()),
        "touch_fail_trigger_rate_main":
            float((main["TOUCH_LIMITUP_FAIL"] == 1).mean()),
        # #28 风险警示状态分布（主池）
        "st_status_dist_main": {
            str(k): float(v) for k, v in
            main["ST_STATUS"].value_counts(normalize=True).items()},
        # #25 市场收益口径拼接标记分布（主池）
        "resid_mksrc_dist_main": {
            str(k): float(v) for k, v in
            main["RESID_MOM60_MKSRC"].value_counts(normalize=True).items()},
        "resid_indna_rate_main":
            float((main["RESID_MOM60_INDNA"] == 1).mean()),
        # #29 面值警戒：事件池中 close<3 元占比（特征启用区间的信息量登记）
        "par_gap_active_rate_main":
            float((main["PAR_VALUE_GAP"] < np.log(3.0)).mean()),
        "consec_susp_gap_rate_main":
            float((main["CONSEC_SUSP_GAP"] == 1).mean()),
    }
    return rep


def prefix_spot_check(feat_full, keys, n_spot):
    """前缀稳定性抽检：截断重算 vs 全历史快照，逐位一致。"""
    rng = np.random.default_rng(SPOT_SEED)
    cells = pd.concat([k.assign(pool=p) for p, k in keys.items()])
    picks = cells.iloc[rng.choice(len(cells), size=min(n_spot, len(cells)),
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
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("构建静态上下文 ctx ...")
    ctx = t3.build_ctx()
    log(f"交易日历 {len(ctx['calendar'])} 天 "
        f"[{pd.Timestamp(ctx['calendar'][0]).date()} .. "
        f"{pd.Timestamp(ctx['calendar'][-1]).date()}] ({time.time()-t0:.0f}s)")

    keys = load_pool_keys()
    log(f"事件键: main={len(keys['main'])} backup={len(keys['backup'])}")

    log(f"装配全市场面板 (workers={args.workers}) ...")
    panel = t3.build_panel(workers=args.workers)
    log(f"面板 {panel.shape} ({time.time()-t0:.0f}s)")

    log("计算 T3 全量特征 ...")
    feat = t3.compute_all(panel, ctx)
    del panel
    log(f"特征面板 {feat.shape} ({time.time()-t0:.0f}s)")

    log("事件日快照 ...")
    snap = {}
    key_cov = {}
    for pool in POOLS:
        s = t3.snapshot(feat, keys[pool])
        path = OUT_DIR / f"t3_snapshot_{pool}.parquet"
        s.to_parquet(path, index=False)
        snap[pool] = s
        n_miss = int(keys[pool].merge(
            feat[["ts_code", "date"]].drop_duplicates(),
            on=["ts_code", "date"], how="left", indicator=True)
            ["_merge"].eq("left_only").sum())
        key_cov[pool] = {"keys": int(len(keys[pool])), "missing": n_miss}
        log(f"[{pool}] 快照 {s.shape} -> {path.name} "
            f"(键缺失 {n_miss}) ({time.time()-t0:.0f}s)")

    rep = coverage_report(snap)
    for pool in POOLS:
        rep["pools"][pool]["key_coverage"] = key_cov[pool]
    log("覆盖率报告已生成")

    if args.skip_spot:
        stab = {"n_checked": 0, "mismatches": [], "skipped": True}
    else:
        log(f"前缀稳定性抽检 {args.spot} 个单元格 ...")
        stab = prefix_spot_check(feat, keys, args.spot)
        log(f"抽检 {stab['n_checked']} 单元, 不一致 {len(stab['mismatches'])} "
            f"({time.time()-t0:.0f}s)")
        assert stab["n_checked"] >= args.spot - 2, "前缀稳定性抽检覆盖不足"
        assert not stab["mismatches"], \
            f"前缀稳定性不一致: {stab['mismatches'][:3]}"

    results = {"prefix_stability": stab, "coverage": rep,
               "columns": t3.T3_COLUMNS, "cn": t3.T3_CN,
               "elapsed_sec": time.time() - t0}
    rpath = OUT_DIR / "t3_results.json"
    rpath.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                default=str))
    log(f"结果 -> {rpath}; 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
