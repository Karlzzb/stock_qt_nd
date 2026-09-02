#!/usr/bin/env python3
"""V4 日频特征全池事件日快照重建驱动（issue #22 来源 3）。

输入:
  v3_pipeline/reports/feature_matrix/{main,backup}_pool_features.parquet  (事件键)
  stock_data/daily/*.parquet                                              (原始日线)
输出:
  v3_pipeline/reports/feature_master/cache/v4daily_snapshot_{main,backup}.parquet
  v3_pipeline/reports/feature_master/cache/v4daily_snapshot_results.json
中间产物:
  cache/v4daily_parts/{code}.parquet  逐股事件日特征行（复跑时跳过已存在）

口径见 v3_pipeline/src/v4_daily_snapshot.py 模块 docstring（含与旧缓存的偏离登记）。
横截面 cs_* 排名群体 = 当日全市场（与原 enrich 一致），因此采用两阶段：
  Pass A 逐股全历史算特征，抽取事件日并集行落盘；
  Pass B 按日期块装配全市场横截面，复用原 _calculate_cross_features 排名，
        只保留两池事件单元格。

用法: python v3_pipeline/scripts/rebuild_v4_daily_snapshot.py [--workers 24] [--sample 0]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import src.feature_pipeline_v2 as fp2  # noqa: E402
import v4_daily_snapshot as v4s  # noqa: E402

FM_DIR = REPO / "v3_pipeline" / "reports" / "feature_matrix"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "feature_master" / "cache"
PARTS_DIR = OUT_DIR / "v4daily_parts"
DAILY_DIR = REPO / "stock_data" / "daily"
POOLS = ("main", "backup")
INDEX_CODES = ("000001.SH", "399001.SZ")
PROGRESS = OUT_DIR / "v4daily_progress.log"

DATE_CHUNK = 250  # 横截面排名的日期块大小

# 前缀稳定性抽检规模（确定性种子）
N_ASSERT_STOCKS = 12
ASSERT_DAYS_PER_STOCK = 3
ASSERT_SEED = 42


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_pool_keys():
    """两池事件键 + 事件日并集（按股）。"""
    keys, per_stock = {}, defaultdict(set)
    for pool in POOLS:
        fm = pd.read_parquet(FM_DIR / f"{pool}_pool_features.parquet",
                             columns=["ts_code", "date"])
        fm["date"] = pd.to_datetime(fm["date"])
        assert not fm.duplicated(["ts_code", "date"]).any()
        keys[pool] = fm
        for code, d in zip(fm["ts_code"], fm["date"]):
            per_stock[code].add(d)
    return keys, per_stock


def market_rank_frame(event_dates):
    """rank_return / rank_volume 的全市场当日横截面百分位（V2 原口径）。

    排名用当日全市场全横截面；只在事件日并集上落盘。
    返回 DataFrame(symbol, timestamp, rank_return, rank_volume)。
    """
    frames = []
    files = sorted(DAILY_DIR.glob("*.parquet"))
    for path in files:
        code = path.stem
        if code in INDEX_CODES:
            continue  # 指数不参与个股横截面排名
        df = pd.read_parquet(path, columns=["trade_date", "close", "vol"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # pct_change 需全历史（组内首日 NaN，与原管线一致），算完再过滤到事件日
        df = df.sort_values("trade_date")
        df["pct_change"] = df["close"].pct_change()
        df = df[df["trade_date"].isin(event_dates)]
        if len(df):
            frames.append(pd.DataFrame({"symbol": code, "timestamp": df["trade_date"],
                                        "pct_change": df["pct_change"].to_numpy(),
                                        "volume": df["vol"].to_numpy()}))
    big = pd.concat(frames, ignore_index=True)
    grp = big.groupby("timestamp")
    big["rank_return"] = grp["pct_change"].rank(pct=True)
    big["rank_volume"] = grp["volume"].rank(pct=True)
    # 原口径: alpha 步骤末尾 fillna(0)
    big[["rank_return", "rank_volume"]] = big[["rank_return", "rank_volume"]].fillna(0.0)
    return big[["symbol", "timestamp", "rank_return", "rank_volume"]]


def _worker(task):
    """Pass A: 单股全历史特征 -> 事件日并集行 -> parts parquet。"""
    code, dates = task
    try:
        path = DAILY_DIR / f"{code}.parquet"
        if not path.exists():
            return code, 0, "缺文件"
        out_path = PARTS_DIR / f"{code}.parquet"
        if out_path.exists():
            return code, -1, None  # 缓存命中
        df = v4s.load_stock_v2(path, code)
        feat = v4s.compute_stock_features(df, pipe=fp2.FeaturePipeline(None, None))
        if feat is None:
            return code, 0, "历史不足100行"
        snap = v4s.snapshot_rows(feat, dates)
        if len(snap) == 0:
            return code, 0, None
        snap = snap.drop(columns=[c for c in v4s.MARKET_RANK_COLS if c in snap.columns])
        snap.to_parquet(out_path, index=False)
        return code, len(snap), None
    except Exception as e:  # noqa: BLE001
        return code, 0, repr(e)


def market_features_for_dates(dates):
    """对全部事件日逐日算大盘特征（V2 原函数，指数截断语义内建）。"""
    df_sh = v4s.load_index_v2(DAILY_DIR / "000001.SH.parquet")
    df_sz = v4s.load_index_v2(DAILY_DIR / "399001.SZ.parquet")
    pipe = fp2.FeaturePipeline(None, None)
    frames = [pipe._calculate_market_features(d, df_sh=df_sh, df_sz=df_sz)
              for d in sorted(pd.to_datetime(list(dates)))]
    out = pd.concat(frames)
    out.index.name = "timestamp"
    return out.reset_index()


def prefix_stability_assert(per_stock):
    """快照口径前缀稳定性: 抽样股票 x 事件日, 全历史值与截断重算必须逐位一致。"""
    rng = np.random.default_rng(ASSERT_SEED)
    stocks = sorted(per_stock)
    picks = rng.choice(len(stocks), size=min(N_ASSERT_STOCKS, len(stocks)),
                       replace=False)
    pipe = fp2.FeaturePipeline(None, None)
    n_checked, mismatches = 0, []
    for i in picks:
        code = stocks[i]
        df = v4s.load_stock_v2(DAILY_DIR / f"{code}.parquet", code)
        feat_full = v4s.compute_stock_features(df, pipe=pipe)
        if feat_full is None:
            continue
        days = sorted(d for d in per_stock[code]
                      if d in set(feat_full["timestamp"]))
        if not days:
            continue
        day_picks = rng.choice(len(days),
                               size=min(ASSERT_DAYS_PER_STOCK, len(days)),
                               replace=False)
        for j in day_picks:
            T = days[j]
            ref = v4s.prefix_recompute_at(DAILY_DIR / f"{code}.parquet", code, T,
                                          pipe=pipe)
            if ref is None:
                continue
            row_full = feat_full[feat_full["timestamp"] == T]
            cmp_cols = [c for c in feat_full.columns
                        if c not in v4s.MARKET_RANK_COLS + v4s.KEY_COLS]
            a = row_full[cmp_cols].iloc[0].to_numpy(np.float64)
            b = ref[cmp_cols].iloc[0].to_numpy(np.float64)
            n_checked += 1
            if not np.allclose(a, b, rtol=1e-9, atol=0, equal_nan=True):
                bad = [cmp_cols[k] for k in np.where(
                    ~np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True))[0]]
                mismatches.append({"ts_code": code, "date": str(T.date()),
                                   "cols": bad[:10]})
    return {"n_checked": n_checked, "mismatches": mismatches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sample", type=int, default=0, help="只取前 N 只股票 (冒烟)")
    args = ap.parse_args()
    t0 = time.time()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    keys, per_stock = load_pool_keys()
    event_dates = set()
    for fm in keys.values():
        event_dates.update(fm["date"])
    log(f"事件键: main={len(keys['main'])} backup={len(keys['backup'])}; "
        f"涉股 {len(per_stock)}, 事件日并集 {len(event_dates)}")

    # ---- 前缀稳定性断言（先验后算, 口径级回归守卫）
    log("前缀稳定性抽检 ...")
    stab = prefix_stability_assert(per_stock)
    log(f"抽检 {stab['n_checked']} 个 (股票,日) 单元, 不一致 {len(stab['mismatches'])}")
    assert stab["n_checked"] > 0, "前缀稳定性抽检为空"
    assert not stab["mismatches"], f"前缀稳定性不一致: {stab['mismatches'][:3]}"

    # ---- 全市场横截面排名 pass (rank_return/rank_volume)
    log("全市场 rank_return/rank_volume pass ...")
    rank_df = market_rank_frame(event_dates)
    log(f"排名面板 {rank_df.shape} ({time.time()-t0:.0f}s)")

    # ---- Pass A: 逐股特征链
    all_codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet")
                       if p.stem not in INDEX_CODES)
    tasks = [(code, sorted(event_dates)) for code in all_codes]
    if args.sample:
        tasks = tasks[: args.sample]
    log(f"Pass A 逐股特征链 {len(tasks)} 只 (workers={args.workers}) ...")
    n_rows, n_cached, errors = 0, 0, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (code, n, err) in enumerate(ex.map(_worker, tasks, chunksize=8)):
            if err:
                errors[code] = err
            elif n == -1:
                n_cached += 1
            else:
                n_rows += n
            if (i + 1) % 500 == 0:
                log(f"  {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)")
    log(f"Pass A 完成: 新算行 {n_rows}, 缓存命中 {n_cached}, 失败/跳过 {len(errors)} "
        f"({time.time()-t0:.0f}s)")

    # ---- 大盘特征
    mkt = market_features_for_dates(event_dates)
    log(f"大盘特征 {mkt.shape} ({time.time()-t0:.0f}s)")

    # ---- Pass B: 日期块横截面排名 + 事件单元格抽取
    dates_sorted = sorted(event_dates)
    chunks = [dates_sorted[i: i + DATE_CHUNK]
              for i in range(0, len(dates_sorted), DATE_CHUNK)]
    part_files = sorted(PARTS_DIR.glob("*.parquet"))
    log(f"Pass B 横截面排名: {len(chunks)} 个日期块 x {len(part_files)} 只股 ...")
    pipe = fp2.FeaturePipeline(None, None)
    pool_blocks = {p: [] for p in POOLS}
    pool_keys = {p: pd.MultiIndex.from_arrays([k["ts_code"], k["date"]])
                 for p, k in keys.items()}
    for ci, cdates in enumerate(chunks):
        cset = set(cdates)
        frames = []
        for pf in part_files:
            d = pd.read_parquet(pf)
            d = d[d["timestamp"].isin(cset)]
            if len(d):
                frames.append(d)
        if not frames:
            continue
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.merge(mkt, on="timestamp", how="left")
        panel = panel.merge(rank_df, on=["symbol", "timestamp"], how="left")
        panel[["rank_return", "rank_volume"]] = \
            panel[["rank_return", "rank_volume"]].fillna(0.0)
        panel = pipe._calculate_cross_features(panel)
        panel_idx = pd.MultiIndex.from_arrays([panel["symbol"], panel["timestamp"]])
        for pool in POOLS:
            mask = panel_idx.isin(pool_keys[pool])
            if mask.any():
                pool_blocks[pool].append(panel.loc[mask])
        log(f"  块 {ci+1}/{len(chunks)}: 面板 {panel.shape} ({time.time()-t0:.0f}s)")
        del panel, frames

    # ---- 装配落盘
    results = {"prefix_stability": stab, "pass_a": {
        "rows_new": n_rows, "cached": n_cached, "errors": errors}, "pools": {}}
    for pool in POOLS:
        df = pd.concat(pool_blocks[pool], ignore_index=True)
        df = v4s.add_calendar_features(df)
        df = df.rename(columns={"timestamp": "date", "symbol": "ts_code"})
        df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
        path = OUT_DIR / f"v4daily_snapshot_{pool}.parquet"
        df.to_parquet(path, index=False)
        n_missed = len(keys[pool]) - len(df)
        results["pools"][pool] = {
            "rows": int(len(df)), "cols": int(df.shape[1]),
            "events_missing_snapshot": int(n_missed),
            "path": str(path.relative_to(REPO)),
        }
        log(f"[{pool}] {df.shape} -> {path} (缺快照事件 {n_missed})")

    rpath = OUT_DIR / "v4daily_snapshot_results.json"
    rpath.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    log(f"结果 -> {rpath}; 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
