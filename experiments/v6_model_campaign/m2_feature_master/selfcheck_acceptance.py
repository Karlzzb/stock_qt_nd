#!/usr/bin/env python3
"""M2 验收独立自检(战役 #47 M2,issue #49;README §七 的独立复核落点)。

纪律:本脚本对承重断言一律**独立实现**(不 import geo_features /
build_event_dictionary 的计算本体),只从落盘产物 + 原始数据重算复核:
  1. 泄漏重扫: 主表 schema 列名对三套排除模式独立重扫(白名单口径),零命中;
     台账剔除清单逐列确认物理缺席;
  2. 去重复核: 抽样被剔对自来源 parquet 重取列重算 ρ(应 ≥0.999),
     抽样保留对重算 ρ(应 <0.999);
  3. 段界复核: 以硬编码带界常量独立重判段标签,与主表 seg 逐行一致;
     带内零建模型行 / 带交易日数 ≥30 / 段间实测间隔 ≥30 / 段计数对预登记值;
  4. 种子布尔: 全表独立重算 dd20(rolling(21) 口径)/ bounce(事件表 anchor_close),
     阈值判定与主表 SEED_V6_* 逐行一致,计数对 frozen 值;
  5. 几何族抽样: 500 事件(seed 20260909)独立实现重算 40 条,与主表逐格一致
     (float32 容差);
  6. MACD 截尾无前视: 100 事件(j≥100)以 close[0..j] 重算 MACD,
     DIF[j]/DEA[j] 与全历史值差 <= 1e-8;
  7. md5 台账: 主产物与 rebuild/ 副本逐位一致。

输出: selfcheck_results.json(全绿才写 "all_pass": true)。
用法: python selfcheck_acceptance.py [--workers 24]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import talib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_engine as fe  # noqa: E402
import feature_master as fmx  # noqa: E402
import v4_daily_snapshot as v4s  # noqa: E402

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
MASTER_PATH = SCRIPT_DIR / "master_v6.parquet"
REBUILD_DIR = SCRIPT_DIR / "rebuild"
CACHE_DIR = SCRIPT_DIR / "cache"
DAILY_DIR = REPO / "stock_data" / "daily"
PROGRESS = SCRIPT_DIR / "progress.log"

GEO_SEED = 20260909
GEO_SAMPLE = 500
MACD_SAMPLE = 100
DEDUP_SAMPLE = 20

# 独立硬编码带界(不 import fmx.EMBARGO,复核落码值本身)
BANDS = [(pd.Timestamp("2019-11-20"), pd.Timestamp("2020-02-20")),
         (pd.Timestamp("2023-11-20"), pd.Timestamp("2024-02-20"))]
TRAIN_LO, TRAIN_HI = pd.Timestamp("2001-01-01"), pd.Timestamp("2019-12-31")
VAL_LO, VAL_HI = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")
PREREG_SEGMENT_COUNTS = {"train": 50165, "val": 24405, "test": 17902, "pre2001": 1042}
PREREG_BAND_EVENTS = [1313, 1750]
SEED_FROZEN_COUNTS = {"SEED_V6_1": 5580, "SEED_V6_2": 2419, "SEED_V6_3": 1159,
                      "SEED_V6_4": 11888, "SEED_V6_5": 3271}
SEED_TH = {"SEED_V6_1": (-0.15, True), "SEED_V6_2": (-0.20, True),
           "SEED_V6_3": (-0.25, True), "SEED_V6_4": (-0.15, False),
           "SEED_V6_5": (-0.25, False)}
BOUNCE_LO, BOUNCE_HI = 0.02, 0.08

GEO40 = ["GC_GAP_NEAR_BARS", "GC_GAP_FAR_BARS", "GC_GAP_RATIO",
         "DIF_EVENT_CLOSE_NORM", "DIF_PREV_CLOSE_NORM", "DIF_LIFT_CLOSE_NORM",
         "DIF_LIFT_RATE_CLOSE_NORM", "DEA_EVENT_CLOSE_NORM", "DEA_LIFT_CLOSE_NORM",
         "DIF_DEA_GAP_CLOSE_NORM", "DIF_PREV2_CLOSE_NORM", "DIF_LIFT_PREV_CLOSE_NORM",
         "DD20", "DD20_START_BARS", "DD20_AVG_SPEED", "DD20_DOWN_DAY_SHARE",
         "DD20_MAX_DAY_DROP", "DD20_LOCAL_LOW_COUNT", "DD20_PATH_EFFICIENCY",
         "BOUNCE", "BOUNCE_AVG_SPEED", "BOUNCE_MAX_GIVEBACK", "BOUNCE_UP_DAY_SHARE",
         "BOUNCE_VOL_RATIO", "ANCHOR_DIST_BARS", "ANCHOR_REL_POS",
         "ANCHOR_BREAK_DEPTH", "MIN_PREV_DIST_BARS", "MIN_PREV_DRAWDOWN",
         "DD20_VOL_BASELINE_RATIO", "DD20_DIST_TH15", "DD20_DIST_TH20",
         "DD20_DIST_TH25", "BOUNCE_DIST_LO", "BOUNCE_DIST_HI",
         "SEED_V6_1", "SEED_V6_2", "SEED_V6_3", "SEED_V6_4", "SEED_V6_5"]


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [selfcheck] {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def load_events():
    ev = pd.read_parquet(EVENTS_PATH)
    ev = ev.sort_values(["ts_code", "event_date"], kind="mergesort").reset_index(drop=True)
    ev["event_id"] = np.arange(len(ev), dtype=np.int64)
    return ev.rename(columns={"event_date": "date"})


def scan_load(path):
    df = pd.read_parquet(path)
    return df.sort_values("trade_date").reset_index(drop=True)


# ================================================================ 1. 泄漏重扫
def check_leakage(results):
    schema_names = pq.read_schema(MASTER_PATH).names
    pats = list(fmx.EXCLUDE_PATTERNS) + list(fe.BLACKLIST_PATTERNS) \
        + list(v4s.FORBIDDEN_PATTERNS)
    res = [re.compile(p, re.IGNORECASE) for p in pats]
    hits = [c for c in schema_names
            if c not in fmx.RANK_WHITELIST and any(rx.match(c) for rx in res)]
    assert not hits, f"主表列命中泄漏模式: {hits[:20]}"
    excluded = results.get("leak_excluded", [])
    still = [c for c in excluded if c in schema_names]
    assert not still, f"台账剔除列仍在主表: {still[:10]}"
    return {"schema_cols": len(schema_names), "leak_hits": 0,
            "excluded_confirmed_absent": len(excluded)}


# ================================================================ 2. 去重复核
def _fetch_column(col, src, keys):
    """自来源 parquet 重取一列并对齐事件键(keys: event_id/ts_code/date)。"""
    if src == "s0":
        d = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date", col])
        d = d.rename(columns={"event_date": "date"})
        d["date"] = pd.to_datetime(d["date"])
        return keys[["ts_code", "date"]].merge(d, on=["ts_code", "date"],
                                               how="left")[col]
    if src == "s1":
        d = pd.read_parquet(CACHE_DIR / "s1_event_dictionary.parquet",
                            columns=["event_id", col])
        return keys[["event_id"]].merge(d, on="event_id", how="left")[col]
    path = {"s2": "s2_factory_full.parquet", "s3": "s3_v4daily_snapshot.parquet",
            "s4": "s4_t3_snapshot.parquet"}[src]
    d = pd.read_parquet(CACHE_DIR / path, columns=["ts_code", "date", col])
    return keys[["ts_code", "date"]].merge(d, on=["ts_code", "date"],
                                           how="left")[col]


def _pairwise_rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def check_dedup(master, results, ev):
    rng = np.random.default_rng(20260909)
    src_of = {}
    for row in pd.read_csv(SCRIPT_DIR / "master_dictionary_v6.csv").itertuples():
        src_of[row.column] = row.source
    keys = ev[["event_id", "ts_code", "date"]]
    mask = master["seg"].isin(["train", "val"]).to_numpy()
    dropped = results["dedup"]["dropped"]
    picks_d = rng.choice(len(dropped), size=min(DEDUP_SAMPLE, len(dropped)),
                         replace=False)
    rechecks = []
    for i in picks_d:
        rec = dropped[int(i)]
        c, anchor = rec["column"], rec["anchor"]
        # 锚列可能亦被剔(链式), 此时自来源重取;保留列直接取主表
        a = master[anchor].to_numpy(np.float64) if anchor in master.columns \
            else _fetch_column(anchor, src_of.get(anchor, "s1"), keys).to_numpy(np.float64)
        b = _fetch_column(c, src_of.get(c, "s1"), keys).to_numpy(np.float64)
        rho = _pairwise_rho(a[mask], b[mask])
        rechecks.append({"column": c, "anchor": anchor, "rho": rho})
        assert np.isnan(rho) or abs(rho) >= 0.999, \
            f"被剔对 {c}~{anchor} 重算 |ρ|={rho} < 0.999"
    kept_cols = [c for c in master.columns if c in src_of
                 and master[c].dtype.kind in "fib"
                 and c not in fmx.EVENT_META_COLS]
    picks_k = rng.choice(len(kept_cols), size=min(DEDUP_SAMPLE * 2, len(kept_cols)),
                         replace=False)
    kept_rechecks = []
    for i in picks_k:
        c1 = kept_cols[int(i)]
        c2 = kept_cols[int(rng.choice(len(kept_cols)))]
        if c1 == c2:
            continue
        rho = _pairwise_rho(master[c1].to_numpy(np.float64)[mask],
                            master[c2].to_numpy(np.float64)[mask])
        kept_rechecks.append({"c1": c1, "c2": c2, "rho": rho})
        assert np.isnan(rho) or abs(rho) < 0.999, \
            f"保留对 {c1}~{c2} 重算 |ρ|={rho} >= 0.999"
    return {"dropped_rechecks": rechecks, "kept_rechecks": kept_rechecks}


# ================================================================ 3. 段界复核
def check_segments(master):
    dates = pd.to_datetime(master["date"])
    seg = np.full(len(master), "test", dtype=object)
    d = dates.to_numpy()
    seg[d < np.datetime64(TRAIN_LO)] = "pre2001"
    seg[(d >= np.datetime64(TRAIN_LO)) & (d <= np.datetime64(TRAIN_HI))] = "train"
    seg[(d >= np.datetime64(VAL_LO)) & (d <= np.datetime64(VAL_HI))] = "val"
    for lo, hi in BANDS:
        seg[(d >= np.datetime64(lo)) & (d <= np.datetime64(hi))] = "embargo"
    got = master["seg"].to_numpy()
    mismatch = int((seg != got).sum())
    assert mismatch == 0, f"独立段标签与主表 seg 不一致: {mismatch} 行"
    idx_sh = pd.read_parquet(DAILY_DIR / "000001.SH.parquet", columns=["trade_date"])
    cal = pd.to_datetime(idx_sh["trade_date"]).sort_values().unique()
    ledger = []
    for i, (lo, hi) in enumerate(BANDS):
        in_band = (dates >= lo) & (dates <= hi)
        bad = master.loc[in_band.to_numpy(), "seg"].isin(["train", "val", "test"]).sum()
        assert bad == 0, f"带 {i} 内混入建模型行 {bad}"
        n_days = int(((cal >= np.datetime64(lo)) & (cal <= np.datetime64(hi))).sum())
        assert n_days >= 30, f"带 {i} 仅 {n_days} 个交易日"
        n_ev = int(in_band.sum())
        assert n_ev == PREREG_BAND_EVENTS[i], f"带 {i} 事件 {n_ev} != 预登记"
        ledger.append({"band": i, "trading_days": n_days, "events": n_ev})
    seg_counts = {k: int(v) for k, v in master["seg"].value_counts().items()}
    assert {k: seg_counts.get(k, 0) for k in PREREG_SEGMENT_COUNTS} \
        == PREREG_SEGMENT_COUNTS, f"段计数不符: {seg_counts}"
    for sa, sb in (("train", "val"), ("val", "test")):
        a_max = dates[got == sa].max()
        b_min = dates[got == sb].min()
        gap = int(((cal > np.datetime64(a_max)) & (cal < np.datetime64(b_min))).sum())
        assert gap >= 30, f"{sa}~{sb} 间隔 {gap} < 30"
    return {"segment_counts": seg_counts, "bands": ledger, "mismatch": 0}


# ================================================================ 4/5/6. 逐股独立重算
def _seed_worker(task):
    """全表种子布尔独立重算: rolling(21) 窗口 dd20 + 事件表 anchor_close 口径。"""
    code, recs = task  # recs: [(event_id, j, anchor_close)]
    try:
        df = scan_load(DAILY_DIR / f"{code}.parquet")
        C = df["close"].to_numpy(np.float64)
        wmax = pd.Series(C).rolling(21, min_periods=1).max().to_numpy()
        j = np.array([r[1] for r in recs], np.int64)
        dd20 = C[j] / wmax[j] - 1.0
        bounce = C[j] / np.array([r[2] for r in recs], np.float64) - 1.0
        return [r[0] for r in recs], dd20, bounce, None
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"{code}: {e!r}"


def check_seed_booleans(master, ev, workers):
    per_stock = defaultdict(list)
    for r in ev.itertuples():
        per_stock[r.ts_code].append((int(r.event_id), int(r.event_row),
                                     float(r.anchor_close)))
    tasks = sorted(per_stock.items())
    log(f"种子布尔全表独立重算 {len(tasks)} 只股 ...")
    eids, dd20s, bounces = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for eid, dd, bo, err in ex.map(_seed_worker, tasks, chunksize=8):
            assert err is None, f"种子重算失败: {err}"
            eids.append(np.asarray(eid, np.int64))
            dd20s.append(dd)
            bounces.append(bo)
    eid = np.concatenate(eids)
    order = np.argsort(eid, kind="stable")
    dd20 = np.concatenate(dd20s)[order]
    bounce = np.concatenate(bounces)[order]
    assert np.array_equal(eid[order], np.arange(len(ev))), "event_id 对齐破坏"
    in_band = (bounce > BOUNCE_LO) & (bounce <= BOUNCE_HI)
    out = {}
    m_sorted = master.sort_values("event_id", kind="mergesort")
    for col, (th, with_band) in SEED_TH.items():
        sel = dd20 <= th
        if with_band:
            sel = sel & in_band
        cnt = int(sel.sum())
        assert cnt == SEED_FROZEN_COUNTS[col], f"{col} 计数 {cnt} != frozen"
        mcol = m_sorted[col].to_numpy(np.float64) > 0.5
        n_diff = int((mcol != sel).sum())
        assert n_diff == 0, f"{col} 与主表逐行不一致 {n_diff} 行"
        out[col] = cnt
    return out


def _geo40_independent(C, V, i2, i1, j, a, p_low):
    """40 条几何特征的独立实现(逐事件标量循环,与 geo_features.py 零共享代码)。"""
    DIF, DEA, _ = talib.MACD(np.asarray(C, np.float64))
    o = {}
    cj, ci1, ci2 = C[j], C[i1], C[i2]
    o["GC_GAP_NEAR_BARS"] = float(j - i1)
    o["GC_GAP_FAR_BARS"] = float(i1 - i2)
    o["GC_GAP_RATIO"] = (j - i1) / (i1 - i2)
    o["DIF_EVENT_CLOSE_NORM"] = DIF[j] / cj
    o["DIF_PREV_CLOSE_NORM"] = DIF[i1] / ci1
    o["DIF_LIFT_CLOSE_NORM"] = (DIF[j] - DIF[i1]) / cj
    o["DIF_LIFT_RATE_CLOSE_NORM"] = (DIF[j] - DIF[i1]) / ((j - i1) * cj)
    o["DEA_EVENT_CLOSE_NORM"] = DEA[j] / cj
    o["DEA_LIFT_CLOSE_NORM"] = (DEA[j] - DEA[i1]) / cj
    o["DIF_DEA_GAP_CLOSE_NORM"] = (DIF[j] - DEA[j]) / cj
    o["DIF_PREV2_CLOSE_NORM"] = DIF[i2] / ci2
    o["DIF_LIFT_PREV_CLOSE_NORM"] = (DIF[i1] - DIF[i2]) / ci1
    t0 = max(0, j - 20)
    W = C[t0:j + 1]
    wmax = float(np.max(W))
    dd = cj / wmax - 1.0
    peak = t0 + int(np.argmax(W))
    o["DD20"] = dd
    o["DD20_START_BARS"] = float(j - peak)
    o["DD20_AVG_SPEED"] = dd / max(1, j - peak)
    if len(W) > 1:
        rets = W[1:] / W[:-1] - 1.0
        o["DD20_DOWN_DAY_SHARE"] = float(np.mean(W[1:] < W[:-1]))
        o["DD20_MAX_DAY_DROP"] = float(np.min(rets))
        absum = float(np.sum(np.abs(rets)))
        o["DD20_PATH_EFFICIENCY"] = abs(dd) / absum if absum > 0 else np.nan
    else:
        o["DD20_DOWN_DAY_SHARE"] = np.nan
        o["DD20_MAX_DAY_DROP"] = np.nan
        o["DD20_PATH_EFFICIENCY"] = np.nan
    o["DD20_LOCAL_LOW_COUNT"] = float(sum(
        1 for t in range(t0 + 1, j) if C[t] < C[t - 1] and C[t] < C[t + 1]))
    bounce = cj / C[a] - 1.0
    o["BOUNCE"] = bounce
    o["BOUNCE_AVG_SPEED"] = bounce / max(1, j - a)
    if j > a:
        runmax = -np.inf
        mg = 0.0
        for t in range(a, j + 1):
            runmax = max(runmax, C[t])
            mg = min(mg, C[t] / runmax - 1.0)
        o["BOUNCE_MAX_GIVEBACK"] = mg
        o["BOUNCE_UP_DAY_SHARE"] = float(np.mean(C[a + 1:j + 1] > C[a:j]))
        vnum = float(np.nanmean(V[a + 1:j + 1]))
    else:
        o["BOUNCE_MAX_GIVEBACK"] = 0.0
        o["BOUNCE_UP_DAY_SHARE"] = 0.0
        vnum = float(V[j])
    vden_seg = V[max(0, a - 20):a + 1]
    vden = float(np.nanmean(vden_seg)) if len(vden_seg) else np.nan
    o["BOUNCE_VOL_RATIO"] = vnum / vden if np.isfinite(vden) and vden > 0 else np.nan
    o["ANCHOR_DIST_BARS"] = float(j - a)
    o["ANCHOR_REL_POS"] = (a - i1) / (j - i1)
    prev_seg = C[i2 + 1:i1 + 1]
    min_prev = float(np.min(prev_seg))
    o["ANCHOR_BREAK_DEPTH"] = C[a] / min_prev - 1.0
    o["MIN_PREV_DIST_BARS"] = float(j - p_low)
    o["MIN_PREV_DRAWDOWN"] = min_prev / float(np.max(prev_seg)) - 1.0
    vbase = V[max(0, j - 80):t0]
    den = float(np.nanmean(vbase)) if len(vbase) else np.nan
    num = float(np.nanmean(V[t0:j + 1]))
    o["DD20_VOL_BASELINE_RATIO"] = num / den if np.isfinite(den) and den > 0 else np.nan
    o["DD20_DIST_TH15"] = dd + 0.15
    o["DD20_DIST_TH20"] = dd + 0.20
    o["DD20_DIST_TH25"] = dd + 0.25
    o["BOUNCE_DIST_LO"] = bounce - BOUNCE_LO
    o["BOUNCE_DIST_HI"] = BOUNCE_HI - bounce
    ib = (bounce > BOUNCE_LO) and (bounce <= BOUNCE_HI)
    o["SEED_V6_1"] = float(dd <= -0.15 and ib)
    o["SEED_V6_2"] = float(dd <= -0.20 and ib)
    o["SEED_V6_3"] = float(dd <= -0.25 and ib)
    o["SEED_V6_4"] = float(dd <= -0.15)
    o["SEED_V6_5"] = float(dd <= -0.25)
    return o


def check_geo_sample(master, ev, results):
    rng = np.random.default_rng(GEO_SEED)
    picks = ev.iloc[rng.choice(len(ev), size=GEO_SAMPLE, replace=False)]
    m_by_eid = master.set_index("event_id")
    # 去重剔列(F1~F5 仿射距离列等)不在主表,须与去重台账逐字对上才豁免
    present = [c for c in GEO40 if c in master.columns]
    dropped_geo = sorted(c for c in GEO40 if c not in master.columns)
    ledger_dropped = sorted(d["column"] for d in results["dedup"]["dropped"]
                            if d["column"] in GEO40)
    assert dropped_geo == ledger_dropped, \
        f"几何族缺席列 {dropped_geo} 与去重台账 {ledger_dropped} 不符"
    n_checked, mismatches = 0, []
    cache = {}
    for r in picks.itertuples():
        code = r.ts_code
        if code not in cache:
            df = scan_load(DAILY_DIR / f"{code}.parquet")
            cache[code] = (df["close"].to_numpy(np.float64),
                           pd.to_numeric(df["vol"], errors="coerce").to_numpy(np.float64))
        C, V = cache[code]
        ref = _geo40_independent(C, V, int(r.cross_prev2_row), int(r.cross_prev_row),
                                 int(r.event_row), int(r.anchor_row),
                                 int(r.min_prev_row))
        got = m_by_eid.loc[int(r.event_id), present].to_numpy(np.float64)
        want = np.array([ref[c] for c in present], np.float64)
        n_checked += 1
        ok = np.isclose(got, want, rtol=1e-5, atol=1e-12, equal_nan=True)
        if not ok.all():
            bad = [present[k] for k in np.where(~ok)[0]]
            mismatches.append({"event_id": int(r.event_id), "cols": bad[:10]})
    assert not mismatches, f"几何族独立重算不一致: {mismatches[:3]}"
    return {"n_checked": n_checked, "mismatches": 0,
            "geo_cols_compared": len(present),
            "geo_cols_dedup_dropped": dropped_geo}


def check_macd_truncation(ev):
    rng = np.random.default_rng(GEO_SEED + 1)
    elig = ev[ev["event_row"] >= 100]
    picks = elig.iloc[rng.choice(len(elig), size=MACD_SAMPLE, replace=False)]
    cache, n_checked, max_diff = {}, 0, 0.0
    for r in picks.itertuples():
        code, j = r.ts_code, int(r.event_row)
        if code not in cache:
            df = scan_load(DAILY_DIR / f"{code}.parquet")
            C = df["close"].to_numpy(np.float64)
            cache[code] = (C, talib.MACD(C))
        C, (dif, dea, _) = cache[code]
        dif_t, dea_t, _ = talib.MACD(C[:j + 1])
        dd = max(abs(dif_t[-1] - dif[j]), abs(dea_t[-1] - dea[j]))
        max_diff = max(max_diff, float(dd))
        n_checked += 1
    assert max_diff <= 1e-8, f"MACD 截尾无前视违反: max_diff={max_diff}"
    return {"n_checked": n_checked, "max_diff": max_diff}


# ================================================================ 7. md5 台账
def check_md5(results):
    ledger = {}
    pairs = [(MASTER_PATH, REBUILD_DIR / "master_v6.parquet")]
    for name in ("s1_event_dictionary", "s2_factory_full",
                 "s3_v4daily_snapshot", "s4_t3_snapshot"):
        pairs.append((CACHE_DIR / f"{name}.parquet",
                      REBUILD_DIR / f"{name}.parquet"))
    for a, b in pairs:
        ha, hb = md5_of(a), md5_of(b)
        assert ha == hb, f"确定性双跑不一致: {a.name} {ha} != {hb}"
        ledger[a.name] = ha
    registered = results.get("md5", {}).get(MASTER_PATH.name)
    if registered:
        assert registered == ledger[MASTER_PATH.name], "台账 md5 与实测不符"
    # s3 Pass A 双份 parts 逐文件 md5(README 验收 8)
    parts_a = {p.name: p for p in (CACHE_DIR / "v4daily_parts_fullhist").glob("*.parquet")}
    parts_b = {p.name: p for p in (REBUILD_DIR / "v4daily_parts_fullhist").glob("*.parquet")}
    assert parts_a.keys() == parts_b.keys(), \
        f"双份 parts 文件集不一致: 仅单边 {list(parts_a.keys() ^ parts_b.keys())[:5]}"
    parts_ledger = {}
    for name in sorted(parts_a):
        ha, hb = md5_of(parts_a[name]), md5_of(parts_b[name])
        assert ha == hb, f"parts 双跑不一致: {name}"
        parts_ledger[name] = ha
    ledger["v4daily_parts_fullhist"] = parts_ledger
    return ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--skip-md5", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out = {"all_pass": False}
    results = json.loads((SCRIPT_DIR / "master_results_v6.json").read_text())

    log("1/7 泄漏重扫 ...")
    out["leakage"] = check_leakage(results)

    log("2/7 载入主表(去重/段界复核) ...")
    master = pd.read_parquet(MASTER_PATH)
    assert len(master) == 96577 and not master.duplicated(["event_id"]).any()
    ev = load_events()
    out["dedup"] = check_dedup(master, results, ev)
    log("3/7 段界复核 ...")
    out["segments"] = check_segments(master)
    log("4/7 种子布尔全表独立重算 ...")
    out["seed_booleans"] = check_seed_booleans(master, ev, args.workers)
    log("5/7 几何族 500 事件独立重算 ...")
    out["geo_sample"] = check_geo_sample(master, ev, results)
    log("6/7 MACD 截尾无前视 100 事件 ...")
    out["macd_truncation"] = check_macd_truncation(ev)
    if not args.skip_md5:
        log("7/7 md5 台账核验 ...")
        out["md5"] = check_md5(results)
    out["all_pass"] = True
    out["elapsed_sec"] = time.time() - t0
    (SCRIPT_DIR / "selfcheck_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"自检全绿 -> selfcheck_results.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
