#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""急跌背离种子全历史审判 —— 预登记 = 同目录 README.md(先于跑数落盘,勿改)。

实现口径:
- 扫描:逐字复刻 divergence_anchor_eval_2026/scan_v1.py 与 scan_v2.py 的单股口径,
  唯一改动 = 去掉 event_date 2026 窗口过滤,保留全历史全部事件。
- 交易模拟:逐条复刻 stage_recognizer/run_recognizer.py 的 simulate(其口径已复核),
  数据加载按 README 披露 2 自写(逐股 parquet + 按需 stk_limit),
  成本/滑点/整手调冻结引擎 strategy_engine 原语。
- 与参照实现的两点必要差异(README 已授权,report 披露):
  1) 索引口径:按个股序列行号(README 第 4 节明文),不建市场日历对齐数组;
     个股序列无停牌行,故"停牌日行计入但不评估"条款自然空置(披露)。
  2) 事件日无下一行(个股数据耗尽,含退市)→ 记 truncated_no_next,并入 n_truncated(披露)。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import talib

REPO = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO, "v3_pipeline", "scripts"))
import strategy_engine as se  # noqa: E402  冻结引擎:只读复用成本原语与常量

DATA_DIR = Path(os.path.join(REPO, "stock_data", "daily"))
LIMIT_DIR = Path(os.path.join(REPO, "stock_data", "stk_limit"))
REF_DIR = Path(os.path.join(REPO, "experiments", "divergence_anchor_eval_2026"))
OUT_DIR = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history"))
LOG_PATH = OUT_DIR / "progress.log"

EVENT_START = pd.Timestamp("2026-01-01")
EVENT_END = pd.Timestamp("2026-08-31")
MIN_ROWS = 100
DIF_LIFT_MIN = 0.001
BUDGET = 100_000.0
TOL = 1e-9
H_LIST = [10, 20, 25]
SEEDS = ["S1", "S2", "S3", "S4", "S5", "ALL"]
VARIANTS = ["v1", "v2"]
SAMPLE_SEED = 20260904
LIMIT_LOOKAHEAD_DAYS = 40  # README:事件日到事件后 40 自然日的 stk_limit 文件

# 供模拟 worker 用的全局(每进程初始化一次)
_G: dict = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 扫描(逐字复刻,去窗口化)
def scan_one_v1(path: str):
    """scan_v1.py::scan_one 逐字口径(区间最低价锚定),去掉 2026 窗口过滤。"""
    try:
        df = pd.read_parquet(path)
        if "ts_code" not in df.columns or "vol" not in df.columns:
            return None, [], "schema"
        df = df.sort_values("trade_date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None, [], "short"

        close = df["close"].to_numpy(dtype=np.float64)
        dates = df["trade_date"].to_numpy()
        dif, dea, _ = talib.MACD(close)  # 默认 12/26/9
        if np.isnan(dif).all():
            return None, [], "nan"

        prev_le = (dif[:-1] <= dea[:-1]) & ~np.isnan(dif[:-1]) & ~np.isnan(dea[:-1])
        now_gt = (dif[1:] > dea[1:]) & ~np.isnan(dif[1:]) & ~np.isnan(dea[1:])
        crosses = np.nonzero(prev_le & now_gt)[0] + 1

        ts_code = str(df["ts_code"].iloc[0])
        events: list[dict] = []
        for k in range(2, len(crosses)):
            c_km2, c_km1, c_k = crosses[k - 2], crosses[k - 1], crosses[k]
            dif_lift = dif[c_k] - dif[c_km1]
            if dif_lift < DIF_LIFT_MIN:
                continue
            seg_prev = close[c_km2 + 1 : c_km1 + 1]
            seg_cur = close[c_km1 + 1 : c_k + 1]
            if len(seg_prev) == 0 or len(seg_cur) == 0:
                continue
            min_prev = seg_prev.min()
            idx_cur_rel = int(np.argmin(seg_cur))
            min_cur = seg_cur[idx_cur_rel]
            if not (min_prev > min_cur):
                continue
            anchor_idx = c_km1 + 1 + idx_cur_rel
            events.append(
                {
                    "ts_code": ts_code,
                    "event_date": pd.Timestamp(dates[c_k]),
                    "anchor_date": pd.Timestamp(dates[anchor_idx]),
                    "anchor_close": float(close[anchor_idx]),
                    "cross_prev_date": pd.Timestamp(dates[c_km1]),
                    "cross_prev_dif": float(dif[c_km1]),
                    "cross_date": pd.Timestamp(dates[c_k]),
                    "cross_dif": float(dif[c_k]),
                    "dif_lift": float(dif_lift),
                }
            )
        return ts_code, events, None
    except Exception as e:  # noqa: BLE001
        return None, [], f"error:{e}"


def scan_one_v2(path: str):
    """scan_v2.py::process_stock 逐字口径(右侧确认低点精确锚定),去掉 2026 窗口过滤。"""
    try:
        df = pd.read_parquet(path)
        cols = set(df.columns)
        if "ts_code" not in cols or "vol" not in cols:
            return None, [], "schema"
        if len(df) < MIN_ROWS:
            return None, [], "short"
        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df["close"].to_numpy(dtype=np.float64)
        dates = df["trade_date"].to_numpy()
        n = len(df)
        ts_code = str(df["ts_code"].iloc[0])

        dif, dea, _ = talib.MACD(close)
        valid = ~(np.isnan(dif) | np.isnan(dea))
        cross_mask = np.zeros(n, dtype=bool)
        cross_mask[1:] = (
            valid[:-1] & valid[1:]
            & (dif[:-1] <= dea[:-1])
            & (dif[1:] > dea[1:])
        )
        crosses = np.flatnonzero(cross_mask)
        if len(crosses) < 3:
            return ts_code, [], None

        def confirmed_lows(lo_idx, hi_idx):
            cands = []
            for t in range(lo_idx + 1, hi_idx + 1):
                if t - 3 < 0 or t + 2 > n - 1:
                    continue
                if close[t] <= close[t - 3:t].min() and close[t] < close[t + 1:t + 3].min():
                    cands.append(t)
            kept = []
            for t in cands:
                if not kept or t - kept[-1] >= 3:
                    kept.append(t)
            return kept

        def find_anchor(k):
            lo = crosses[k - 1]
            hi = min(crosses[k] + 5, n - 1)
            if hi <= lo:
                return None
            lows = confirmed_lows(lo, hi)
            if lows:
                best = lows[0]
                for t in lows[1:]:
                    if close[t] < close[best]:
                        best = t
                return best
            seg = close[lo + 1:hi + 1]
            return lo + 1 + int(np.argmin(seg))

        events = []
        anchors = {}
        for k in range(1, len(crosses)):
            anchors[k] = find_anchor(k)
        for k in range(2, len(crosses)):
            a_prev, a_cur = anchors.get(k - 1), anchors.get(k)
            if a_prev is None or a_cur is None:
                continue
            dif_lift = dif[crosses[k]] - dif[crosses[k - 1]]
            if dif_lift < DIF_LIFT_MIN:
                continue
            if not (close[a_cur] < close[a_prev]):
                continue
            if a_cur + 2 > n - 1:
                continue
            event_idx = max(crosses[k], a_cur + 2)
            events.append({
                "ts_code": ts_code,
                "event_date": pd.Timestamp(dates[event_idx]),
                "anchor_date": pd.Timestamp(dates[a_cur]),
                "anchor_close": float(close[a_cur]),
                "cross_prev_date": pd.Timestamp(dates[crosses[k - 1]]),
                "cross_prev_dif": float(dif[crosses[k - 1]]),
                "cross_date": pd.Timestamp(dates[crosses[k]]),
                "cross_dif": float(dif[crosses[k]]),
                "dif_lift": float(dif_lift),
                "anchor_prev_date": pd.Timestamp(dates[a_prev]),
                "anchor_prev_close": float(close[a_prev]),
            })
        return ts_code, events, None
    except Exception:  # noqa: BLE001
        return None, [], "error:" + traceback.format_exc().splitlines()[-1]


def run_scan(name: str, fn, files: list[str]) -> pd.DataFrame:
    t0 = time.time()
    stats = {"schema": 0, "short": 0, "nan": 0, "error": 0}
    n_stocks = 0
    all_events: list[dict] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (ts_code, evs, reason) in enumerate(
            pool.imap_unordered(fn, files, chunksize=16)
        ):
            if reason:
                key = reason.split(":")[0]
                stats[key if key in stats else "error"] += 1
            if ts_code is not None:
                n_stocks += 1
            all_events.extend(evs)
            if (i + 1) % 500 == 0:
                log(f"heartbeat: scan_{name} {i + 1}/{len(files)} 文件 "
                    f"({time.time() - t0:.0f}s)")
    ev = pd.DataFrame(all_events)
    if len(ev) > 0:
        ev = ev.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    log(f"[scan_{name}] 个股 {n_stocks}/{len(files)},跳过 {stats},"
        f"全历史事件 {len(ev)} 起 ({time.time() - t0:.0f}s)")
    return ev


def check_subset_equal(ev_hist: pd.DataFrame, ref_path: Path, name: str) -> dict:
    """自检 1:全历史事件的 2026 子集与既有 events_{name}.parquet 逐行一致。"""
    ref = pd.read_parquet(ref_path)
    sub = ev_hist[(ev_hist["event_date"] >= EVENT_START)
                  & (ev_hist["event_date"] <= EVENT_END)].reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    detail: dict = {"n_subset": int(len(sub)), "n_ref": int(len(ref))}
    if len(sub) != len(ref) or list(sub.columns) != list(ref.columns):
        detail["ok"] = False
        detail["reason"] = "行数或列不一致"
        return detail
    ok = True
    max_diff = 0.0
    for col in ref.columns:
        a, b = sub[col], ref[col]
        if a.dtype.kind == "f":
            d = float(np.abs(a.to_numpy() - b.to_numpy()).max()) if len(a) else 0.0
            max_diff = max(max_diff, d)
            if not bool(np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-9)):
                ok = False
                detail["fail_col"] = col
        else:
            if not bool((a == b).all()):
                ok = False
                detail["fail_col"] = col
    detail["max_abs_float_diff"] = max_diff
    detail["ok"] = bool(ok)
    return detail


# ---------------------------------------------------------------- 涨跌停按需加载
def _limit_worker_init(codes: frozenset) -> None:
    _G["limit_codes"] = codes


def _limit_read_one(d: str):
    fp = LIMIT_DIR / f"{d}.parquet"
    if not fp.exists():
        return d, None
    lf = pd.read_parquet(fp)
    lf = lf[lf["ts_code"].isin(_G["limit_codes"])]
    return d, (lf["ts_code"].to_numpy(),
               lf["up_limit"].to_numpy(dtype=np.float64),
               lf["down_limit"].to_numpy(dtype=np.float64))


def load_limits(needed_dates: list[str], needed_codes: set[str]) -> dict:
    """加载指定日期的 stk_limit 文件,过滤到有事件的个股。
    返回 {ts_code: (dates_int32 有序, up float64, dn float64)}。缺文件数由调用方统计。"""
    t0 = time.time()
    acc_codes: list = []
    acc_d: list = []
    acc_up: list = []
    acc_dn: list = []
    missing = 0
    with mp.Pool(processes=max(1, mp.cpu_count() - 1),
                 initializer=_limit_worker_init,
                 initargs=(frozenset(needed_codes),)) as pool:
        for i, (d, res) in enumerate(pool.imap_unordered(_limit_read_one,
                                                         needed_dates,
                                                         chunksize=16)):
            if res is None:
                missing += 1
            elif len(res[0]):
                codes, ups, dns = res
                acc_codes.append(codes)
                acc_d.append(np.full(len(codes), int(d), dtype=np.int32))
                acc_up.append(ups)
                acc_dn.append(dns)
            if (i + 1) % 500 == 0:
                log(f"heartbeat: stk_limit 加载 {i + 1}/{len(needed_dates)} 文件 "
                    f"({time.time() - t0:.0f}s)")
    big = pd.DataFrame({
        "code": pd.Categorical(np.concatenate(acc_codes)),
        "d": np.concatenate(acc_d),
        "up": np.concatenate(acc_up),
        "dn": np.concatenate(acc_dn),
    }).sort_values(["code", "d"])
    out: dict[str, tuple] = {}
    for code, grp in big.groupby("code", observed=True):
        out[str(code)] = (grp["d"].to_numpy(), grp["up"].to_numpy(),
                          grp["dn"].to_numpy())
    log(f"[limits] 加载 {len(needed_dates) - missing}/{len(needed_dates)} 文件,"
        f"缺文件 {missing} 天,覆盖 {len(out)} 股 ({time.time() - t0:.0f}s)")
    return out, missing


# ---------------------------------------------------------------- 模拟(逐股 worker)
def _sim_worker_init():
    _G["fallback_cache"] = {}
    _G["fallback_loads"] = 0


def _lookup_limit(lim, date_int: int, which: int, ts_code: str):
    """lim = (dates_int32, up, dn) 或 None;which: 1=up 2=dn。
    预载数组命中则直接返回;未命中(文件缺失/无该股行/顺延超出 +40 自然日窗口)
    按需回退读当日文件(进程内缓存),仍无则 NaN = 无约束。"""
    if lim is not None:
        dates, up, dn = lim
        i = int(np.searchsorted(dates, date_int))
        if i < len(dates) and dates[i] == date_int:
            return float(up[i] if which == 1 else dn[i])
    cache = _G["fallback_cache"]
    if date_int not in cache:
        fp = LIMIT_DIR / f"{date_int}.parquet"
        if fp.exists():
            lf = pd.read_parquet(fp)
            cache[date_int] = dict(zip(lf["ts_code"],
                                       zip(lf["up_limit"], lf["down_limit"])))
        else:
            cache[date_int] = None
        _G["fallback_loads"] += 1
    day = cache[date_int]
    if day is None:
        return np.nan
    row = day.get(ts_code)
    if row is None:
        return np.nan
    return float(row[0] if which == 1 else row[1])


def simulate_stock(task: dict) -> list[dict]:
    """单股:加载日线,对该股全部事件(两变体)算 dd20/bounce,逐 H 模拟。"""
    ts_code = task["ts_code"]
    lim = task["limits"]  # (dates, up, dn) 或 None
    df = pd.read_parquet(DATA_DIR / f"{ts_code}.parquet",
                         columns=["trade_date", "open", "close"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(df)
    open_ = df["open"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    dts = pd.DatetimeIndex(df["trade_date"])
    d_int = (dts.year * 10000 + dts.month * 100 + dts.day).to_numpy(dtype=np.int32)
    pos = {d: i for i, d in enumerate(dts)}

    recs: list[dict] = []
    for variant in VARIANTS:
        for ev in task["events"].get(variant, []):
            j = pos.get(ev["event_date"], -1)
            if j < 0:
                continue  # 不会发生(事件由同一文件生成),防御
            close_ev = close[j]
            dd20 = close_ev / close[max(0, j - 20):j + 1].max() - 1.0
            bounce = close_ev / ev["anchor_close"] - 1.0
            base = dict(variant=variant, ts_code=ts_code,
                        event_date=ev["event_date"],
                        dd20=float(dd20), bounce=float(bounce))
            for H in H_LIST:
                recs.append({**base, "H": H,
                             **_simulate_one(open_, close, dts, d_int, lim,
                                             ts_code, j, H)})
    fb = _G["fallback_loads"]
    _G["fallback_loads"] = 0
    return recs, fb


def _simulate_one(open_, close, dts, d_int, lim, ts_code, j, H) -> dict:
    """逐条复刻已复核口径:次日开盘买(涨停/无报价拒买不递补,整手现金),
    入场日记第 1 日,第 H 个交易日(个股序列行号)收盘卖,跌停顺延,耗尽 truncated。"""
    n = len(close)
    if j + 1 >= n:
        return dict(status="truncated_no_next")  # 事件后无下一行(含退市),披露口径
    e = j + 1
    o = open_[e]
    if not np.isfinite(o):
        return dict(status="dropped_no_quote")
    up = _lookup_limit(lim, int(d_int[e]), 1, ts_code)
    if np.isfinite(up) and o >= up - TOL:
        return dict(status="dropped_limitup")
    px = o * (1.0 + se.SLIPPAGE)
    sh = int(BUDGET / px / se.BOARD_LOT) * se.BOARD_LOT
    if sh < se.BOARD_LOT:
        sh = se.BOARD_LOT
    comm = se.buy_cost(sh, px)
    while sh > 0 and sh * px + comm > BUDGET + 1e-6:
        sh -= se.BOARD_LOT
        comm = se.buy_cost(sh, px) if sh > 0 else 0.0
    if sh <= 0:
        return dict(status="dropped_cash")
    base = dict(entry_date=dts[e], entry_raw=float(o), entry_exec=float(px),
                shares=int(sh), buy_comm=float(comm))
    deferred = 0
    for r in range(e + H - 1, n):
        c = close[r]
        if not np.isfinite(c):
            continue  # 行计入但不评估(个股序列实际无停牌行,防御保留)
        dn = _lookup_limit(lim, int(d_int[r]), 2, ts_code)
        if np.isfinite(dn) and c <= dn + TOL:
            deferred += 1
            continue
        xs = c * (1.0 - se.SLIPPAGE)
        xcomm, stamp = se.sell_costs(sh, xs, dts[r])
        net_pnl = sh * (xs - px) - comm - xcomm - stamp
        net_ret = net_pnl / (sh * px + comm)
        return dict(status="closed", exit_date=dts[r], exit_raw=float(c),
                    exit_exec=float(xs), sell_comm=float(xcomm), stamp=float(stamp),
                    gross_ret=float(xs / px - 1.0), net_ret=float(net_ret),
                    net_pnl=float(net_pnl), held_rows=int(r - e + 1),
                    deferred_days=int(deferred), **base)
    return dict(status="truncated_exhausted", deferred_days=int(deferred), **base)


# ---------------------------------------------------------------- 聚类稳健 t(Liang-Zeger,预登记口径)
def cluster_t(x: np.ndarray, clusters: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return np.nan
    xbar = x.mean()
    s = x - xbar
    df = pd.DataFrame({"s": s, "c": clusters})
    sums = df.groupby("c")["s"].sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan
    var = (g / (g - 1.0)) * float((sums ** 2).sum()) / (n * n)
    if var <= 0:
        return np.nan
    return float(xbar / np.sqrt(var))


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t_all = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("SEED TRIAL HISTORY START | 预登记=README.md(冻结) | "
        "预计总时长约 15-25 分钟(扫描 ~3min ×2,涨跌停 ~4min,模拟 ~6min,汇总 ~2min)")

    files = sorted(str(p) for p in DATA_DIR.glob("*.parquet"))
    log(f"[init] 个股 parquet {len(files)} 个;CPU {mp.cpu_count()}")

    # ---------------- 阶段 1:全历史扫描(两变体) ----------------
    events = {}
    for name, fn in (("v1", scan_one_v1), ("v2", scan_one_v2)):
        events[name] = run_scan(name, fn, files)
        events[name].to_parquet(OUT_DIR / f"events_history_{name}.parquet", index=False)
        log(f"[dump] events_history_{name}.parquet {len(events[name])} 行")

    # ---------------- 自检 1:2026 子集与既有事件表逐行一致(硬断言) ----------------
    check1 = {}
    for name in VARIANTS:
        check1[name] = check_subset_equal(
            events[name], REF_DIR / f"events_{name}.parquet", name)
        d = check1[name]
        log(f"[check1] {name}: 2026 子集 {d['n_subset']} 行 vs 参照 {d['n_ref']} 行,"
            f"浮点最大绝对差 {d.get('max_abs_float_diff', float('nan')):.2e}"
            f" -> {'PASS' if d['ok'] else 'FAIL'}")
    check1_ok = all(d["ok"] for d in check1.values())

    # ---------------- 阶段 2:涨跌停按需加载 ----------------
    all_ev_dates = pd.concat([events[n]["event_date"] for n in VARIANTS]).sort_values()
    limit_files = sorted(p.stem for p in LIMIT_DIR.glob("*.parquet"))
    lf_dates = pd.to_datetime(pd.Series(limit_files), format="%Y%m%d")
    lo = lf_dates - pd.Timedelta(days=LIMIT_LOOKAHEAD_DAYS)
    # 文件日 d 被需要 ⟺ 存在事件日 ∈ [d-40, d]
    ev_arr = all_ev_dates.to_numpy()
    idx_hi = np.searchsorted(ev_arr, lf_dates.to_numpy(), side="right")
    idx_lo = np.searchsorted(ev_arr, lo.to_numpy(), side="left")
    needed_mask = idx_hi > idx_lo
    needed_dates = [d for d, m in zip(limit_files, needed_mask) if m]
    needed_codes = set(pd.concat([events[n]["ts_code"] for n in VARIANTS]).unique())
    log(f"[limits] 需求日期(事件日~+40 自然日并集){int(needed_mask.sum())}/{len(limit_files)}"
        f" 文件;有事件个股 {len(needed_codes)} 只")
    limits_by_code, n_limit_missing = load_limits(needed_dates, needed_codes)

    # ---------------- 阶段 3:逐股模拟(每事件×变体×H) ----------------
    ev_by_code: dict[str, dict] = {}
    for name in VARIANTS:
        for r in events[name].itertuples(index=False):
            d = ev_by_code.setdefault(r.ts_code, {})
            d.setdefault(name, []).append(
                dict(event_date=r.event_date, anchor_close=r.anchor_close))
    tasks = [dict(ts_code=code, events=evd, limits=limits_by_code.get(code))
             for code, evd in sorted(ev_by_code.items())]
    log(f"[simulate] 待模拟个股 {len(tasks)} 只,事件 "
        f"{sum(len(v) for e in ev_by_code.values() for v in e.values())} 起 × H{H_LIST}")

    t0 = time.time()
    trade_recs: list[dict] = []
    fallback_total = 0
    n_done = 0
    with mp.Pool(processes=max(1, mp.cpu_count() - 1),
                 initializer=_sim_worker_init) as pool:
        for recs, fb in pool.imap_unordered(simulate_stock, tasks, chunksize=4):
            trade_recs.extend(recs)
            fallback_total += fb
            n_done += 1
            if n_done % 500 == 0:
                log(f"heartbeat: 模拟 {n_done}/{len(tasks)} 股,"
                    f"累计记录 {len(trade_recs)} ({time.time() - t0:.0f}s)")
    trades = pd.DataFrame(trade_recs)
    log(f"[simulate] 完成:{len(trades)} 行(事件×变体×H) ({time.time() - t0:.0f}s)")

    # ---------------- 种子选择布尔列 ----------------
    dd, bo = trades["dd20"], trades["bounce"]
    trades["sel_S1"] = (dd <= -0.15) & (bo > 0.02) & (bo <= 0.08)
    trades["sel_S2"] = (dd <= -0.20) & (bo > 0.02) & (bo <= 0.08)
    trades["sel_S3"] = (dd <= -0.25) & (bo > 0.02) & (bo <= 0.08)
    trades["sel_S4"] = dd <= -0.15
    trades["sel_S5"] = dd <= -0.25
    trades["sel_ALL"] = True
    trades.to_parquet(OUT_DIR / "trades_seed.parquet", index=False)
    log(f"[dump] trades_seed.parquet {len(trades)} 行")
    vc = trades.groupby(["variant", "H"])["status"].value_counts()
    log(f"[simulate] 状态分布:\n{vc.to_string()}")

    # ---------------- 阶段 4:汇总 36 行 ----------------
    sum_rows: list[dict] = []
    yearly: dict[tuple, pd.Series] = {}
    for variant in VARIANTS:
        tv = trades[trades["variant"] == variant]
        n_uni = int(len(tv) / len(H_LIST))
        for seed in SEEDS:
            sel_col = f"sel_{seed}"
            for H in H_LIST:
                g = tv[(tv["H"] == H) & tv[sel_col]]
                st = g["status"].value_counts().to_dict()
                cl = g[g["status"] == "closed"]
                ret = cl["net_ret"].to_numpy(dtype=float)
                n_closed = len(cl)
                ct = cluster_t(ret, cl["entry_date"].astype(str).to_numpy()) \
                    if n_closed else np.nan
                # 逐年净笔均(按入场年)
                yr_stats = {}
                if n_closed:
                    yr = cl["entry_date"].dt.year
                    grp = cl.groupby(yr)["net_ret"]
                    ymean = grp.mean()
                    ycount = grp.count()
                    yr_stats = {int(y): (float(ymean.loc[y]), int(ycount.loc[y]))
                                for y in ymean.index}
                    qual = [y for y, (m, c) in yr_stats.items() if c >= 30]
                    win_year_share = (float(np.mean([yr_stats[y][0] > 0
                                                     for y in qual]))
                                      if qual else np.nan)
                else:
                    win_year_share = np.nan
                # 日期集中度(top5 入场日净盈亏合计 / 全部净盈亏合计)
                if n_closed:
                    pnl_by_day = cl.groupby(cl["entry_date"].dt.date)["net_pnl"].sum()
                    total_pnl = float(pnl_by_day.sum())
                    top5 = float(pnl_by_day.sort_values(ascending=False).head(5).sum())
                    date_conc = top5 / total_pnl if total_pnl > 0 else np.nan
                else:
                    date_conc = np.nan
                sum_rows.append(dict(
                    variant=variant, seed=seed, H=H,
                    n_universe=n_uni, n_selected=int(len(g)),
                    n_closed=n_closed,
                    n_dropped_limitup=int(st.get("dropped_limitup", 0)),
                    n_dropped_no_quote=int(st.get("dropped_no_quote", 0)),
                    n_dropped_cash=int(st.get("dropped_cash", 0)),
                    n_truncated=int(st.get("truncated_no_next", 0)
                                    + st.get("truncated_exhausted", 0)),
                    net_mean=float(ret.mean()) if n_closed else np.nan,
                    net_median=float(np.median(ret)) if n_closed else np.nan,
                    win_rate=float((ret > 0).mean()) if n_closed else np.nan,
                    cluster_t=ct,
                    win_year_share=win_year_share,
                    date_conc=date_conc))
                yearly[(variant, seed, H)] = yr_stats
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(OUT_DIR / "summary_seed.csv", index=False, float_format="%.6f")
    log(f"[dump] summary_seed.csv {len(summary)} 行")

    # ---------------- 自检 2:交易因果 ----------------
    closed = trades[trades["status"] == "closed"]
    n_viol_entry = int((closed["entry_date"] <= closed["event_date"]).sum())
    n_viol_exit = int((closed["exit_date"] <= closed["entry_date"]).sum())
    check2_ok = n_viol_entry == 0 and n_viol_exit == 0
    log(f"[check2] 交易因果:entry<=event {n_viol_entry} 笔,exit<=entry {n_viol_exit} 笔"
        f" -> {'PASS' if check2_ok else 'FAIL'}")

    # ---------------- 自检 3:计数守恒 ----------------
    c3a = bool((summary["n_selected"] <= summary["n_universe"]).all())
    all_rows = summary[summary["seed"] == "ALL"]
    c3b = bool((all_rows["n_selected"] == all_rows["n_universe"]).all())
    cons = (summary["n_closed"] + summary["n_dropped_limitup"]
            + summary["n_dropped_no_quote"] + summary["n_dropped_cash"]
            + summary["n_truncated"])
    c3c = bool((cons == summary["n_selected"]).all())
    check3_ok = c3a and c3b and c3c
    log(f"[check3] 计数守恒:n_selected<=n_universe={c3a},"
        f"ALL 行相等={c3b},分量加总={c3c} -> {'PASS' if check3_ok else 'FAIL'}")

    # ---------------- 自检 4:随机 5 笔成交完整生命周期(不同年代) ----------------
    rng = np.random.default_rng(SAMPLE_SEED)
    eras = [(2007, 2010), (2011, 2014), (2015, 2018), (2019, 2022), (2023, 2026)]
    c4_samples = []
    cl_pool = closed.assign(yr=closed["entry_date"].dt.year)
    for lo_y, hi_y in eras:
        pool_df = cl_pool[(cl_pool["yr"] >= lo_y) & (cl_pool["yr"] <= hi_y)]
        if len(pool_df) == 0:
            c4_samples.append(dict(era=f"{lo_y}-{hi_y}", note="该年代无成交"))
            continue
        r = pool_df.iloc[int(rng.integers(0, len(pool_df)))]
        c4_samples.append(dict(
            era=f"{lo_y}-{hi_y}", variant=r["variant"], ts_code=r["ts_code"],
            H=int(r["H"]), event_date=str(r["event_date"].date()),
            dd20=float(r["dd20"]), bounce=float(r["bounce"]),
            entry_date=str(r["entry_date"].date()), entry_raw=float(r["entry_raw"]),
            entry_exec=float(r["entry_exec"]), shares=int(r["shares"]),
            buy_comm=float(r["buy_comm"]), exit_date=str(r["exit_date"].date()),
            exit_raw=float(r["exit_raw"]), exit_exec=float(r["exit_exec"]),
            sell_comm=float(r["sell_comm"]), stamp=float(r["stamp"]),
            gross_ret=float(r["gross_ret"]), net_pnl=float(r["net_pnl"]),
            net_ret=float(r["net_ret"]), held_rows=int(r["held_rows"]),
            deferred_days=int(r["deferred_days"])))
        log(f"[check4-sample] {lo_y}-{hi_y} | {r['variant']} {r['ts_code']} "
            f"ev={r['event_date'].date()} H={r['H']} "
            f"入{r['entry_date'].date()}@{r['entry_raw']:.3f}→{r['entry_exec']:.3f}×{r['shares']} "
            f"出{r['exit_date'].date()}@{r['exit_raw']:.3f}→{r['exit_exec']:.3f} "
            f"净{r['net_ret']:+.4f} 持有{r['held_rows']}行 顺延{r['deferred_days']}")
    check4_ok = all("ts_code" in s for s in c4_samples)
    log(f"[check4] 抽样 5 笔(年代×1,种子 {SAMPLE_SEED})"
        f" -> {'PASS' if check4_ok else 'FAIL(某年代无成交)'}")

    # ---------------- 自检 5:缺文件天数 + truncated 计数披露 ----------------
    trunc_tab = (trades[trades["status"].str.startswith("truncated")]
                 .groupby(["variant", "H", "status"]).size().to_dict())
    LIMIT_EARLIEST = pd.Timestamp("2007-01-04")  # stk_limit 文件起点(README 披露 4)
    n_pre_limit_entry = int((closed["entry_date"] < LIMIT_EARLIEST).sum())
    check5 = dict(n_limit_missing_days=int(n_limit_missing),
                  n_limit_files_needed=int(len(needed_dates)),
                  n_limit_files_total=int(len(limit_files)),
                  n_fallback_loads=int(fallback_total),
                  n_closed_entry_before_limit_era=n_pre_limit_entry,
                  truncated_by_variant_H={f"{k[0]}|H{k[1]}|{k[2]}": int(v)
                                          for k, v in trunc_tab.items()})
    log(f"[check5] stk_limit 缺文件 {n_limit_missing} 天(需求 {len(needed_dates)} 天);"
        f"入场早于 2007-01-04(全程无涨跌停约束)的成交 {n_pre_limit_entry} 笔;"
        f"truncated 计数={check5['truncated_by_variant_H']} -> PASS(披露项)")

    checks_all = check1_ok and check2_ok and check3_ok and check4_ok

    # ---------------- 宣判(每种子只在 v1;v2 参照) ----------------
    verdicts: dict = {}
    for seed in SEEDS:
        per_H = {}
        for H in H_LIST:
            row = summary[(summary["variant"] == "v1") & (summary["seed"] == seed)
                          & (summary["H"] == H)].iloc[0]
            crit = dict(
                c1_n_closed_ge_300=bool(row["n_closed"] >= 300),
                c2_net_mean_pos=bool(np.isfinite(row["net_mean"])
                                     and row["net_mean"] > 0),
                c3_cluster_t_ge_2=bool(np.isfinite(row["cluster_t"])
                                       and row["cluster_t"] >= 2),
                c4_win_year_share_ge_60pct=bool(
                    np.isfinite(row["win_year_share"])
                    and row["win_year_share"] >= 0.6),
                c5_date_conc_le_50pct=bool(np.isfinite(row["date_conc"])
                                           and row["date_conc"] <= 0.5),
            )
            per_H[str(H)] = dict(
                **{k: (float(row[mk]) if np.isfinite(row[mk]) else None)
                   for k, mk in [("n_closed", "n_closed"),
                                 ("net_mean", "net_mean"),
                                 ("cluster_t", "cluster_t"),
                                 ("win_year_share", "win_year_share"),
                                 ("date_conc", "date_conc")]},
                checks=crit, passed=bool(all(crit.values())))
        alive = any(v["passed"] for v in per_H.values())
        verdicts[seed] = dict(variant_judged="v1", per_H=per_H, alive=bool(alive),
                              verdict="活(存在 H 五线全满)" if alive else "死")
        log(f"[verdict] {seed}: "
            + " ".join(f"H{h}:{'过' if v['passed'] else '否'}" for h, v in per_H.items())
            + f" -> {verdicts[seed]['verdict']}")
    alive_seeds = [s for s in SEEDS if verdicts[s]["alive"] and s != "ALL"]
    overall = ("种子存活:" + ",".join(alive_seeds) + "(给定版候选)"
               if alive_seeds else
               "全部不过线 —— 2026 沙盒的肉是全历史不存在的运气,该方向封档")
    log(f"[verdict] 总评:{overall}")

    # ---------------- verdict.json ----------------
    vout = dict(
        experiment="急跌背离种子全历史审判",
        prereadme="README.md 先于任何跑数落盘(冻结)",
        disclosure=[
            "数据加载自写(逐股 parquet,不复权);成本/滑点/整手调冻结引擎原语",
            "退市股:事件后无下一行记 truncated_no_next(按没买成处理,结论偏乐观方向,README 披露 3)",
            "个股序列无停牌行,'停牌日计入但不评估'条款自然空置",
            "stk_limit 缺文件日=无涨跌停约束;顺延超出事件+40 自然日的日期按需回退加载",
            "dd20 窗口 j<20 时按 max(0, j-20) 截断(上市初期事件,占比极小)",
            "索引口径为个股序列行号(README 第 4 节),非市场日历对齐",
        ],
        seeds=verdicts,
        pass_criteria="v1 同一 H 五线全满:n_closed>=300 / 净笔均>0 / cluster_t>=2 / "
                      "盈利年占比>=60% / 日期集中度<=50%",
        alive_seeds=alive_seeds,
        overall=overall,
        checks=dict(
            check1_scan_consistency=dict(ok=bool(check1_ok), detail=check1),
            check2_trade_causality=dict(ok=bool(check2_ok),
                                        n_entry_viol=n_viol_entry,
                                        n_exit_viol=n_viol_exit),
            check3_count_conservation=dict(ok=bool(check3_ok), sel_le_uni=bool(c3a),
                                           all_eq_uni=bool(c3b), sum_eq=bool(c3c)),
            check4_samples=dict(ok=bool(check4_ok), seed=SAMPLE_SEED,
                                samples=c4_samples),
            check5_disclosure=dict(ok=True, **check5),
        ),
        checks_all_pass=bool(checks_all),
        duration_sec=round(time.time() - t_all, 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(vout, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] verdict.json")

    # ---------------- report.md ----------------
    write_report(summary, verdicts, yearly, vout["checks"], checks_all, overall)
    log("[dump] report.md")
    log(f"SEED TRIAL HISTORY DONE ({time.time() - t_all:.0f}s) "
        f"自检总评={'ALL PASS' if checks_all else 'HAS FAIL'}")

    pd.set_option("display.width", 300)
    print("\n===== SUMMARY(36 行全出数) =====")
    print(summary.to_string(index=False))


def write_report(summary, verdicts, yearly, checks, checks_all, overall) -> None:
    checks_all_str = "ALL PASS" if checks_all else "HAS FAIL(如实交付)"
    L: list[str] = []
    L.append("# 急跌背离种子全历史审判报告(预登记一发,2026-09-04)")
    L.append("")
    L.append("> **披露 1**:种子来自 2026 沙盒的探索性扫描(看图挑出),本实验是它们第一次见到全历史。")
    L.append("> **披露 2**:数据加载自写(逐股 parquet,不复权);成本、滑点、整手调冻结引擎原语。")
    L.append("> **披露 3**:退市股事件后数据不足记 truncated(按没买成处理,结论偏乐观方向)。")
    L.append("> **披露 4**:stk_limit 缺文件日视为无涨跌停约束;个股序列无停牌行,'停牌日计入但不评估'条款空置。")
    L.append("")
    L.append(f"**自检总评:{checks_all_str}**")
    L.append("")
    L.append(f"**总评:{overall}**")
    L.append("")
    L.append("## 1. 宣判(v1,判活线:同一 H 五线全满)")
    L.append("")
    L.append("| 种子 | H | n_closed | 净笔均 | cluster_t | 盈利年占比 | 日期集中度 | 五线(12345) | 过线 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for seed, v in verdicts.items():
        for h, pv in v["per_H"].items():
            fmt = lambda x, p="+.4f": (format(x, p) if x is not None else "NaN")  # noqa: E731
            crit = "".join("1" if k else "0" for k in pv["checks"].values())
            L.append(f"| {seed} | {h} | {int(pv['n_closed'])} | {fmt(pv['net_mean'])} | "
                     f"{fmt(pv['cluster_t'], '.3f')} | {fmt(pv['win_year_share'], '.3f')} | "
                     f"{fmt(pv['date_conc'], '.3f')} | {crit} | "
                     f"{'过' if pv['passed'] else '否'} |")
    L.append("")
    L.append("五线编码顺序:1=n_closed>=300 / 2=净笔均>0 / 3=cluster_t>=2 / 4=盈利年占比>=60% / 5=日期集中度<=50%。")
    L.append("")
    for seed, v in verdicts.items():
        L.append(f"- {seed}: **{v['verdict']}**")
    L.append("")
    L.append("## 2. 36 行全表(配置为行,全出数)")
    L.append("")
    L.append("| 变体 | 种子 | H | n_universe | n_selected | n_closed | drop涨停 | drop无报价 | drop现金 | n_truncated | 净笔均 | 净中位 | 胜率 | cluster_t | 盈利年占比 | 日期集中度 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary.itertuples(index=False):
        fmt = lambda x, p="+.4f": (format(x, p) if np.isfinite(x) else "NaN")  # noqa: E731
        L.append(f"| {r.variant} | {r.seed} | {r.H} | {r.n_universe} | {r.n_selected} | "
                 f"{r.n_closed} | {r.n_dropped_limitup} | {r.n_dropped_no_quote} | "
                 f"{r.n_dropped_cash} | {r.n_truncated} | {fmt(r.net_mean)} | "
                 f"{fmt(r.net_median)} | {fmt(r.win_rate, '.3f')} | "
                 f"{fmt(r.cluster_t, '.3f')} | {fmt(r.win_year_share, '.3f')} | "
                 f"{fmt(r.date_conc, '.3f')} |")
    L.append("")
    L.append("## 3. v1 各种子逐年净笔均(按入场年;括号内为当年成交笔数)")
    L.append("")
    years = sorted({y for (variant, seed, H), ys in yearly.items()
                    if variant == "v1" for y in ys})
    for seed in SEEDS:
        L.append(f"### {seed}(v1)")
        L.append("")
        L.append("| 年份 | H10 净笔均(n) | H20 净笔均(n) | H25 净笔均(n) |")
        L.append("|---|---|---|---|")
        for y in years:
            cells = []
            for H in H_LIST:
                ys = yearly.get(("v1", seed, H), {})
                if y in ys:
                    m, c = ys[y]
                    cells.append(f"{m:+.4f}({c})")
                else:
                    cells.append("—")
            L.append(f"| {y} | {cells[0]} | {cells[1]} | {cells[2]} |")
        L.append("")
    L.append("## 4. 自检结果(README 第 7 节)")
    L.append("")
    c1 = checks["check1_scan_consistency"]["detail"]
    for name in VARIANTS:
        d = c1[name]
        L.append(f"1. 扫描一致性 {name}:2026 子集 {d['n_subset']} 行 vs 参照 "
                 f"{d['n_ref']} 行,浮点最大绝对差 {d.get('max_abs_float_diff', float('nan')):.2e}"
                 f" -> {'PASS' if d['ok'] else 'FAIL'}")
    c2 = checks["check2_trade_causality"]
    L.append(f"2. 交易因果:entry<=event {c2['n_entry_viol']} 笔,exit<=entry "
             f"{c2['n_exit_viol']} 笔 -> {'PASS' if c2['ok'] else 'FAIL'}")
    c3 = checks["check3_count_conservation"]
    L.append(f"3. 计数守恒:n_selected<=n_universe={c3['sel_le_uni']},"
             f"ALL 行 n_selected==n_universe={c3['all_eq_uni']},"
             f"分量加总={c3['sum_eq']} -> {'PASS' if c3['ok'] else 'FAIL'}")
    c4 = checks["check4_samples"]
    L.append(f"4. 随机 5 笔成交完整生命周期(种子 {c4['seed']},每年代 1 笔):")
    L.append("")
    L.append("| 年代 | 变体 | 代码 | H | 事件日 | dd20 | bounce | 入场日 | 入场价(原始/执行) | 股数 | 买佣 | 出场日 | 出场价(原始/执行) | 卖佣 | 印花税 | 净盈亏 | 净收益 | 持有行 | 顺延 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in c4["samples"]:
        if "ts_code" not in s:
            L.append(f"| {s['era']} | — | {s['note']} |")
            continue
        L.append(f"| {s['era']} | {s['variant']} | {s['ts_code']} | {s['H']} | "
                 f"{s['event_date']} | {s['dd20']:+.4f} | {s['bounce']:+.4f} | "
                 f"{s['entry_date']} | {s['entry_raw']:.3f}/{s['entry_exec']:.3f} | "
                 f"{s['shares']} | {s['buy_comm']:.2f} | {s['exit_date']} | "
                 f"{s['exit_raw']:.3f}/{s['exit_exec']:.3f} | {s['sell_comm']:.2f} | "
                 f"{s['stamp']:.2f} | {s['net_pnl']:+.2f} | {s['net_ret']:+.4f} | "
                 f"{s['held_rows']} | {s['deferred_days']} |")
    L.append("")
    c5 = checks["check5_disclosure"]
    L.append(f"5. stk_limit 缺文件 {c5['n_limit_missing_days']} 天"
             f"(需求 {c5['n_limit_files_needed']}/{c5['n_limit_files_total']} 天,"
             f"缺文件日=无约束);入场早于 2007-01-04(stk_limit 起点,全程无约束)的成交 "
             f"{c5['n_closed_entry_before_limit_era']} 笔;"
             f"truncated 逐变体×H 计数:")
    L.append("")
    L.append("```json")
    L.append(json.dumps(c5["truncated_by_variant_H"], ensure_ascii=False, indent=2))
    L.append("```")
    L.append("")
    L.append(f"**自检总评:{'ALL PASS' if checks_all else 'HAS FAIL'}**")
    L.append("")
    L.append("## 5. 宣判结语")
    L.append("")
    L.append(overall)
    L.append("")
    with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
