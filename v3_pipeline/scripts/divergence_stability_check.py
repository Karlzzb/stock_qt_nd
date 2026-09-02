#!/usr/bin/env python3
"""背离稳定性验收(issue #21): 主池 m_fractal15_full 与备池 m_zigzag05_nofilter。

双重验收:
  1) 时点一致性(前缀稳定性对拍): 按季度截断网格(每 63 个交易日一个截断点, 覆盖到数据
     末日), 每个截断点 T 只用 <=T 的数据重跑全市场检测, 断言信号日落入 (prev_T, T]
     窗口的事件与全量跑完全一致(零不一致)。网格覆盖使每个事件恰好被验证一次。
  2) 参数敏感性: 单因子扰动窗口参数(fractal order/min_sep, zigzag pct/min_sep,
     主池 regime 窗口), 报事件集重合率(同股同信号日口径)。

辅助断言:
  - 可复现性: 本脚本全量跑的事件表必须与归档 events.parquet 逐事件一致;
  - regime 截断等价: 全量 regime 的前缀切片 == 对截断股票群重算(抽样截断点实证)。

CLI:
  python divergence_stability_check.py [--smoke N] [--workers W] [--grid-step 63]
输出: v3_pipeline/reports/divergence_stability/{results.json, stability_report.md}
"""
import argparse
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import talib

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import divergence_lab as lab  # noqa: E402

REPO = SCRIPT_DIR.parents[1]
DATA_DIR = REPO / "stock_data" / "daily"
MAIN_CFG_PATH = REPO / "v3_pipeline/configs/divergence_lab/m_scan/m_fractal15_full.json"
BACKUP_CFG_PATH = REPO / "v3_pipeline/configs/divergence_lab/m_scan/m_zigzag05_nofilter.json"
OUT_DIR = REPO / "v3_pipeline/reports/divergence_stability"
REGIME_CODES = {"up": 1, "down": 2, "sideways": 0}

# 单因子扰动(窗口参数 ±20% 取整)
PERTURBATIONS = {
    "m_fractal15_full": [
        ("order_12", {"lows": {"order": 12}}),
        ("order_18", {"lows": {"order": 18}}),
        ("min_sep_16", {"lows": {"min_sep": 16}}),
        ("min_sep_24", {"lows": {"min_sep": 24}}),
        ("regime_96", {"regime": {"window": 96}}),
        ("regime_144", {"regime": {"window": 144}}),
    ],
    "m_zigzag05_nofilter": [
        ("pct_004", {"lows": {"pct": 0.04}}),
        ("pct_006", {"lows": {"pct": 0.06}}),
        ("min_sep_4", {"lows": {"min_sep": 4}}),
        ("min_sep_6", {"lows": {"min_sep": 6}}),
    ],
}


# ================================================================ 单股检测(数组版)
def detect_events_from_arrays(close, vol, cfg, low=None):
    """与 divergence_lab.load_stock 的检测分支逐行对应(fractal/zigzag), 输入为数组。
    用于全量跑与截断前缀跑共用同一代码路径。"""
    lcfg, dcfg = cfg["lows"], cfg["divergence"]
    close = np.asarray(close, np.float64)
    vol = np.asarray(vol, np.float64)
    if lcfg["price"] == "low":
        if low is None:
            raise ValueError("price=low 需要 low 数组")
        p = np.asarray(low, np.float64)
    else:
        p = close
    dif, _, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    warmup = int(cfg["universe"]["min_day"])
    method = lcfg["method"]
    if method == "fractal":
        low_idx, sig_idx = lab.fractal_lows(p, int(lcfg["order"]))
    elif method == "zigzag":
        low_idx, sig_idx = lab.zigzag_lows(p, float(lcfg["pct"]))
    else:
        raise ValueError(f"稳定性验收只支持 fractal/zigzag, 收到 {method}")
    low_idx, sig_idx = lab.apply_min_sep(low_idx, sig_idx, int(lcfg["min_sep"]))
    ind = dif if dcfg["indicator"] == "dif" else hist
    return lab.detect_divergence_events(low_idx, sig_idx, close, ind, dif, vol, dcfg, warmup)


# ================================================================ 截断网格与对拍
def truncation_grid(dates, step=63):
    """交易日轴上的截断点: 每 step 根一个, 末点恒为最后交易日。返回日值数组。"""
    n = len(dates)
    idx = list(range(step - 1, n, step))
    if not idx or idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray([dates[i] for i in idx])


def _event_keys(ev):
    return set(zip(ev["sig"].tolist(), ev["low"].tolist(), ev["prev"].tolist(),
                   ev["rank"].tolist(), ev["form"].tolist()))


def prefix_mismatches(full, dates, detect_fn, grid):
    """full: 全量跑事件 dict(索引语义); dates: 该股交易日轴; detect_fn(t_idx):
    用前缀 [:t_idx+1] 重跑检测。逐截断窗口 (prev_T, T] 对拍, 返回不一致列表。"""
    mismatches = []
    full_keys = _event_keys(full)
    sig_dates = dates[full["sig"]] if len(full["sig"]) else np.empty(0, np.int32)
    prev_t = None
    for T in grid:
        t_idx = prefix_end(dates, T)
        pref_keys = _event_keys(detect_fn(t_idx))
        # 前缀跑的事件按定义 sig<=T; 过滤到本窗口
        if prev_t is not None:
            pref_keys = {k for k in pref_keys if dates[k[0]] > prev_t}
        if prev_t is None:
            win_mask = sig_dates <= T
        else:
            win_mask = (sig_dates > prev_t) & (sig_dates <= T)
        full_win = {k for k, m in zip(zip(full["sig"].tolist(), full["low"].tolist(),
                                          full["prev"].tolist(), full["rank"].tolist(),
                                          full["form"].tolist()), win_mask) if m}
        if pref_keys != full_win:
            mismatches.append({
                "T": str(np.datetime64(int(T), "D")),
                "prefix_only": sorted(pref_keys - full_win)[:10],
                "full_only": sorted(full_win - pref_keys)[:10],
            })
        prev_t = T
    return mismatches


def overlap_metrics(base_keys, pert_keys):
    """事件集重合率: base/pert 为 (symbol, sig_date) 可哈希键列表。"""
    b, p = set(base_keys), set(pert_keys)
    inter = len(b & p)
    return {
        "n_base": len(b), "n_pert": len(p), "n_inter": inter,
        "recall_base": inter / len(b) if b else np.nan,
        "recall_pert": inter / len(p) if p else np.nan,
    }


SPOT_CHECK_DATES = 25  # 逐日粒度抽查的非网格交易日数(确定性种子)


def _regime_prefix_at(sums, cnts, all_dates, T, window, th):
    """截断点 T 的 regime(前缀切片语义), 返回全局轴等长的 int8 数组。"""
    iT = int(np.searchsorted(all_dates, T))
    _, reg_t = _regime_from_sums(sums[: iT + 1], cnts[: iT + 1], window, th)
    full_reg = np.full(len(all_dates), -1, np.int8)
    full_reg[: iT + 1] = reg_t
    return full_reg


# ================================================================ 工作进程
def prefix_end(dates, T):
    """该股在全局日 T 的时点一致前缀末尾: 最后一个 <= T 的交易日下标(无则 -1)。
    个股在 T 日停牌时 T 不在 dates 中, 必须回退到最后一个 <= T 的交易日,
    否则前缀会纳入 > T 的未来数据。"""
    return int(np.searchsorted(dates, T, "right")) - 1


def _load_one(path):
    """读单股, 返回检测与 regime 所需的紧凑数组。
    预处理与 divergence_lab.load_stock 逐行对应: pct_chg 列存在时优先使用
    (与归档 events.parquet 的口径一致), 否则由收盘价自算; 成交量兼容 vol/volume。"""
    symbol = Path(path).stem
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    if len(df) < 30:
        return {"symbol": symbol, "error": "too_short"}
    close = df["close"].to_numpy(np.float64)
    if "pct_chg" in df.columns:
        pct = df["pct_chg"].to_numpy(np.float64)
    else:
        pct = np.concatenate([[np.nan], close[1:] / close[:-1] * 100.0 - 100.0])
    vol_col = "vol" if "vol" in df.columns else ("volume" if "volume" in df.columns else None)
    vol = df[vol_col].to_numpy(np.float64) if vol_col else np.full(len(df), np.nan)
    dates = lab._to_days(df["trade_date"])
    cf = np.cumprod(1.0 + np.where(np.isfinite(pct), pct, 0.0) / 100.0)
    return {"symbol": symbol, "dates": dates, "close": close, "vol": vol,
            "cf": cf.astype(np.float32)}


def _worker_prefix(args):
    """股票块 × 全部截断点: 返回 {symbol: {T: [事件键...]}}(已按窗口与 regime 过滤)。"""
    (paths, cfg, grid, regime_by_t, all_dates, prev_bounds) = args
    want_regime = cfg["filters"]["regime"]
    out = {}
    for path in paths:
        st = _load_one(path)
        if "error" in st:
            continue
        dates, close, vol = st["dates"], st["close"], st["vol"]
        per = {}
        for gi, T in enumerate(grid):
            t_idx = prefix_end(dates, T)
            if t_idx < int(cfg["universe"]["min_day"]):
                continue
            ev = detect_events_from_arrays(close[: t_idx + 1], vol[: t_idx + 1], cfg)
            if not len(ev["sig"]):
                continue
            sd = dates[ev["sig"]]
            lo_b = prev_bounds[gi]
            win = sd > lo_b if lo_b is not None else np.ones(len(sd), bool)
            if want_regime != "all":
                reg = regime_by_t[gi]
                want = REGIME_CODES[want_regime]
                win &= reg[np.searchsorted(all_dates, sd)] == want
            keys = [(int(ev["sig"][i]), int(ev["low"][i]), int(ev["prev"][i]),
                     int(ev["rank"][i]), int(ev["form"][i]))
                    for i in np.nonzero(win)[0]]
            if keys:
                per[int(T)] = keys
        if per:
            out[st["symbol"]] = per
    return out


def _worker_full(args):
    """股票块全量跑(含 regime 过滤), 返回 {symbol: [事件键(含信号日期)]}。"""
    paths, cfg, regime_arr, all_dates = args
    want_regime = cfg["filters"]["regime"]
    out = {}
    for path in paths:
        st = _load_one(path)
        if "error" in st:
            continue
        ev = detect_events_from_arrays(st["close"], st["vol"], cfg)
        if not len(ev["sig"]):
            continue
        sd = st["dates"][ev["sig"]]
        keep = np.ones(len(sd), bool)
        if want_regime != "all":
            want = REGIME_CODES[want_regime]
            keep &= regime_arr[np.searchsorted(all_dates, sd)] == want
        keys = [(int(ev["sig"][i]), int(ev["low"][i]), int(ev["prev"][i]),
                 int(ev["rank"][i]), int(ev["form"][i]), int(sd[i]))
                for i in np.nonzero(keep)[0]]
        if keys:
            out[st["symbol"]] = keys
    return out


# ================================================================ regime 工具
def _regime_from_sums(sums, cnts, window, th):
    """与 lab.build_market_regime 后半段同逻辑, 但输入为预聚合的逐日收益和/计数。"""
    mean = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
    idx = np.cumprod(1.0 + mean)
    regime = np.full(len(sums), -1, np.int8)
    if len(sums) > window:
        roll = idx[window:] / idx[:-window] - 1.0
        regime[window:] = np.where(roll > th, 1, np.where(roll < -th, 2, 0)).astype(np.int8)
    return idx, regime


def _aggregate_returns(stocks, all_dates):
    sums = np.zeros(len(all_dates), np.float64)
    cnts = np.zeros(len(all_dates), np.float64)
    for st in stocks:
        pos = np.searchsorted(all_dates, st["dates"])
        c = st["cf"].astype(np.float64)
        r = c[1:] / c[:-1] - 1.0
        np.add.at(sums, pos[1:], r)
        np.add.at(cnts, pos[1:], 1.0)
    return sums, cnts


# ================================================================ 归档对照
def _archived_keys(cfg_name):
    """归档 events.parquet -> {(symbol, sig, low, prev, rank, form)}(日值语义)。"""
    fp = REPO / "v3_pipeline/reports/divergence_lab/m_scan" / cfg_name / "events.parquet"
    df = pd.read_parquet(fp)
    sig_days = lab._to_days(df["date"])
    return set(zip(df["ts_code"].tolist(), sig_days.astype(int).tolist(),
                   df["low_date"].astype(int).tolist(),
                   df["prev_low_date"].astype(int).tolist(),
                   df["compare_rank"].astype(int).tolist(),
                   df["formation"].astype(int).tolist()))


def _fullrun_keys(full_by_symbol, stocks_dates):
    """_worker_full 产出 -> {(symbol, sig_date, low_date, prev_date, rank, form)}。"""
    out = set()
    for sym, keys in full_by_symbol.items():
        d = stocks_dates[sym]
        for sig, low, prev, rank, form, sd in keys:
            out.add((sym, int(sd), int(d[low]), int(d[prev]), int(rank), int(form)))
    return out


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="只取前 N 只股票(冒烟)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--grid-step", type=int, default=63)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)  # noqa: E731

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    if args.smoke:
        files = files[: args.smoke]
    log(f"股票文件 {len(files)} 只, workers={args.workers}")

    # ---- Phase 1: 载入(并行) + 全局日期轴 + regime 聚合
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        stocks = [r for r in ex.map(_load_one, files, chunksize=32) if "error" not in r]
    log(f"Phase1 载入完成: 有效 {len(stocks)} 只")
    all_dates = np.unique(np.concatenate([st["dates"] for st in stocks]))
    sums, cnts = _aggregate_returns(stocks, all_dates)
    log(f"全局交易日 {len(all_dates)} 天 "
        f"({np.datetime64(int(all_dates[0]),'D')} ~ {np.datetime64(int(all_dates[-1]),'D')})")

    cfgs = {"m_fractal15_full": lab.load_config(str(MAIN_CFG_PATH)),
            "m_zigzag05_nofilter": lab.load_config(str(BACKUP_CFG_PATH))}
    grid = truncation_grid(all_dates, step=args.grid_step)
    log(f"截断网格 {len(grid)} 个点 (step={args.grid_step})")

    # regime: 主配置窗口的全量值 + 每个截断点的前缀切片 + 扰动窗口
    reg_cfgs = {}
    for name, cfg in cfgs.items():
        w, th = int(cfg["regime"]["window"]), float(cfg["regime"]["threshold"])
        _, reg_cfgs[name] = _regime_from_sums(sums, cnts, w, th)
    # 截断等价实证: 抽 3 个截断点用 lab.build_market_regime 对截断股票群重算
    equiv_checks = []
    for gi in (0, len(grid) // 2, len(grid) - 1):
        T = grid[gi]
        trunc = [{"dates": st["dates"][st["dates"] <= T],
                  "cf": st["cf"][st["dates"] <= T]} for st in stocks]
        trunc = [s for s in trunc if len(s["dates"]) > 1]
        for name, cfg in cfgs.items():
            w, th = int(cfg["regime"]["window"]), float(cfg["regime"]["threshold"])
            ad_t, reg_t = lab.build_market_regime(trunc, w, th)
            pos = np.searchsorted(all_dates, ad_t)
            ok = np.array_equal(reg_cfgs[name][pos], reg_t)
            equiv_checks.append({"grid_i": gi, "T": str(np.datetime64(int(T), "D")),
                                 "config": name, "equal": bool(ok)})
    log(f"regime 截断等价实证: {equiv_checks}")

    stocks_dates = {st["symbol"]: st["dates"] for st in stocks}
    chunks = [files[i:: args.workers] for i in range(args.workers)]
    results = {"equiv_checks": equiv_checks, "grid": [str(np.datetime64(int(t), "D"))
                                                      for t in grid]}

    for name, cfg in cfgs.items():
        log(f"===== {name} 全量跑 =====")
        reg = reg_cfgs[name]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            parts = list(ex.map(_worker_full,
                                [(c, cfg, reg, all_dates) for c in chunks]))
        full_by_symbol = {}
        for p in parts:
            full_by_symbol.update(p)
        log(f"{name} 全量事件股票数 {len(full_by_symbol)}")

        # 可复现性硬断言: 与归档 events.parquet 逐事件一致
        if not args.smoke:
            mine = _fullrun_keys(full_by_symbol, stocks_dates)
            arch = _archived_keys(name)
            diff_a, diff_b = mine - arch, arch - mine
            results.setdefault("reproduce", {})[name] = {
                "n_mine": len(mine), "n_archived": len(arch),
                "only_mine": len(diff_a), "only_archived": len(diff_b),
                "sample_only_mine": sorted(map(str, diff_a))[:5],
                "sample_only_archived": sorted(map(str, diff_b))[:5],
            }
            log(f"{name} 可复现性: mine={len(mine)} archived={len(arch)} "
                f"only_mine={len(diff_a)} only_archived={len(diff_b)}")

        # ---- 前缀稳定性对拍
        log(f"===== {name} 前缀稳定性({len(grid)} 截断点) =====")
        regime_by_t = []
        prev_bounds = []
        for gi, T in enumerate(grid):
            iT = int(np.searchsorted(all_dates, T))
            w = int(cfg["regime"]["window"])
            th = float(cfg["regime"]["threshold"])
            _, reg_t = _regime_from_sums(sums[: iT + 1], cnts[: iT + 1], w, th)
            full_reg = np.full(len(all_dates), -1, np.int8)
            full_reg[: iT + 1] = reg_t
            regime_by_t.append(full_reg)
            prev_bounds.append(grid[gi - 1] if gi > 0 else None)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            parts = list(ex.map(_worker_prefix,
                                [(c, cfg, grid, regime_by_t, all_dates, prev_bounds)
                                 for c in chunks]))
        # 汇总前缀事件并逐股对拍(对拍人口 = 全量 ∪ 前缀, 防全量零事件股票成为盲区)
        pref_by_symbol = {}
        for p in parts:
            for sym, per in p.items():
                pref_by_symbol.setdefault(sym, {}).update(per)
        n_mismatch, mismatch_detail = 0, []
        for sym in sorted(set(full_by_symbol) | set(pref_by_symbol)):
            keys = full_by_symbol.get(sym, [])
            full_win = {}
            for sig, low, prev, rank, form, sd in keys:
                # 窗口 i 语义为 (grid[i-1], grid[i]]: 取首个 grid[gi] >= sd
                gi = int(np.searchsorted(grid, sd, "left"))
                if gi >= len(grid):
                    gi = len(grid) - 1
                full_win.setdefault(int(grid[gi]), set()).add((sig, low, prev, rank, form))
            per = pref_by_symbol.get(sym, {})
            all_t = sorted(set(full_win) | set(per))
            for T in all_t:
                a, b = full_win.get(T, set()), set(map(tuple, per.get(T, [])))
                if a != b:
                    n_mismatch += 1
                    if len(mismatch_detail) < 20:
                        mismatch_detail.append({
                            "symbol": sym, "T": str(np.datetime64(T, "D")),
                            "full_only": sorted(map(str, a - b))[:5],
                            "prefix_only": sorted(map(str, b - a))[:5]})
        results.setdefault("prefix", {})[name] = {
            "n_windows_mismatch": n_mismatch, "detail": mismatch_detail}
        log(f"{name} 前缀对拍: 不一致窗口数={n_mismatch}")

        # ---- 逐日粒度抽查: 确定性种子抽 25 个非网格交易日, 全事件集(sig<=T)对拍
        w = int(cfg["regime"]["window"])
        th = float(cfg["regime"]["threshold"])
        rng = np.random.default_rng(42)
        non_grid = np.setdiff1d(all_dates[int(cfg["universe"]["min_day"]):], grid)
        spot = np.sort(rng.choice(non_grid, size=min(SPOT_CHECK_DATES, len(non_grid)),
                                  replace=False))
        spot_mismatch_stocks, spot_event_diff = 0, 0
        for T in spot:
            reg_t = _regime_prefix_at(sums, cnts, all_dates, T, w, th)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                parts = list(ex.map(_worker_prefix,
                                    [(c, cfg, [T], [reg_t], all_dates, [None])
                                     for c in chunks]))
            pref = {}
            for p in parts:
                for sym, per in p.items():
                    pref.setdefault(sym, set()).update(map(tuple, per[int(T)]))
            full_side = {}
            for sym, keys in full_by_symbol.items():
                s = {(sig, low, prev, rank, form)
                     for sig, low, prev, rank, form, sd in keys if sd <= T}
                if s:
                    full_side[sym] = s
            for sym in set(full_side) | set(pref):
                a, b = full_side.get(sym, set()), pref.get(sym, set())
                if a != b:
                    spot_mismatch_stocks += 1
                    spot_event_diff += len(a ^ b)
        results.setdefault("spot_check", {})[name] = {
            "n_dates": len(spot), "dates": [str(np.datetime64(int(t), "D")) for t in spot],
            "n_stocks_mismatch": spot_mismatch_stocks, "n_events_diff": spot_event_diff}
        log(f"{name} 逐日抽查: {len(spot)} 个非网格交易日, "
            f"不一致股票数={spot_mismatch_stocks}, 事件级差异={spot_event_diff}")

        # ---- 参数敏感性(信号日口径 + 低点日口径双重重合率; 后者对 order 类
        # 扰动才有信息量——order 变化把信号日机械平移, 信号日口径结构性恒为零)
        log(f"===== {name} 参数敏感性 =====")
        base_sig = [(sym, sd) for sym, keys in full_by_symbol.items()
                    for *_, sd in keys]
        base_low = [(sym, int(stocks_dates[sym][low]))
                    for sym, keys in full_by_symbol.items()
                    for _, low, *_ in keys]
        sens = {}
        for pert_name, over in PERTURBATIONS[name]:
            pcfg = lab._deep_merge(cfg, over)
            w, th = int(pcfg["regime"]["window"]), float(pcfg["regime"]["threshold"])
            _, reg_p = _regime_from_sums(sums, cnts, w, th)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                parts = list(ex.map(_worker_full,
                                    [(c, pcfg, reg_p, all_dates) for c in chunks]))
            pert_by_symbol = {}
            for p in parts:
                pert_by_symbol.update(p)
            pert_sig = [(sym, sd) for sym, keys in pert_by_symbol.items()
                        for *_, sd in keys]
            pert_low = [(sym, int(stocks_dates[sym][low]))
                        for sym, keys in pert_by_symbol.items()
                        for _, low, *_ in keys]
            sens[pert_name] = {"sig_date": overlap_metrics(base_sig, pert_sig),
                               "low_date": overlap_metrics(base_low, pert_low)}
            log(f"  {pert_name}: 信号日口径 {sens[pert_name]['sig_date']['recall_base']:.3f} "
                f"低点口径 {sens[pert_name]['low_date']['recall_base']:.3f}")
        results.setdefault("sensitivity", {})[name] = sens

    results["meta"] = {"n_files": len(files), "n_stocks": len(stocks),
                       "grid_step": args.grid_step, "smoke": args.smoke,
                       "elapsed_s": round(time.time() - t0, 1),
                       "finished_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    log(f"结果已写入 {OUT_DIR/'results.json'}, 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
