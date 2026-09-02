#!/usr/bin/env python3
"""背离实验室: 配置驱动的背离信号事件研究基建(库 + CLI), 供多路并行参数扫描共用。

评估协议(与 v3_pipeline/scripts/divergence_event_study.py 一致):
  - 双对照: C1=同股随机非事件日(等量, seed+crc32(symbol)), C2=同日随机非事件股(每事件1只, 固定seed);
  - 指标: 胜率、均值、均超额(vs C1/C2)、Welch t / Mann-Whitney p 值;
  - 探索期(事件日 <= explore_end, 默认 2020-12-31)与验证期分开报告; 分年度与分市场阶段稳定性切分;
  - 收益口径沿用 pct_chg 链式累乘(规避除权跳变); 入场 close_T=T收盘 / open_T1=T+1开盘。

因果性:
  - fractal: 低点 i 需右窗口走完才确认 => 信号日 = i+order, 严格无未来函数;
  - zigzag: 流式状态机, 反向运动达 pct% 当日确认前低 => 信号日=确认日;
  - legacy: 复刻 V1 生产截断语义(直接复用 divergence_event_study.simulate_events_idx);
  - tests/test_divergence_lab.py 用暴力逐日截断法对 3 只股全量对拍 fractal/zigzag, 0 mismatch。

配置 schema (JSON/YAML, 缺省项见 DEFAULT_CONFIG):
  name: 配置名(输出目录名)
  seed: 42
  explore_end: "2020-12-31"
  lows:
    method: fractal | zigzag | legacy
    price: close | low            # 取低点用的价格序列(默认 close)
    order: 10                     # fractal 左右窗口 k; 信号日=低点+k
    pct: 0.05                     # zigzag 摆动阈值(小数)
    min_sep: 5                    # 相邻低点最小间隔(根), 作用于 fractal/zigzag 的低点序列
  divergence:
    indicator: dif | hist         # 比较量: MACD DIF 线 / MACD histogram(talib 口径)
    min_change: 0.001             # 指标高出前低的最小幅度
    below_zero: false             # 两低点 DIF 均 < 0
    min_decline: 0.0              # 次低相对前低的最小价格跌幅 {0,0.03,0.05,0.08}
    lookback: 2                   # 回看低点数(取首个满足的排名)
    volume_confirm: false         # 次低缩量(次低当日成交量 < 前低)
    volume_ratio: null            # 缩量强度档: 非空时要求 次低量 < 前低量*ratio (优先于 volume_confirm)
    multi: 1                      # 1=普通双低点背离; 3=连续 3 低点多重背离
  entry: close_T | open_T1
  labels:
    fixed: [5,10,30]              # 固定 horizon 收益 ret_h{h}
    dynamic: null | {c: 1.0, cap: 60}   # h = c*formation_period(封顶 cap), 收益列 dyn
    sniper: null | {Ns: [10,20,40], ks: [1.5,2,3]}  # T+1开盘入场, N日内触及+k*ATR(14) => hit_N{N}_k{k}
    mfe: null | [10,30]           # 入场次日起 w 日内最大有利变动 mfe_h{w}
  filters:
    regime: all | up | down | sideways   # 市场阶段(全样本等权代理, 120日滚动收益±10%)
    above_ma200: null | true | false     # 信号日收盘 vs MA200
  universe: {sample: 0, min_day: 60}     # sample>0 只取前 N 个文件(冒烟用); min_day=信号日最小索引
  regime: {window: 120, threshold: 0.10}
  output: {root: v3_pipeline/reports/divergence_lab}

CLI:
  python divergence_lab.py --config cfg.json [--sample N] [--workers W] [--out-root DIR]
输出: {out_root}/{name}/events.parquet, labels.parquet, stats.json;
      汇总行追加到 {out_root}/summary.csv (fcntl 加锁, 可并行扫描后合并)。
"""
import argparse
import csv
import fcntl
import glob
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats as sc_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import divergence_event_study as des  # noqa: E402  (复用 V1 因果模拟与统计函数)

REPO = SCRIPT_DIR.parents[1]
DATA_DIR = REPO / "stock_data" / "daily"

DEFAULT_CONFIG = {
    "name": "unnamed",
    "seed": 42,
    "explore_end": "2020-12-31",
    "lows": {"method": "fractal", "price": "close", "order": 10, "pct": 0.05, "min_sep": 5},
    "divergence": {"indicator": "dif", "min_change": 0.001, "below_zero": False,
                   "min_decline": 0.0, "lookback": 2, "volume_confirm": False,
                   "volume_ratio": None, "multi": 1},
    "entry": "close_T",
    "labels": {"fixed": [5, 10, 30], "dynamic": None, "sniper": None, "mfe": None},
    "filters": {"regime": "all", "above_ma200": None},
    "intersect_with": None,  # 可选: 另一 run 的 events.parquet 路径, 事件取同股同日交集
    "universe": {"sample": 0, "min_day": 60},
    "regime": {"window": 120, "threshold": 0.10},
    "output": {"root": str(REPO / "v3_pipeline" / "reports" / "divergence_lab")},
}
REGIME_NAMES = {-1: "unknown", 0: "sideways", 1: "up", 2: "down"}
LEGACY_WARMUP = des.MIN_LEN - 1  # 99


# ================================================================ 配置
def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        user = yaml.safe_load(text)
    else:
        user = json.loads(text)
    cfg = _deep_merge(DEFAULT_CONFIG, user or {})
    if not cfg["name"] or cfg["name"] == "unnamed":
        cfg["name"] = p.stem
    return cfg


# ================================================================ 低点识别
def fractal_lows(p, order):
    """因果分形低点: i 为 p[i-k..i+k] 严格最小值, 信号日 = i+k (右窗口走完日)。
    返回 (low_idx, sig_idx) int64 数组。"""
    n = len(p)
    w = 2 * order + 1
    if n < w:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    sw = sliding_window_view(p, w)                      # sw[j] = p[j .. j+2k]
    center = sw[:, order]
    strict = (sw > center[:, None]).sum(axis=1) == (w - 1)  # 其余 2k 根均严格更高
    low_idx = np.nonzero(strict)[0] + order
    sig_idx = low_idx + order                           # <= n-1 恒成立
    return low_idx.astype(np.int64), sig_idx.astype(np.int64)


@njit(cache=True)
def zigzag_lows(p, pct):
    """因果 ZigZag 低点: 流式状态机。下跌段跟踪运行最低点, 价格自最低点反弹 >= pct
    当日确认该低点(信号日=确认日)。返回 (low_idx, sig_idx)。"""
    n = len(p)
    low_idx = np.empty(n, np.int64)
    sig_idx = np.empty(n, np.int64)
    cnt = 0
    hi, lo = p[0], p[0]
    hi_i, lo_i = 0, 0
    direction = 0  # 0=未定, 1=上涨段(找高), -1=下跌段(找低)
    for i in range(1, n):
        x = p[i]
        if direction <= 0 and x < lo:
            lo, lo_i = x, i
        if direction >= 0 and x > hi:
            hi, hi_i = x, i
        if direction <= 0 and lo_i < i and x >= lo * (1.0 + pct):
            low_idx[cnt], sig_idx[cnt] = lo_i, i
            cnt += 1
            direction = 1
            hi, hi_i = x, i
        elif direction >= 0 and hi_i < i and x <= hi * (1.0 - pct):
            direction = -1
            lo, lo_i = x, i
    return low_idx[:cnt], sig_idx[:cnt]


def apply_min_sep(low_idx, sig_idx, min_sep):
    """低点序列最小间隔过滤: 按时间顺序, 距上一保留低点 < min_sep 根的低点丢弃。"""
    if min_sep is None or min_sep <= 1 or len(low_idx) == 0:
        return low_idx, sig_idx
    keep = np.zeros(len(low_idx), bool)
    last = -10 ** 9
    for i in range(len(low_idx)):
        if low_idx[i] - last >= min_sep:
            keep[i] = True
            last = low_idx[i]
    return low_idx[keep], sig_idx[keep]


# ================================================================ 背离事件
def _pair_ok(j, p, low_idx, close, ind, dif, vol, dcfg):
    """低点 j 相对前低 p 的基础背离条件(价格新低 + 指标抬高 + 可选过滤)。"""
    ic, ip = low_idx[j], low_idx[p]
    mc, mp = ind[ic], ind[ip]
    if not (np.isfinite(mc) and np.isfinite(mp)):
        return False
    if not (close[ic] < close[ip] * (1.0 - dcfg["min_decline"])):
        return False
    if not (mc > mp + dcfg["min_change"]):
        return False
    if dcfg["below_zero"] and not (dif[ic] < 0 and dif[ip] < 0):
        return False
    vr = dcfg.get("volume_ratio")
    if vr is not None:
        vc, vp = vol[ic], vol[ip]
        if not (np.isfinite(vc) and np.isfinite(vp)) or not (vc < vp * float(vr)):
            return False
    elif dcfg["volume_confirm"]:
        vc, vp = vol[ic], vol[ip]
        if not (np.isfinite(vc) and np.isfinite(vp)) or not (vc < vp):
            return False
    return True


def detect_divergence_events(low_idx, sig_idx, close, ind, dif, vol, dcfg, warmup):
    """对低点序列做背离判定。返回 dict of arrays: sig/low/prev/rank/form。
    同一低点首次满足的 compare_rank(1..lookback)记一个事件; multi=3 时要求
    连续 3 低点两两背离(rank 恒为 1)。"""
    m = len(low_idx)
    lookback = int(dcfg["lookback"])
    multi = int(dcfg["multi"])
    ev_sig, ev_low, ev_prev, ev_rank, ev_form = [], [], [], [], []
    if multi == 3:
        for j in range(2, m):
            if sig_idx[j] < warmup:
                continue
            if _pair_ok(j, j - 1, low_idx, close, ind, dif, vol, dcfg) and \
               _pair_ok(j - 1, j - 2, low_idx, close, ind, dif, vol, dcfg):
                ev_sig.append(sig_idx[j]); ev_low.append(low_idx[j])
                ev_prev.append(low_idx[j - 1]); ev_rank.append(1)
                ev_form.append(low_idx[j] - low_idx[j - 1])
    else:
        for j in range(1, m):
            if sig_idx[j] < warmup:
                continue
            for r in range(1, min(lookback, j) + 1):
                if _pair_ok(j, j - r, low_idx, close, ind, dif, vol, dcfg):
                    ev_sig.append(sig_idx[j]); ev_low.append(low_idx[j])
                    ev_prev.append(low_idx[j - r]); ev_rank.append(r)
                    ev_form.append(low_idx[j] - low_idx[j - r])
                    break
    return {
        "sig": np.asarray(ev_sig, np.int32), "low": np.asarray(ev_low, np.int32),
        "prev": np.asarray(ev_prev, np.int32), "rank": np.asarray(ev_rank, np.int8),
        "form": np.asarray(ev_form, np.int32),
    }


# ================================================================ 数据加载(工作进程)
def _to_days(s):
    arr = s.to_numpy()
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]").astype(np.int32)
    return pd.to_datetime(pd.Series(arr).astype(str)).to_numpy("datetime64[D]").astype(np.int32)


def load_stock(args):
    """读取单股, 计算指标与事件。返回紧凑字典(f32/i32) 或 {"symbol","error"}。"""
    path, cfg = args
    symbol = Path(path).stem
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    if len(df) < 30:
        return {"symbol": symbol, "error": "too_short"}
    close = df["close"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    if "pct_chg" in df.columns:
        pct = df["pct_chg"].to_numpy(np.float64)
    else:
        pct = np.concatenate([[np.nan], close[1:] / close[:-1] * 100.0 - 100.0])
    vol_col = "vol" if "vol" in df.columns else ("volume" if "volume" in df.columns else None)
    vol = df[vol_col].to_numpy(np.float64) if vol_col else np.full(len(df), np.nan)
    dates = _to_days(df["trade_date"])

    dif, _, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    atr = talib.ATR(high, low, close, timeperiod=14)
    ma200 = talib.SMA(close, timeperiod=200)
    pct_safe = np.where(np.isfinite(pct), pct, 0.0)
    cf = np.cumprod(1.0 + pct_safe / 100.0)

    lcfg, dcfg = cfg["lows"], cfg["divergence"]
    method = lcfg["method"]
    if method == "legacy":
        warmup = LEGACY_WARMUP
        evs = des.simulate_events_idx(close, dif)  # [(t, rank, prev), ...]
        events = {
            "sig": np.asarray([e[0] for e in evs], np.int32),
            "low": np.asarray([e[0] for e in evs], np.int32),
            "prev": np.asarray([e[2] for e in evs], np.int32),
            "rank": np.asarray([e[1] for e in evs], np.int8),
            "form": np.asarray([e[0] - e[2] for e in evs], np.int32),
        }
    else:
        warmup = int(cfg["universe"]["min_day"])
        p = close if lcfg["price"] == "close" else low
        if method == "fractal":
            low_idx, sig_idx = fractal_lows(p, int(lcfg["order"]))
        elif method == "zigzag":
            low_idx, sig_idx = zigzag_lows(p, float(lcfg["pct"]))
        else:
            return {"symbol": symbol, "error": f"unknown method {method}"}
        low_idx, sig_idx = apply_min_sep(low_idx, sig_idx, int(lcfg["min_sep"]))
        ind = dif if dcfg["indicator"] == "dif" else hist
        events = detect_divergence_events(low_idx, sig_idx, close, ind, dif, vol, dcfg, warmup)

    return {
        "symbol": symbol, "dates": dates,
        "open": open_.astype(np.float32), "high": high.astype(np.float32),
        "low": low.astype(np.float32), "close": close.astype(np.float32),
        "vol": vol.astype(np.float32), "cf": cf.astype(np.float32),
        "dif": dif.astype(np.float32), "atr": atr.astype(np.float32),
        "ma200": ma200.astype(np.float32), "events": events,
    }


# ================================================================ 标签
def label_columns(cfg):
    lcfg = cfg["labels"]
    cols = [f"ret_h{h}" for h in (lcfg["fixed"] or [])]
    if lcfg["dynamic"]:
        cols.append("dyn")
    if lcfg["sniper"]:
        for N in lcfg["sniper"]["Ns"]:
            for k in lcfg["sniper"]["ks"]:
                cols.append(f"hit_N{N}_k{k}")
    for w in (lcfg["mfe"] or []):
        cols.append(f"mfe_h{w}")
    return cols


def compute_labels(st, sig, form, cfg):
    """对信号日数组 sig(升序不要求)计算全部标签列。form 为各信号对应的 formation_period
    (dynamic 标签用; 无 dynamic 时可为 None)。返回 {col: f32 array}。"""
    n = len(st["close"])
    sig = np.asarray(sig, np.int64)
    m = len(sig)
    lcfg = cfg["labels"]
    entry_open = cfg["entry"] == "open_T1"
    close, open_, high, cf, atr = st["close"], st["open"], st["high"], st["cf"], st["atr"]
    out = {}
    if m == 0:
        return {c: np.empty(0, np.float32) for c in label_columns(cfg)}
    t1 = np.minimum(sig + 1, n - 1)

    def ret_for(h):  # h: int 或等长 int 数组
        r = np.full(m, np.nan, np.float32)
        if entry_open:
            ok = (sig + 1 + h <= n - 1) & (open_[t1] > 0)
            i = np.nonzero(ok)[0]
            hj = h if np.isscalar(h) else np.asarray(h)[i]
            r[i] = close[sig[i] + 1] / open_[sig[i] + 1] * (
                cf[sig[i] + 1 + hj] / cf[sig[i] + 1]) - 1.0
        else:
            ok = sig + h <= n - 1
            i = np.nonzero(ok)[0]
            hj = h if np.isscalar(h) else np.asarray(h)[i]
            r[i] = cf[sig[i] + hj] / cf[sig[i]] - 1.0
        return r

    for h in (lcfg["fixed"] or []):
        out[f"ret_h{h}"] = ret_for(int(h))
    dyn = lcfg["dynamic"]
    if dyn:
        if form is None:
            raise ValueError("dynamic 标签需要 formation_period")
        hh = np.clip(np.rint(float(dyn["c"]) * np.asarray(form, np.float64)), 1,
                     int(dyn.get("cap", 60))).astype(np.int64)
        out["dyn"] = ret_for(hh)
    sn = lcfg["sniper"]
    if sn:  # 口径固定: T+1 开盘入场, 窗口 = high[T+1 .. T+N], ATR(14) 取信号日 T
        for N in sn["Ns"]:
            N = int(N)
            if n >= N:
                H = sliding_window_view(high, N)  # H[i] = high[i..i+N-1]
            for k in sn["ks"]:
                col = np.full(m, np.nan, np.float32)
                if n >= N:
                    ok = (sig + N <= n - 1) & (open_[t1] > 0) & np.isfinite(atr[sig])
                    i = np.nonzero(ok)[0]
                    if len(i):
                        wmax = H[sig[i] + 1].max(axis=1)
                        target = open_[sig[i] + 1] + float(k) * atr[sig[i]]
                        col[i] = (wmax >= target).astype(np.float32)
                out[f"hit_N{N}_k{k}"] = col
    for w in (lcfg["mfe"] or []):  # 窗口 = 入场次日起 w 根 high
        w = int(w)
        col = np.full(m, np.nan, np.float32)
        if n >= w:
            H = sliding_window_view(high, w)
            e = open_[t1] if entry_open else close[sig]
            ok = (sig + w <= n - 1) & (e > 0)
            i = np.nonzero(ok)[0]
            if len(i):
                col[i] = H[sig[i] + 1].max(axis=1) / e[i] - 1.0
        out[f"mfe_h{w}"] = col
    return out


# ================================================================ 市场阶段
def build_market_regime(stocks, window, th):
    """全样本等权日收益代理; 滚动 window 日收益 >+th 为 up(1), <-th 为 down(2),
    其余 sideways(0), 前 window 日 unknown(-1)。返回 (all_dates i32, regime i8)。"""
    all_dates = np.unique(np.concatenate([st["dates"] for st in stocks]))
    sums = np.zeros(len(all_dates), np.float64)
    cnts = np.zeros(len(all_dates), np.float64)
    for st in stocks:
        pos = np.searchsorted(all_dates, st["dates"])
        c = st["cf"].astype(np.float64)
        r = c[1:] / c[:-1] - 1.0
        np.add.at(sums, pos[1:], r)
        np.add.at(cnts, pos[1:], 1.0)
    mean = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
    idx = np.cumprod(1.0 + mean)
    regime = np.full(len(all_dates), -1, np.int8)
    if len(all_dates) > window:
        roll = idx[window:] / idx[:-window] - 1.0
        reg = np.where(roll > th, 1, np.where(roll < -th, 2, 0)).astype(np.int8)
        regime[window:] = reg
    return all_dates, regime


# ================================================================ 统计
def group_stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, win=np.nan, mean=np.nan, median=np.nan)
    return dict(n=n, win=float(np.mean(x > 0)), mean=float(np.mean(x)),
                median=float(np.median(x)))


def compare(d, c):
    d = np.asarray(d, np.float64); c = np.asarray(c, np.float64)
    d = d[np.isfinite(d)]; c = c[np.isfinite(c)]
    out = {}
    if len(d) >= 10 and len(c) >= 10:
        out["p_t"] = float(sc_stats.ttest_ind(d, c, equal_var=False).pvalue)
        out["p_mw"] = float(sc_stats.mannwhitneyu(d, c, alternative="two-sided").pvalue)
    else:
        out["p_t"] = out["p_mw"] = np.nan
    return out


def _block(d, c1, c2):
    gs_d, gs1, gs2 = group_stats(d), group_stats(c1), group_stats(c2)
    out = dict(div=gs_d, c1=gs1, c2=gs2)
    for tag, g, c in (("c1", gs1, c1), ("c2", gs2, c2)):
        out[f"ex_mean_{tag}"] = (gs_d["mean"] - g["mean"]) if (
            np.isfinite(gs_d["mean"]) and np.isfinite(g["mean"])) else np.nan
        out[f"ex_win_{tag}"] = (gs_d["win"] - g["win"]) if (
            np.isfinite(gs_d["win"]) and np.isfinite(g["win"])) else np.nan
        out.update({f"{k}_{tag}": v for k, v in compare(d, c).items()})
    return out


def stats_by_period(lab_ev, lab_c1, lab_c2, dates_ev, dates_c1, dates_c2, cols, explore_end):
    """主统计: 每个标签列 × {all, explore, validate}。"""
    def masks(dates):
        e = dates <= explore_end
        return {"all": np.ones(len(dates), bool), "explore": e, "validate": ~e}
    me, m1, m2 = masks(dates_ev), masks(dates_c1), masks(dates_c2)
    out = {}
    for c in cols:
        out[c] = {p: _block(lab_ev[c][me[p]], lab_c1[c][m1[p]], lab_c2[c][m2[p]])
                  for p in ("all", "explore", "validate")}
    return out


def stats_by_split(lab_ev, lab_c1, lab_c2, key_ev, key_c1, key_c2, cols, min_n=100):
    """稳定性切分: key 为每个样本的分组键(年份或 regime 码)。"""
    out = {}
    keys = sorted(set(np.unique(key_ev).tolist()))
    for c in cols:
        per = {}
        for k in keys:
            if k < 0:  # regime unknown
                continue
            d = lab_ev[c][key_ev == k]
            c1 = lab_c1[c][key_c1 == k]
            c2 = lab_c2[c][key_c2 == k]
            if np.isfinite(d).sum() < min_n:
                continue
            b = _block(d, c1, c2)
            per[str(k)] = {kk: b[kk] for kk in
                           ("ex_mean_c1", "ex_win_c1", "p_mw_c1", "ex_mean_c2", "ex_win_c2")}
            per[str(k)]["n"] = b["div"]["n"]
            per[str(k)]["win"] = b["div"]["win"]
            per[str(k)]["c1_win"] = b["c1"]["win"]
            per[str(k)]["c2_win"] = b["c2"]["win"]
        out[c] = per
    return out


# ================================================================ 汇总 CSV
SUMMARY_LABEL_KEYS = [("r5", "ret_h5"), ("r10", "ret_h10"), ("r30", "ret_h30"),
                      ("dyn", "dyn"), ("hit", None), ("mfe", None)]  # None => 取该前缀首列


def _canon_cols(cols):
    m = {}
    for key, col in SUMMARY_LABEL_KEYS:
        if col and col in cols:
            m[key] = col
        elif col is None:
            pref = "hit_" if key == "hit" else "mfe_"
            cands = [c for c in cols if c.startswith(pref)]
            if cands:
                m[key] = cands[0]
    return m


def summary_row(cfg, meta, st):
    row = {
        "name": cfg["name"], "method": cfg["lows"]["method"],
        "price": cfg["lows"]["price"], "order": cfg["lows"]["order"],
        "pct": cfg["lows"]["pct"], "min_sep": cfg["lows"]["min_sep"],
        "indicator": cfg["divergence"]["indicator"],
        "min_change": cfg["divergence"]["min_change"],
        "below_zero": cfg["divergence"]["below_zero"],
        "min_decline": cfg["divergence"]["min_decline"],
        "lookback": cfg["divergence"]["lookback"],
        "volume_confirm": cfg["divergence"]["volume_confirm"],
        "multi": cfg["divergence"]["multi"], "entry": cfg["entry"],
        "regime_filter": cfg["filters"]["regime"],
        "ma200_filter": cfg["filters"]["above_ma200"],
        "sample": meta["sample"], "n_events": meta["n_events"],
        "n_event_stocks": meta["n_event_stocks"],
        "freq_per_stock_year": round(meta["freq_per_stock_year"], 4),
        "elapsed_s": round(meta["elapsed_s"], 1),
    }
    canon = _canon_cols(list(st.keys()))
    for key, _ in SUMMARY_LABEL_KEYS:
        for p in ("e", "v"):
            for m_ in ("win", "exc1", "pmw1"):
                row[f"{key}_{p}_{m_}"] = ""
        if key in canon:
            for p, pname in (("e", "explore"), ("v", "validate")):
                b = st[canon[key]][pname]
                row[f"{key}_{p}_win"] = _r4(b["div"]["win"])
                row[f"{key}_{p}_exc1"] = _r4(b["ex_mean_c1"])
                row[f"{key}_{p}_pmw1"] = _r4(b["p_mw_c1"])
    return row


def _r4(x):
    return "" if x is None or not np.isfinite(x) else round(float(x), 4)


def append_summary(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        empty = f.tell() == 0
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if empty:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


# ================================================================ 主流程
def run(cfg, sample=0, workers=8, verbose=True):
    t0 = time.time()
    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    n_sample = int(sample or cfg["universe"]["sample"] or 0)
    if n_sample:
        files = files[:n_sample]
    log = print if verbose else (lambda *a, **k: None)
    log(f"[{cfg['name']}] 加载并检测 {len(files)} 只股票 ...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(load_stock, [(f, cfg) for f in files], chunksize=16)):
            results.append(r)
            if verbose and (i + 1) % 1000 == 0:
                log(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    stocks = [r for r in results if "error" not in r]
    log(f"有效股票 {len(stocks)}, 剔除 {len(results)-len(stocks)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 市场阶段
    all_dates, regime_arr = build_market_regime(stocks, int(cfg["regime"]["window"]),
                                                float(cfg["regime"]["threshold"]))

    # ---- 事件表 + 标签
    explore_end = np.datetime64(cfg["explore_end"]).astype("datetime64[D]").astype(np.int32)
    cols = label_columns(cfg)
    ev_stock, ev_sig, ev_low, ev_prev, ev_rank, ev_form, ev_reg, ev_am = \
        [], [], [], [], [], [], [], []
    lab_ev = {c: [] for c in cols}
    lab_c1 = {c: [] for c in cols}
    c1_stock, c1_day, c1_reg = [], [], []
    seed = int(cfg["seed"])
    warmup = LEGACY_WARMUP if cfg["lows"]["method"] == "legacy" else int(cfg["universe"]["min_day"])
    regime_filter = cfg["filters"]["regime"]
    ma_filter = cfg["filters"]["above_ma200"]
    # 可选: 与另一 run 的事件取交集(同股共振), 用于多方法交集层
    # intersect_on: "sig"=信号日同日(默认) | "low"=同一低点(跨方法确认延迟不同, 同日近似空集)
    intersect_map = None
    iw = cfg.get("intersect_with")
    iw_on = cfg.get("intersect_on", "sig")
    iw_causal = bool(cfg.get("intersect_causal", True))
    if iw:
        idf = pd.read_parquet(iw, columns=["ts_code", "low_date", "date"])
        def _ref_days(s):  # events.parquet: date 为 datetime64, low_date 为 int32 日序
            arr = s.to_numpy()
            return arr.astype(np.int32) if np.issubdtype(arr.dtype, np.integer) else _to_days(s)
        if iw_on == "low":
            # 同股同低点共振; intersect_causal=true 时要求参考信号日 <= 本方信号日
            intersect_map = {}
            for sym, g in idf.groupby("ts_code"):
                ld = _ref_days(g["low_date"]); sd = _ref_days(g["date"])
                d = {}
                for l, s in zip(ld.tolist(), sd.tolist()):
                    d[l] = min(d.get(l, 1 << 30), s) if iw_causal else 0
                intersect_map[sym] = d
        else:
            intersect_map = {sym: _ref_days(g["date"]) for sym, g in idf.groupby("ts_code")}

    for si, st in enumerate(stocks):
        n = len(st["close"])
        ev = st["events"]
        reg_st = regime_arr[np.searchsorted(all_dates, st["dates"])]
        if len(ev["sig"]):
            lab = compute_labels(st, ev["sig"], ev["form"], cfg)
            keep = np.ones(len(ev["sig"]), bool)
            if regime_filter != "all":
                want = {"up": 1, "down": 2, "sideways": 0}[regime_filter]
                keep &= reg_st[ev["sig"]] == want
            if ma_filter is not None:
                above = st["close"][ev["sig"]] > st["ma200"][ev["sig"]]
                keep &= above if ma_filter else ~above
            if intersect_map is not None:
                ref = intersect_map.get(st["symbol"])
                if iw_on == "low":
                    ld = st["dates"][ev["low"]].tolist(); sd = st["dates"][ev["sig"]]
                    ik = np.array([ref is not None and ref.get(l, 1 << 30) <= s
                                   for l, s in zip(ld, sd.tolist())], bool)
                else:
                    ik = np.isin(st["dates"][ev["sig"]], ref) if ref is not None \
                        else np.zeros(len(ev["sig"]), bool)
                keep &= ik
            for c in cols:
                lab_ev[c].append(lab[c][keep])
            ev_stock.extend([si] * int(keep.sum()))
            for arr, lst in ((ev["sig"], ev_sig), (ev["low"], ev_low), (ev["prev"], ev_prev),
                             (ev["rank"], ev_rank), (ev["form"], ev_form)):
                lst.extend(arr[keep].tolist())
            ev_reg.extend(reg_st[ev["sig"][keep]].tolist())
            ev_am.extend((st["close"][ev["sig"][keep]] > st["ma200"][ev["sig"][keep]]).tolist())
            # C1: 同股随机非事件日(等量, 确定性种子), 标签的 form 与事件逐对配对
            mask = np.ones(n, bool)
            mask[:warmup] = False
            mask[ev["sig"]] = False
            days = np.nonzero(mask)[0]
            if len(days):
                rng = np.random.default_rng(seed + zlib.crc32(st["symbol"].encode()))
                pick = rng.choice(days, size=len(ev["sig"]), replace=len(days) < len(ev["sig"]))
                lab1 = compute_labels(st, pick, ev["form"], cfg)
                for c in cols:
                    lab_c1[c].append(lab1[c])
                c1_stock.extend([si] * len(pick))
                c1_day.extend(pick.tolist())
                c1_reg.extend(reg_st[pick].tolist())
    n_events = len(ev_sig)
    for c in cols:
        lab_ev[c] = np.concatenate(lab_ev[c]) if lab_ev[c] else np.empty(0, np.float32)
        lab_c1[c] = np.concatenate(lab_c1[c]) if lab_c1[c] else np.empty(0, np.float32)
    ev_stock = np.asarray(ev_stock, np.int32)
    ev_sig = np.asarray(ev_sig, np.int32)
    ev_form = np.asarray(ev_form, np.int32)
    ev_dates = np.array([stocks[s]["dates"][t] for s, t in zip(ev_stock, ev_sig)], np.int32) \
        if n_events else np.empty(0, np.int32)
    c1_day = np.asarray(c1_day, np.int32)
    c1_dates = np.array([stocks[s]["dates"][t] for s, t in zip(c1_stock, c1_day)], np.int32) \
        if c1_day.size else np.empty(0, np.int32)
    log(f"背离事件数: {n_events} ({time.time()-t0:.0f}s)", flush=True)

    # ---- C2: 同日随机非事件股(逐事件 1 只, 与事件行对齐)
    g_parts, s_parts, r_parts = [], [], []
    for si, st in enumerate(stocks):
        g = np.searchsorted(all_dates, st["dates"])
        g_parts.append(g.astype(np.int32))
        s_parts.append(np.full(len(g), si, np.int32))
        r_parts.append(np.arange(len(g), dtype=np.int32))
    G = np.concatenate(g_parts); S = np.concatenate(s_parts); R = np.concatenate(r_parts)
    order = np.argsort(G, kind="stable")
    G, S, R = G[order], S[order], R[order]
    bounds = np.searchsorted(G, np.arange(len(all_dates) + 1))
    ev_g = np.searchsorted(all_dates, ev_dates) if n_events else np.empty(0, np.int32)
    banned = {}
    for gv, si in zip(ev_g.tolist(), ev_stock.tolist()):
        banned.setdefault(gv, set()).add(si)
    lab_c2 = {c: np.full(n_events, np.nan, np.float32) for c in cols}
    rng2 = np.random.default_rng(seed)
    picks = []  # (event_i, stock, day)
    no_pool = 0
    for i in range(n_events):
        gv = int(ev_g[i])
        lo_b, hi_b = bounds[gv], bounds[gv + 1]
        ban = banned.get(gv)
        if ban:
            keep = np.isin(S[lo_b:hi_b], list(ban), invert=True)
            ps, pr = S[lo_b:hi_b][keep], R[lo_b:hi_b][keep]
        else:
            ps, pr = S[lo_b:hi_b], R[lo_b:hi_b]
        if len(ps) == 0:
            no_pool += 1
            continue
        j = int(rng2.integers(len(ps)))
        picks.append((i, int(ps[j]), int(pr[j])))
    # 按股票分组批量计算标签
    if picks:
        pidx = np.argsort([p[1] for p in picks], kind="stable")
        picks = [picks[j] for j in pidx]
        blk = 0
        while blk < len(picks):
            si = picks[blk][1]
            end = blk
            while end < len(picks) and picks[end][1] == si:
                end += 1
            rows = picks[blk:end]
            days = np.array([r[2] for r in rows], np.int64)
            forms = ev_form[[r[0] for r in rows]]
            lab2 = compute_labels(stocks[si], days, forms, cfg)
            for r_i, r in enumerate(rows):
                for c in cols:
                    lab_c2[c][r[0]] = lab2[c][r_i]
            blk = end
    log(f"C2 完成 (no_pool={no_pool}) ({time.time()-t0:.0f}s)", flush=True)

    # ---- 统计
    st_main = stats_by_period(lab_ev, lab_c1, lab_c2, ev_dates, c1_dates, ev_dates,
                              cols, explore_end)
    year_ev = (ev_dates.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970) \
        if n_events else np.empty(0, int)
    year_c1 = (c1_dates.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970) \
        if c1_dates.size else np.empty(0, int)
    yearly = stats_by_split(lab_ev, lab_c1, lab_c2, year_ev, year_c1, year_ev, cols)
    reg_stats = stats_by_split(lab_ev, lab_c1, lab_c2,
                               np.asarray(ev_reg, np.int8) if n_events else np.empty(0, np.int8),
                               np.asarray(c1_reg, np.int8) if c1_day.size else np.empty(0, np.int8),
                               np.asarray(ev_reg, np.int8) if n_events else np.empty(0, np.int8),
                               cols)
    reg_stats = {c: {REGIME_NAMES[int(k)]: v for k, v in per.items()}
                 for c, per in reg_stats.items()}

    stock_years = sum((st["dates"][-1] - st["dates"][0]) / 365.25 for st in stocks)
    meta = dict(
        name=cfg["name"], timestamp=pd.Timestamp.now().isoformat(timespec="seconds"),
        config=cfg, sample=n_sample, seed=seed,
        n_files=len(files), n_stocks=len(stocks), n_skipped=len(results) - len(stocks),
        n_events=n_events, n_event_stocks=len(set(ev_stock.tolist())),
        freq_per_stock_year=float(n_events / stock_years) if stock_years > 0 else np.nan,
        c2_no_pool=no_pool, elapsed_s=time.time() - t0,
    )

    # ---- 落盘
    out_dir = Path(cfg["output"]["root"]) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(
        event_id=np.arange(n_events, dtype=np.int64),
        ts_code=[stocks[s]["symbol"] for s in ev_stock],
        sig_idx=ev_sig,
        date=ev_dates.astype("datetime64[D]") if n_events else np.array([], dtype="datetime64[ns]"),
        low_date=[stocks[s]["dates"][l] for s, l in zip(ev_stock, ev_low)],
        prev_low_date=[stocks[s]["dates"][p] for s, p in zip(ev_stock, ev_prev)],
        compare_rank=np.asarray(ev_rank, np.int8),
        formation=ev_form,
        regime=[REGIME_NAMES[r] for r in ev_reg],
        above_ma200=np.asarray(ev_am, bool),
    )).to_parquet(out_dir / "events.parquet", index=False)
    wide = pd.concat([
        pd.DataFrame(dict(group="div", **{c: lab_ev[c] for c in cols})),
        pd.DataFrame(dict(group="c1", **{c: lab_c1[c] for c in cols})),
        pd.DataFrame(dict(group="c2", **{c: lab_c2[c] for c in cols})),
    ], ignore_index=True)
    wide.to_parquet(out_dir / "labels.parquet", index=False)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(dict(meta=meta, periods=st_main, yearly=yearly, regime=reg_stats),
                  f, ensure_ascii=False, indent=1, default=float)
    row = summary_row(cfg, meta, st_main)
    append_summary(Path(cfg["output"]["root"]) / "summary.csv", row)
    log(f"[{cfg['name']}] 完成 {time.time()-t0:.0f}s -> {out_dir}", flush=True)
    return dict(meta=meta, periods=st_main, out_dir=str(out_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out_root:
        cfg["output"]["root"] = args.out_root
    res = run(cfg, sample=args.sample, workers=args.workers)
    # 摘要打印(冒烟用)
    m = res["meta"]
    print(f"n_events={m['n_events']}  freq={m['freq_per_stock_year']:.3f}/股/年  "
          f"elapsed={m['elapsed_s']:.0f}s")
    for key, col in _canon_cols(list(res["periods"].keys())).items():
        for p in ("explore", "validate"):
            b = res["periods"][col][p]
            print(f"  {col:12s} {p:9s} n={b['div']['n']:7d} win={_fmt(b['div']['win'])} "
                  f"ex_c1={_fmt(b['ex_mean_c1'])} p_mw={_fmt(b['p_mw_c1'])}")


def _fmt(x):
    return "NA" if x is None or not np.isfinite(x) else f"{x:.4f}"


if __name__ == "__main__":
    main()
