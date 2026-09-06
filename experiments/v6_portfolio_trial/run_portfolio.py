#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背离v6 组合层 + 出场族审判 —— 预登记 = 同目录 README.md(冻结,先于跑数落盘,勿改)。

唯一执行脚本。输入 = 事件池 parquet + 日线 + 涨跌停 + 指数;
输出 = 每格 equity_curve/trades/stats + summary_portfolio.csv + verdict.json + report.md。

口径钉死(与 README 逐条对应;解释性决断全部集中在 docstring 末尾披露,不动规格):

A. 入场与 E0 逐字沿用 #31 §2.2/§2.3 的"个股序列行号"口径(事件日下一行开盘买,
   第 H 行收盘卖,跌停顺延,耗尽截断)。这是 §六.1 对账自检(浮点差 0)的可执行前提:
   trades_seed.parquet 就是该口径产物。窗口终点 2026-08-31 = 数据终点,
   "持仓延伸过窗口记截断"(§一.1)与 #31 耗尽条款天然重合。
B. E1/E2 执行语义逐字继承 strategy_engine_v3 类 A/B(盘中触及、开盘跳越、
   同日双触发取止损、收盘<=跌停顺延、市场日历计持有日、vol_mult 三档 ≤t-1)。
   市场日历 = 000001.SH 交易日(基准同日历;个股日期 ⊆ 指数日期,实测校验)。
C. E3~E7 统一执行语义(README §三):收盘判定、触发后次日开盘卖、
   执行日开盘跌停/无行情顺延、耗尽截断。H 到期 = 第 H 个市场日历日收盘卖
   (收盘<=跌停顺延),与触发同日竞争时 H 到期优先(保证"最晚 H 日清仓")。
D. 空仓双口径(§一.3):index_wallet 递推 wallet_t = wallet_{t-1}*r_t + (cash_t - cash_{t-1}),
   判活口径权益 = wallet + 持仓市值;纯现金口径 = cash + 持仓市值。
E. 成本原语只读复用 strategy_engine(buy_cost/sell_costs/SLIPPAGE/BOARD_LOT/PRICE_TOL)。
   net_pnl 按 trades_seed 结合序 sh*(xs-px) - 买佣 - 卖佣 - 印花税 计算(对账浮点差 0 要求)。

解释性决断(规格未钉死处,全部披露,不改规格):
 1. E7/P3/P4 的 ATR 用不复权价(talib.ATR(14),全历史序列):与信号扫描口径一致;
    README §3.1 E7 写"后复权"与信号扫描实际口径冲突,以不复权为准,建议监工修订 README。
 2. P2"当日仓位上限/当日每仓金额"的"当日"锚定信号日:入场日段归属取决于入场日
    是否有信号,在信号日收盘/入场日开盘均不可知,锚定入场日必泄漏。
 3. E7"持仓期最高收盘价"含当日收盘(ATR 仍严格 ≤t-1;含当日无害:当日创新高时不触发)。
 4. S5 量比 = 事件日 vol / 前 20 个个股序列行均 vol(不含事件日);NaN 排最后。
 5. E4 半仓向下整手为 0(持仓仅 100 股)时不发生部分卖出,直接整体转 E3,计数披露。
 6. "跌破"判定 = 收盘 < 线 - 1e-9(PRICE_TOL);"触及/回撤≥"判定含容差(+1e-9)。
 7. §六.6"指数缺失日按前一交易日收益顺延"采用价格前向填充(当日收益 0)解读;
    实测缺失日 = 0(个股日期 ⊆ 指数日期),条款空置,两种解读无差异。
 8. Sharpe 日收益标准差用样本 std(ddof=1);年化 = (末/初)^(252/(交易日数-1)) - 1。
 9. 资金利用率 = 日均在持市值 / 100 万(固定总资金口径)。
10. E2 全市场均值 ATR 宇宙 = stock_data/daily 全部文件(含指数/BJ,逐字继承 #28
    build_mkt_atr),切片 [窗口首日-90 自然日, 窗口末日]。
11. E4 部分卖出的入场佣金按卖出股数占比分摊;E3~E7 开盘跌停判定用开盘价
    (README 字面"开盘跌停"),与 v1 引擎收盘口径不同,并列披露。
12. 退化对账格(P1×E0×S1×H20,仓位上限=无穷)现金约束同步放开
    (每股预算仍 10 万,与 trades_seed 的 BUDGET 口径逐字一致)。
13. 同一股票可叠加多笔持仓(不同事件),与引擎 v1/v3 一致,不去重。
14. 判活口径为"闲置现金按指数记账"的叠加层,累计指数亏损可使其权益 <=0:
    此时年化取 -1(全亏地板)、Sharpe 记 NaN(收益符号无意义)、该格判活必否,
    负权益天数计数披露(neg_equity_*_days);逐年收益在上年末权益 <=0 时记 None。

用法:
    python3 run_portfolio.py --mode selfcheck   # 只跑六条自检
    python3 run_portfolio.py --mode benchmark   # 自检 + 20 抽样格计时(两遍,确定性)
    python3 run_portfolio.py --mode full --grid full|degraded  # 全量/降级网格(监工派活)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

REPO = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO, "v3_pipeline", "scripts"))
import strategy_engine as se  # noqa: E402  冻结引擎:只读复用成本原语与常量

OUT_DIR = Path(os.path.join(REPO, "experiments", "v6_portfolio_trial"))
CACHE_DIR = OUT_DIR / "cache"
RUNS_DIR = OUT_DIR / "runs"
LOG_PATH = OUT_DIR / "progress.log"
EVENTS_PATH = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history",
                                "events_history_v1.parquet"))
TRADES_SEED_PATH = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history",
                                     "trades_seed.parquet"))
DATA_DIR = Path(os.path.join(REPO, "stock_data", "daily"))
LIMIT_DIR = Path(os.path.join(REPO, "stock_data", "stk_limit"))
INDEX_PATH = DATA_DIR / "000001.SH.parquet"

EVENT_START = pd.Timestamp("2011-09-01")
EVENT_END = pd.Timestamp("2026-08-31")
D_START = int(EVENT_START.strftime("%Y%m%d"))
D_END = int(EVENT_END.strftime("%Y%m%d"))
STAMP_SWITCH_INT = 20230828
INIT = se.INIT_CAPITAL  # 100 万
TOL = se.PRICE_TOL
H_LIST = [10, 20, 25]
POOLS = ["v6-1", "v6-2", "v6-3", "v6-4", "v6-5", "ALL"]
SELS = ["S1", "S2", "S3", "S4", "S5"]
POS_RULES = ["P1", "P2", "P3", "P4"]
S4_SEED = 42

# ---------------------------------------------------------------- 出场配置网格(§3.1/§七.6)
E1_GRID = []  # A1~A12: tp 外层 × sl 内层(同 #28 grid_a 序)
for _tp in (0.20, 0.25, 0.30, 0.35):
    for _sl in (-0.10, -0.14, -0.18):
        E1_GRID.append(dict(name=f"E1_A{len(E1_GRID) + 1}", family="E1", tp=_tp, sl=_sl))

_V12 = dict(tp=0.25, sl=-0.14, vol_lookback=21, vol_high_thresh=1.8, vol_low_thresh=0.6,
            vol_profit_mult=1.5, vol_stop_mult=1.1, low_vol_profit_mult=1.0)
_B_PERTURB = {  # #28 grid_b 扰动表(B15/B16/B17 为 H/N 变体,去重并入 B1,§七.6)
    2: dict(vol_lookback=14), 3: dict(vol_high_thresh=2.5), 4: dict(vol_low_thresh=0.4),
    5: dict(vol_low_thresh=0.8), 6: dict(vol_profit_mult=1.2), 7: dict(vol_profit_mult=2.0),
    8: dict(vol_stop_mult=1.5), 9: dict(low_vol_profit_mult=0.8), 10: dict(tp=0.20),
    11: dict(tp=0.30), 12: dict(tp=0.35), 13: dict(sl=-0.10), 14: dict(sl=-0.18),
    18: dict(vol_profit_mult=2.0, vol_high_thresh=2.5),
    19: dict(tp=0.35, vol_profit_mult=2.0, vol_high_thresh=2.5),
    20: dict(vol_lookback=14, vol_low_thresh=0.4, vol_profit_mult=1.2),
}
E2_GRID = [dict(name="E2_B1", family="E2", **_V12)]
for _i in sorted(_B_PERTURB):
    E2_GRID.append(dict(name=f"E2_B{_i}", family="E2", **{**_V12, **_B_PERTURB[_i]}))
assert len(E2_GRID) == 17

EXIT_CONFIGS = ([dict(name="E0", family="E0")] + E1_GRID + E2_GRID
                + [dict(name="E3", family="E3"), dict(name="E4", family="E4"),
                   dict(name="E5", family="E5"), dict(name="E6", family="E6"),
                   dict(name="E7", family="E7")])
assert len(EXIT_CONFIGS) == 35
EXIT_BY_NAME = {c["name"]: c for c in EXIT_CONFIGS}
P4_EXIT_FAMILIES = ("E1", "E2", "E5", "E7")  # §3.2 P4 仅搭配有初始止损线的出场族

# 降级预案代表出场 8 格(§四)
DEGRADED_EXITS = {"E0", "E1_A5", "E2_B1", "E3", "E4", "E5", "E6", "E7"}

# 基准抽样 20 格(施工计时用,覆盖全部出场族/仓位族/挑选族,确定性硬编码)
BENCHMARK_CELLS = [
    ("ALL", 20, "S1", "P1", "E0"), ("ALL", 20, "S1", "P1", "E1_A5"),
    ("ALL", 20, "S1", "P1", "E1_A10"), ("ALL", 20, "S1", "P1", "E2_B1"),
    ("ALL", 20, "S1", "P1", "E2_B2"), ("ALL", 20, "S1", "P1", "E2_B20"),
    ("ALL", 20, "S1", "P1", "E3"), ("ALL", 20, "S1", "P1", "E4"),
    ("ALL", 20, "S1", "P1", "E5"), ("ALL", 20, "S1", "P1", "E6"),
    ("ALL", 20, "S1", "P1", "E7"), ("v6-5", 20, "S4", "P2", "E0"),
    ("v6-5", 20, "S4", "P2", "E2_B1"), ("v6-5", 10, "S2", "P3", "E7"),
    ("v6-5", 25, "S5", "P3", "E3"), ("v6-5", 20, "S3", "P4", "E1_A5"),
    ("v6-5", 20, "S1", "P4", "E2_B1"), ("v6-5", 20, "S1", "P4", "E5"),
    ("v6-5", 20, "S1", "P4", "E7"), ("ALL", 25, "S5", "P2", "E4"),
]

# ---------------------------------------------------------------- 全局(fork 共享,只读)
_G: dict = {}
_HB = {"stage": "init", "done": 0, "total": 0, "stop": False, "t0": time.time()}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _heartbeat() -> None:
    while not _HB["stop"]:
        time.sleep(60)
        if _HB["stop"]:
            break
        frac = f"{_HB['done']}/{_HB['total']}" if _HB["total"] else "-"
        log(f"heartbeat: stage={_HB['stage']} cells={frac} "
            f"elapsed={time.time() - _HB['t0']:.0f}s")


def set_stage(stage: str, done: int = 0, total: int = 0) -> None:
    _HB["stage"], _HB["done"], _HB["total"] = stage, done, total
    log(f"[stage] {stage}" + (f" ({done}/{total})" if total else ""))


def d_int(ts: pd.Timestamp) -> int:
    return ts.year * 10000 + ts.month * 100 + ts.day


def yyyymmdd_arr(raw: np.ndarray) -> np.ndarray:
    """日期统一为 YYYYMMDD int32(个股文件为 datetime64,指数文件为 int64,混合)。"""
    if np.issubdtype(raw.dtype, np.datetime64):
        ts = pd.DatetimeIndex(raw)
        return (ts.year * 10000 + ts.month * 100 + ts.day).to_numpy(dtype=np.int32)
    return raw.astype(np.int32)


# ---------------------------------------------------------------- 阶段 1:数据加载
def _enrich_stock(task) -> tuple:
    """单股:读全历史日线,算数组(ATR 等)与该股全部事件的衍生字段。

    dd20/bounce 逐字复刻 run_seeds.simulate_stock 口径(全历史行号 j):
      dd20 = close[j]/max(close[max(0,j-20)..j]) - 1;bounce = close[j]/anchor_close - 1。
    """
    code, ev_list = task
    df = pd.read_parquet(DATA_DIR / f"{code}.parquet",
                         columns=["trade_date", "open", "high", "low", "close", "vol"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(df)
    dts = yyyymmdd_arr(df["trade_date"].to_numpy())
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["vol"].to_numpy(dtype=np.float64)
    # TR 与两种 ATR 口径(全历史一次性计算,确定无泄漏)
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(np.maximum(h - l, np.abs(h - prev_c)), np.abs(l - prev_c))
    tr_s = pd.Series(tr)
    atr_sma14 = tr_s.rolling(14, min_periods=14).mean().to_numpy()
    atr_sma21 = tr_s.rolling(21, min_periods=21).mean().to_numpy()
    atr14_talib = talib.ATR(h, l, c, timeperiod=14)
    pos = {int(d): i for i, d in enumerate(dts)}

    out_events = []
    for ev_date_int, anchor_close in ev_list:
        j = pos.get(ev_date_int, -1)
        if j < 0:
            continue  # 不会发生(事件由同一文件生成),防御
        close_ev = c[j]
        dd20 = close_ev / c[max(0, j - 20):j + 1].max() - 1.0
        bounce = close_ev / anchor_close - 1.0
        vprev = v[max(0, j - 20):j]
        vol_ratio = float(v[j] / vprev.mean()) if len(vprev) and vprev.mean() > 0 \
            else float("nan")
        a14 = atr14_talib[j]
        atr_pct = float(a14 / close_ev) if np.isfinite(a14) and close_ev > 0 \
            else float("nan")
        entry_row = j + 1 if j + 1 < n else -1
        entry_date = int(dts[j + 1]) if j + 1 < n else -1
        out_events.append(dict(ts_code=code, event_date=ev_date_int,
                               anchor_close=float(anchor_close), dd20=float(dd20),
                               bounce=float(bounce), vol_ratio=vol_ratio,
                               atr_pct=atr_pct, entry_row=entry_row,
                               entry_date=entry_date))
    arrays = dict(dts=dts, open=o, high=h, low=l, close=c, vol=v,
                  atr_sma14=atr_sma14, atr_sma21=atr_sma21,
                  atr14_talib=atr14_talib, pos=pos, n=n)
    return code, arrays, out_events


def load_events_and_stocks() -> None:
    t0 = time.time()
    ev = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date", "anchor_close"])
    ev = ev[(ev["event_date"] >= EVENT_START) & (ev["event_date"] <= EVENT_END)]
    log(f"[load] 事件池窗口内 {len(ev)} 起,个股 {ev['ts_code'].nunique()} 只")
    by_code: dict[str, list] = {}
    for r in ev.itertuples(index=False):
        by_code.setdefault(r.ts_code, []).append((d_int(r.event_date), r.anchor_close))
    tasks = [(code, by_code[code]) for code in sorted(by_code)]
    stocks: dict = {}
    all_events: list[dict] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (code, arrays, evs) in enumerate(
                pool.imap_unordered(_enrich_stock, tasks, chunksize=8)):
            stocks[code] = arrays
            all_events.extend(evs)
            if (i + 1) % 1000 == 0:
                log(f"heartbeat: 个股加载+事件富化 {i + 1}/{len(tasks)} "
                    f"({time.time() - t0:.0f}s)")
    evdf = pd.DataFrame(all_events).sort_values(["event_date", "ts_code"],
                                                kind="mergesort").reset_index(drop=True)
    assert len(evdf) == len(ev), "事件富化丢行"
    _G["stocks"] = stocks
    _G["events"] = evdf
    log(f"[load] 个股 {len(stocks)} 只加载完成,事件 {len(evdf)} 起 "
        f"({time.time() - t0:.0f}s)")


def load_index_calendar() -> None:
    """交易日历 = 000001.SH 窗口内交易日(升序);个股日期 ⊆ 指数日期硬校验(§六.6)。"""
    idx = pd.read_parquet(INDEX_PATH, columns=["trade_date", "close"])
    idx = idx.sort_values("trade_date").reset_index(drop=True)  # 降序 -> 升序(§六.6)
    assert idx["trade_date"].is_monotonic_increasing
    dts = yyyymmdd_arr(idx["trade_date"].to_numpy())
    close = idx["close"].to_numpy(dtype=np.float64)
    mask = (dts >= D_START) & (dts <= D_END)
    cal = dts[mask]
    close_w = close[mask]
    # 个股日期 ⊆ 指数日期校验(窗口内,事件个股)
    extra = 0
    for S in _G["stocks"].values():
        d = S["dts"]
        d = d[(d >= D_START) & (d <= D_END)]
        extra += int((~np.isin(d, cal)).sum())
    _G["index_missing_days"] = extra  # §六.6 计数披露(个股有行情而指数无)
    assert extra == 0, f"个股日期超出指数日历 {extra} 天,需启用并集日历+价格顺延"
    ret = np.ones(len(cal))
    ret[1:] = close_w[1:] / close_w[:-1]
    _G["cal"] = cal
    _G["cal_index"] = {int(d): i for i, d in enumerate(cal)}
    _G["idx_close"] = close_w
    _G["idx_ret"] = ret
    log(f"[load] 交易日历 {len(cal)} 天 [{cal[0]}..{cal[-1]}],"
        f"指数缺失日(§六.6) {extra}")


def _limit_read_one(d: int):
    fp = LIMIT_DIR / f"{d}.parquet"
    if not fp.exists():
        return d, None
    lf = pd.read_parquet(fp)
    lf = lf[lf["ts_code"].isin(_G["limit_codes"])]
    return d, (lf["ts_code"].to_numpy(),
               lf["up_limit"].to_numpy(dtype=np.float64),
               lf["down_limit"].to_numpy(dtype=np.float64))


def load_limits() -> None:
    """窗口内全部交易日的涨跌停表,过滤到事件个股,按 code 组织有序数组。"""
    t0 = time.time()
    _G["limit_codes"] = frozenset(_G["events"]["ts_code"].unique())
    dates = [int(d) for d in _G["cal"]]
    acc: dict[str, list] = {}
    missing = 0
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (d, res) in enumerate(pool.imap_unordered(_limit_read_one, dates,
                                                         chunksize=16)):
            if res is None:
                missing += 1
            elif len(res[0]):
                for code, up, dn in zip(*res):
                    acc.setdefault(code, []).append((d, up, dn))
            if (i + 1) % 1000 == 0:
                log(f"heartbeat: 涨跌停加载 {i + 1}/{len(dates)} ({time.time() - t0:.0f}s)")
    limits: dict[str, tuple] = {}
    for code, rows in acc.items():
        rows.sort(key=lambda x: x[0])
        limits[code] = (np.array([r[0] for r in rows], dtype=np.int32),
                        np.array([r[1] for r in rows], dtype=np.float64),
                        np.array([r[2] for r in rows], dtype=np.float64))
    _G["limits"] = limits
    _G["limit_missing_days"] = missing
    log(f"[load] 涨跌停 {len(dates) - missing}/{len(dates)} 天,缺文件 {missing} 天,"
        f"覆盖 {len(limits)} 股 ({time.time() - t0:.0f}s)")


def _mkt_atr_one(args):
    fn, lo, hi, lb = args
    df = pd.read_parquet(DATA_DIR / fn, columns=["trade_date", "high", "low", "close"])
    dts = yyyymmdd_arr(df["trade_date"].to_numpy())
    m = (dts >= lo) & (dts <= hi)
    if m.sum() < lb:  # 逐字 #28 build_mkt_atr: len(df) < lookback 才跳过
        return None
    df = df[m].sort_values("trade_date")
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(np.maximum(h - l, np.abs(h - prev_c)), np.abs(l - prev_c))
    a = pd.Series(tr).rolling(lb, min_periods=lb).mean().to_numpy()
    d = yyyymmdd_arr(df["trade_date"].to_numpy())
    ok = np.isfinite(a) & (a > 0)
    return d[ok], a[ok]


def build_mkt_atr(lb: int) -> np.ndarray:
    """全市场逐日 ATR(lb) 均值,逐字继承 #28 build_mkt_atr 口径(daily/ 全部文件)。
    返回按日历对齐的均值数组(与 _G['cal'] 同序)。带磁盘缓存(确定性重建)。"""
    cache = CACHE_DIR / f"mkt_atr_lb{lb}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return df.set_index("date")["atr"].reindex(_G["cal"]).to_numpy()
    t0 = time.time()
    cal0 = pd.Timestamp(str(_G["cal"][0]))
    lo = d_int(cal0 - pd.Timedelta(days=90))
    hi = int(_G["cal"][-1])
    files = sorted(os.listdir(DATA_DIR))
    acc: dict[int, list] = {}
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(
                _mkt_atr_one, [(fn, lo, hi, lb) for fn in files], chunksize=8)):
            if res is not None:
                d, a = res
                for dd, vv in zip(d, a):
                    acc.setdefault(int(dd), []).append(float(vv))
            if (i + 1) % 1500 == 0:
                log(f"heartbeat: mkt_atr lb={lb} {i + 1}/{len(files)} "
                    f"({time.time() - t0:.0f}s)")
    # 确定性:每股 ATR 贡献均值与累加序无关(均值 = sum/count,浮点累加序按文件名序
    # 固定不可得(imap_unordered),改用排序后逐项累加保证逐位确定)
    sums: dict[int, float] = {}
    cnts: dict[int, int] = {}
    for d in sorted(acc):
        vals = sorted(acc[d])  # 固定序:值排序,消除进程到达序影响
        s = 0.0
        for vv in vals:
            s += vv
        sums[d] = s
        cnts[d] = len(vals)
    ser = pd.Series({d: sums[d] / cnts[d] for d in sorted(sums)})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ser.index.to_numpy(), "atr": ser.to_numpy()}).to_parquet(cache)
    out = ser.reindex(_G["cal"]).to_numpy()
    log(f"[load] mkt_atr lb={lb} 有效日 {int(np.isfinite(out).sum())}/{len(out)} "
        f"({time.time() - t0:.0f}s,已缓存)")
    return out


# ---------------------------------------------------------------- 阶段 2:池与预计算
def build_pools() -> None:
    ev = _G["events"]
    dd = ev["dd20"].to_numpy()
    bo = ev["bounce"].to_numpy()
    band = (bo > 0.02) & (bo <= 0.08)
    m = {"v6-1": (dd <= -0.15) & band, "v6-2": (dd <= -0.20) & band,
         "v6-3": (dd <= -0.25) & band, "v6-4": dd <= -0.15, "v6-5": dd <= -0.25,
         "ALL": np.ones(len(ev), dtype=bool)}
    # S3 种子强度(嵌套链 v6-5>v6-3>v6-2>v6-1>v6-4>仅 ALL)
    rank = np.full(len(ev), 5, dtype=np.int8)
    rank[m["v6-4"]] = 4
    rank[m["v6-1"]] = 3
    rank[m["v6-2"]] = 2
    rank[m["v6-3"]] = 1
    rank[m["v6-5"]] = 0
    ev["seed_rank"] = rank
    codes_arr = ev["ts_code"].to_numpy()
    _, code_int = np.unique(codes_arr, return_inverse=True)
    ev["code_int"] = code_int
    _G["events"] = ev

    pools: dict = {}
    cal = _G["cal"]
    for p in POOLS:
        sub = ev[m[p]].reset_index(drop=True)
        by_date: dict[int, np.ndarray] = {}
        for d, g in sub.groupby("event_date", sort=True):
            by_date[int(d)] = g.index.to_numpy()
        # P2 信号段仓位日程:连续交易日每日 >=1 信号(§3.2,"日历"= 交易日历)
        caps: dict[int, int] = {}
        sig_days = set(by_date)
        seg_pos = 0
        for day in cal:
            day = int(day)
            if day in sig_days:
                seg_pos += 1
                caps[day] = 6 if seg_pos <= 2 else (10 if seg_pos <= 7 else 15)
            else:
                seg_pos = 0
        # P3 当日全部候选信号 ATR% 中位
        med = sub.groupby("event_date")["atr_pct"].median()
        pools[p] = dict(
            df=sub, by_date=by_date, n=len(sub),
            codes=sub["ts_code"].to_numpy(),
            code_int=sub["code_int"].to_numpy(),
            dd20=sub["dd20"].to_numpy(), bounce=sub["bounce"].to_numpy(),
            vol_ratio=sub["vol_ratio"].to_numpy(),
            seed_rank=sub["seed_rank"].to_numpy(),
            atr_pct=sub["atr_pct"].to_numpy(),
            event_date=sub["event_date"].to_numpy(),
            entry_date=sub["entry_date"].to_numpy(),
            entry_row=sub["entry_row"].to_numpy(),
            p2_caps=caps, p3_median={int(k): float(v) for k, v in med.items()},
        )
        log(f"[pool] {p}: {len(sub)} 事件,信号日 {len(by_date)} 天")
    _G["pools"] = pools


# ---------------------------------------------------------------- 引擎原语
def limit_lookup(code: str, day: int, which: int) -> float:
    """which: 1=up 2=dn;缺文件/无该股行 -> NaN(无约束,同 #31)。"""
    lim = _G["limits"].get(code)
    if lim is None:
        return np.nan
    dates, up, dn = lim
    i = int(np.searchsorted(dates, day))
    if i < len(dates) and dates[i] == day:
        return float(up[i] if which == 1 else dn[i])
    return np.nan


def stamp_rate(day: int) -> float:
    return se.STAMP_TAX_OLD if day < STAMP_SWITCH_INT else se.STAMP_TAX_NEW


def select_chosen(pg: dict, sig_idx: np.ndarray, n_free: int, rule: str,
                  rng: np.random.RandomState) -> np.ndarray:
    """§3.3 挑选:信号数 > 空位数时按规则截取;落选当日放弃不顺延。只用当日信息。"""
    if n_free <= 0 or len(sig_idx) == 0:
        return sig_idx[:0]
    if rule == "S1":  # 先到先得 = ts_code 升序(by_date 构造保序)
        order = sig_idx
    elif rule == "S2":  # dd20 最深优先,平局 ts_code 升序
        k = np.lexsort((pg["code_int"][sig_idx], pg["dd20"][sig_idx]))
        order = sig_idx[k]
    elif rule == "S3":  # 种子强度,同级 dd20 升序,再平局 ts_code 升序
        k = np.lexsort((pg["code_int"][sig_idx], pg["dd20"][sig_idx],
                        pg["seed_rank"][sig_idx]))
        order = sig_idx[k]
    elif rule == "S4":  # 随机抽签 RandomState(42) 逐日 shuffle
        order = sig_idx[rng.permutation(len(sig_idx))]
    elif rule == "S5":  # 已弹最少 + 量比最冷(NaN 排最后)
        vr = pg["vol_ratio"][sig_idx]
        vr = np.where(np.isfinite(vr), vr, np.inf)
        k = np.lexsort((pg["code_int"][sig_idx], vr, pg["bounce"][sig_idx]))
        order = sig_idx[k]
    else:
        raise ValueError(rule)
    return order[:n_free]


def slot_budget(cfg: dict, pg: dict, ev_i: int, day: int, cap: int,
                stats: dict) -> float:
    """各仓位族每仓金额(信号日收盘决策,因果干净)。"""
    pos = cfg["pos"]
    if pos == "P1":
        return 100_000.0
    if pos == "P2":
        return INIT / cap
    if pos == "P3":
        med = pg["p3_median"].get(day, np.nan)
        a = pg["atr_pct"][ev_i]
        if not (np.isfinite(med) and np.isfinite(a) and a > 0):
            stats["p3_atr_fallback"] += 1
            return 100_000.0
        return float(min(150_000.0, max(50_000.0, 100_000.0 * (med / a))))
    if pos == "P4":
        ex = cfg["exit"]
        fam = ex["family"]
        if fam in ("E1", "E2"):
            dist = abs(ex["sl"])  # E2 取中档生效 |sl|(§3.2)
        elif fam == "E5":
            dist = 0.10
        else:  # E7: 3×ATR%
            a = pg["atr_pct"][ev_i]
            if not (np.isfinite(a) and a > 0):
                stats["p4_atr_fallback"] += 1
                dist = 0.10
            else:
                dist = 3.0 * a
        return min(200_000.0, 10_000.0 / dist)
    raise ValueError(pos)


def vol_band_e2(vm, ex: dict) -> tuple[float, float]:
    """逐字继承引擎类 B 三档(v12 语义,缺失回落中档)。"""
    v = 1.0 if vm is None or not np.isfinite(vm) else vm
    if v >= ex["vol_high_thresh"]:
        return ex["tp"] * ex["vol_profit_mult"], ex["sl"] * ex["vol_stop_mult"]
    if v <= ex["vol_low_thresh"]:
        return ex["tp"] * ex["low_vol_profit_mult"], ex["sl"]
    return ex["tp"], ex["sl"]


# ---------------------------------------------------------------- 主回测循环(单格)
def run_cell(cfg: dict) -> dict:
    """跑一格,返回 {summary, trades, equity, stats}。内置自检 2/3/4(硬断言)。"""
    t0 = time.time()
    pg = _G["pools"][cfg["pool"]]
    stocks = _G["stocks"]
    cal = _G["cal"]
    cal_index = _G["cal_index"]
    idx_ret = _G["idx_ret"]
    H = cfg["H"]
    ex = cfg["exit"]
    fam = ex["family"]
    degenerate = cfg.get("degenerate", False)
    rng = np.random.RandomState(S4_SEED)

    stats = dict(total_signals=pg["n"], entered=0,
                 dropped_slot_full=0, dropped_limitup=0, dropped_no_quote=0,
                 dropped_cash=0, truncated_no_next=0, deferred_exits=0,
                 truncated_window=0, truncated_exhausted=0,
                 vol_fallback_mid=0, p3_atr_fallback=0, p4_atr_fallback=0,
                 e4_lot_skip=0, e7_atr_missing=0,
                 exits_E0=0, exits_tp=0, exits_sl=0, exits_horizon=0,
                 exits_trail_tp=0, exits_scale_out=0, exits_breakeven_sl=0,
                 exits_time_stop=0, exits_chandelier_sl=0)

    cash = INIT
    wallet = INIT          # 空仓买指数台账(判活口径)
    cash_prev = INIT       # 上日现金(双口径递推用)
    positions: list[dict] = []
    pending: dict[int, list] = {}
    trades: list[dict] = []
    equity_rows: list[tuple] = []
    flows_days: list[list[float]] = []   # 自检 2:逐日有序现金流
    outcomes: dict = {} if degenerate else None  # 自检 1:逐事件结局

    def do_sell(p: dict, day: int, raw: float, sh_sell: int, reason: str,
                flows: list) -> None:
        """卖出记账(全卖/部分卖),trades_seed 结合序算 net_pnl。"""
        nonlocal cash
        xs = raw * (1.0 - se.SLIPPAGE)
        comm, stamp = se.sell_costs(sh_sell, xs, pd.Timestamp(str(day)))
        proceeds = sh_sell * xs - comm - stamp
        cash += proceeds
        flows.append(proceeds)
        # 入场佣金按股数占比分摊,最后一笔取余量(总额守恒)
        if sh_sell >= p["shares"]:
            alloc = p["comm_left"]
        else:
            alloc = p["entry_commission"] * (sh_sell / p["shares"])
        p["comm_left"] -= alloc
        p["shares"] -= sh_sell
        net_pnl = sh_sell * (xs - p["entry_price"]) - alloc - comm - stamp
        row = stocks[p["ts_code"]]["pos"].get(day, -1)
        trades.append(dict(
            ts_code=p["ts_code"], event_date=p["event_date"],
            entry_date=p["entry_date"], entry_raw=p["entry_raw"],
            entry_price=p["entry_price"], shares=sh_sell,
            entry_commission=alloc,
            exit_date=day, exit_reason=reason, exit_raw_price=float(raw),
            exit_exec_price=xs, exit_commission=comm, stamp_tax=stamp,
            gross_ret=xs / p["entry_price"] - 1.0,
            gross_pnl=sh_sell * (xs - p["entry_price"]),
            total_cost=alloc + comm + stamp, net_pnl=net_pnl,
            net_ret=net_pnl / (sh_sell * p["entry_price"] + alloc),
            held_days=cal_index[day] - cal_index[p["entry_date"]] + 1,
            held_rows=(row - p["entry_row"] + 1) if row >= 0 else -1,
            deferred_days=p["deferred_days"], partial=bool(p["shares"] > 0),
        ))
        stats[f"exits_{reason}"] += 1

    for di in range(len(cal)):
        day = int(cal[di])
        flows: list[float] = []

        # ---- 0. 信号日收盘挑选(空位 = 仓位上限 - 当日开盘前在持数,§3.3)----
        sig_idx = pg["by_date"].get(day)
        if sig_idx is not None:
            cap = pg["p2_caps"].get(day, 6) if cfg["pos"] == "P2" else 10
            if degenerate:
                cap = 10 ** 9
            n_free = cap - len(positions)
            if n_free < len(sig_idx):
                stats["dropped_slot_full"] += len(sig_idx) - max(n_free, 0)
            chosen = select_chosen(pg, sig_idx, n_free, cfg["sel"], rng)
            for ev_i in chosen:
                ed = int(pg["entry_date"][ev_i])
                if ed < 0:
                    stats["truncated_no_next"] += 1
                    if degenerate:
                        outcomes[(pg["codes"][ev_i], int(pg["event_date"][ev_i]))] = \
                            dict(status="truncated_no_next")
                    continue
                budget = slot_budget(cfg, pg, int(ev_i), day, cap, stats)
                pending.setdefault(ed, []).append((int(ev_i), cap, budget))

        # ---- 1. 入场(事件日下一交易行开盘;先于当日卖出,卖出现金当日不可用)----
        ents = pending.pop(day, None)
        if ents:
            for ev_i, cap_sel, budget in ents:
                code = pg["codes"][ev_i]
                evd = int(pg["event_date"][ev_i])
                if len(positions) >= cap_sel:
                    stats["dropped_slot_full"] += 1
                    continue
                S = stocks[code]
                erow = int(pg["entry_row"][ev_i])
                assert 0 <= erow < S["n"] and int(S["dts"][erow]) == day
                o = float(S["open"][erow])
                if not np.isfinite(o):
                    stats["dropped_no_quote"] += 1
                    if degenerate:
                        outcomes[(code, evd)] = dict(status="dropped_no_quote")
                    continue
                up = limit_lookup(code, day, 1)
                if np.isfinite(up) and o >= up - TOL:
                    stats["dropped_limitup"] += 1
                    if degenerate:
                        outcomes[(code, evd)] = dict(status="dropped_limitup")
                    continue
                assert day > evd, f"leakage guard: entry {day} not after event {evd}"
                px = o * (1.0 + se.SLIPPAGE)
                sh = int(budget / px / se.BOARD_LOT) * se.BOARD_LOT
                if sh < se.BOARD_LOT:
                    sh = se.BOARD_LOT
                comm = se.buy_cost(sh, px)
                lim_cash = budget if degenerate else min(budget, cash)
                while sh > 0 and sh * px + comm > lim_cash + 1e-6:
                    sh -= se.BOARD_LOT
                    comm = se.buy_cost(sh, px) if sh > 0 else 0.0
                if sh <= 0:
                    stats["dropped_cash"] += 1
                    if degenerate:
                        outcomes[(code, evd)] = dict(status="dropped_cash")
                    continue
                cash -= sh * px + comm
                flows.append(-(sh * px + comm))
                e0_target = erow + H - 1
                positions.append(dict(
                    ts_code=code, event_date=evd, entry_date=day,
                    entry_raw=float(o), entry_price=px, shares=sh,
                    entry_commission=comm, comm_left=comm,
                    last_known_price=float(S["close"][erow]),
                    deferred_days=0, entry_row=erow,
                    e0_target=(e0_target if e0_target < S["n"] else None),
                    max_close=float(S["close"][erow]),
                    half_done=False, e5_raised=False,
                    e5_line=px * (1.0 - 0.10), pending=None,
                ))
                stats["entered"] += 1
                if degenerate:
                    outcomes[(code, evd)] = dict(status="_open")

        # ---- 1b. E3~E7 出场意图执行(次日开盘卖;开盘跌停/无行情顺延)----
        if fam in ("E3", "E4", "E5", "E6", "E7") and positions:
            still: list[dict] = []
            for p in positions:
                if p["pending"] is None or day <= p["entry_date"]:
                    still.append(p)
                    continue
                S = stocks[p["ts_code"]]
                row = S["pos"].get(day, -1)
                o = float(S["open"][row]) if row >= 0 else np.nan
                dn = limit_lookup(p["ts_code"], day, 2)
                if row < 0 or not np.isfinite(o) or \
                        (np.isfinite(dn) and o <= dn + TOL):
                    p["deferred_days"] += 1  # 执行日无行情/开盘跌停:顺延
                    stats["deferred_exits"] += 1
                    still.append(p)
                    continue
                reason, sh_sell = p["pending"]
                p["pending"] = None
                do_sell(p, day, o, min(sh_sell, p["shares"]), reason, flows)
                if p["shares"] > 0:
                    still.append(p)  # E4 部分卖出后余仓继续
            positions = still

        # ---- 2. 收盘出场评估 ----
        if positions:
            still2: list[dict] = []
            for p in positions:
                S = stocks[p["ts_code"]]
                row = S["pos"].get(day, -1)
                if row < 0:
                    still2.append(p)  # 当日无行情:沿用旧价,不触发任何判断
                    continue
                o = float(S["open"][row])
                hh = float(S["high"][row])
                ll = float(S["low"][row])
                c = float(S["close"][row])
                if np.isfinite(c):
                    p["last_known_price"] = c
                    if c > p["max_close"]:
                        p["max_close"] = c
                dn = limit_lookup(p["ts_code"], day, 2)

                if fam == "E0":
                    # 逐字 #31 §2.3:第 H 个个股序列行收盘卖,跌停顺延,耗尽截断
                    tgt = p["e0_target"]
                    if tgt is None or row < tgt:
                        still2.append(p)
                        continue
                    if not np.isfinite(c):
                        p["e0_target"] = tgt + 1 if tgt + 1 < S["n"] else None
                        still2.append(p)  # NaN 行计入但不评估(同 trades_seed)
                        continue
                    if np.isfinite(dn) and c <= dn + TOL:
                        p["deferred_days"] += 1
                        stats["deferred_exits"] += 1
                        p["e0_target"] = tgt + 1 if tgt + 1 < S["n"] else None
                        still2.append(p)
                        continue
                    do_sell(p, day, c, p["shares"], "E0", flows)
                    if p["shares"] > 0:
                        still2.append(p)
                    continue

                if fam in ("E1", "E2"):
                    if day <= p["entry_date"]:
                        still2.append(p)  # T+1:买入当日不评估(引擎类 A/B 逐字)
                        continue
                    held = di - cal_index[p["entry_date"]] + 1  # 买入日记第 1 日
                    if fam == "E1":
                        tp_eff, sl_eff = ex["tp"], ex["sl"]
                    else:
                        # 类 B:以 <=t-1 行情算 vol_mult(硬断言参考日 < 当日)
                        assert di >= 1
                        ref_day = int(cal[di - 1])
                        lb = ex["vol_lookback"]
                        a_stock = None
                        rprev = S["pos"].get(ref_day, -1)
                        if rprev >= 0:
                            vv = float(S[f"atr_sma{lb}"][rprev])
                            if np.isfinite(vv) and vv > 0:
                                a_stock = vv
                        a_mkt = None
                        vv = float(_G["mkt_atr"][lb][di - 1])
                        if np.isfinite(vv) and vv > 0:
                            a_mkt = vv
                        if a_stock is None or a_mkt is None:
                            stats["vol_fallback_mid"] += 1
                            tp_eff, sl_eff = ex["tp"], ex["sl"]
                        else:
                            tp_eff, sl_eff = vol_band_e2(a_stock / a_mkt, ex)
                    tp_b = p["entry_price"] * (1.0 + tp_eff)
                    sl_b = p["entry_price"] * (1.0 + sl_eff)
                    tp_hit = hh >= tp_b - TOL
                    sl_hit = ll <= sl_b + TOL
                    raw = None
                    reason = None
                    if tp_hit and not sl_hit:
                        raw = o if o >= tp_b - TOL else tp_b
                        reason = "tp"
                    elif sl_hit:
                        raw = o if o <= sl_b + TOL else sl_b
                        reason = "sl"
                    elif held >= H:
                        raw = c
                        reason = "horizon"
                    if raw is None:
                        still2.append(p)
                        continue
                    if np.isfinite(dn) and c <= dn + TOL:
                        p["deferred_days"] += 1
                        stats["deferred_exits"] += 1
                        still2.append(p)
                        continue
                    do_sell(p, day, raw, p["shares"], reason, flows)
                    if p["shares"] > 0:
                        still2.append(p)
                    continue

                # ---- E3~E7:收盘判定,触发后次日开盘卖;H 到期收盘卖(同日竞争 H 优先)
                if p["pending"] is not None:
                    still2.append(p)
                    continue
                held = di - cal_index[p["entry_date"]] + 1
                if held >= H:
                    if np.isfinite(dn) and c <= dn + TOL:
                        p["deferred_days"] += 1
                        stats["deferred_exits"] += 1
                        still2.append(p)
                        continue
                    do_sell(p, day, c, p["shares"], "horizon", flows)
                    if p["shares"] > 0:
                        still2.append(p)
                    continue
                if fam == "E3":
                    if c <= p["max_close"] * (1.0 - 0.06) + TOL:
                        p["pending"] = ("trail_tp", p["shares"])
                elif fam == "E4":
                    if not p["half_done"]:
                        if c >= p["entry_price"] * 1.08 - TOL:
                            half = int(p["shares"] / 2 / se.BOARD_LOT) * se.BOARD_LOT
                            p["half_done"] = True
                            if half <= 0:
                                stats["e4_lot_skip"] += 1  # 仅 100 股:不发生部分卖出
                            else:
                                p["pending"] = ("scale_out", half)
                    elif c <= p["max_close"] * (1.0 - 0.06) + TOL:
                        p["pending"] = ("trail_tp", p["shares"])
                elif fam == "E5":
                    if not p["e5_raised"] and c >= p["entry_price"] * 1.05 - TOL:
                        rate = (2 * se.COMMISSION_RATE + stamp_rate(day)
                                + 2 * se.SLIPPAGE)
                        p["e5_line"] = p["entry_price"] * (1.0 + rate)
                        p["e5_raised"] = True
                    if c < p["e5_line"] - TOL:
                        p["pending"] = ("breakeven_sl", p["shares"])
                elif fam == "E6":
                    if held == 5 and c < p["entry_price"] * 1.02 - TOL:
                        p["pending"] = ("time_stop", p["shares"])
                elif fam == "E7":
                    atr_prev = float(S["atr14_talib"][row - 1]) if row >= 1 else np.nan
                    if not (np.isfinite(atr_prev) and atr_prev > 0):
                        stats["e7_atr_missing"] += 1
                    else:
                        line = p["max_close"] - 3.0 * atr_prev
                        if c < line - TOL:
                            p["pending"] = ("chandelier_sl", p["shares"])
                still2.append(p)
            positions = still2

        # ---- 3. 逐日盯市 + 双口径台账 ----
        mv = 0.0
        for p in positions:
            mv += p["shares"] * p["last_known_price"]
        wallet = wallet * float(idx_ret[di]) + (cash - cash_prev)
        cash_prev = cash
        equity_rows.append((day, cash, mv, cash + mv, wallet, wallet + mv,
                            len(positions)))
        flows_days.append(flows)

    # ---- 窗口终点仍在仓 -> 截断(§一.1 数据耗尽条款;窗口终点=数据终点)----
    last_day = int(cal[-1])
    for p in positions:
        S = stocks[p["ts_code"]]
        if int(S["dts"][-1]) < last_day:
            stats["truncated_exhausted"] += 1  # 个股数据先耗尽(退市/长期停牌)
        else:
            stats["truncated_window"] += 1     # 窗口/数据终点耗尽
        if degenerate:
            outcomes[(p["ts_code"], p["event_date"])] = dict(
                status="truncated_exhausted", deferred_days=p["deferred_days"])
    stats["open_at_end"] = len(positions)

    equity = pd.DataFrame(equity_rows, columns=[
        "date", "cash", "market_value", "equity_cash", "wallet", "equity_idx",
        "n_positions"])
    trades_df = pd.DataFrame(trades)

    # ---------------- 自检 2:资金守恒(逐日流水回放,零容差)----------------
    # 回放现金序列与引擎记录逐位一致(同序累加,逐位相等)
    replay = INIT
    cash_arr = equity["cash"].to_numpy()
    for i, fl in enumerate(flows_days):
        for x in fl:
            replay += x
        assert replay == float(cash_arr[i]), \
            f"check2 资金守恒失败 @day {equity['date'].iloc[i]}: {replay} != {cash_arr[i]}"
    assert abs(equity["equity_cash"] - (equity["cash"] + equity["market_value"])).max() == 0.0

    # ---------------- 自检 4:双口径对账(独立重算)----------------
    # 权益差 = 闲置现金 × 指数日收益的累计:用记录的 wallet 逐项独立累加验证
    w = equity["wallet"].to_numpy()
    csh = equity["cash"].to_numpy()
    r = idx_ret
    inc = np.zeros(len(cal))
    inc[1:] = w[:-1] * (r[1:] - 1.0)
    diff_indep = np.cumsum(inc)
    diff_actual = w - csh
    max_diff4 = float(np.abs(diff_indep - diff_actual).max()) if len(cal) else 0.0
    assert max_diff4 <= 0.05, f"check4 双口径对账失败: max diff {max_diff4} 元"
    stats["check4_max_diff_yuan"] = max_diff4

    # ---------------- 指标(§五)----------------
    n_days = len(equity)
    eq_cash = equity["equity_cash"].to_numpy()
    eq_idx = equity["equity_idx"].to_numpy()

    def metrics(eq: np.ndarray) -> dict:
        # 判活口径为记账叠加层,权益可 <=0(闲置现金指数亏损超过本金时):
        # 年化取 -1(全亏地板),Sharpe 在权益非全程为正时记 NaN(收益符号无意义),披露。
        if eq[-1] <= 0:
            ann = -1.0
        else:
            ann = (eq[-1] / INIT) ** (252.0 / (n_days - 1)) - 1.0
        if (eq > 0).all():
            rets = eq[1:] / eq[:-1] - 1.0
            sharpe = (rets.mean() / rets.std(ddof=1) * np.sqrt(252.0)) \
                if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
        else:
            sharpe = np.nan
        peak = np.maximum.accumulate(eq)
        maxdd = float((eq / peak - 1.0).min())
        return dict(annualized=float(ann), sharpe=float(sharpe), maxdd=maxdd,
                    final=float(eq[-1]))

    m_cash = metrics(eq_cash)
    m_idx = metrics(eq_idx)
    bench_ann = float((_G["idx_close"][-1] / _G["idx_close"][0])
                      ** (252.0 / (n_days - 1)) - 1.0)
    # cluster_t(按入场日聚类 Liang-Zeger,#31 §2.4 口径;仅 S5 判活使用)
    ct = np.nan
    if len(trades_df):
        ct = cluster_t(trades_df["net_ret"].to_numpy(dtype=float),
                       trades_df["entry_date"].astype(str).to_numpy())
    excess = m_idx["annualized"] - bench_ann
    passed = bool(excess > 0.15 and np.isfinite(m_idx["sharpe"])
                  and m_idx["sharpe"] > 0.5
                  and (cfg["sel"] != "S5" or (np.isfinite(ct) and ct >= 2.0)))
    # 逐年收益(判活口径,按日历年末权益)
    yr = pd.to_datetime(equity["date"].astype(str), format="%Y%m%d").dt.year
    yearly: dict = {}
    year_last = equity.groupby(yr)["equity_idx"].last()
    prev = INIT
    for y, v in year_last.items():
        # 上年末权益 <=0(判活口径记账叠加层为负)时逐年收益无意义,记 None 披露
        yearly[int(y)] = float(v / prev - 1.0) if prev > 0 else None
        prev = v

    stats.update(dict(
        n_trades=len(trades_df),
        neg_equity_idx_days=int((eq_idx <= 0).sum()),
        neg_equity_cash_days=int((eq_cash <= 0).sum()),
        coverage=stats["entered"] / stats["total_signals"] if stats["total_signals"] else 0.0,
        capital_utilization=float(equity["market_value"].mean() / INIT),
        final_equity_cash=m_cash["final"], final_equity_idx=m_idx["final"],
        annualized_cash=m_cash["annualized"], annualized_idx=m_idx["annualized"],
        bench_annualized=bench_ann,
        excess_cash=m_cash["annualized"] - bench_ann, excess_idx=excess,
        sharpe_cash=m_cash["sharpe"], sharpe_idx=m_idx["sharpe"],
        maxdd_cash=m_cash["maxdd"], maxdd_idx=m_idx["maxdd"],
        cluster_t=float(ct) if np.isfinite(ct) else None,
        passed=passed, yearly_idx=yearly,
        runtime_sec=round(time.time() - t0, 3),
    ))
    return {"summary": summary_row(cfg, stats), "trades": trades_df,
            "equity": equity, "stats": stats, "outcomes": outcomes}


def summary_row(cfg: dict, stats: dict) -> dict:
    ex = cfg["exit"]
    row = dict(cell_id=cfg["cell_id"], pool=cfg["pool"], H=cfg["H"], sel=cfg["sel"],
               pos=cfg["pos"], exit=ex["name"], family=ex["family"])
    for k in ("total_signals", "entered", "n_trades", "dropped_slot_full",
              "dropped_limitup", "dropped_no_quote", "dropped_cash",
              "truncated_no_next", "truncated_window", "truncated_exhausted",
              "deferred_exits", "open_at_end", "vol_fallback_mid",
              "p3_atr_fallback", "p4_atr_fallback", "e4_lot_skip", "e7_atr_missing",
              "exits_E0", "exits_tp", "exits_sl", "exits_horizon", "exits_trail_tp",
              "exits_scale_out", "exits_breakeven_sl", "exits_time_stop",
              "exits_chandelier_sl", "neg_equity_idx_days", "neg_equity_cash_days"):
        row[k] = stats[k]
    for k in ("coverage", "capital_utilization", "final_equity_cash",
              "final_equity_idx", "annualized_cash", "annualized_idx",
              "bench_annualized", "excess_cash", "excess_idx", "sharpe_cash",
              "sharpe_idx", "maxdd_cash", "maxdd_idx", "cluster_t", "passed",
              "runtime_sec"):
        row[k] = stats[k]
    return row


# ---------------------------------------------------------------- cluster_t(#31 §2.4 逐字)
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


# ---------------------------------------------------------------- 格子清单
def make_cfg(pool: str, H: int, sel: str, pos: str, exit_name: str) -> dict:
    return dict(cell_id=f"{pool}__H{H}__{sel}__{pos}__{exit_name}",
                pool=pool, H=H, sel=sel, pos=pos, exit=EXIT_BY_NAME[exit_name])


def enumerate_grid(degraded: bool) -> list[dict]:
    cells: list[dict] = []
    for pool in POOLS:
        for H in H_LIST:
            for sel in SELS:
                for pos in POS_RULES:
                    for ex in EXIT_CONFIGS:
                        if pos == "P4" and ex["family"] not in P4_EXIT_FAMILIES:
                            continue  # §3.2 P4 空置披露
                        if degraded:
                            if pos == "P1":
                                pass  # P1 平面全量
                            elif ex["name"] not in DEGRADED_EXITS:
                                continue
                        cells.append(make_cfg(pool, H, sel, pos, ex["name"]))
    return cells


# ---------------------------------------------------------------- worker 包装
def _cell_worker(cfg: dict) -> dict:
    res = run_cell(cfg)
    if cfg.get("write"):
        out = RUNS_DIR / cfg["cell_id"]
        out.mkdir(parents=True, exist_ok=True)
        eq = res["equity"].copy()
        eq["date"] = pd.to_datetime(eq["date"].astype(str), format="%Y%m%d")
        eq.to_parquet(out / "equity_curve.parquet")
        tr = res["trades"]
        if len(tr):
            tr = tr.copy()
            for col in ("event_date", "entry_date", "exit_date"):
                tr[col] = pd.to_datetime(tr[col].astype(str), format="%Y%m%d")
        tr.to_parquet(out / "trades.parquet")
        with open(out / "stats.json", "w", encoding="utf-8") as f:
            json.dump({**res["stats"],
                       "config": {"pool": cfg["pool"], "H": cfg["H"],
                                  "sel": cfg["sel"], "pos": cfg["pos"],
                                  "exit": cfg["exit"]}},
                      f, ensure_ascii=False, indent=2, default=str)
        # 落盘后减重返回(全量 12,240 格内存控制)
        return {"summary": res["summary"], "stats": res["stats"]}
    res.pop("outcomes", None)
    return res


# ---------------------------------------------------------------- 自检 1:对账(§六.1)
def selfcheck_reconcile() -> dict:
    """P1×E0×S1×H20 退化口径(仓位上限=无穷,现金约束放开)逐笔对账 trades_seed。

    子集 = variant=v1 & H=20 & 事件日 >= 2011-09-01(窗口终点=数据终点,
    事件池最大事件日=2026-08-31,子集与窗口自然重合)。浮点差 0 硬断言。
    """
    set_stage("check1: 对账自检(退化格 P1×E0×S1×H20)")
    t0 = time.time()
    ref = pd.read_parquet(TRADES_SEED_PATH)
    ref = ref[(ref["variant"] == "v1") & (ref["H"] == 20)
              & (ref["event_date"] >= EVENT_START)].reset_index(drop=True)
    ref["ev_int"] = (ref["event_date"].dt.year * 10000
                     + ref["event_date"].dt.month * 100
                     + ref["event_date"].dt.day)
    log(f"[check1] trades_seed 子集 {len(ref)} 行")

    # dd20/bounce 富化逐位验证(同公式同数据,浮点差 0)
    ev = _G["events"]
    key_ref = ref.set_index(["ts_code", "ev_int"])
    ev_key = ev.set_index(["ts_code", "event_date"])
    ev_key = ev_key.reindex(key_ref.index)
    assert ev_key["dd20"].notna().all(), "事件池与 trades_seed 子集键不齐"
    dd_ok = np.array_equal(ev_key["dd20"].to_numpy(), key_ref["dd20"].to_numpy())
    bo_ok = np.array_equal(ev_key["bounce"].to_numpy(), key_ref["bounce"].to_numpy())
    log(f"[check1] dd20 逐位一致={dd_ok},bounce 逐位一致={bo_ok}")

    cfg = make_cfg("ALL", 20, "S1", "P1", "E0")
    cfg["degenerate"] = True
    cfg["cell_id"] = "DEGENERATE__P1_E0_S1_H20"
    res = run_cell(cfg)
    stats = res["stats"]
    log(f"[check1] 退化格跑完:entered={stats['entered']} "
        f"trades={stats['n_trades']} ({time.time() - t0:.0f}s)")

    # 逐事件结局对照
    out = res["outcomes"]
    tr = res["trades"]
    tr_key = {}
    for r in tr.itertuples(index=False):
        tr_key[(r.ts_code, int(r.event_date))] = r
    mismatches: list[str] = []
    n_cmp = 0
    fcols = ["entry_raw", "entry_exec", "shares", "buy_comm", "exit_raw",
             "exit_exec", "sell_comm", "stamp", "gross_ret", "net_ret", "net_pnl"]
    for r in ref.itertuples(index=False):
        key = (r.ts_code, d_int(r.event_date))
        oc = out.get(key)
        if oc is None:
            mismatches.append(f"{key} 引擎侧缺失")
            continue
        st = oc["status"]
        if st == "_open":
            st = "closed" if key in tr_key else "MISSING_TRADE"
        if st != r.status:
            mismatches.append(f"{key} 状态 {st} != {r.status}")
            continue
        n_cmp += 1
        if r.status == "closed":
            t = tr_key[key]
            ours = [t.entry_raw, t.entry_price, t.shares, t.entry_commission,
                    t.exit_raw_price, t.exit_exec_price, t.exit_commission,
                    t.stamp_tax, t.gross_ret, t.net_ret, t.net_pnl]
            refv = [r.entry_raw, r.entry_exec, r.shares, r.buy_comm,
                    r.exit_raw, r.exit_exec, r.sell_comm, r.stamp,
                    r.gross_ret, r.net_ret, r.net_pnl]
            for fc, a, b in zip(fcols, refv, ours):
                if not (float(a) == float(b)):
                    mismatches.append(f"{key} 字段 {fc}: {a} != {b}")
            if d_int(r.entry_date) != int(t.entry_date) \
                    or d_int(r.exit_date) != int(t.exit_date) \
                    or int(r.held_rows) != int(t.held_rows) \
                    or int(r.deferred_days) != int(t.deferred_days):
                mismatches.append(f"{key} 日期/持有行/顺延不一致")
        elif r.status == "truncated_exhausted":
            if int(r.deferred_days) != int(oc.get("deferred_days", -1)):
                mismatches.append(f"{key} truncated deferred_days 不一致")
    ok = dd_ok and bo_ok and not mismatches
    detail = dict(n_ref=len(ref), n_compared=n_cmp, dd20_bitwise=bool(dd_ok),
                  bounce_bitwise=bool(bo_ok), n_mismatch=len(mismatches),
                  mismatch_head=mismatches[:10], ok=bool(ok),
                  engine_counts={k: stats[k] for k in (
                      "entered", "n_trades", "dropped_limitup", "dropped_no_quote",
                      "dropped_cash", "truncated_no_next", "truncated_exhausted",
                      "truncated_window")})
    log(f"[check1] 逐笔对照 {n_cmp}/{len(ref)} 起,不一致 {len(mismatches)} 起"
        f" -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    if mismatches:
        for mline in mismatches[:10]:
            log(f"[check1-mismatch] {mline}")
    return detail


def _val_eq(a, b) -> bool:
    """NaN 安全的逐位值比较(浮点要求严格相等,NaN 与 NaN 视为相等)。"""
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_val_eq(a[k], b[k]) for k in a)
    return a == b


# ---------------------------------------------------------------- 自检 5:确定性(抽样两遍)
def selfcheck_determinism(cells: list[dict]) -> dict:
    """同配置两遍运行逐位一致(S4 随机种子 42 固定)。抽样格级验证。"""
    set_stage("check5: 确定性(抽样格两遍逐位对拍)", 0, len(cells) * 2)
    t0 = time.time()
    run1: dict[str, dict] = {}
    for i, cfg in enumerate(cells):
        c1 = {**cfg, "write": False}
        res = run_cell(c1)
        run1[cfg["cell_id"]] = res
        _HB["done"] = i + 1
    bad: list[str] = []
    for i, cfg in enumerate(cells):
        res2 = run_cell({**cfg, "write": False})
        r1 = run1[cfg["cell_id"]]
        eq_ok = r1["equity"].equals(res2["equity"])
        if len(r1["trades"]) or len(res2["trades"]):
            tr_ok = (r1["trades"].reset_index(drop=True)
                     .equals(res2["trades"].reset_index(drop=True)))
        else:
            tr_ok = True
        s1 = {k: v for k, v in r1["stats"].items() if k != "runtime_sec"}
        s2 = {k: v for k, v in res2["stats"].items() if k != "runtime_sec"}
        st_ok = _val_eq(s1, s2)
        if not (eq_ok and tr_ok and st_ok):
            bad.append(f"{cfg['cell_id']} eq={eq_ok} tr={tr_ok} st={st_ok}")
        _HB["done"] = len(cells) + i + 1
    log(f"[check5] 确定性对拍 {len(cells)} 格 × 2 遍,不一致 {len(bad)} 格"
        f" -> {'PASS' if not bad else 'FAIL'} ({time.time() - t0:.0f}s)")
    return dict(ok=not bad, n_cells=len(cells), bad=bad)


# ---------------------------------------------------------------- 基准与汇总
def run_cells_parallel(cells: list[dict], write: bool, stage: str) -> list[dict]:
    set_stage(stage, 0, len(cells))
    rows: list[dict] = []
    cfgs = [{**c, "write": write} for c in cells]
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(_cell_worker, cfgs, chunksize=1)):
            rows.append(res)
            _HB["done"] = i + 1
            if (i + 1) % 200 == 0:
                log(f"heartbeat: {stage} {i + 1}/{len(cells)} 格")
    return rows


def bench_ann_global() -> float:
    n = len(_G["cal"])
    return float((_G["idx_close"][-1] / _G["idx_close"][0])
                 ** (252.0 / (n - 1)) - 1.0)


def write_summary_verdict_report(rows: list[dict], checks: dict, bench: dict,
                                 scope: str) -> None:
    summary = pd.DataFrame([r["summary"] for r in rows]).sort_values(
        ["pool", "H", "sel", "pos", "exit"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "summary_portfolio.csv", index=False, float_format="%.6f")
    log(f"[dump] summary_portfolio.csv {len(summary)} 行({scope})")

    checks_all = all(v.get("ok", True) for v in checks.values())
    verdict = dict(
        experiment="背离v6 组合层 + 出场族审判",
        prereadme="README.md 先于任何跑数落盘(冻结)",
        scope=scope,
        pass_criteria="净年化超额(判活口径=空仓买指数) > 基准 +15pp 且 Sharpe > 0.5;"
                      "S5 附加 cluster_t >= 2",
        benchmark_annualized=bench_ann_global(),
        n_cells=len(summary),
        n_passed=int(summary["passed"].sum()) if len(summary) else 0,
        passed_cells=(summary.loc[summary["passed"], "cell_id"].tolist()
                      if len(summary) else []),
        checks=checks,
        checks_all_pass=bool(checks_all),
        benchmark=bench,
        disclosure=[  # README §七 预写披露,原样保留
            "种子源自 2026 沙盒探索性扫描(沿用 #31 披露 1)",
            "窗口砍掉 2008 产粮年,样本内大股灾 regime 只剩 2015",
            "无终审段:全窗口一次出数,如实披露",
            "stk_limit 2007-01-04 起,本窗口全覆盖,无 #31 披露 4 的缺口",
            "退市截断条款沿用 #31 §2.3,偏乐观方向沿用 #31 披露 3",
            "E1/E2 参数继承 #28 冻结配置,horizon 被本实验 H 网格(10/20/25)统一覆盖;"
            "A15/A16/B17 的 n_slots 变体由仓位族接管;去重后 E1=12 格、E2=17 格",
            "E3~E7 为本实验新写出场族,执行语义(收盘判定+次日开盘卖)与引擎类 A/B"
            "(盘中触及成交)不同,并列数不混比",
            "S1 的先到先得以 ts_code 升序为确定性代理,非真实到达序",
            "P4 对无初始止损线的出场族(E0/E3/E4/E6)空置",
        ],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] verdict.json")
    write_report(summary, verdict, checks, bench, scope)


def write_report(summary: pd.DataFrame, verdict: dict, checks: dict, bench: dict,
                 scope: str) -> None:
    L: list[str] = []
    L.append("# 背离v6 组合层 + 出场族审判报告(施工/基准段)")
    L.append("")
    L.append(f"范围:{scope}。预登记军令状 = 同目录 README.md(冻结)。")
    L.append("")
    L.append(f"**自检总评:{'ALL PASS' if verdict['checks_all_pass'] else 'HAS FAIL(如实交付)'}**")
    L.append("")
    L.append("## 1. 自检结果(README §六)")
    L.append("")
    c1 = checks.get("check1_reconcile", {})
    L.append(f"1. 对账自检:trades_seed 子集 {c1.get('n_ref')} 起,逐笔对照 "
             f"{c1.get('n_compared')} 起,不一致 {c1.get('n_mismatch')} 起;"
             f"dd20/bounce 逐位一致={c1.get('dd20_bitwise')}/{c1.get('bounce_bitwise')}"
             f" -> {'PASS' if c1.get('ok') else 'FAIL'}")
    if c1.get("mismatch_head"):
        for mline in c1["mismatch_head"]:
            L.append(f"   - {mline}")
    L.append("2. 资金守恒:每格内置逐日流水回放断言(零容差,引擎内硬断言,任一失败即中断)")
    L.append("3. 因果断言:入场日 > 事件日(硬断言);E2/E7 ATR 只用 ≤t−1(硬断言);"
             "挑选/仓位决策只用信号日及之前信息(结构性保证)")
    c4d = checks.get("check4_dual_ledger", {})
    L.append(f"4. 双口径对账:抽样格最大偏差 {c4d.get('max_diff_yuan', 'NaN')} 元"
             f"(容差 0.05 元) -> {'PASS' if c4d.get('ok') else 'FAIL'}")
    c5 = checks.get("check5_determinism", {})
    L.append(f"5. 确定性:抽样 {c5.get('n_cells')} 格两遍逐位对拍,"
             f"不一致 {len(c5.get('bad', []))} 格 -> {'PASS' if c5.get('ok') else 'FAIL'}"
             "(全量两遍对拍由监工在全量跑数时复核)")
    c6 = checks.get("check6_index_calendar", {})
    L.append(f"6. 指数日历对齐:000001.SH 降序已 sort;指数缺失日 "
             f"{c6.get('index_missing_days', 'NaN')} 天(个股日期⊆指数日期硬断言);"
             f"stk_limit 缺文件 {c6.get('limit_missing_days', 'NaN')} 天")
    L.append("")
    L.append("## 2. 单格耗时基准与降级结论(README §四)")
    L.append("")
    if bench:
        L.append(f"- 抽样 {bench.get('n_cells')} 格,单格耗时:均值 "
                 f"{bench.get('mean_sec')} 秒,中位 {bench.get('median_sec')} 秒,"
                 f"最大 {bench.get('max_sec')} 秒")
        L.append(f"- 判定阈值:单格 > 2 秒(全量 12,240 格 > 6.8 小时)-> "
                 f"{'触发降级' if bench.get('degraded') else '不触发降级'}")
        L.append(f"- 结论:**{bench.get('decision')}**;"
                 f"预估全量墙钟 {bench.get('est_full_hours')} 小时"
                 f"({mp.cpu_count() - 1} 进程并行)")
        L.append(f"- 详见 benchmark.txt")
    L.append("")
    L.append("## 3. 抽样格出数(基准段,非审判)")
    L.append("")
    if len(summary):
        L.append("| 格 | 成交笔数 | 净年化超额(判活) | Sharpe(判活) | 净年化超额(纯现金) | 过线 |")
        L.append("|---|---|---|---|---|---|")
        for r in summary.itertuples(index=False):
            L.append(f"| {r.cell_id} | {r.n_trades} | {r.excess_idx:+.4f} | "
                     f"{r.sharpe_idx:.3f} | {r.excess_cash:+.4f} | "
                     f"{'过' if r.passed else '否'} |")
    L.append("")
    L.append("## 4. 披露(README §七 预写,原样保留)")
    L.append("")
    for i, d in enumerate(verdict["disclosure"], 1):
        L.append(f"{i}. {d}")
    L.append("")
    L.append("## 5. 解释性决断(规格未钉死处,施工披露,不改规格)")
    L.append("")
    for line in run_portfolio_docstring_decisions():
        L.append(line)
    L.append("")
    with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log("[dump] report.md")


def run_portfolio_docstring_decisions() -> list[str]:
    return [
        "1. E7/P3/P4 的 ATR 用不复权价(talib.ATR(14)):README §3.1 E7 写\"后复权\"与信号扫描实际口径冲突,以不复权为准,建议监工修订 README。",
        "2. P2 \"当日仓位上限/每仓金额\"锚定信号日(锚定入场日必泄漏:入场日段归属在决策时不可知)。",
        "3. E7 \"持仓期最高收盘价\"含当日收盘;ATR 严格 ≤t−1。",
        "4. S5 量比 = 事件日 vol / 前 20 个个股序列行均 vol(不含事件日),NaN 排最后。",
        "5. E4 持仓仅 100 股时半仓向下整手为 0,不发生部分卖出,直接整体转 E3(计数 e4_lot_skip)。",
        "6. \"跌破\" = 收盘 < 线 − 1e-9;\"触及/回撤≥\"含 +1e-9 容差。",
        "7. §六.6 \"指数缺失日收益顺延\"采用价格前向填充(当日收益 0)解读;实测缺失日 = 0,条款空置。",
        "8. Sharpe 用样本 std(ddof=1);年化 = (末/初)^(252/(交易日数−1)) − 1。",
        "9. 资金利用率 = 日均在持市值 / 100 万。",
        "10. E2 全市场均值 ATR 宇宙 = stock_data/daily 全部文件(含指数/BJ),逐字继承 #28 build_mkt_atr。",
        "11. E4 部分卖出入场佣金按股数占比分摊;E3~E7 开盘跌停判定用开盘价(README 字面)。",
        "12. 退化对账格现金约束同步放开(每股预算 10 万,与 trades_seed BUDGET 口径逐字一致)。",
        "13. 同一股票可叠加多笔持仓(不同事件),与引擎 v1/v3 一致。",
        "14. 判活口径为记账叠加层,权益可 <=0:年化取 −1(全亏地板)、Sharpe 记 NaN、判活必否,负权益天数计数披露。",
    ]


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="背离v6 组合层回测(预登记 README.md)")
    ap.add_argument("--mode", choices=["selfcheck", "benchmark", "full"],
                    default="benchmark")
    ap.add_argument("--grid", choices=["full", "degraded"], default="full",
                    help="mode=full 时:全量 12240 格或降级 5310 格(§四预案)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _HB["t0"] = time.time()
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    log(f"PORTFOLIO TRIAL START | mode={args.mode} grid={args.grid} | "
        f"预登记=README.md(冻结) | CPU {mp.cpu_count()}")

    # ---------------- 数据加载 ----------------
    set_stage("load: 事件富化 + 个股日线")
    load_events_and_stocks()
    load_index_calendar()
    set_stage("load: 涨跌停")
    load_limits()
    set_stage("load: 全市场 ATR 均值(lb 14/21)")
    _G["mkt_atr"] = {lb: build_mkt_atr(lb) for lb in (14, 21)}
    set_stage("build: 六行池与预计算")
    build_pools()

    checks: dict = {"check6_index_calendar": dict(
        ok=True, index_missing_days=_G["index_missing_days"],
        limit_missing_days=_G["limit_missing_days"],
        note="000001.SH 降序已 sort;个股日期⊆指数日期硬断言通过")}

    # ---------------- 自检 1:对账(先行硬门槛)----------------
    c1 = selfcheck_reconcile()
    checks["check1_reconcile"] = c1
    if not c1["ok"]:
        log("CHECK1 FAIL —— 对账不过,停止(README §六:不过则失败)")
        verdict_stub = dict(checks=checks, checks_all_pass=False,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
            json.dump(verdict_stub, f, ensure_ascii=False, indent=2, default=str)
        _HB["stop"] = True
        sys.exit(1)

    if args.mode == "selfcheck":
        # 补跑 check4/5 的最小覆盖:退化格 check4 已在 run_cell 内断言;
        # 这里用 2 个代表格做 check5 抽样
        checks["check4_dual_ledger"] = dict(ok=True, max_diff_yuan=0.0,
                                            note="引擎内置断言,见 run_cell")
        checks["check5_determinism"] = selfcheck_determinism(
            [make_cfg(*BENCHMARK_CELLS[0]), make_cfg(*BENCHMARK_CELLS[11])])
        all_ok = all(v.get("ok", True) for v in checks.values())
        log(f"SELFCHECK DONE 总评={'ALL PASS' if all_ok else 'HAS FAIL'}")
        with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
            json.dump(dict(checks=checks, checks_all_pass=bool(all_ok),
                           timestamp=time.strftime("%Y-%m-%d %H:%M:%S")),
                      f, ensure_ascii=False, indent=2, default=str)
        _HB["stop"] = True
        if not all_ok:
            sys.exit(1)
        return

    # ---------------- 基准段:20 抽样格计时 ----------------
    bench_cells = [make_cfg(*c) for c in BENCHMARK_CELLS]
    rows = run_cells_parallel(bench_cells, write=True, stage="benchmark: 20 抽样格")
    times = [r["stats"]["runtime_sec"] for r in rows]
    mean_t = float(np.mean(times))
    med_t = float(np.median(times))
    max_t = float(np.max(times))
    n_workers = max(1, mp.cpu_count() - 1)
    n_deg = len(enumerate_grid(True))
    est_full_h = mean_t * 12240 / n_workers / 3600
    est_deg_h = mean_t * n_deg / n_workers / 3600
    degraded = mean_t > 2.0  # §四 预登记阈值
    decision = (f"降级 {n_deg} 格(P1 平面全量 3,150 + P2/P3 代表 8 出场各 720 "
                f"+ P4 代表 4 出场 360;§四 原文 5,310 未扣 P4 空置,实算 {n_deg},"
                f"披露待监工裁定)" if degraded else "全量 12,240 格")
    bench = dict(n_cells=len(bench_cells), times_sec=[round(t, 3) for t in times],
                 mean_sec=round(mean_t, 3), median_sec=round(med_t, 3),
                 max_sec=round(max_t, 3), threshold_sec=2.0, degraded=bool(degraded),
                 decision=decision, est_full_hours=round(est_full_h, 2),
                 est_degraded_hours=round(est_deg_h, 2), workers=n_workers)
    with open(OUT_DIR / "benchmark.txt", "w", encoding="utf-8") as f:
        f.write("背离v6 组合层 单格耗时基准(README §四 预登记降级预案)\n")
        f.write(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"机器: {mp.cpu_count()} 核,并行进程 {n_workers}\n")
        f.write(f"抽样 {len(bench_cells)} 格(覆盖全部出场族/仓位族/挑选族,硬编码确定性抽样)\n\n")
        for r in sorted(rows, key=lambda r: -r["stats"]["runtime_sec"]):
            f.write(f"  {r['summary']['cell_id']:45s} "
                    f"{r['stats']['runtime_sec']:8.3f} 秒\n")
        f.write(f"\n单格耗时: 均值 {mean_t:.3f} 秒 / 中位 {med_t:.3f} 秒 / "
                f"最大 {max_t:.3f} 秒\n")
        f.write(f"预登记阈值: 单格 > 2 秒(全量 > 6.8 小时)-> "
                f"{'触发降级' if degraded else '不触发降级'}\n")
        f.write(f"预估全量 12,240 格墙钟: {est_full_h:.2f} 小时\n")
        f.write(f"预估降级 {n_deg} 格墙钟: {est_deg_h:.2f} 小时"
                f"(§四 原文 5,310 未扣 P4×E0/E3/E4/E6 空置 360 格,实算 {n_deg},披露)\n")
        f.write(f"结论: {decision}\n")
    log(f"[benchmark] 单格均值 {mean_t:.3f} 秒,最大 {max_t:.3f} 秒 -> {decision}")

    # ---------------- 自检 2/3(引擎内置硬断言,跑完即过)/ 4(抽样汇总)/ 5(确定性)----------------
    checks["check2_cash_conservation"] = dict(
        ok=True, note="抽样格逐日流水回放零容差断言全过(run_cell 内置,不过即中断)")
    checks["check3_causality"] = dict(
        ok=True, note="入场日>事件日硬断言;E2/E7/P3/P4 ATR ≤t−1 或 ≤事件日;"
                      "挑选/仓位决策只用信号日及之前信息(结构性)")
    max_d4 = max(r["stats"]["check4_max_diff_yuan"] for r in rows)
    checks["check4_dual_ledger"] = dict(ok=max_d4 <= 0.05,
                                        max_diff_yuan=round(max_d4, 6))
    c5 = selfcheck_determinism(bench_cells)
    checks["check5_determinism"] = c5
    if not c5["ok"]:
        log("CHECK5 FAIL —— 确定性不过,停止(README §六)")
        _HB["stop"] = True
        sys.exit(1)

    write_summary_verdict_report(rows, checks, bench,
                                 scope="基准段(20 抽样格,非审判)")

    if args.mode == "full":
        cells = enumerate_grid(degraded=(args.grid == "degraded"))
        log(f"[full] 网格 {len(cells)} 格(grid={args.grid})")
        rows_full = run_cells_parallel(cells, write=True,
                                       stage=f"full: {len(cells)} 格")
        write_summary_verdict_report(rows_full, checks, bench,
                                     scope=f"全量审判(grid={args.grid},{len(cells)} 格)")

    _HB["stop"] = True
    log(f"PORTFOLIO TRIAL DONE ({time.time() - _HB['t0']:.0f}s)")


if __name__ == "__main__":
    main()
