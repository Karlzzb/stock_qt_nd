"""真实交易回测引擎 —— issue #18 预登记规格实现（冻结版）。

事件驱动逐日模拟，把背离信号池事件转成资金曲线与交易明细。
规格逐条对应：
 1. 本金 100 万，N 仓并发，单仓等权 = 信号日(T)收盘权益 / N，现金不生息。
    A 股整手约束：股数向下取整到 100 股；不足一手且现金够则买一手，不够则放弃。
 2. 入场 T+1 开盘价；T+1 开盘价 >= 当日涨停价 → 放弃（不排队）；
    T+1 无行情 → 放弃。买入价含滑点 x1.001 与佣金（万 2.5，最低 5 元）。
 3. T+1 规则：买入当日不可卖，最早卖出 = 买入后首个交易日。
 4. 出场：E1 持有满 H 个交易日（买入日记第 1 日）收盘卖；
    E2 止盈屏障 = 入场执行价 + 2*ATR，当日 high >= 屏障 → 若 open >= 屏障按开盘价，否则按屏障价；
    E3 止损屏障 = 入场执行价 - 1*ATR，当日 low <= 屏障 → 若 open <= 屏障按开盘价，否则按屏障价；
    E4 = E2+E3 同日双触发保守取止损；E2/E3/E4 到期未触及按 E1。
    卖出日跌停（当日收盘 <= 跌停价）→ 不可卖，顺延至首个非跌停日。
    ATR = 事件日特征矩阵 ATRN × T 日收盘价（T 日已知信息，无泄漏）。
    卖价减滑点 x0.999 + 佣金 + 印花税（< 2023-08-28 收 0.1%，>= 2023-08-28 收 0.05%）。
 5. 取舍（信号日 T 收盘按空位数截取，空位 = N - T 收盘持仓数）：
    S0 随机（固定 seed）；S1 时间优先（sig_idx 升序，平局 ts_code 升序）；
    S2 超跌最深（T 日 RET20 升序）；S3 波动最大（T 日 ATRN 降序）。
    被截取的信号在 T+1 若涨停/无行情被放弃，不递补（保守）。
 6. 逐日盯市：持仓按当日收盘估值（当日无行情沿用最近已知价），出资金曲线。
 7. 防泄漏断言：买入日 > 事件日；特征行按 (ts_code, date=T) 合并；
    出场评估日 > 买入日；涨跌停表缺失日期视为无约束并计 warning。

数据源：
 日线 stock_data/daily/<ts_code>.parquet（trade_date 为日期索引）
 涨跌停 stock_data/stk_limit/YYYYMMDD.parquet（缺日 -> 无约束 + warning）
 事件宇宙 v3_pipeline/reports/pool_cleaning/excluded_events_{pool}.parquet 的 ~f_any
 特征 v3_pipeline/reports/feature_matrix/{pool}_pool_features.parquet（ATRN/RET20/sig_idx）
 交易日历 stock_data/index/000905.SH.parquet 的交易日

用法（供下游代理）：
    from strategy_engine import load_events, MarketData, run_config
    result = run_config(pool="main", selection="S0", exit_rule="E1", horizon=20,
                        n_slots=10, start="2019-01-01", end="2022-10-31", seed=42,
                        repo_root="/home/karl/repos/personal/stock_qt_nd",
                        out_dir=None, log_path=None)
    equity_df = result["equity"]   # date/cash/market_value/equity/n_positions
    trades_df = result["trades"]   # 逐笔：成交价/成本/持有天数/退出原因
    stats     = result["stats"]    # 覆盖率/资金利用率/放弃计数/warning 计数

CLI:
    python strategy_engine.py --pool main --selection S0 --exit-rule E1 --horizon 20 \
        --n-slots 10 --start 2019-01-01 --end 2022-10-31 --seed 42 \
        --out-dir v3_pipeline/reports/strategy_real_trading/runs/S0_E1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量（规格第 1/5 条）
INIT_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.00025          # 佣金双边万 2.5
COMMISSION_MIN = 5.0               # 单笔最低 5 元
SLIPPAGE = 0.001                   # 滑点单边 0.1%
STAMP_TAX_OLD = 0.001              # 印花税 0.1%（至 2023-08-27）
STAMP_TAX_NEW = 0.0005             # 印花税 0.05%（2023-08-28 起）
STAMP_TAX_SWITCH = pd.Timestamp("2023-08-28")
BOARD_LOT = 100                    # A 股整手
TP_ATR_MULT = 2.0                  # E2 止盈 = 入场价 + 2*ATR
SL_ATR_MULT = 1.0                  # E3 止损 = 入场价 - 1*ATR
PRICE_TOL = 1e-9                   # 价格比较容差

VALID_SELECTIONS = ("S0", "S1", "S2", "S3")
VALID_EXITS = ("E1", "E2", "E3", "E4")

POOLS = ("main", "backup")


def _log(log_path: str | None, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- 事件加载（规格：事件宇宙 + 特征日期==T）
def load_events(pool: str, start: str, end: str, repo_root: str,
                log_path: str | None = None) -> pd.DataFrame:
    """加载事件宇宙（~f_any）并按 (ts_code, date) 合并特征 ATRN/RET20/sig_idx。

    防泄漏：特征行以事件日 T 为合并键，特征日期恒等于 T；任何缺失视为数据错误直接拒绝。
    返回列：ts_code, event_date, ATRN, RET20, sig_idx，按 event_date 排序。
    """
    assert pool in POOLS, f"unknown pool {pool}"
    ex_path = os.path.join(repo_root, "v3_pipeline/reports/pool_cleaning",
                           f"excluded_events_{pool}.parquet")
    fm_path = os.path.join(repo_root, "v3_pipeline/reports/feature_matrix",
                           f"{pool}_pool_features.parquet")
    ex = pd.read_parquet(ex_path, columns=["ts_code", "date", "f_any"])
    ev = ex.loc[~ex["f_any"], ["ts_code", "date"]].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    ev = ev[(ev["date"] >= start_ts) & (ev["date"] <= end_ts)]

    fm = pd.read_parquet(fm_path, columns=["ts_code", "date", "ATRN", "RET20", "sig_idx"])
    fm["date"] = pd.to_datetime(fm["date"])
    m = ev.merge(fm, on=["ts_code", "date"], how="left", validate="one_to_one")
    n_missing = int(m["ATRN"].isna().sum() + m["RET20"].isna().sum() + m["sig_idx"].isna().sum())
    assert m[["ATRN", "RET20", "sig_idx"]].notna().all().all(), \
        f"feature join incomplete for {n_missing} events (feature date must equal event date)"
    m = m.rename(columns={"date": "event_date"})
    m = m.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    _log(log_path, f"load_events pool={pool} window=[{start},{end}] events={len(m)} "
                   f"unique_stocks={m['ts_code'].nunique()}")
    return m


# ---------------------------------------------------------------- 行情数据容器
@dataclass
class MarketData:
    """内存行情容器。daily: ts_code -> DataFrame(索引=trade_date, 列 open/high/low/close)。

    limits: trade_date(Timestamp) -> DataFrame(索引=ts_code, 列 up_limit/down_limit)。
    calendar: 排序后的交易日列表。
    limit_missing_dates: 涨跌停表缺失日期计数（warning）。
    """
    daily: dict[str, pd.DataFrame]
    limits: dict[pd.Timestamp, pd.DataFrame]
    calendar: list[pd.Timestamp]
    limit_missing_dates: int = 0


def load_market_data(ts_codes: list[str], start: str, end: str, repo_root: str,
                     calendar_start: str | None = None,
                     log_path: str | None = None) -> MarketData:
    """加载日线、涨跌停表与交易日历。每 500 只股票打一行心跳。"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cal_start_ts = pd.Timestamp(calendar_start) if calendar_start else start_ts

    idx_path = os.path.join(repo_root, "stock_data/index/000905.SH.parquet")
    idx = pd.read_parquet(idx_path, columns=["trade_date"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    calendar = sorted(d for d in idx["trade_date"].unique()
                      if cal_start_ts <= d <= end_ts)
    calendar = [pd.Timestamp(d) for d in calendar]
    _log(log_path, f"calendar loaded: {len(calendar)} trading days "
                   f"[{calendar[0].date()} .. {calendar[-1].date()}]")

    # 日线向前多取 60 个自然日，保证 T 日（可能略早于窗口起点的入场参考）与延期卖出有余量
    load_start = start_ts - pd.Timedelta(days=60)
    load_end = end_ts + pd.Timedelta(days=60)
    daily: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for i, code in enumerate(sorted(set(ts_codes))):
        p = os.path.join(repo_root, "stock_data/daily", f"{code}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["trade_date", "open", "high", "low", "close"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= load_start) & (df["trade_date"] <= load_end)]
        if df.empty:
            continue
        daily[code] = df.set_index("trade_date").sort_index()
        if (i + 1) % 500 == 0:
            _log(log_path, f"heartbeat: loaded {i + 1}/{len(set(ts_codes))} daily files "
                           f"({time.time() - t0:.1f}s)")
    _log(log_path, f"daily loaded: {len(daily)} stocks ({time.time() - t0:.1f}s)")

    # 涨跌停表：按日历日逐文件加载；缺文件 -> 该日无涨跌停约束（warning 计数）
    limits: dict[pd.Timestamp, pd.DataFrame] = {}
    limit_dir = os.path.join(repo_root, "stock_data/stk_limit")
    missing = 0
    for d in calendar:
        fp = os.path.join(limit_dir, d.strftime("%Y%m%d") + ".parquet")
        if not os.path.exists(fp):
            missing += 1
            continue
        lf = pd.read_parquet(fp)
        limits[d] = lf.set_index("ts_code")[["up_limit", "down_limit"]]
    if missing:
        _log(log_path, f"WARNING: stk_limit missing for {missing}/{len(calendar)} days "
                       f"-> treated as no limit constraint")
    return MarketData(daily=daily, limits=limits, calendar=calendar,
                      limit_missing_dates=missing)


# ---------------------------------------------------------------- 取舍规则（规格第 6 条）
def select_signals(day_signals: pd.DataFrame, n_free: int, rule: str,
                   rng: np.random.RandomState) -> pd.DataFrame:
    """当日信号数 > 空位数时按取舍规则截取前 n_free 个。只用 T 日及以前信息。"""
    if n_free <= 0 or day_signals.empty:
        return day_signals.iloc[0:0]
    df = day_signals
    if rule == "S0":
        order = rng.permutation(len(df))
        df = df.iloc[order]
    elif rule == "S1":
        df = df.sort_values(["sig_idx", "ts_code"], kind="mergesort")
    elif rule == "S2":
        df = df.sort_values(["RET20", "ts_code"], ascending=[True, True], kind="mergesort")
    elif rule == "S3":
        df = df.sort_values(["ATRN", "ts_code"], ascending=[False, True], kind="mergesort")
    else:
        raise ValueError(f"unknown selection rule {rule}")
    return df.head(n_free)


# ---------------------------------------------------------------- 持仓
@dataclass
class Position:
    ts_code: str
    event_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float          # 含滑点执行价
    shares: int
    entry_commission: float
    atr: float | None           # None -> 无屏障（退化为 E1）
    tp_barrier: float | None
    sl_barrier: float | None
    last_known_price: float     # 盯市沿用价
    deferred_days: int = 0      # 跌停顺延天数


# ---------------------------------------------------------------- 成本函数
def buy_cost(shares: int, exec_price: float) -> float:
    return max(COMMISSION_MIN, shares * exec_price * COMMISSION_RATE)


def sell_costs(shares: int, exec_price: float, date: pd.Timestamp) -> tuple[float, float]:
    gross = shares * exec_price
    commission = max(COMMISSION_MIN, gross * COMMISSION_RATE)
    stamp = gross * (STAMP_TAX_OLD if date < STAMP_TAX_SWITCH else STAMP_TAX_NEW)
    return commission, stamp


# ---------------------------------------------------------------- 主回测循环
def run_backtest(events: pd.DataFrame, md: MarketData, n_slots: int,
                 selection: str, exit_rule: str, horizon: int,
                 seed: int = 42, init_capital: float = INIT_CAPITAL,
                 log_path: str | None = None) -> dict:
    """事件驱动逐日模拟。返回 equity/trades/stats。

    events 必须含列 ts_code/event_date/ATRN/RET20/sig_idx，且 event_date 在 md.calendar 内。
    """
    assert selection in VALID_SELECTIONS and exit_rule in VALID_EXITS
    assert n_slots >= 1 and horizon >= 2
    events = events.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    by_date: dict[pd.Timestamp, pd.DataFrame] = {
        d: g for d, g in events.groupby("event_date")
    }
    pending: dict[pd.Timestamp, pd.DataFrame] = {}   # 入场日 -> 已截取信号
    cal_index = {d: i for i, d in enumerate(md.calendar)}

    rng = np.random.RandomState(seed)
    cash = init_capital
    positions: list[Position] = []
    trades: list[dict] = []
    equity_rows: list[dict] = []
    stats = dict(
        total_signals=len(events), entered=0,
        dropped_slot_full=0, dropped_limitup=0, dropped_no_quote=0,
        dropped_cash=0, dropped_no_close_T=0,
        limit_missing_days=md.limit_missing_dates,
        deferred_exits=0, atr_missing_fallback_E1=0,
    )

    for di, day in enumerate(md.calendar):
        # ---- 0. 信号日（T=day）收盘：按空位截取信号，挂到下一交易日入场 ----
        if day in by_date:
            n_free = n_slots - len(positions)
            day_sigs = by_date[day]
            if n_free < len(day_sigs):
                stats["dropped_slot_full"] += len(day_sigs) - max(n_free, 0)
            chosen = select_signals(day_sigs, n_free, selection, rng)
            if not chosen.empty and di + 1 < len(md.calendar):
                nxt = md.calendar[di + 1]
                if nxt in pending:
                    pending[nxt] = pd.concat([pending[nxt], chosen])
                else:
                    pending[nxt] = chosen

        # 当日（T+1 入场日）收盘权益快照 = 昨收权益（入场预算基准，规格第 1 条）
        if di > 0:
            prev_day = md.calendar[di - 1]
            prev_mv = 0.0
            for p in positions:
                row = md.daily.get(p.ts_code)
                px = None
                if row is not None and prev_day in row.index:
                    px = float(row.loc[prev_day, "close"])
                prev_mv += p.shares * (px if px is not None else p.last_known_price)
            equity_prev_close = cash + prev_mv
        else:
            equity_prev_close = cash

        # ---- 1. 入场（T+1 开盘；先于当日卖出，因卖出现金当日不可用）----
        todays = pending.pop(day, None)
        if todays is not None and len(positions) < n_slots:
            limit_tbl = md.limits.get(day)
            for sig in todays.itertuples():
                if len(positions) >= n_slots:
                    stats["dropped_slot_full"] += 1  # 防御：截取后仍满员
                    continue
                row = md.daily.get(sig.ts_code)
                bar = row.loc[day] if (row is not None and day in row.index) else None
                if bar is None:
                    stats["dropped_no_quote"] += 1
                    continue
                open_px = float(bar["open"])
                if limit_tbl is not None and sig.ts_code in limit_tbl.index:
                    up_lim = float(limit_tbl.loc[sig.ts_code, "up_limit"])
                    if open_px >= up_lim - PRICE_TOL:
                        stats["dropped_limitup"] += 1
                        continue
                # T 日收盘价（ATR 基准，T 日已知信息）
                t_close = float(row.loc[sig.event_date, "close"]) \
                    if sig.event_date in row.index else None
                if t_close is None:
                    stats["dropped_no_close_T"] += 1
                    continue
                # 防泄漏断言：买入日必须严格晚于事件日
                assert day > sig.event_date, \
                    f"leakage guard: entry {day} not after event {sig.event_date}"
                exec_price = open_px * (1.0 + SLIPPAGE)
                budget = equity_prev_close / n_slots
                shares = int(budget / exec_price / BOARD_LOT) * BOARD_LOT
                if shares < BOARD_LOT:
                    shares = BOARD_LOT  # 不足一手买一手（若现金够）
                comm = buy_cost(shares, exec_price)
                while shares > 0 and shares * exec_price + comm > cash + 1e-6:
                    shares -= BOARD_LOT
                    comm = buy_cost(shares, exec_price) if shares > 0 else 0.0
                if shares <= 0:
                    stats["dropped_cash"] += 1
                    continue
                cash -= shares * exec_price + comm
                atr = float(sig.ATRN) * t_close
                if not np.isfinite(atr) or atr <= 0:
                    atr = None
                    stats["atr_missing_fallback_E1"] += 1
                pos = Position(
                    ts_code=sig.ts_code, event_date=sig.event_date, entry_date=day,
                    entry_price=exec_price, shares=shares, entry_commission=comm,
                    atr=atr,
                    tp_barrier=(exec_price + TP_ATR_MULT * atr) if atr else None,
                    sl_barrier=(exec_price - SL_ATR_MULT * atr) if atr else None,
                    last_known_price=float(bar["close"]),
                )
                positions.append(pos)
                stats["entered"] += 1

        # ---- 2. 出场（T+1 规则：entry_date < day 才可卖）----
        limit_tbl = md.limits.get(day)
        still_open: list[Position] = []
        for p in positions:
            row = md.daily.get(p.ts_code)
            bar = row.loc[day] if (row is not None and day in row.index) else None
            if bar is None:
                still_open.append(p)  # 当日无行情：沿用旧价，不触发任何判断
                continue
            p.last_known_price = float(bar["close"])
            if day <= p.entry_date:
                still_open.append(p)  # 买入当日不可卖
                continue
            held_days = cal_index[day] - cal_index[p.entry_date] + 1  # 买入日记第 1 日
            o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))

            raw_exit: float | None = None
            reason: str | None = None
            tp_hit = p.tp_barrier is not None and h >= p.tp_barrier - PRICE_TOL
            sl_hit = p.sl_barrier is not None and l <= p.sl_barrier + PRICE_TOL
            if exit_rule in ("E2", "E4") and tp_hit and not (
                    exit_rule == "E4" and sl_hit):
                raw_exit = o if o >= p.tp_barrier - PRICE_TOL else p.tp_barrier
                reason = "E2"
            elif exit_rule in ("E3", "E4") and sl_hit:
                raw_exit = o if o <= p.sl_barrier + PRICE_TOL else p.sl_barrier
                reason = "E3"
            elif held_days >= horizon:
                raw_exit = c
                reason = "E1"

            if raw_exit is None:
                still_open.append(p)
                continue
            # 跌停不可卖：保守口径 收盘 <= 跌停价 -> 顺延
            if limit_tbl is not None and p.ts_code in limit_tbl.index:
                dn = float(limit_tbl.loc[p.ts_code, "down_limit"])
                if c <= dn + PRICE_TOL:
                    p.deferred_days += 1
                    stats["deferred_exits"] += 1
                    still_open.append(p)
                    continue
            exec_sell = raw_exit * (1.0 - SLIPPAGE)
            comm, stamp = sell_costs(p.shares, exec_sell, day)
            cash += p.shares * exec_sell - comm - stamp
            # 毛利按执行价差（含滑点）计，费用单列，口径透明
            gross_pnl = p.shares * (exec_sell - p.entry_price)
            total_cost = p.entry_commission + comm + stamp
            net_pnl = gross_pnl - total_cost
            trades.append(dict(
                ts_code=p.ts_code, event_date=p.event_date, entry_date=p.entry_date,
                entry_price=p.entry_price, shares=p.shares,
                entry_commission=p.entry_commission,
                exit_date=day, exit_reason=reason, exit_raw_price=raw_exit,
                exit_exec_price=exec_sell, exit_commission=comm, stamp_tax=stamp,
                gross_pnl=gross_pnl, total_cost=total_cost, net_pnl=net_pnl,
                ret=net_pnl / (p.shares * p.entry_price + p.entry_commission),
                held_days=held_days, deferred_days=p.deferred_days,
                atr=p.atr, tp_barrier=p.tp_barrier, sl_barrier=p.sl_barrier,
            ))
        positions = still_open

        # ---- 3. 逐日盯市（收盘估值）----
        mv = 0.0
        for p in positions:
            mv += p.shares * p.last_known_price
        equity_rows.append(dict(date=day, cash=cash, market_value=mv,
                                equity=cash + mv, n_positions=len(positions)))

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    if not equity_df.empty:
        equity_df["utilization"] = np.where(equity_df["equity"] > 0,
                                            equity_df["market_value"] / equity_df["equity"], 0.0)
        stats["capital_utilization"] = float(equity_df["utilization"].mean())
        stats["coverage"] = stats["entered"] / stats["total_signals"] if stats["total_signals"] else 0.0
        stats["final_equity"] = float(equity_df["equity"].iloc[-1])
    _log(log_path, f"run_backtest done: signals={stats['total_signals']} entered={stats['entered']} "
                   f"coverage={stats.get('coverage', 0):.4f} trades={len(trades_df)} "
                   f"final_equity={stats.get('final_equity', float('nan')):.2f}")
    return {"equity": equity_df, "trades": trades_df, "stats": stats}


# ---------------------------------------------------------------- 顶层配置入口（供下游代理调用）
def run_config(pool: str, selection: str, exit_rule: str, horizon: int, n_slots: int,
               start: str, end: str, seed: int = 42,
               repo_root: str = "/home/karl/repos/personal/stock_qt_nd",
               out_dir: str | None = None,
               log_path: str | None = None) -> dict:
    """跑一个 池×取舍×出场×N×H×切分段 配置，返回 {equity, trades, stats, config}。

    out_dir 非空时落盘 equity_curve.parquet / trades.parquet / stats.json。
    """
    t0 = time.time()
    _log(log_path, f"run_config pool={pool} sel={selection} exit={exit_rule} H={horizon} "
                   f"N={n_slots} [{start}..{end}] seed={seed}")
    events = load_events(pool, start, end, repo_root, log_path)
    md = load_market_data(events["ts_code"].unique().tolist(), start, end, repo_root,
                          log_path=log_path)
    # 事件可能落在窗口首日、T+1 入场需要后续交易日；calendar 已含 [start..end]
    events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
    result = run_backtest(events, md, n_slots=n_slots, selection=selection,
                          exit_rule=exit_rule, horizon=horizon, seed=seed,
                          log_path=log_path)
    result["config"] = dict(pool=pool, selection=selection, exit_rule=exit_rule,
                            horizon=horizon, n_slots=n_slots, start=start, end=end,
                            seed=seed)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        result["equity"].to_parquet(os.path.join(out_dir, "equity_curve.parquet"))
        result["trades"].to_parquet(os.path.join(out_dir, "trades.parquet"))
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({**result["stats"], **result["config"]}, f, ensure_ascii=False,
                      indent=2, default=str)
    _log(log_path, f"run_config finished in {time.time() - t0:.1f}s")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="真实交易回测引擎（issue #18）")
    ap.add_argument("--pool", choices=POOLS, required=True)
    ap.add_argument("--selection", choices=VALID_SELECTIONS, required=True)
    ap.add_argument("--exit-rule", choices=VALID_EXITS, required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-slots", type=int, default=10)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_config(pool=args.pool, selection=args.selection, exit_rule=args.exit_rule,
               horizon=args.horizon, n_slots=args.n_slots, start=args.start, end=args.end,
               seed=args.seed, repo_root=repo_root, out_dir=args.out_dir,
               log_path=args.log)


if __name__ == "__main__":
    main()
