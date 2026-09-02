"""真实交易回测引擎 v2 —— issue #19 池级敞口择时覆盖层（冻结候选）。

在 #18 冻结的 strategy_engine（下称 v1）之上叠加择时层，v1 文件一字不动。
择时规格逐条对应 issue #19 预登记：

 1. 择时信号源：stock_data/index/000905.SH.parquet 收盘价。
    t 日敞口决策只用 trade_date <= t-1 的指数收盘信息；
    引擎内埋硬断言：敞口序列对 t 的取值依赖的最大指数日期 < t。
 2. 择时规则（均线/最高点/分位窗口同样只用 <= t-1 数据）：
    T1 趋势200：t-1 收盘 > 其 200 日均线 -> 全敞口（1.0），否则 0。
    T2 趋势60：同上，60 日均线。
    T3 回撤阶梯：t-1 收盘距其 250 日最高收盘的回撤 <10% -> 全；10%~20%（含端点）-> 半（0.5）；>20% -> 0。
    T4 波动闸门：20 日已实现波动 = 日收益（close.pct_change）20 日滚动样本 std（ddof=1）× sqrt(252)；
      t-1 的该波动 > 其过去 250 日（含 t-1 当日，共 250 个波动值）的 90 分位 -> 半，否则全。
    降级规则：窗口历史不足（T1<200、T2<60、T3<250、T4<270 个 <=t-1 收盘）时视为全敞口，
      记 warning（stats["timing_fallback_days"] 计数 + 日志）。
 3. 敞口 -> 槽位：当日可用槽位 = round_half_up(N × 敞口)（即 floor(x + 0.5)，避免银行家舍入歧义）。
    G1 软闸门：只拦新入场（信号截取与 T+1 入场两处都按当日槽位卡），
      已有持仓按原出场规则自然退出，持仓数可暂时超过槽位。
    G2 硬闸门：G1 之上再加强制退出——每日（以 t-1 信息定 t 日行动）若持仓数 > 当日槽位，
      超出部分在 t 日按收盘价强制退出（exit_reason="G2_forced"），遵守 T+1 不可卖与跌停顺延
      （当日不可卖的持仓留在仓内，次日继续按同一规则重检；敞口回升则强制意图自然失效）。
      选择退出哪些持仓：持有浮盈最低者优先（浮盈 = last_known_price / entry_price - 1，
      平局按 ts_code 升序保证确定性），文档在此写明。
    注：G2 强制退出不可能落在买入当日——降档日持仓数 >= 新槽位时新入场已被闸门拦截，
      买入当日强制退出的分支仅为防御性保留。
 4. 引擎 v2 API：run_config 同 v1 签名外加 timing=None|"T1"|"T2"|"T3"|"T4"、gate="G1"|"G2"。
    timing=None 时直接委托 v1 的 run_config，行为与 v1 完全一致（回归断言以此复现 #18 归档数值）。

其余口径（本金/成本/滑点/涨跌停/ATR 屏障/逐日盯市/防泄漏断言）全部继承 v1，不重述。
单仓预算仍为 信号日 T 收盘权益 / N（N 为名义槽位数，不随敞口变化）。

用法（供下游代理）：
    from strategy_engine_v2 import run_config, compute_exposure, load_index_close
    result = run_config(pool="backup", selection="S3", exit_rule="E1", horizon=20,
                        n_slots=10, start="2019-01-01", end="2022-10-31", seed=42,
                        timing="T1", gate="G1")
    result["equity"] / result["trades"] / result["stats"] 同 v1；
    result["exposure"] 为逐日敞口表（date/exposure/max_src_date/fallback/slots）。

CLI:
    python strategy_engine_v2.py --pool backup --selection S3 --exit-rule E1 --horizon 20 \
        --n-slots 10 --start 2019-01-01 --end 2022-10-31 --seed 42 --timing T1 --gate G1 \
        --out-dir v3_pipeline/reports/exposure_timing/runs/val_backup_S3_E1_N10_T1_G1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy_engine as se  # noqa: E402  # v1 冻结引擎，只复用不修改

VALID_TIMINGS = ("T1", "T2", "T3", "T4")
VALID_GATES = ("G1", "G2")

INDEX_PATH = "stock_data/index/000905.SH.parquet"
VOL_WINDOW = 20               # T4 已实现波动窗口（交易日）
VOL_ANNUALIZE = float(np.sqrt(252.0))
T4_Q_WINDOW = 250             # T4 分位回望窗口（波动值个数）
T4_QUANTILE = 0.9
MIN_HISTORY = {"T1": 200, "T2": 60, "T3": 250, "T4": T4_Q_WINDOW + VOL_WINDOW}


# ---------------------------------------------------------------- 槽位换算（规格第 3 条）
def round_slots(n_slots: int, exposure: float) -> int:
    """round_half_up(N × 敞口)：floor(x + 0.5)。10×0.5=5，5×0.5=3。"""
    return int(np.floor(n_slots * exposure + 0.5))


# ---------------------------------------------------------------- 指数收盘加载（择时信号源）
def load_index_close(repo_root: str) -> pd.Series:
    """000905.SH 收盘价，索引=trade_date（Timestamp），升序。"""
    idx = pd.read_parquet(os.path.join(repo_root, INDEX_PATH),
                          columns=["trade_date", "close"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    s = idx.set_index("trade_date")["close"].sort_index()
    return s


# ---------------------------------------------------------------- 敞口序列计算（规格第 1/2 条，无泄漏硬断言）
def compute_exposure(calendar: list[pd.Timestamp], index_close: pd.Series,
                     rule: str, log_path: str | None = None) -> pd.DataFrame:
    """逐交易日敞口。t 日取值只用 trade_date < t 的指数收盘（<= t-1 信息）。

    返回 DataFrame：date / exposure(1.0|0.5|0.0) / max_src_date(取值依赖的最大指数日期)
    / fallback(窗口不足降级为全敞口) / slots 不在此列（依赖 N，由引擎换算）。
    硬断言：每一天 max_src_date < date。
    """
    assert rule in VALID_TIMINGS, f"unknown timing rule {rule}"
    closes = index_close.sort_index()
    dates = closes.index.to_numpy()          # datetime64[ns]，升序
    vals = closes.to_numpy(dtype=float)
    need = MIN_HISTORY[rule]

    vol: np.ndarray | None = None
    if rule == "T4":
        ret = pd.Series(vals).pct_change()
        vol = (ret.rolling(VOL_WINDOW).std(ddof=1) * VOL_ANNUALIZE).to_numpy()

    rows: list[dict] = []
    n_fallback = 0
    for t in calendar:
        t64 = np.datetime64(t)
        pos = int(np.searchsorted(dates, t64, side="left")) - 1  # 最后一个 < t 的指数日期下标
        if pos < 0 or pos + 1 < need:
            n_fallback += 1
            rows.append(dict(date=t, exposure=1.0,
                             max_src_date=pd.Timestamp(dates[pos]) if pos >= 0 else pd.NaT,
                             fallback=True))
            continue
        hist = vals[:pos + 1]                    # 全部 <= t-1 的收盘
        if rule in ("T1", "T2"):
            w = 200 if rule == "T1" else 60
            expo = 1.0 if hist[-1] > hist[-w:].mean() else 0.0
        elif rule == "T3":
            dd = 1.0 - hist[-1] / hist[-250:].max()
            expo = 1.0 if dd < 0.10 else (0.5 if dd <= 0.20 else 0.0)
        else:  # T4
            window = vol[pos - T4_Q_WINDOW + 1: pos + 1]   # 250 个波动值，含 t-1 当日
            expo = 0.5 if vol[pos] > np.quantile(window, T4_QUANTILE) else 1.0
        src = pd.Timestamp(dates[pos])
        assert src < t, f"leakage guard: exposure({t.date()}) used index date {src.date()}"
        rows.append(dict(date=t, exposure=expo, max_src_date=src, fallback=False))

    if n_fallback:
        se._log(log_path, f"WARNING: timing {rule} insufficient index history on "
                          f"{n_fallback}/{len(calendar)} days -> treated as full exposure")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 带择时的主回测循环
def run_backtest_timed(events: pd.DataFrame, md: se.MarketData,
                       exposure: pd.DataFrame, n_slots: int,
                       selection: str, exit_rule: str, horizon: int,
                       gate: str, seed: int = 42,
                       init_capital: float = se.INIT_CAPITAL,
                       log_path: str | None = None) -> dict:
    """v1 主循环 + 择时层。exposure 须含列 date/exposure/max_src_date/fallback。

    与 v1 的差异（其余逐行一致）：
      - 每日槽位 slots_t = round_slots(n_slots, exposure_t) 替代固定 n_slots；
      - 信号截取空位按入场日（T+1）槽位计（T 收盘时 T+1 槽位已由 <=T 信息确定）；
      - T+1 入场执行时按当日槽位二次拦截（计数 dropped_timing_gate）；
      - G2：正常出场判定后，若持仓数 > slots_t，浮盈最低者优先按收盘价强制退出，
        遵守 T+1 与跌停顺延（计数 forced_exits / forced_deferred）。
    """
    assert selection in se.VALID_SELECTIONS and exit_rule in se.VALID_EXITS
    assert gate in VALID_GATES, f"unknown gate {gate}"
    assert n_slots >= 1 and horizon >= 2
    expo = exposure.set_index("date")
    # 无泄漏硬断言（规格第 1 条）：敞口取值依赖的最大指数日期 < 决策日
    for t, r in expo.iterrows():
        assert pd.isna(r["max_src_date"]) or r["max_src_date"] < t, \
            f"leakage guard: exposure({t}) depends on {r['max_src_date']}"
    missing_days = [d for d in md.calendar if d not in expo.index]
    assert not missing_days, f"exposure series missing {len(missing_days)} calendar days"

    events = events.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    by_date: dict[pd.Timestamp, pd.DataFrame] = {
        d: g for d, g in events.groupby("event_date")
    }
    pending: dict[pd.Timestamp, pd.DataFrame] = {}
    cal_index = {d: i for i, d in enumerate(md.calendar)}

    rng = np.random.RandomState(seed)
    cash = init_capital
    positions: list[se.Position] = []
    trades: list[dict] = []
    equity_rows: list[dict] = []
    stats = dict(
        total_signals=len(events), entered=0,
        dropped_slot_full=0, dropped_limitup=0, dropped_no_quote=0,
        dropped_cash=0, dropped_no_close_T=0,
        limit_missing_days=md.limit_missing_dates,
        deferred_exits=0, atr_missing_fallback_E1=0,
        dropped_timing_gate=0, forced_exits=0, forced_deferred=0,
        timing_fallback_days=int(expo["fallback"].sum()),
    )

    def slots_of(day: pd.Timestamp) -> int:
        return round_slots(n_slots, float(expo.loc[day, "exposure"]))

    for di, day in enumerate(md.calendar):
        slots_t = slots_of(day)
        # ---- 0. 信号日（T=day）收盘：按入场日槽位截取信号，挂到下一交易日入场 ----
        if day in by_date:
            slots_next = slots_of(md.calendar[di + 1]) if di + 1 < len(md.calendar) else 0
            n_free = slots_next - len(positions)
            day_sigs = by_date[day]
            if n_free < len(day_sigs):
                stats["dropped_slot_full"] += len(day_sigs) - max(n_free, 0)
            chosen = se.select_signals(day_sigs, n_free, selection, rng)
            if not chosen.empty and di + 1 < len(md.calendar):
                nxt = md.calendar[di + 1]
                if nxt in pending:
                    pending[nxt] = pd.concat([pending[nxt], chosen])
                else:
                    pending[nxt] = chosen

        # 当日（T+1 入场日）收盘权益快照 = 昨收权益（入场预算基准，同 v1）
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

        # ---- 1. 入场（T+1 开盘；闸门按当日槽位拦截新入场）----
        todays = pending.pop(day, None)
        if todays is not None and len(positions) < slots_t:
            limit_tbl = md.limits.get(day)
            for sig in todays.itertuples():
                if len(positions) >= slots_t:
                    stats["dropped_timing_gate"] += 1  # 择时闸门拦截新入场
                    continue
                row = md.daily.get(sig.ts_code)
                bar = row.loc[day] if (row is not None and day in row.index) else None
                if bar is None:
                    stats["dropped_no_quote"] += 1
                    continue
                open_px = float(bar["open"])
                if limit_tbl is not None and sig.ts_code in limit_tbl.index:
                    up_lim = float(limit_tbl.loc[sig.ts_code, "up_limit"])
                    if open_px >= up_lim - se.PRICE_TOL:
                        stats["dropped_limitup"] += 1
                        continue
                t_close = float(row.loc[sig.event_date, "close"]) \
                    if sig.event_date in row.index else None
                if t_close is None:
                    stats["dropped_no_close_T"] += 1
                    continue
                assert day > sig.event_date, \
                    f"leakage guard: entry {day} not after event {sig.event_date}"
                exec_price = open_px * (1.0 + se.SLIPPAGE)
                budget = equity_prev_close / n_slots   # 单仓预算恒为权益/N，不随敞口变
                shares = int(budget / exec_price / se.BOARD_LOT) * se.BOARD_LOT
                if shares < se.BOARD_LOT:
                    shares = se.BOARD_LOT
                comm = se.buy_cost(shares, exec_price)
                while shares > 0 and shares * exec_price + comm > cash + 1e-6:
                    shares -= se.BOARD_LOT
                    comm = se.buy_cost(shares, exec_price) if shares > 0 else 0.0
                if shares <= 0:
                    stats["dropped_cash"] += 1
                    continue
                cash -= shares * exec_price + comm
                atr = float(sig.ATRN) * t_close
                if not np.isfinite(atr) or atr <= 0:
                    atr = None
                    stats["atr_missing_fallback_E1"] += 1
                pos = se.Position(
                    ts_code=sig.ts_code, event_date=sig.event_date, entry_date=day,
                    entry_price=exec_price, shares=shares, entry_commission=comm,
                    atr=atr,
                    tp_barrier=(exec_price + se.TP_ATR_MULT * atr) if atr else None,
                    sl_barrier=(exec_price - se.SL_ATR_MULT * atr) if atr else None,
                    last_known_price=float(bar["close"]),
                )
                positions.append(pos)
                stats["entered"] += 1
        elif todays is not None:
            stats["dropped_timing_gate"] += len(todays)  # 槽位为 0 或满仓，整批拦截

        # ---- 2. 出场（T+1 规则与 v1 完全一致）----
        limit_tbl = md.limits.get(day)
        still_open: list[se.Position] = []
        for p in positions:
            row = md.daily.get(p.ts_code)
            bar = row.loc[day] if (row is not None and day in row.index) else None
            if bar is None:
                still_open.append(p)
                continue
            p.last_known_price = float(bar["close"])
            if day <= p.entry_date:
                still_open.append(p)
                continue
            held_days = cal_index[day] - cal_index[p.entry_date] + 1
            o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))

            raw_exit: float | None = None
            reason: str | None = None
            tp_hit = p.tp_barrier is not None and h >= p.tp_barrier - se.PRICE_TOL
            sl_hit = p.sl_barrier is not None and l <= p.sl_barrier + se.PRICE_TOL
            if exit_rule in ("E2", "E4") and tp_hit and not (
                    exit_rule == "E4" and sl_hit):
                raw_exit = o if o >= p.tp_barrier - se.PRICE_TOL else p.tp_barrier
                reason = "E2"
            elif exit_rule in ("E3", "E4") and sl_hit:
                raw_exit = o if o <= p.sl_barrier + se.PRICE_TOL else p.sl_barrier
                reason = "E3"
            elif held_days >= horizon:
                raw_exit = c
                reason = "E1"

            if raw_exit is None:
                still_open.append(p)
                continue
            if limit_tbl is not None and p.ts_code in limit_tbl.index:
                dn = float(limit_tbl.loc[p.ts_code, "down_limit"])
                if c <= dn + se.PRICE_TOL:
                    p.deferred_days += 1
                    stats["deferred_exits"] += 1
                    still_open.append(p)
                    continue
            exec_sell = raw_exit * (1.0 - se.SLIPPAGE)
            comm, stamp = se.sell_costs(p.shares, exec_sell, day)
            cash += p.shares * exec_sell - comm - stamp
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

        # ---- 2b. G2 硬闸门：持仓数 > 当日槽位 -> 浮盈最低者优先强制退出 ----
        if gate == "G2":
            excess = len(positions) - slots_t
            if excess > 0:
                cands = sorted(
                    positions,
                    key=lambda p: (p.last_known_price / p.entry_price - 1.0, p.ts_code),
                )
                removed_ids: set[int] = set()
                for p in cands:
                    if excess <= 0:
                        break
                    if day <= p.entry_date:
                        stats["forced_deferred"] += 1   # T+1 不可卖（防御分支）
                        continue
                    row = md.daily.get(p.ts_code)
                    bar = row.loc[day] if (row is not None and day in row.index) else None
                    if bar is None:
                        stats["forced_deferred"] += 1   # 当日无行情无法卖，次日重检
                        continue
                    c = float(bar["close"])
                    if limit_tbl is not None and p.ts_code in limit_tbl.index:
                        dn = float(limit_tbl.loc[p.ts_code, "down_limit"])
                        if c <= dn + se.PRICE_TOL:
                            p.deferred_days += 1
                            stats["forced_deferred"] += 1  # 跌停顺延，次日重检
                            continue
                    exec_sell = c * (1.0 - se.SLIPPAGE)
                    comm, stamp = se.sell_costs(p.shares, exec_sell, day)
                    cash += p.shares * exec_sell - comm - stamp
                    gross_pnl = p.shares * (exec_sell - p.entry_price)
                    total_cost = p.entry_commission + comm + stamp
                    net_pnl = gross_pnl - total_cost
                    held_days = cal_index[day] - cal_index[p.entry_date] + 1
                    trades.append(dict(
                        ts_code=p.ts_code, event_date=p.event_date,
                        entry_date=p.entry_date, entry_price=p.entry_price,
                        shares=p.shares, entry_commission=p.entry_commission,
                        exit_date=day, exit_reason="G2_forced", exit_raw_price=c,
                        exit_exec_price=exec_sell, exit_commission=comm, stamp_tax=stamp,
                        gross_pnl=gross_pnl, total_cost=total_cost, net_pnl=net_pnl,
                        ret=net_pnl / (p.shares * p.entry_price + p.entry_commission),
                        held_days=held_days, deferred_days=p.deferred_days,
                        atr=p.atr, tp_barrier=p.tp_barrier, sl_barrier=p.sl_barrier,
                    ))
                    removed_ids.add(id(p))
                    excess -= 1
                    stats["forced_exits"] += 1
                if removed_ids:
                    positions = [p for p in positions if id(p) not in removed_ids]

        # ---- 3. 逐日盯市（同 v1）----
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
    se._log(log_path, f"run_backtest_timed done: gate={gate} signals={stats['total_signals']} "
                      f"entered={stats['entered']} gate_blocked={stats['dropped_timing_gate']} "
                      f"forced_exits={stats['forced_exits']} forced_deferred={stats['forced_deferred']} "
                      f"fallback_days={stats['timing_fallback_days']} trades={len(trades_df)} "
                      f"final_equity={stats.get('final_equity', float('nan')):.2f}")
    return {"equity": equity_df, "trades": trades_df, "stats": stats}


# ---------------------------------------------------------------- 顶层配置入口（v1 签名 + timing/gate）
def run_config(pool: str, selection: str, exit_rule: str, horizon: int, n_slots: int,
               start: str, end: str, seed: int = 42,
               repo_root: str = "/home/karl/repos/personal/stock_qt_nd",
               out_dir: str | None = None,
               log_path: str | None = None,
               timing: str | None = None,
               gate: str = "G1") -> dict:
    """跑一个配置。timing=None 时委托 v1 run_config，行为与 v1 完全一致；
    timing 为 T1..T4 时叠加择时层（gate=G1 软 / G2 硬），结果附 exposure 表。
    """
    if timing is None:
        result = se.run_config(pool=pool, selection=selection, exit_rule=exit_rule,
                               horizon=horizon, n_slots=n_slots, start=start, end=end,
                               seed=seed, repo_root=repo_root, out_dir=out_dir,
                               log_path=log_path)
        # 仅内存中补充配置键；落盘内容（out_dir 由 v1 写）与 v1 逐字节一致
        result["config"]["timing"] = None
        result["config"]["gate"] = gate
        return result

    assert timing in VALID_TIMINGS, f"unknown timing {timing}"
    assert gate in VALID_GATES, f"unknown gate {gate}"
    t0 = time.time()
    se._log(log_path, f"run_config v2 pool={pool} sel={selection} exit={exit_rule} "
                      f"H={horizon} N={n_slots} [{start}..{end}] seed={seed} "
                      f"timing={timing} gate={gate}")
    events = se.load_events(pool, start, end, repo_root, log_path)
    md = se.load_market_data(events["ts_code"].unique().tolist(), start, end, repo_root,
                             log_path=log_path)
    events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
    index_close = load_index_close(repo_root)
    exposure = compute_exposure(md.calendar, index_close, timing, log_path=log_path)
    result = run_backtest_timed(events, md, exposure, n_slots=n_slots,
                                selection=selection, exit_rule=exit_rule,
                                horizon=horizon, gate=gate, seed=seed,
                                log_path=log_path)
    result["exposure"] = exposure
    result["config"] = dict(pool=pool, selection=selection, exit_rule=exit_rule,
                            horizon=horizon, n_slots=n_slots, start=start, end=end,
                            seed=seed, timing=timing, gate=gate)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        result["equity"].to_parquet(os.path.join(out_dir, "equity_curve.parquet"))
        result["trades"].to_parquet(os.path.join(out_dir, "trades.parquet"))
        exposure.to_parquet(os.path.join(out_dir, "exposure.parquet"))
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({**result["stats"], **result["config"]}, f, ensure_ascii=False,
                      indent=2, default=str)
    se._log(log_path, f"run_config v2 finished in {time.time() - t0:.1f}s")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="真实交易回测引擎 v2（issue #19 择时覆盖层）")
    ap.add_argument("--pool", choices=se.POOLS, required=True)
    ap.add_argument("--selection", choices=se.VALID_SELECTIONS, required=True)
    ap.add_argument("--exit-rule", choices=se.VALID_EXITS, required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-slots", type=int, default=10)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timing", choices=[None, *VALID_TIMINGS], default=None)
    ap.add_argument("--gate", choices=VALID_GATES, default="G1")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_config(pool=args.pool, selection=args.selection, exit_rule=args.exit_rule,
               horizon=args.horizon, n_slots=args.n_slots, start=args.start, end=args.end,
               seed=args.seed, repo_root=repo_root, out_dir=args.out_dir,
               log_path=args.log, timing=args.timing, gate=args.gate)


if __name__ == "__main__":
    main()
