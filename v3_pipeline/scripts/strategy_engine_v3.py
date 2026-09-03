"""真实交易回测引擎 v3 —— issue #28 三类策略验证段调优（冻结候选）。

在 v1（strategy_engine.py）/v2（strategy_engine_v2.py）冻结引擎之上叠加
v5 定版分数信号源与三类出场策略；v1/v2 文件一字不动。
本金/整手/滑点/佣金/印花税/T+1/涨停拒买/跌停顺延/逐日盯市口径逐条复用 v1 机制，
入口处的信号源与出场判断为本层新增，规格逐条对应 issue #28 预登记：

 1. 信号源：v3_pipeline/reports/feature_selection/scores_final.parquet（issue #27 定版）。
    merged pool = main ∪ backup 行；(ts_code, date) 跨池重复时取 prob 高者
    （平局 pool="main" 优先，再平局 event_id 升序，保证确定性）。
    取舍规则固定为分数优先：当日信号按 prob 降序、ts_code 升序截取空位数。
    被截取信号 T+1 涨停/无行情被放弃不递补（同 v1）。
 2. 出场策略三类（仅出场判断不同，成交/顺延/成本机制同 v1）：
    A 固定止盈止损类（fixed_tp_sl）：屏障 = 入场执行价 × (1+tp) / ×(1+sl)，
      当日 high/low 触及即成交（open 越过屏障按 open，否则按屏障价），
      同日双触发保守取止损；到期未触及按持有满 H 日收盘卖（同 v1 E1，买入日记第 1 日）。
    B 波动率自适应动态类（vol_adaptive，历史 v12 语义）：
      每日（以 <=t-1 行情）重算 vol_mult = 个股 ATR(vol_lookback) / 全市场均值 ATR(vol_lookback)；
      vol_mult >= vol_high_thresh：tp_eff = tp × vol_profit_mult，sl_eff = sl × vol_stop_mult；
      vol_mult <= vol_low_thresh ：tp_eff = tp × low_vol_profit_mult，sl_eff = sl；
      其余 tp_eff = tp，sl_eff = sl。屏障价 = 入场执行价 × (1+eff)，触发口径同 A。
      ATR = 简单均值 TR 窗口（v12 PrecomputedATR 口径：TR=max(h-l,|h-c'|,|l-c'|)，
      窗口不足 vol_lookback 个 TR 则当日 vol_mult 记缺失并回落为中档）。
      市场均值 = 当日全部有有效 ATR 的 stock_data/daily 股票的算术平均（预计算序列传入）。
    C 分数衰减退出类（score_decay）：每日（t > 买入日）以 t 日收盘信息重算持仓股分数
      score_t（日频打分面板查表，面板构建见 build_daily_score_panel.py）；
      当日候选集 = 当日新鲜信号（prob 取自 scores_final）∪ 当前持仓股（score_t）；
      score_t 跌出候选集前 top_k 名（prob 降序、ts_code 升序）或
      score_t < 买入时分数 × (1 - score_margin) → 记出场意图，
      下一交易日开盘卖出（信号在 t 收盘产生，t+1 开盘执行，与入场 T+1 对称）；
      执行日无行情或执行日收盘 <= 跌停价 → 顺延至首个可卖日（同 v1 保守口径）；
      最大持仓 H 日兜底（持有满 H 日收盘卖，同 v1 E1 确定性口径）。
 3. 防泄漏断言（继承并扩展 v1）：买入日 > 事件日；出场评估日 > 买入日；
    类 B 的 ATR 只用 <=t-1 行情（引擎内硬断言窗口右端 < t）；
    类 C 的 score_t 面板值硬断言其特征时点 == t（面板构建侧保证因果，引擎侧抽查
    由 build_daily_score_panel 的前缀截断断言兜底）。

回归约定：v3 提供 selection="sig_idx" 兼容模式与 horizon-only 出场（exit_rule="E1"），
给定与 v1 相同的事件表（含 sig_idx 列）时必须逐位复现 v1 run_backtest 的交易明细
（tests/test_strategy_engine_v3.py 中有合成与真实数据两级对拍）。

用法（供下游代理）：
    from strategy_engine_v3 import load_score_events, run_config_v3, ExitSpec
    events = load_score_events("2019-01-01", "2022-10-31", repo_root)
    spec = ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=16)
    result = run_config_v3(events, md, n_slots=3, exit_spec=spec)
    result["equity"] / result["trades"] / result["stats"] 同 v1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy_engine as se  # noqa: E402  # v1 冻结引擎，只复用不修改

SCORES_PATH = "v3_pipeline/reports/feature_selection/scores_final.parquet"

VALID_STRATEGY_CLASSES = ("fixed_tp_sl", "vol_adaptive", "score_decay", "E1")


# ---------------------------------------------------------------- 出场规格
@dataclass(frozen=True)
class ExitSpec:
    """三类策略出场参数。strategy_class ∈ VALID_STRATEGY_CLASSES。

    horizon: 最大持仓交易日数（买入日记第 1 日，同 v1 E1）。
    tp/sl: 固定百分比止盈止损（类 A/B 基准值）。
    vol_*: 类 B 分档参数（v12 语义，见模块 docstring 第 2 条）。
    top_k/score_margin: 类 C 参数（见第 2 条 C 款）。
    """
    strategy_class: str
    horizon: int
    tp: float | None = None
    sl: float | None = None
    vol_lookback: int | None = None
    vol_high_thresh: float | None = None
    vol_low_thresh: float | None = None
    vol_profit_mult: float | None = None
    vol_stop_mult: float | None = None
    low_vol_profit_mult: float | None = None
    top_k: int | None = None
    score_margin: float | None = None

    @staticmethod
    def fixed_tp_sl(tp: float, sl: float, horizon: int) -> "ExitSpec":
        return ExitSpec(strategy_class="fixed_tp_sl", tp=tp, sl=sl, horizon=horizon)

    @staticmethod
    def vol_adaptive(tp: float, sl: float, horizon: int, vol_lookback: int,
                     vol_high_thresh: float, vol_low_thresh: float,
                     vol_profit_mult: float, vol_stop_mult: float,
                     low_vol_profit_mult: float) -> "ExitSpec":
        return ExitSpec(strategy_class="vol_adaptive", tp=tp, sl=sl, horizon=horizon,
                        vol_lookback=vol_lookback, vol_high_thresh=vol_high_thresh,
                        vol_low_thresh=vol_low_thresh, vol_profit_mult=vol_profit_mult,
                        vol_stop_mult=vol_stop_mult,
                        low_vol_profit_mult=low_vol_profit_mult)

    @staticmethod
    def score_decay(horizon: int, top_k: int = 5, score_margin: float = 0.0) -> "ExitSpec":
        return ExitSpec(strategy_class="score_decay", horizon=horizon,
                        top_k=top_k, score_margin=score_margin)

    @staticmethod
    def horizon_only(horizon: int) -> "ExitSpec":
        """回归用：等价 v1 E1（无屏障，持有满 horizon 日收盘卖）。"""
        return ExitSpec(strategy_class="E1", horizon=horizon)


# ---------------------------------------------------------------- 信号源：定版分数序列
def load_score_events(start: str, end: str, repo_root: str,
                      log_path: str | None = None) -> pd.DataFrame:
    """加载 merged pool 定版分数信号并按窗口过滤、跨池去重。

    返回列：ts_code, event_date, prob, event_id，按 (event_date, ts_code) 排序。
    去重规则：同 (ts_code, date) 取 prob 高者；平局 pool main 优先；再平局 event_id 升序。
    """
    df = pd.read_parquet(os.path.join(repo_root, SCORES_PATH))
    df["date"] = pd.to_datetime(df["date"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    assert df["prob"].notna().all(), "score events with NaN prob in window"
    df["_pool_rank"] = (df["pool"] != "main").astype(int)
    df = df.sort_values(["ts_code", "date", "prob", "_pool_rank", "event_id"],
                        ascending=[True, True, False, True, True], kind="mergesort")
    n_before = len(df)
    df = df.drop_duplicates(subset=["ts_code", "date"], keep="first")
    df = df.rename(columns={"date": "event_date"})
    df = df[["ts_code", "event_date", "prob", "event_id"]]
    df = df.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    se._log(log_path, f"load_score_events window=[{start},{end}] events={len(df)} "
                      f"(dedup dropped {n_before - len(df)}) "
                      f"unique_stocks={df['ts_code'].nunique()}")
    return df


def select_signals_score(day_signals: pd.DataFrame, n_free: int) -> pd.DataFrame:
    """分数优先取舍：prob 降序、ts_code 升序截取前 n_free 个。只用 T 日信息。"""
    if n_free <= 0 or day_signals.empty:
        return day_signals.iloc[0:0]
    df = day_signals.sort_values(["prob", "ts_code"], ascending=[False, True],
                                 kind="mergesort")
    return df.head(n_free)


# ---------------------------------------------------------------- 类 B：ATR 与波动分档
def compute_tr(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """TR 序列（首行 TR=h-l）。v12 PrecomputedATR 口径。"""
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(np.maximum(high - low, np.abs(high - prev_close)),
                    np.abs(low - prev_close))
    return tr


def atr_series(df: pd.DataFrame, lookback: int) -> pd.Series:
    """单股 ATR 简单均值序列（索引=trade_date）。ATR[t] = mean(TR[t-LB+1..t])，
    窗口不足 LB 个 TR 记 NaN（v12 口径：idx >= lb-1 才出值）。"""
    tr = compute_tr(df["close"].to_numpy(dtype=np.float64),
                    df["high"].to_numpy(dtype=np.float64),
                    df["low"].to_numpy(dtype=np.float64))
    s = pd.Series(tr, index=df.index)
    return s.rolling(lookback, min_periods=lookback).mean()


def vol_band(vol_mult: float | None, spec: ExitSpec) -> tuple[float, float]:
    """v12 分档：返回 (tp_eff, sl_eff)。vol_mult 缺失回落中档（v12 返回 1.0 口径）。"""
    assert spec.strategy_class == "vol_adaptive"
    vm = 1.0 if vol_mult is None or not np.isfinite(vol_mult) else vol_mult
    if vm >= spec.vol_high_thresh:
        return spec.tp * spec.vol_profit_mult, spec.sl * spec.vol_stop_mult
    if vm <= spec.vol_low_thresh:
        return spec.tp * spec.low_vol_profit_mult, spec.sl
    return spec.tp, spec.sl


# ---------------------------------------------------------------- 持仓（扩展 v1.Position）
@dataclass
class PositionV3:
    ts_code: str
    event_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    entry_commission: float
    last_known_price: float
    buy_prob: float | None = None        # 类 C 买入时分数阈值
    exit_intent: str | None = None       # 类 C：前一收盘产生的出场意图（reason）
    deferred_days: int = 0


# ---------------------------------------------------------------- 主回测循环
_EXIT_STAT_KEY = {"tp": "exits_tp", "sl": "exits_sl", "horizon": "exits_horizon",
                  "rank_out": "exits_rank_out", "score_drop": "exits_score_drop"}


def _record_sell(p: "PositionV3", day: pd.Timestamp, raw_exit: float, reason: str,
                 stats: dict, trades: list[dict],
                 cal_index: dict[pd.Timestamp, int]) -> float:
    """卖出成交记账（滑点/佣金/印花税/盈亏/交易行），返回现金增量。"""
    exec_sell = raw_exit * (1.0 - se.SLIPPAGE)
    comm, stamp = se.sell_costs(p.shares, exec_sell, day)
    gross_pnl = p.shares * (exec_sell - p.entry_price)
    total_cost = p.entry_commission + comm + stamp
    net_pnl = gross_pnl - total_cost
    held_days = cal_index[day] - cal_index[p.entry_date] + 1
    stats[_EXIT_STAT_KEY[reason]] += 1
    trades.append(dict(
        ts_code=p.ts_code, event_date=p.event_date, entry_date=p.entry_date,
        entry_price=p.entry_price, shares=p.shares,
        entry_commission=p.entry_commission,
        exit_date=day, exit_reason=reason, exit_raw_price=raw_exit,
        exit_exec_price=exec_sell, exit_commission=comm, stamp_tax=stamp,
        gross_pnl=gross_pnl, total_cost=total_cost, net_pnl=net_pnl,
        ret=net_pnl / (p.shares * p.entry_price + p.entry_commission),
        held_days=held_days, deferred_days=p.deferred_days,
        buy_prob=p.buy_prob,
    ))
    return p.shares * exec_sell - comm - stamp


def run_backtest_v3(events: pd.DataFrame, md: se.MarketData, n_slots: int,
                    exit_spec: ExitSpec,
                    selection: str = "score",
                    init_capital: float = se.INIT_CAPITAL,
                    score_panel: pd.DataFrame | None = None,
                    mkt_atr: pd.Series | None = None,
                    stock_atr: dict[str, pd.Series] | None = None,
                    log_path: str | None = None) -> dict:
    """事件驱动逐日模拟（v5 分数信号源 + 三类出场）。返回 equity/trades/stats。

    events 必须含列 ts_code/event_date/prob（selection="sig_idx" 回归模式另需 sig_idx）。
    selection: "score" = 分数优先（prob 降序）；"sig_idx" = 回归模式（复刻 v1 S1）。
    类 B 需传 mkt_atr（逐日全市场 ATR 均值）；类 C 需传 score_panel
    （列 ts_code/date/prob，(ts_code,date) 唯一）。
    """
    assert exit_spec.strategy_class in VALID_STRATEGY_CLASSES
    assert n_slots >= 1 and exit_spec.horizon >= 2
    cls = exit_spec.strategy_class
    if cls == "vol_adaptive":
        assert mkt_atr is not None, "vol_adaptive requires mkt_atr series"
    if cls == "score_decay":
        assert score_panel is not None, "score_decay requires score_panel"
        sp = score_panel.set_index(["ts_code", "date"])["prob"]
    else:
        sp = None

    events = events.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    by_date: dict[pd.Timestamp, pd.DataFrame] = {
        d: g for d, g in events.groupby("event_date")
    }
    pending: dict[pd.Timestamp, pd.DataFrame] = {}
    cal_index = {d: i for i, d in enumerate(md.calendar)}
    cal_arr = list(md.calendar)

    cash = init_capital
    positions: list[PositionV3] = []
    trades: list[dict] = []
    equity_rows: list[dict] = []
    stats = dict(
        total_signals=len(events), entered=0,
        dropped_slot_full=0, dropped_limitup=0, dropped_no_quote=0,
        dropped_cash=0, dropped_no_close_T=0,
        limit_missing_days=md.limit_missing_dates,
        deferred_exits=0,
        exits_tp=0, exits_sl=0, exits_horizon=0,
        exits_rank_out=0, exits_score_drop=0,
        vol_fallback_mid=0, score_lookup_missing=0,
    )

    def _band_at(pos: PositionV3, day: pd.Timestamp) -> tuple[float, float]:
        """类 B：以 <=day-1 行情计算当日有效 tp/sl 百分比（防泄漏硬断言）。"""
        di = cal_index[day]
        assert di >= 1
        ref_day = cal_arr[di - 1]
        lb = exit_spec.vol_lookback
        s_atr = (stock_atr or {}).get(pos.ts_code)
        a_stock = None
        if s_atr is not None and ref_day in s_atr.index:
            v = float(s_atr.loc[ref_day])
            if np.isfinite(v) and v > 0:
                a_stock = v
        a_mkt = None
        if ref_day in mkt_atr.index:
            v = float(mkt_atr.loc[ref_day])
            if np.isfinite(v) and v > 0:
                a_mkt = v
        if a_stock is None or a_mkt is None:
            stats["vol_fallback_mid"] += 1
            return exit_spec.tp, exit_spec.sl
        return vol_band(a_stock / a_mkt, exit_spec)

    for di, day in enumerate(md.calendar):
        # ---- 0. 信号日收盘：分数优先截取，挂到下一交易日入场 ----
        if day in by_date:
            n_free = n_slots - len(positions)
            day_sigs = by_date[day]
            if n_free < len(day_sigs):
                stats["dropped_slot_full"] += len(day_sigs) - max(n_free, 0)
            if selection == "score":
                chosen = select_signals_score(day_sigs, n_free)
            elif selection == "sig_idx":  # 回归模式：复刻 v1 S1
                chosen = day_sigs.sort_values(["sig_idx", "ts_code"], kind="mergesort") \
                                 .head(max(n_free, 0)) if n_free > 0 else day_sigs.iloc[0:0]
            else:
                raise ValueError(f"unknown selection {selection}")
            if not chosen.empty and di + 1 < len(cal_arr):
                nxt = cal_arr[di + 1]
                pending[nxt] = pd.concat([pending[nxt], chosen]) if nxt in pending else chosen

        # 当日收盘权益快照 = 昨收权益（入场预算基准，同 v1）
        if di > 0:
            prev_day = cal_arr[di - 1]
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

        # ---- 1. 入场（T+1 开盘；先于当日卖出，同 v1）----
        todays = pending.pop(day, None)
        if todays is not None and len(positions) < n_slots:
            limit_tbl = md.limits.get(day)
            for sig in todays.itertuples():
                if len(positions) >= n_slots:
                    stats["dropped_slot_full"] += 1
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
                budget = equity_prev_close / n_slots
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
                positions.append(PositionV3(
                    ts_code=sig.ts_code, event_date=sig.event_date, entry_date=day,
                    entry_price=exec_price, shares=shares, entry_commission=comm,
                    last_known_price=float(bar["close"]),
                    buy_prob=float(sig.prob),
                ))
                stats["entered"] += 1

        # ---- 1b. 类 C：执行前一收盘产生的出场意图（T+1 开盘）----
        if cls == "score_decay":
            limit_tbl = md.limits.get(day)
            still: list[PositionV3] = []
            for p in positions:
                if p.exit_intent is None or day <= p.entry_date:
                    still.append(p)
                    continue
                row = md.daily.get(p.ts_code)
                bar = row.loc[day] if (row is not None and day in row.index) else None
                if bar is None:
                    still.append(p)  # 执行日无行情：顺延
                    continue
                o, c = float(bar["open"]), float(bar["close"])
                if limit_tbl is not None and p.ts_code in limit_tbl.index:
                    dn = float(limit_tbl.loc[p.ts_code, "down_limit"])
                    if c <= dn + se.PRICE_TOL:
                        p.deferred_days += 1
                        stats["deferred_exits"] += 1
                        still.append(p)
                        continue
                reason = p.exit_intent
                p.exit_intent = None
                cash += _record_sell(p, day, o, reason, stats, trades, cal_index)
            positions = still

        # ---- 2. 出场评估（T+1 规则：entry_date < day 才可卖）----
        limit_tbl = md.limits.get(day)
        still_open: list[PositionV3] = []
        # 类 C：当日候选集 = 当日新鲜信号 ∪ 当前持仓
        if cls == "score_decay":
            fresh = by_date[day] if day in by_date else None
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

            if cls in ("fixed_tp_sl", "vol_adaptive"):
                if cls == "fixed_tp_sl":
                    tp_eff, sl_eff = exit_spec.tp, exit_spec.sl
                else:
                    tp_eff, sl_eff = _band_at(p, day)
                tp_barrier = p.entry_price * (1.0 + tp_eff)
                sl_barrier = p.entry_price * (1.0 + sl_eff)
                tp_hit = h >= tp_barrier - se.PRICE_TOL
                sl_hit = l <= sl_barrier + se.PRICE_TOL
                if tp_hit and not sl_hit:
                    raw_exit = o if o >= tp_barrier - se.PRICE_TOL else tp_barrier
                    reason = "tp"
                elif sl_hit:
                    raw_exit = o if o <= sl_barrier + se.PRICE_TOL else sl_barrier
                    reason = "sl"
                elif held_days >= exit_spec.horizon:
                    raw_exit = c
                    reason = "horizon"

            elif cls == "score_decay":
                if held_days >= exit_spec.horizon:
                    raw_exit = c
                    reason = "horizon"
                else:
                    s_t = sp.get((p.ts_code, day), np.nan) if sp is not None else np.nan
                    if not np.isfinite(s_t):
                        stats["score_lookup_missing"] += 1
                        s_t = None
                    if s_t is not None:
                        # 候选集：当日新鲜信号 prob ∪ 持仓股当日重算分数（按 ts_code 去重取高）
                        cand: dict[str, float] = {}
                        if fresh is not None:
                            for r in fresh.itertuples():
                                v = float(r.prob)
                                if v > cand.get(r.ts_code, -np.inf):
                                    cand[r.ts_code] = v
                        for q in positions:
                            if q.ts_code == p.ts_code:
                                q_s = s_t
                            else:
                                q_row = md.daily.get(q.ts_code)
                                if q_row is None or day not in q_row.index:
                                    continue
                                q_s = sp.get((q.ts_code, day), np.nan)
                            if np.isfinite(q_s):
                                cand[q.ts_code] = max(cand.get(q.ts_code, -np.inf),
                                                      float(q_s))
                        ranked = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))
                        top_codes = [code for code, _ in ranked[:exit_spec.top_k]]
                        threshold = p.buy_prob * (1.0 - exit_spec.score_margin)
                        if p.ts_code not in top_codes:
                            p.exit_intent = "rank_out"
                        elif s_t < threshold:
                            p.exit_intent = "score_drop"
                    still_open.append(p)
                    continue

            else:  # E1 回归模式
                if held_days >= exit_spec.horizon:
                    raw_exit = c
                    reason = "horizon"

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
            cash += _record_sell(p, day, raw_exit, reason, stats, trades, cal_index)
        positions = still_open

        # ---- 3. 逐日盯市 ----
        mv = sum(p.shares * p.last_known_price for p in positions)
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
    se._log(log_path, f"run_backtest_v3 done: class={cls} signals={stats['total_signals']} "
                      f"entered={stats['entered']} trades={len(trades_df)} "
                      f"final_equity={stats.get('final_equity', float('nan')):.2f}")
    # 诊断列（纯附加，不影响任何交易行为）：窗口末仍在仓的持仓（含停牌僵尸仓）。
    # 用途：类 C 逐笔重放断言重建候选集时，在仓集合 = trades ∪ open_positions。
    open_df = pd.DataFrame([dict(ts_code=p.ts_code, event_date=p.event_date,
                                 entry_date=p.entry_date, buy_prob=p.buy_prob)
                            for p in positions],
                           columns=["ts_code", "event_date", "entry_date", "buy_prob"])
    stats["open_at_end"] = int(len(open_df))
    return {"equity": equity_df, "trades": trades_df, "stats": stats,
            "open_positions": open_df}


# ---------------------------------------------------------------- 顶层配置入口
def run_config_v3(events: pd.DataFrame, md: se.MarketData, n_slots: int,
                  exit_spec: ExitSpec, start: str, end: str,
                  selection: str = "score",
                  score_panel: pd.DataFrame | None = None,
                  mkt_atr: pd.Series | None = None,
                  stock_atr: dict[str, pd.Series] | None = None,
                  out_dir: str | None = None,
                  log_path: str | None = None) -> dict:
    """跑一个 信号源×出场规格×N 配置，返回 {equity, trades, stats, config}。"""
    t0 = time.time()
    se._log(log_path, f"run_config_v3 class={exit_spec.strategy_class} N={n_slots} "
                      f"spec={exit_spec} [{start}..{end}]")
    events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
    result = run_backtest_v3(events, md, n_slots=n_slots, exit_spec=exit_spec,
                             selection=selection,
                             score_panel=score_panel, mkt_atr=mkt_atr,
                             stock_atr=stock_atr, log_path=log_path)
    result["config"] = dict(strategy_class=exit_spec.strategy_class,
                            spec=exit_spec.__dict__, n_slots=n_slots,
                            start=start, end=end, selection=selection)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        result["equity"].to_parquet(os.path.join(out_dir, "equity_curve.parquet"))
        result["trades"].to_parquet(os.path.join(out_dir, "trades.parquet"))
        result["open_positions"].to_parquet(os.path.join(out_dir, "open_positions.parquet"))
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({**result["stats"], **result["config"]}, f, ensure_ascii=False,
                      indent=2, default=str)
    se._log(log_path, f"run_config_v3 finished in {time.time() - t0:.1f}s")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="真实交易回测引擎 v3（issue #28）")
    ap.add_argument("--strategy-class", choices=VALID_STRATEGY_CLASSES, required=True)
    ap.add_argument("--tp", type=float, default=0.25)
    ap.add_argument("--sl", type=float, default=-0.14)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-slots", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--score-margin", type=float, default=0.0)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2022-10-31")
    ap.add_argument("--score-panel", default=None,
                    help="score_decay 必填：日频打分面板 parquet（ts_code/date/prob）")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    events = load_score_events(args.start, args.end, repo_root, args.log)
    md = se.load_market_data(events["ts_code"].unique().tolist(), args.start, args.end,
                             repo_root, log_path=args.log)
    if args.strategy_class == "fixed_tp_sl":
        spec = ExitSpec.fixed_tp_sl(tp=args.tp, sl=args.sl, horizon=args.horizon)
    elif args.strategy_class == "score_decay":
        spec = ExitSpec.score_decay(horizon=args.horizon, top_k=args.top_k,
                                    score_margin=args.score_margin)
    elif args.strategy_class == "E1":
        spec = ExitSpec.horizon_only(horizon=args.horizon)
    else:
        raise SystemExit("vol_adaptive 请用 run_config_v3 库调用（需 mkt_atr 序列）")
    panel = None
    if args.score_panel:
        panel = pd.read_parquet(args.score_panel)
        panel["date"] = pd.to_datetime(panel["date"])
    run_config_v3(events, md, n_slots=args.n_slots, exit_spec=spec,
                  start=args.start, end=args.end,
                  score_panel=panel, out_dir=args.out_dir, log_path=args.log)


if __name__ == "__main__":
    main()
