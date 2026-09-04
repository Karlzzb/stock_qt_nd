#!/usr/bin/env python3
"""手工重放抽查：从 runs/events_v2__A13 抽 1 笔交易，用 stock_data/daily 原始行情
独立重算入场日开盘价、出场触发扫描、成本与净盈亏，与引擎明细逐字段比对。
不读引擎内部状态，只用原始日线 + 涨跌停表 + 指数日历。全程禁网络。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline", "scripts"))
import strategy_engine as se  # noqa: E402  只读复用冻结常量

BT_DIR = os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/backtest")
TRADES_PATH = os.path.join(BT_DIR, "runs/events_v2__A13/trades.parquet")
START, END = "2026-01-01", "2026-08-31"
TP, SL, HORIZON = 0.25, -0.14, 12

TOL = 1e-6


def close_enough(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def main() -> None:
    trades = pd.read_parquet(TRADES_PATH)
    assert not trades.empty
    # 抽第一笔无跌停顺延的交易（顺延路径已由引擎单测覆盖）
    tr = trades[trades["deferred_days"] == 0].iloc[0]
    code, ev_d = tr["ts_code"], pd.Timestamp(tr["event_date"])
    print(f"抽查交易: {code} event={ev_d.date()} entry={pd.Timestamp(tr['entry_date']).date()} "
          f"exit={pd.Timestamp(tr['exit_date']).date()} reason={tr['exit_reason']}")

    # 日历（与引擎同源：000905.SH 交易日）
    idx = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index/000905.SH.parquet"),
                          columns=["trade_date"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    cal = sorted(d for d in idx["trade_date"].unique()
                 if pd.Timestamp(START) <= d <= pd.Timestamp(END))
    cal = [pd.Timestamp(d) for d in cal]
    cal_index = {d: i for i, d in enumerate(cal)}

    # 原始日线
    df = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/daily", f"{code}.parquet"),
                         columns=["trade_date", "open", "high", "low", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()

    checks: list[tuple[str, bool, str]] = []

    # ---- 入场：event_date 的下一交易日开盘 ×(1+滑点) ----
    ei = cal_index[ev_d]
    entry_day = cal[ei + 1]
    bar = df.loc[entry_day]
    exp_entry = float(bar["open"]) * (1.0 + se.SLIPPAGE)
    checks.append(("entry_date = event 后一交易日", entry_day == pd.Timestamp(tr["entry_date"]),
                   f"{entry_day.date()} vs {pd.Timestamp(tr['entry_date']).date()}"))
    checks.append(("entry_price = open×1.001", close_enough(exp_entry, tr["entry_price"]),
                   f"{exp_entry:.6f} vs {tr['entry_price']:.6f}"))

    # ---- 入场佣金 ----
    exp_ecomm = max(se.COMMISSION_MIN, tr["shares"] * tr["entry_price"] * se.COMMISSION_RATE)
    checks.append(("entry_commission", close_enough(exp_ecomm, tr["entry_commission"]),
                   f"{exp_ecomm:.4f} vs {tr['entry_commission']:.4f}"))

    # ---- 出场触发扫描（T+1 起，同日双触发保守取止损，到期收盘卖，跌停顺延）----
    tp_b = tr["entry_price"] * (1.0 + TP)
    sl_b = tr["entry_price"] * (1.0 + SL)
    found = None
    for d in cal[cal_index[entry_day] + 1:]:
        if d not in df.index:
            continue
        o, h, l, c = (float(df.loc[d, k]) for k in ("open", "high", "low", "close"))
        held = cal_index[d] - cal_index[entry_day] + 1
        tp_hit = h >= tp_b - se.PRICE_TOL
        sl_hit = l <= sl_b + se.PRICE_TOL
        raw, reason = None, None
        if tp_hit and not sl_hit:
            raw, reason = (o if o >= tp_b - se.PRICE_TOL else tp_b), "tp"
        elif sl_hit:
            raw, reason = (o if o <= sl_b + se.PRICE_TOL else sl_b), "sl"
        elif held >= HORIZON:
            raw, reason = c, "horizon"
        if raw is None:
            continue
        # 跌停顺延检查
        lim_fp = os.path.join(REPO_ROOT, "stock_data/stk_limit", d.strftime("%Y%m%d") + ".parquet")
        if os.path.exists(lim_fp):
            lim = pd.read_parquet(lim_fp).set_index("ts_code")
            if code in lim.index and c <= float(lim.loc[code, "down_limit"]) + se.PRICE_TOL:
                continue  # 顺延
        found = (d, raw, reason, held)
        break
    assert found is not None, "重放未找到出场"
    d, raw, reason, held = found
    checks.append(("exit_date", d == pd.Timestamp(tr["exit_date"]),
                   f"{d.date()} vs {pd.Timestamp(tr['exit_date']).date()}"))
    checks.append(("exit_reason", reason == tr["exit_reason"], f"{reason} vs {tr['exit_reason']}"))
    checks.append(("exit_raw_price", close_enough(raw, tr["exit_raw_price"]),
                   f"{raw:.6f} vs {tr['exit_raw_price']:.6f}"))
    checks.append(("held_days", held == tr["held_days"], f"{held} vs {tr['held_days']}"))

    # ---- 卖出成本与净盈亏 ----
    exp_exec = raw * (1.0 - se.SLIPPAGE)
    checks.append(("exit_exec_price", close_enough(exp_exec, tr["exit_exec_price"]),
                   f"{exp_exec:.6f} vs {tr['exit_exec_price']:.6f}"))
    exp_xcomm = max(se.COMMISSION_MIN, tr["shares"] * exp_exec * se.COMMISSION_RATE)
    exp_stamp = tr["shares"] * exp_exec * se.STAMP_TAX_NEW  # 窗口全在 2023-08-28 后
    checks.append(("exit_commission", close_enough(exp_xcomm, tr["exit_commission"]),
                   f"{exp_xcomm:.4f} vs {tr['exit_commission']:.4f}"))
    checks.append(("stamp_tax", close_enough(exp_stamp, tr["stamp_tax"]),
                   f"{exp_stamp:.4f} vs {tr['stamp_tax']:.4f}"))
    exp_net = tr["shares"] * (exp_exec - tr["entry_price"]) - \
        (tr["entry_commission"] + exp_xcomm + exp_stamp)
    checks.append(("net_pnl", close_enough(exp_net, tr["net_pnl"]),
                   f"{exp_net:.4f} vs {tr['net_pnl']:.4f}"))
    exp_ret = exp_net / (tr["shares"] * tr["entry_price"] + tr["entry_commission"])
    checks.append(("ret", close_enough(exp_ret, tr["ret"]), f"{exp_ret:.6f} vs {tr['ret']:.6f}"))

    n_fail = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        n_fail += 0 if ok else 1
    print(f"重放抽查: {len(checks) - n_fail}/{len(checks)} 项一致 -> "
          f"{'PASS' if n_fail == 0 else 'FAIL'}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
