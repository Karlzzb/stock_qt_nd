#!/usr/bin/env python3
"""事件层手工重放抽查（README.md「预登记增补 2026-09-04」E-4 节）。

抽 2 笔事件层已闭合交易（v1 一笔、v2 一笔，均走过多日跌停顺延路径），
从 stock_data/daily 原始日线 + stock_data/stk_limit 原始涨跌停表 + 000905.SH 指数日历
独立重算：入场日/入场价（×1.001）、10 万预算整手股数、屏障触发扫描（同日双触发取止损）、
跌停顺延、卖出滑点/佣金/印花税、net_pnl 与 ret，与 event_study.parquet 明细逐字段比对。

独立性：不 import 引擎与 event_study.py 的任何函数/常量；全部常量按预登记文本
（README 第 7 节与增补 D 节）硬编码：滑点 0.001、佣金 0.00025 最低 5 元、
印花税 0.0005、整手 100、容差 1e-9、A13 屏障 tp=+25%/sl=-14%/H=12、预算 100,000。
全程禁网络。
"""
from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
BT_DIR = os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/backtest")
EVENT_PQ = os.path.join(BT_DIR, "event_study.parquet")

# ---- 预登记常量（硬编码，不读引擎源码）----
SLIPPAGE = 0.001
COMM_RATE = 0.00025
COMM_MIN = 5.0
STAMP = 0.0005
LOT = 100
TOL = 1e-9
TP, SL, HORIZON = 0.25, -0.14, 12
BUDGET = 100_000.0
START, END = "2026-01-01", "2026-08-31"

PICKS = [  # (signal, config, ts_code, event_date)
    ("events_v1", "A13", "600355.SH", "2026-03-11"),
    ("events_v2", "A13", "600608.SH", "2026-03-24"),
]


def load_calendar() -> list[pd.Timestamp]:
    df = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index/000905.SH.parquet"),
                         columns=["trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return sorted(pd.Timestamp(d) for d in df["trade_date"].unique()
                  if pd.Timestamp(START) <= d <= pd.Timestamp(END))


def load_daily(code: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/daily", f"{code}.parquet"),
                         columns=["trade_date", "open", "high", "low", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date").sort_index()


def down_limit(code: str, day: pd.Timestamp) -> float | None:
    fp = os.path.join(REPO_ROOT, "stock_data/stk_limit", day.strftime("%Y%m%d") + ".parquet")
    if not os.path.exists(fp):
        return None
    t = pd.read_parquet(fp).set_index("ts_code")
    if code not in t.index:
        return None
    return float(t.loc[code, "down_limit"])


def replay_one(code: str, event_date: pd.Timestamp, cal: list[pd.Timestamp]) -> dict:
    cal_index = {d: i for i, d in enumerate(cal)}
    df = load_daily(code)
    ei = cal_index[event_date]
    entry_day = cal[ei + 1]
    assert entry_day > event_date
    ebar = df.loc[entry_day]
    entry_price = float(ebar["open"]) * (1.0 + SLIPPAGE)

    # 10 万预算整手：向下取整到手，不足一手买一手，现金不足逐手递减
    shares = int(BUDGET / entry_price / LOT) * LOT
    if shares < LOT:
        shares = LOT
    comm = max(COMM_MIN, shares * entry_price * COMM_RATE)
    while shares > 0 and shares * entry_price + comm > BUDGET + 1e-6:
        shares -= LOT
        comm = max(COMM_MIN, shares * entry_price * COMM_RATE) if shares > 0 else 0.0
    assert shares > 0, "现金不足（dropped_cash），本笔不应闭合"
    entry_comm = comm

    tp_b = entry_price * (1.0 + TP)
    sl_b = entry_price * (1.0 + SL)
    deferred = 0
    for di in range(cal_index[entry_day] + 1, len(cal)):
        day = cal[di]
        if day not in df.index:
            continue
        o, h, l, c = (float(df.loc[day, k]) for k in ("open", "high", "low", "close"))
        held = di - cal_index[entry_day] + 1
        tp_hit = h >= tp_b - TOL
        sl_hit = l <= sl_b + TOL
        raw, reason = None, None
        if tp_hit and not sl_hit:
            raw, reason = (o if o >= tp_b - TOL else tp_b), "tp"
        elif sl_hit:
            raw, reason = (o if o <= sl_b + TOL else sl_b), "sl"
        elif held >= HORIZON:
            raw, reason = c, "horizon"
        if raw is None:
            continue
        dn = down_limit(code, day)
        if dn is not None and c <= dn + TOL:
            deferred += 1
            continue
        exec_sell = raw * (1.0 - SLIPPAGE)
        xcomm = max(COMM_MIN, shares * exec_sell * COMM_RATE)
        stamp = shares * exec_sell * STAMP
        net = shares * (exec_sell - entry_price) - (entry_comm + xcomm + stamp)
        return dict(entry_date=entry_day, entry_price=entry_price, shares=shares,
                    entry_commission=entry_comm, exit_date=day, exit_reason=reason,
                    exit_raw_price=raw, exit_exec_price=exec_sell,
                    exit_commission=xcomm, stamp_tax=stamp, net_pnl=net,
                    ret=net / (shares * entry_price + entry_comm),
                    held_days=held, deferred_days=deferred)
    raise AssertionError("重放未找到出场（应为 incomplete）")


def main() -> None:
    es = pd.read_parquet(EVENT_PQ)
    es["event_date"] = pd.to_datetime(es["event_date"])
    cal = load_calendar()
    n_fail = 0
    for sig, cfg, code, ev_d in PICKS:
        row = es[(es["signal"] == sig) & (es["config"] == cfg)
                 & (es["ts_code"] == code) & (es["event_date"] == pd.Timestamp(ev_d))]
        assert len(row) == 1, f"{sig}/{cfg}/{code}/{ev_d} 事件层行数={len(row)}"
        r = row.iloc[0]
        assert r["status"] == "closed"
        print(f"抽查: {sig} {cfg} {code} event={ev_d} "
              f"(事件层 deferred_days={int(r['deferred_days'])}, reason={r['exit_reason']})")
        rep = replay_one(code, pd.Timestamp(ev_d), cal)
        fields = ["entry_date", "entry_price", "shares", "entry_commission",
                  "exit_date", "exit_reason", "exit_raw_price", "exit_exec_price",
                  "exit_commission", "stamp_tax", "net_pnl", "ret",
                  "held_days", "deferred_days"]
        for f in fields:
            a, b = rep[f], r[f]
            if f in ("entry_date", "exit_date"):
                ok = pd.Timestamp(a) == pd.Timestamp(b)
            elif f == "exit_reason":
                ok = a == b
            elif f in ("shares", "held_days", "deferred_days"):
                ok = int(a) == int(b)
            else:
                ok = abs(float(a) - float(b)) <= 1e-9
            print(f"  [{'PASS' if ok else 'FAIL'}] {f}: replay={a} vs event_study={b}")
            n_fail += 0 if ok else 1
    print(f"手工重放 2 笔: {'全部字段一致 PASS' if n_fail == 0 else f'{n_fail} 处不符 FAIL'}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
