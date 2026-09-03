#!/usr/bin/env python3
"""T9 复核项 1：裁决数字独立重算。

自写实现（不调用 run_strategy_family.compute_metrics / index_metrics）：
  - 年化 = (final_equity / 1e6) ** (1/years) - 1，years = 自然日/365.25
  - Sharpe = 日频收益 mean/std * sqrt(252)（首日对初始本金 1e6）
  - 最大回撤 = equity / cummax - 1 最小值
  - 逐年收益 = 年界权益链式（首年对 1e6）
  - 基准 = stock_data/index/000905.SH.parquet 同窗口 close 年化 + 回撤
  - 超额 = 策略年化 - 基准年化（算术差，pp）
输入仅 runs/test_*/equity_curve.parquet、trades.parquet 与指数原始数据；
与 adjudication.json 逐项对比（容差 1e-9，落盘为全精度 float）。
"""
import json
import os

import numpy as np
import pandas as pd

REPO = "/home/karl/repos/personal/stock_qt_nd"
ADJ = os.path.join(REPO, "v3_pipeline/reports/test_adjudication/adjudication.json")
RUNS = os.path.join(REPO, "v3_pipeline/reports/test_adjudication/runs")
INIT = 1_000_000.0
TOL = 1e-9


def my_metrics(eq_df: pd.DataFrame) -> dict:
    eq = eq_df["equity"].astype(float).to_numpy()
    dates = pd.to_datetime(eq_df["date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    ann = (eq[-1] / INIT) ** (1.0 / years) - 1.0
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    prev = np.concatenate([[INIT], eq[:-1]])
    ret = eq / prev - 1.0
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252))
    calmar = float(ann / abs(mdd)) if mdd < 0 else float("nan")
    yearly = {}
    s = pd.Series(eq, index=dates)
    prev_eq = INIT
    for y, v in s.groupby(s.index.year).last().items():
        yearly[str(y)] = float(v / prev_eq - 1.0)
        prev_eq = v
    return dict(ann_return=float(ann), max_dd=mdd, sharpe=sharpe, calmar=calmar,
                yearly_returns=yearly, final_equity=float(eq[-1]))


def my_bench(code: str, start: str, end: str) -> dict:
    df = pd.read_parquet(os.path.join(REPO, "stock_data/index", f"{code}.parquet"))
    # 自写：不假定列名口径，先探查
    dcol = "trade_date" if "trade_date" in df.columns else "date"
    df[dcol] = pd.to_datetime(df[dcol])
    df = df[(df[dcol] >= start) & (df[dcol] <= end)].sort_values(dcol)
    close = df["close"].astype(float).to_numpy()
    years = (df[dcol].iloc[-1] - df[dcol].iloc[0]).days / 365.25
    ann = (close[-1] / close[0]) ** (1.0 / years) - 1.0
    mdd = float(np.min(close / np.maximum.accumulate(close) - 1.0))
    return dict(ann_return=float(ann), max_dd=mdd,
                first=str(df[dcol].iloc[0].date()), last=str(df[dcol].iloc[-1].date()),
                n_days=len(df))


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


def main():
    adj = json.load(open(ADJ))
    start, end = adj["window"]
    bench = my_bench(adj["benchmark"], start, end)
    print(f"[bench] 自算 {adj['benchmark']} [{bench['first']}..{bench['last']}] "
          f"n={bench['n_days']} ann={bench['ann_return']:.10f} mdd={bench['max_dd']:.10f}")
    ok = True
    for k, v in (("benchmark_ann", bench["ann_return"]),
                 ("benchmark_mdd", bench["max_dd"])):
        match = close(adj[k], v)
        ok &= match
        print(f"  {k}: adj={adj[k]:.12f} mine={v:.12f} -> {'OK' if match else 'MISMATCH'}")

    for name in ("A13", "B15", "C15"):
        eq = pd.read_parquet(os.path.join(RUNS, f"test_{name}/equity_curve.parquet"))
        tr = pd.read_parquet(os.path.join(RUNS, f"test_{name}/trades.parquet"))
        mine = my_metrics(eq)
        ref = adj["results"][name]
        excess = mine["ann_return"] - bench["ann_return"]
        pass_mine = bool(excess > adj["thresholds"]["excess_ann_gt"]
                         and mine["sharpe"] > adj["thresholds"]["sharpe_gt"])
        checks = [
            ("ann_return", mine["ann_return"], ref["ann_return"]),
            ("excess_ann", excess, ref["excess_ann"]),
            ("max_dd", mine["max_dd"], ref["max_dd"]),
            ("sharpe", mine["sharpe"], ref["sharpe"]),
            ("calmar", mine["calmar"], ref["calmar"]),
            ("n_trades", float(len(tr)), float(ref["n_trades"])),
        ]
        print(f"[{name}] eq_rows={len(eq)} trades={len(tr)}")
        for k, a, b in checks:
            m = close(a, b)
            ok &= m
            print(f"  {k}: adj={b:.12f} mine={a:.12f} -> {'OK' if m else 'MISMATCH'}")
        for y, v in ref["yearly_returns"].items():
            m = close(mine["yearly_returns"].get(y, float("nan")), v)
            ok &= m
            if not m:
                print(f"  yearly[{y}]: adj={v} mine={mine['yearly_returns'].get(y)} -> MISMATCH")
        m = pass_mine == ref["pass_threshold"]
        ok &= m
        print(f"  pass_threshold: adj={ref['pass_threshold']} mine={pass_mine} -> "
              f"{'OK' if m else 'MISMATCH'}")
        yr = ",".join(f"{y}:{v:+.4f}" for y, v in sorted(mine["yearly_returns"].items()))
        print(f"  yearly mine: {yr}")

    print("REVIEW1:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
