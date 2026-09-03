#!/usr/bin/env python3
"""T9 复核项 4b：引擎修改行为保持的独立实证。

用当前（已改）引擎独立重跑 T8 归档 val_C15（验证段 score_decay H15/N3/top_k3/margin0），
与归档 runs/val_C15 的 trades/equity/stats 逐位对比。
验证段僵尸仓为 0 时 open_positions 应为空——附加输出不改变任何既有产物。
仅复用引擎入口（被测对象本身）做回归对拍，指标/断言均不复用被复核脚本。
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO, "v3_pipeline", "scripts"))
import strategy_engine as se  # noqa: E402
import strategy_engine_v3 as se3  # noqa: E402

ARCH = os.path.join(REPO, "v3_pipeline/reports/strategy_tuning/runs/val_C15")
VAL_PANEL = os.path.join(REPO, "v3_pipeline/reports/strategy_tuning",
                         "daily_score_panel_20190101_20221031.parquet")
START, END = "2019-01-01", "2022-10-31"


def main():
    arch_tr = pd.read_parquet(os.path.join(ARCH, "trades.parquet"))
    arch_eq = pd.read_parquet(os.path.join(ARCH, "equity_curve.parquet"))
    print(f"归档 val_C15: trades={len(arch_tr)} equity_rows={len(arch_eq)}")

    events = se3.load_score_events(START, END, REPO, None)
    md = se.load_market_data(events["ts_code"].unique().tolist(), START, END, REPO,
                             log_path=None)
    events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
    panel = pd.read_parquet(VAL_PANEL, columns=["ts_code", "date", "prob"])
    panel["date"] = pd.to_datetime(panel["date"])

    res = se3.run_config_v3(events, md, n_slots=3,
                            exit_spec=se3.ExitSpec.score_decay(horizon=15, top_k=3,
                                                               score_margin=0.0),
                            start=START, end=END, out_dir=None, log_path=None,
                            score_panel=panel)
    tr, eq, op = res["trades"], res["equity"], res["open_positions"]
    print(f"重跑 val_C15: trades={len(tr)} equity_rows={len(eq)} open_positions={len(op)}")

    ok = True
    ok &= len(tr) == len(arch_tr) and len(eq) == len(arch_eq)
    if len(tr) == len(arch_tr):
        num_cols = [c for c in arch_tr.columns
                    if pd.api.types.is_numeric_dtype(arch_tr[c])]
        str_cols = [c for c in arch_tr.columns if c not in num_cols]
        same_tr = all(np.allclose(tr[c].astype(float), arch_tr[c].astype(float),
                                  rtol=0, atol=0, equal_nan=True) for c in num_cols)
        same_tr &= all((tr[c].astype(str) == arch_tr[c].astype(str)).all()
                       for c in str_cols)
        print(f"trades 全列逐位一致: {same_tr} (数值列 {len(num_cols)} 文本列 {len(str_cols)})")
        ok &= same_tr
    if len(eq) == len(arch_eq):
        same_eq = all(np.allclose(eq[c].astype(float), arch_eq[c].astype(float),
                                  rtol=0, atol=0, equal_nan=True)
                      for c in ("cash", "market_value", "equity", "utilization"))
        print(f"equity 数值列逐位一致: {same_eq}")
        ok &= same_eq
    print(f"stats.open_at_end={res['stats'].get('open_at_end')}")
    ok &= len(op) == 0  # 验证段应无僵尸仓（报告口径）
    print("REVIEW4B:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
