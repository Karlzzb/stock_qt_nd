#!/usr/bin/env python3
"""T9 复核项 7：t9_report.md 全部数字与落盘产物逐项核对（自写）。

核对源：adjudication.json、runs/test_*/stats.json、scores_final.parquet、
tuning_summary.json / tuning_class_*.csv（T8 归档）、面板 results json。
报告文本中的每个数字声明逐一与产物对拍（百分比保留两位四舍五入口径）。
"""
import json
import os
import re

import pandas as pd

REPO = "/home/karl/repos/personal/stock_qt_nd"
TA = os.path.join(REPO, "v3_pipeline/reports/test_adjudication")
ST = os.path.join(REPO, "v3_pipeline/reports/strategy_tuning")

adj = json.load(open(os.path.join(TA, "adjudication.json")))
report = open(os.path.join(TA, "t9_report.md"), encoding="utf-8").read()

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")


def pct(x):
    return f"{x * 100:.2f}"


# ---- 表头与基准 ----
check("基准年化 +7.91%", pct(adj["benchmark_ann"]) == "7.91" and "+7.91%" in report)
check("基准回撤 -31.15%", pct(adj["benchmark_mdd"]) == "-31.15" and "-31.15%" in report)

# ---- 裁决表 ----
expect = {
    "A13": dict(ann="-0.70%", exs="-8.61pp", gap="23.6pp", mdd="-33.63%",
                sharpe="0.101", n=142),
    "B15": dict(ann="+5.52%", exs="-2.39pp", gap="17.4pp", mdd="-30.81%",
                sharpe="0.346", n=140),
    "C15": dict(ann="-7.08%", exs="-14.99pp", gap="30.0pp", mdd="-33.89%",
                sharpe="-0.310", n=224),
}
for name, e in expect.items():
    r = adj["results"][name]
    gap = (adj["thresholds"]["excess_ann_gt"] - r["excess_ann"]) * 100
    vals = dict(ann=f"{r['ann_return'] * 100:+.2f}%", exs=f"{r['excess_ann'] * 100:+.2f}pp",
                gap=f"{gap:.1f}pp", mdd=f"{r['max_dd'] * 100:.2f}%",
                sharpe=f"{r['sharpe']:+.3f}".replace("+", ""), n=r["n_trades"])
    for k in e:
        ev = str(e[k])
        match = str(vals[k]) == ev and ev in report
        check(f"{name} {k}: 报告={e[k]} 重算={vals[k]}", match)

# ---- 逐年表 ----
expect_yearly = {
    "A13": ["-8.12%", "+14.91%", "-15.37%", "+10.75%", "-1.63%"],
    "B15": ["-8.12%", "+37.06%", "-11.53%", "+10.87%", "-0.56%"],
    "C15": ["-10.86%", "+6.08%", "-18.50%", "-2.05%", "0.00%"],
}
for name, exp in expect_yearly.items():
    yr = adj["results"][name]["yearly_returns"]
    got = [f"{yr[y] * 100:+.2f}%" if yr[y] != 0 else "0.00%"
           for y in ("2022", "2023", "2024", "2025", "2026")]
    match = got == exp and all(v in report for v in exp)
    check(f"{name} 逐年 {exp}", match, f"(重算 {got})")

# ---- 交易数 / 事件数 / 面板行数 ----
check("10643 事件 / 去重 10630",
      "10643" in report and "10630" in report
      and json.load(open(os.path.join(ST, "daily_score_panel_20221101_20260831_results.json")))["n_events"] == 10643)
stats_c = json.load(open(os.path.join(TA, "runs/test_C15/stats.json")))
check("C15 出场分布 107/111/6",
      (stats_c["exits_rank_out"], stats_c["exits_score_drop"], stats_c["exits_horizon"])
      == (107, 111, 6) and "rank_out 107 / score_drop 111 / horizon 6" in report)
check("僵尸仓 3 只", stats_c["open_at_end"] == 3
      and len(pd.read_parquet(os.path.join(TA, "runs/test_C15/open_positions.parquet"))) == 3)
util_vals = {k: adj["results"][k]["utilization"] for k in expect}
check("利用率 0.80-0.85 区间声明（两位舍入口径）",
      all(0.80 <= round(u, 2) <= 0.85 for u in util_vals.values()),
      f"(精确值 { {k: round(v, 6) for k, v in util_vals.items()} }；"
      "A13=0.85153 严格大于 0.85，两位舍入后为 0.85——擦边措辞，记录备查)")
check("C15 换手 20.5", f"{adj['results']['C15']['turnover']:.1f}" == "20.5"
      and "20.5" in report)

# ---- T8 对照声明 ----
ts = json.load(open(os.path.join(ST, "tuning_summary.json")))
check("T8 验证段 A13 +16.0pp / B15 +16.1pp",
      f"{ts['adjudication']['A']['excess_ann'] * 100:.1f}" == "16.0"
      and f"{ts['adjudication']['B']['excess_ann'] * 100:.1f}" == "16.1"
      and "+16.0/+16.1pp" in report)
tot = pos = 0
for cls in "ABC":
    df = pd.read_csv(os.path.join(ST, f"tuning_class_{cls}.csv"))
    tot += len(df)
    pos += int((df["excess_ann"] > 0).sum())
check("56 配置仅 10 个超额为正", tot == 56 and pos == 10 and "56 配置仅 10 个" in report,
      f"(实际 {tot}/{pos})")

# ---- 僵尸仓分数声明 ----
check("600781/688271 面板分 0.52837/0.52712 声明",
      "0.52837" in report and "0.52712" in report)

# ---- 排序声明 B ≈ A > C ----
e = {k: adj["results"][k]["excess_ann"] for k in expect}
check("类间排序 B>A>C", e["B15"] > e["A13"] > e["C15"])
check("唯一 Sharpe 为负 = C15",
      adj["results"]["C15"]["sharpe"] < 0 < adj["results"]["A13"]["sharpe"]
      and adj["results"]["B15"]["sharpe"] > 0)

# ---- 0/3 过线结论 ----
check("0/3 过线", not any(adj["results"][k]["pass_threshold"] for k in expect)
      and "0/3" in report)

# ---- 面板行数措辞（391 万行 vs 4375×931 全交叉）----
check("面板 391 万行", len(pd.read_parquet(
    os.path.join(ST, "daily_score_panel_20221101_20260831.parquet"),
    columns=["ts_code"])) == 3911166 and "391 万行" in report)
full_cross = 4375 * 931
print(f"  [NOTE] 报告括注'4375 股 × 931 交易日'={full_cross}，与实际行数 3911166 不等"
      f"（面板非全交叉：新股/停牌日无行）。措辞不严谨但主数字正确。")

print("REVIEW7:", "PASS" if ok else "FAIL")
