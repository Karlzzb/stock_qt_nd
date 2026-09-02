"""策略家族全量回测 runner —— issue #18 预登记 + 2026-09-02 执行口径补充。

执行清单：
 1. 主家族 32 配置（池 main/backup × 取舍 S0-S3 × 出场 E1-E4，N=10 H=20）val 段。
 2. 同 32 配置 train 后段（稳健性参考，只出汇总表）。
 3. 指标：扣成本年化/最大回撤/夏普(日频√252)/卡玛/年换手/成本占毛利比/逐年收益/资金利用率/入金覆盖率。
 4. val 卡玛降序（平局取年化高者）定前 3 名。
 5. 前 3 名 test 段一次性确认（只跑 3 个）。
 6. 消融 ≤9：前 3 名 × {N=5, N=20, H=10}，仅 val 段。
 7. 汇总 family_summary.csv + strategy_family_report.md。

引擎 strategy_engine.py 已冻结，本脚本只读调用，不做任何修改。
行情/事件数据按 (pool, segment) 缓存复用（引擎 load_* 为纯函数，run_backtest 每配置独立 seed，
与逐配置调用 run_config 完全等价，仅省去重复 IO）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline/scripts"))
from strategy_engine import (  # noqa: E402
    INIT_CAPITAL, POOLS, VALID_EXITS, VALID_SELECTIONS,
    load_events, load_market_data, run_backtest,
)

OUT_ROOT = os.path.join(REPO_ROOT, "v3_pipeline/reports/strategy_real_trading")
RUNS_DIR = os.path.join(OUT_ROOT, "runs")
PROGRESS_LOG = os.path.join(OUT_ROOT, "progress.log")
SUMMARY_CSV = os.path.join(OUT_ROOT, "family_summary.csv")
REPORT_MD = os.path.join(OUT_ROOT, "strategy_family_report.md")

SEED = 42
SEGMENTS = {
    "train_late": ("2015-01-01", "2018-12-31"),
    "val": ("2019-01-01", "2022-10-31"),
    "test": ("2022-11-01", "2026-05-31"),
}
INDEX_MAIN = "000905.SH"       # 中证500（主对照）
INDEX_REFS = ["000300.SH", "000852.SH"]  # 沪深300 / 中证1000（参考）


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 指标
def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, stats: dict) -> dict:
    """对单配置单段计算全部指标。equity 需含 date/equity 列。"""
    m: dict = {}
    if equity.empty:
        return m
    eq = equity["equity"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(equity["date"])
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1e-9)

    final_eq = float(eq.iloc[-1])
    ann = (final_eq / INIT_CAPITAL) ** (1.0 / years) - 1.0
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    # 日频收益（首日对初始本金）
    prev = pd.concat([pd.Series([INIT_CAPITAL]), eq.iloc[:-1]]).reset_index(drop=True)
    ret_d = eq / prev - 1.0
    sharpe = float(ret_d.mean() / ret_d.std() * np.sqrt(252)) if ret_d.std() > 0 else np.nan
    calmar = float(ann / abs(max_dd)) if max_dd < 0 else np.nan

    # 年换手 = 年卖出笔数 × 单仓名义（平均入场名义）/ 平均权益
    n_sells = len(trades)
    if n_sells > 0:
        mean_notional = float((trades["shares"] * trades["entry_price"]).mean())
        turnover = (n_sells / years) * mean_notional / float(eq.mean())
        gross_sum = float(trades["gross_pnl"].sum())
        cost_sum = float(trades["total_cost"].sum())
        cost_ratio = cost_sum / gross_sum if gross_sum > 0 else np.nan
    else:
        turnover, cost_ratio = 0.0, np.nan

    # 逐年收益：年界权益（首年对初始本金）
    eq_by_date = pd.Series(eq.values, index=dates)
    yearly: dict[str, float] = {}
    year_ends = eq_by_date.groupby(eq_by_date.index.year).last()
    prev_eq = INIT_CAPITAL
    for y, v in year_ends.items():
        yearly[str(y)] = float(v / prev_eq - 1.0)
        prev_eq = v

    m.update(
        ann_return=float(ann), max_dd=max_dd, sharpe=sharpe, calmar=calmar,
        turnover=float(turnover), cost_ratio=cost_ratio,
        utilization=float(stats.get("capital_utilization", np.nan)),
        coverage=float(stats.get("coverage", np.nan)),
        final_equity=final_eq, n_trades=int(n_sells),
        entered=int(stats.get("entered", 0)),
        total_signals=int(stats.get("total_signals", 0)),
        yearly_returns=yearly,
    )
    return m


def index_metrics(code: str, start: str, end: str) -> dict:
    df = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index", f"{code}.parquet"),
                         columns=["trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].sort_values("trade_date")
    if df.empty:
        return {"ann_return": np.nan, "max_dd": np.nan}
    close = df["close"].astype(float).reset_index(drop=True)
    years = max((df["trade_date"].iloc[-1] - df["trade_date"].iloc[0]).days / 365.25, 1e-9)
    ann = (close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0
    mdd = float((close / close.cummax() - 1.0).min())
    return {"ann_return": float(ann), "max_dd": mdd}


# ---------------------------------------------------------------- 运行框架
class SegmentCache:
    """按 (pool, segment) 缓存事件与行情，避免 76 次重复 IO。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], tuple[pd.DataFrame, object]] = {}

    def get(self, pool: str, segment: str):
        key = (pool, segment)
        if key not in self._cache:
            start, end = SEGMENTS[segment]
            events = load_events(pool, start, end, REPO_ROOT, PROGRESS_LOG)
            md = load_market_data(events["ts_code"].unique().tolist(), start, end,
                                  REPO_ROOT, log_path=PROGRESS_LOG)
            events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
            self._cache[key] = (events, md)
        return self._cache[key]


def config_name(pool: str, sel: str, exit_rule: str, n_slots: int, horizon: int) -> str:
    base = f"{pool}_{sel}_{exit_rule}"
    parts = []
    if horizon != 20:
        parts.append(f"H{horizon}")
    parts.append(f"N{n_slots}")
    return f"{base}_{'_'.join(parts)}"


def run_one(cache: SegmentCache, segment: str, pool: str, sel: str, exit_rule: str,
            n_slots: int, horizon: int, persist: bool) -> dict:
    """跑单配置单段；persist=True 时落盘 runs/<segment>_<name>/。返回汇总行。"""
    name = config_name(pool, sel, exit_rule, n_slots, horizon)
    events, md = cache.get(pool, segment)
    t0 = time.time()
    res = run_backtest(events, md, n_slots=n_slots, selection=sel, exit_rule=exit_rule,
                       horizon=horizon, seed=SEED, log_path=None)
    elapsed = time.time() - t0
    stats = res["stats"]
    metrics = compute_metrics(res["equity"], res["trades"], stats)

    if persist:
        out_dir = os.path.join(RUNS_DIR, f"{segment}_{name}")
        os.makedirs(out_dir, exist_ok=True)
        res["equity"].to_parquet(os.path.join(out_dir, "equity_curve.parquet"))
        res["trades"].to_parquet(os.path.join(out_dir, "trades.parquet"))
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({**stats,
                       "config": dict(pool=pool, selection=sel, exit_rule=exit_rule,
                                      horizon=horizon, n_slots=n_slots,
                                      start=SEGMENTS[segment][0], end=SEGMENTS[segment][1],
                                      seed=SEED)},
                      f, ensure_ascii=False, indent=2, default=str)

    row = dict(segment=segment, config=name, pool=pool, selection=sel,
               exit_rule=exit_rule, n_slots=n_slots, horizon=horizon, seed=SEED,
               elapsed_s=round(elapsed, 2), **{k: v for k, v in metrics.items()
                                               if k != "yearly_returns"},
               yearly_returns=metrics.get("yearly_returns", {}))
    log(f"FAMILY {segment} {name} done in {elapsed:.1f}s | "
        f"ann={metrics.get('ann_return', float('nan')):+.4f} "
        f"mdd={metrics.get('max_dd', float('nan')):.4f} "
        f"calmar={metrics.get('calmar', float('nan')):.3f} "
        f"trades={metrics.get('n_trades', 0)} cov={metrics.get('coverage', float('nan')):.4f}")
    return row


# ---------------------------------------------------------------- 报告辅助
def pct(x, digits=2):
    return "NaN" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{digits}f}"


def ratio(x, digits=3):
    return "NaN" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


METRIC_COLS = ["ann_return", "max_dd", "sharpe", "calmar", "turnover", "cost_ratio",
               "coverage", "utilization"]
METRIC_HEAD = ["配置", "年化%", "最大回撤%", "夏普", "卡玛", "年换手", "成本/毛利",
               "覆盖率%", "利用率%"]


def family_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in df.iterrows():
        rows.append([r["config"], pct(r["ann_return"]), pct(r["max_dd"]),
                     ratio(r["sharpe"]), ratio(r["calmar"]), ratio(r["turnover"]),
                     ratio(r["cost_ratio"]), pct(r["coverage"]), pct(r["utilization"])])
    return rows


def main() -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    t_all = time.time()
    log("RUN_FAMILY START (issue #18): 32 主家族 val + train_late 参考 + 前3 test + 消融<=9")

    cache = SegmentCache()
    summary_rows: list[dict] = []

    # ---- 1+2. 主家族 32 配置：val（落盘）+ train_late（仅汇总）----
    family = list(product(POOLS, VALID_SELECTIONS, VALID_EXITS))
    for pool, sel, exit_rule in family:
        summary_rows.append(run_one(cache, "val", pool, sel, exit_rule,
                                    n_slots=10, horizon=20, persist=True))
    log("RUN_FAMILY val 32/32 done")
    for pool, sel, exit_rule in family:
        summary_rows.append(run_one(cache, "train_late", pool, sel, exit_rule,
                                    n_slots=10, horizon=20, persist=False))
    log("RUN_FAMILY train_late 32/32 done（参考段，不落盘 runs）")

    # ---- 4. 排名：val 卡玛降序，平局取年化高者 ----
    val_df = pd.DataFrame([r for r in summary_rows if r["segment"] == "val"])
    ranked = val_df.sort_values(["calmar", "ann_return"],
                                ascending=[False, False], na_position="last")
    top3 = ranked.head(3)
    top3_names = top3["config"].tolist()
    log("RUN_FAMILY top3 by val calmar: " + ", ".join(
        f"{r.config}(calmar={r.calmar:.3f})" for r in top3.itertuples()))

    # ---- 5. 测试段一次性确认：仅前 3 名 ----
    for r in top3.itertuples():
        summary_rows.append(run_one(cache, "test", r.pool, r.selection, r.exit_rule,
                                    n_slots=10, horizon=20, persist=True))
    log("RUN_FAMILY test top3 done")

    # ---- 6. 消融 ≤9：前 3 名 × {N=5, N=20, H=10}，仅 val ----
    for r in top3.itertuples():
        for n_slots, horizon in [(5, 20), (20, 20), (10, 10)]:
            summary_rows.append(run_one(cache, "val", r.pool, r.selection, r.exit_rule,
                                        n_slots=n_slots, horizon=horizon, persist=True))
    log("RUN_FAMILY ablation 9/9 done")

    # ---- 指数对照 ----
    index_rows = {}
    for seg, (s, e) in SEGMENTS.items():
        for code in [INDEX_MAIN] + INDEX_REFS:
            index_rows[(seg, code)] = index_metrics(code, s, e)
    log("RUN_FAMILY index benchmarks done")

    # ---- 7. family_summary.csv ----
    all_years = sorted({y for r in summary_rows for y in r["yearly_returns"]})
    flat_rows = []
    for r in summary_rows:
        row = {k: v for k, v in r.items() if k != "yearly_returns"}
        for y in all_years:
            row[f"ret_{y}"] = r["yearly_returns"].get(y, np.nan)
        flat_rows.append(row)
    summary_df = pd.DataFrame(flat_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    log(f"RUN_FAMILY summary saved: {SUMMARY_CSV} rows={len(summary_df)}")

    # ---- strategy_family_report.md ----
    write_report(summary_df, top3_names, index_rows)
    log(f"RUN_FAMILY report saved: {REPORT_MD}")
    log(f"RUN_FAMILY DONE total_elapsed={time.time() - t_all:.1f}s")


def write_report(summary_df: pd.DataFrame, top3_names: list[str],
                 index_rows: dict) -> None:
    L: list[str] = []
    val = summary_df[summary_df["segment"] == "val"].copy()
    tl = summary_df[summary_df["segment"] == "train_late"].copy()
    test = summary_df[summary_df["segment"] == "test"].copy()
    main32_val = val[(val["n_slots"] == 10) & (val["horizon"] == 20)].copy()
    main32_val = main32_val.sort_values(["calmar", "ann_return"], ascending=[False, False])

    L.append("# 策略家族全量回测报告（issue #18）")
    L.append("")
    L.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}。")
    L.append("")
    L.append("## 1. 协议与口径")
    L.append("")
    L.append("协议来源：issue #18 预登记 + 2026-09-02 执行口径补充 comment。")
    L.append("引擎：v3_pipeline/scripts/strategy_engine.py（独立校验 4/4 通过后冻结，本次未做任何修改）。")
    L.append("本金 100 万，N=10 仓并发，单仓等权 = 信号日收盘权益/N，A 股整手，现金不生息。")
    L.append("入场 T+1 开盘价（涨停/无行情放弃不排队），T+1 规则买入当日不可卖。")
    L.append("出场 E1 满 H=20 日收盘卖；E2 止盈 +2×ATR；E3 止损 -1×ATR；E4=E2+E3（同日双触发取止损）；跌停顺延。")
    L.append("成本：佣金双边万 2.5（最低 5 元）+ 印花税卖出 0.1%（2023-08-28 起 0.05%）+ 滑点单边 0.1%。")
    L.append("S0 随机取舍 seed=42 单次运行（预登记口径，不作多 seed 平均）。")
    L.append("切分：val 2019-01-01~2022-10-31（主迭代）、train 后段 2015-01-01~2018-12-31（稳健性参考）、test 2022-11-01~2026-05-31（仅前 3 名一次性确认）。")
    L.append("年换手工径 = 年卖出笔数 × 平均单仓入场名义 / 平均权益。")
    L.append("成本占毛利比 = 总费用（佣金+印花税）÷ 总毛利（含滑点价差的毛盈亏），毛利 ≤0 记 NaN。")
    L.append("夏普为日频收益年化（√252）。")
    L.append("排名口径：val 段卡玛降序，平局取年化高者。")
    L.append("")
    L.append("## 2. 指数对照（同段同口径）")
    L.append("")
    idx_rows = []
    for seg, label in [("train_late", "train后段"), ("val", "val"), ("test", "test")]:
        for code, nm in [("000905.SH", "中证500"), ("000300.SH", "沪深300"), ("000852.SH", "中证1000")]:
            m = index_rows[(seg, code)]
            idx_rows.append([label, f"{nm}({code})", pct(m["ann_return"]), pct(m["max_dd"])])
    L.append(md_table(["段", "指数", "年化%", "最大回撤%"], idx_rows))
    L.append("")
    L.append("## 3. 主家族 32 配置 val 段对比（按卡玛降序）")
    L.append("")
    L.append(md_table(METRIC_HEAD, family_rows(main32_val)))
    L.append("")
    L.append("## 4. train 后段稳健性参考（2015-2018，仅参考性质，未参与排名）")
    L.append("")
    tl_sorted = tl.sort_values(["calmar", "ann_return"], ascending=[False, False])
    L.append(md_table(METRIC_HEAD, family_rows(tl_sorted)))
    L.append("")
    L.append("## 5. 前 3 名测试段一次性确认（2022-11-01~2026-05-31）")
    L.append("")
    zz500_test = index_rows[("test", "000905.SH")]
    s0_test = summary_df[(summary_df["segment"] == "test")]
    # 超额 vs 同池 S0：test 段只跑了前 3 名，若同池 S0 未进前 3 则标注未跑（预登记限制）。
    test_rows = []
    for _, r in test.iterrows():
        s0_same_pool = s0_test[(s0_test["pool"] == r["pool"]) &
                               (s0_test["selection"] == "S0") &
                               (s0_test["n_slots"] == 10) & (s0_test["horizon"] == 20)]
        if len(s0_same_pool):
            ex_s0 = r["ann_return"] - s0_same_pool["ann_return"].iloc[0]
            ex_s0_s = pct(ex_s0)
        else:
            ex_s0_s = "未跑(非前3)"
        ex_idx = r["ann_return"] - zz500_test["ann_return"]
        test_rows.append([r["config"], pct(r["ann_return"]), pct(r["max_dd"]),
                          ratio(r["sharpe"]), ratio(r["calmar"]), ratio(r["turnover"]),
                          ratio(r["cost_ratio"]), pct(r["coverage"]), pct(r["utilization"]),
                          pct(ex_idx), ex_s0_s])
    L.append(md_table(METRIC_HEAD + ["超额vs中证500(pp)", "超额vs同池S0(pp)"], test_rows))
    L.append("")
    L.append("### 5.1 前 3 名测试段逐年收益（%）")
    L.append("")
    all_yr = sorted([c for c in test.columns if c.startswith("ret_") and not test[c].isna().all()])
    yrows = []
    for _, r in test.iterrows():
        yrows.append([r["config"]] + [pct(r[c]) if pd.notna(r[c]) else "-" for c in all_yr])
    L.append(md_table(["配置"] + [c.replace("ret_", "") for c in all_yr], yrows))
    L.append("")
    # 中证500 test 逐年（供对照）
    idx_df = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index/000905.SH.parquet"),
                             columns=["trade_date", "close"])
    idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"])
    s, e = SEGMENTS["test"]
    idx_df = idx_df[(idx_df["trade_date"] >= s) & (idx_df["trade_date"] <= e)].sort_values("trade_date")
    close_s = pd.Series(idx_df["close"].values, index=idx_df["trade_date"])
    yr_ends = close_s.groupby(close_s.index.year).last()
    prev_v = close_s.iloc[0]
    irows = []
    for y, v in yr_ends.items():
        irows.append([str(y), pct(v / prev_v - 1.0)])
        prev_v = v
    L.append("### 5.2 中证500 测试段逐年收益（%，对照）")
    L.append("")
    L.append(md_table(["年份", "中证500%"], irows))
    L.append("")
    L.append("## 6. 消融（前 3 名 × {N=5, N=20, H=10}，仅 val 段）")
    L.append("")
    abl = val[~val["config"].isin(main32_val["config"])].copy()

    # 按所属基座（前 3 名顺序）排
    def base_of(c):
        for t in top3_names:
            if c.startswith(t.rsplit("_N", 1)[0]):
                return top3_names.index(t)
        return 99
    abl["base"] = abl["config"].apply(base_of)
    abl = abl.sort_values(["base", "n_slots", "horizon"])
    L.append(md_table(METRIC_HEAD, family_rows(abl)))
    L.append("")
    L.append("## 7. 验收线判定")
    L.append("")
    L.append("验收线（用户标准操作化）：测试段扣成本年化 > 中证500 年化 且 最大回撤 ≤15% 且 扣成本收益为正。")
    L.append("")
    verdict_rows = []
    n_pass = 0
    for _, r in test.iterrows():
        c1 = r["ann_return"] > zz500_test["ann_return"]
        c2 = abs(r["max_dd"]) <= 0.15
        c3 = r["ann_return"] > 0
        ok = c1 and c2 and c3
        n_pass += int(ok)
        verdict_rows.append([r["config"], pct(r["ann_return"]),
                             pct(zz500_test["ann_return"]),
                             "是" if c1 else "否",
                             pct(r["max_dd"]), "是" if c2 else "否",
                             "是" if c3 else "否",
                             "PASS" if ok else "FAIL"])
    L.append(md_table(["配置", "年化%", "中证500年化%", "年化>指数", "最大回撤%", "回撤≤15%",
                       "收益为正", "判定"], verdict_rows))
    L.append("")
    if n_pass == 0:
        L.append("结论：前 3 名无一通过验收线 —— 按预登记，池级策略方向关闭，战役终局。")
    elif n_pass == len(test):
        L.append("结论：前 3 名全部通过验收线 —— GO，进入实盘论证。")
    else:
        L.append(f"结论：{n_pass}/{len(test)} 通过验收线 —— 部分通过，按预登记未定义全满足/全不满足之外情形，需用户拍板。")
    L.append("")
    L.append("## 8. 产物清单")
    L.append("")
    L.append("- runs/val_<池>_<取舍>_<出场>_N10/ × 32：equity_curve.parquet / trades.parquet / stats.json。")
    L.append("- runs/test_<配置>_N10/ × 3：前 3 名测试段落盘。")
    L.append("- runs/val_<消融配置>/ × 9：消融落盘。")
    L.append("- family_summary.csv：全量配置 × 段指标表。")
    L.append("- progress.log：逐配置心跳。")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
