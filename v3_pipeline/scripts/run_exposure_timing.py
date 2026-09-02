"""池级敞口择时家族全量回测 runner —— issue #19 预登记执行。

执行清单（顺序严格）：
 1. val 段 16 配置：底座(B1=backup_S3_E1_N10_H20 / B2=backup_S1_E1_N10_H20)
    × timing(T1/T2/T3/T4) × gate(G1/G2)，2019-01-01~2022-10-31，seed=42，
    落盘 runs/val_<底座>_<timing>_<gate>/。
 2. 稳健性段 16 配置：同矩阵，2015-01-01~2018-12-31，只进汇总表（不落盘 runs）。
 3. 指标：复用 #18 run_strategy_family.compute_metrics 口径（扣成本年化/最大回撤/夏普/
    卡玛/年换手/成本占毛利比/逐年收益/资金利用率/覆盖率），外加择时专项：
    敞口时间占比（全/半/0 各多少天）、闸门拦截笔数、G2 强制退出笔数、降级天数。
 4. 对照：无择时底座 B1/B2 各段数值直接引用 #18 归档 family_summary.csv（不重跑）；
    中证500 各段年化/回撤引用 #18 归档 strategy_family_report.md §2。
 5. 排名：val 卡玛降序（平局取年化高者）定前 3 名。
 6. 测试段一次性：前 3 名跑 2022-11-01~2026-05-31，落盘 runs/test_<配置>/，只跑 3 个。
 7. 汇总：exposure_timing_summary.csv + exposure_timing_report.md。

引擎 strategy_engine_v2.py 已冻结（校验 4/4 通过，见 validation_v2_report.md），
本脚本只读调用 run_config，不修改任何已有文件。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline/scripts"))
from strategy_engine_v2 import run_config  # noqa: E402
from run_strategy_family import compute_metrics  # noqa: E402  # 复用 #18 指标口径

OUT_ROOT = os.path.join(REPO_ROOT, "v3_pipeline/reports/exposure_timing")
RUNS_DIR = os.path.join(OUT_ROOT, "runs")
PROGRESS_LOG = os.path.join(OUT_ROOT, "progress.log")
SUMMARY_CSV = os.path.join(OUT_ROOT, "exposure_timing_summary.csv")
REPORT_MD = os.path.join(OUT_ROOT, "exposure_timing_report.md")
FAMILY18_CSV = os.path.join(REPO_ROOT,
                            "v3_pipeline/reports/strategy_real_trading/family_summary.csv")

SEED = 42
SEGMENTS = {
    "train_late": ("2015-01-01", "2018-12-31"),
    "val": ("2019-01-01", "2022-10-31"),
    "test": ("2022-11-01", "2026-05-31"),
}
BASES = {
    "B1": dict(pool="backup", selection="S3", exit_rule="E1", horizon=20, n_slots=10),
    "B2": dict(pool="backup", selection="S1", exit_rule="E1", horizon=20, n_slots=10),
}
BASE_CONFIG18 = {"B1": "backup_S3_E1_N10", "B2": "backup_S1_E1_N10"}
TIMINGS = ("T1", "T2", "T3", "T4")
GATES = ("G1", "G2")

# 中证500（000905.SH）同段年化/最大回撤 —— 引用 #18 归档 strategy_family_report.md §2，不重算
ZZ500_REF = {
    "train_late": {"ann_return": -0.0638, "max_dd": -0.6520},
    "val": {"ann_return": 0.0930, "max_dd": -0.3157},
    "test": {"ann_return": 0.1003, "max_dd": -0.3115},
}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 择时专项指标
def timing_metrics(result: dict) -> dict:
    """敞口时间占比（全/半/0 天数与占比）+ 闸门/强制退出计数。"""
    expo = result["exposure"]
    n = len(expo)
    vc = expo["exposure"].value_counts()
    full = int(vc.get(1.0, 0))
    half = int(vc.get(0.5, 0))
    zero = int(vc.get(0.0, 0))
    stats = result["stats"]
    return dict(
        expo_days=n,
        expo_full_days=full, expo_half_days=half, expo_zero_days=zero,
        expo_full_frac=full / n if n else np.nan,
        expo_half_frac=half / n if n else np.nan,
        expo_zero_frac=zero / n if n else np.nan,
        gate_blocked=int(stats.get("dropped_timing_gate", 0)),
        forced_exits=int(stats.get("forced_exits", 0)),
        forced_deferred=int(stats.get("forced_deferred", 0)),
        timing_fallback_days=int(stats.get("timing_fallback_days", 0)),
    )


# ---------------------------------------------------------------- 单配置执行
def run_one(base: str, timing: str, gate: str, segment: str, persist: bool) -> dict:
    name = f"{base}_{timing}_{gate}"
    start, end = SEGMENTS[segment]
    out_dir = os.path.join(RUNS_DIR, f"{segment}_{name}") if persist else None
    t0 = time.time()
    log(f"RUN {segment} {name} start [{start}..{end}] persist={persist}")
    result = run_config(start=start, end=end, seed=SEED, repo_root=REPO_ROOT,
                        out_dir=out_dir, log_path=PROGRESS_LOG,
                        timing=timing, gate=gate, **BASES[base])
    elapsed = time.time() - t0
    metrics = compute_metrics(result["equity"], result["trades"], result["stats"])
    tm = timing_metrics(result)
    row = dict(segment=segment, config=name, base=base, timing=timing, gate=gate,
               pool=BASES[base]["pool"], selection=BASES[base]["selection"],
               exit_rule=BASES[base]["exit_rule"], n_slots=BASES[base]["n_slots"],
               horizon=BASES[base]["horizon"], seed=SEED, source="run",
               elapsed_s=round(elapsed, 2),
               **{k: v for k, v in metrics.items() if k != "yearly_returns"},
               **tm,
               yearly_returns=metrics.get("yearly_returns", {}))
    log(f"RUN {segment} {name} done in {elapsed:.1f}s | "
        f"ann={metrics.get('ann_return', float('nan')):+.4f} "
        f"mdd={metrics.get('max_dd', float('nan')):.4f} "
        f"calmar={metrics.get('calmar', float('nan')):.3f} "
        f"expo(F/H/0)={tm['expo_full_days']}/{tm['expo_half_days']}/{tm['expo_zero_days']} "
        f"gate_blocked={tm['gate_blocked']} forced={tm['forced_exits']} "
        f"trades={metrics.get('n_trades', 0)}")
    return row


# ---------------------------------------------------------------- #18 归档基线引用（不重跑）
def load_baselines_18() -> list[dict]:
    df18 = pd.read_csv(FAMILY18_CSV)
    ret_cols = [c for c in df18.columns if c.startswith("ret_")]
    rows: list[dict] = []
    for base, cfg18 in BASE_CONFIG18.items():
        for seg in SEGMENTS:
            r = df18[(df18["segment"] == seg) & (df18["config"] == cfg18)]
            assert len(r) == 1, f"#18 archive missing {seg}/{cfg18}"
            r = r.iloc[0]
            yearly = {c.replace("ret_", ""): float(r[c]) for c in ret_cols
                      if pd.notna(r[c])}
            rows.append(dict(
                segment=seg, config=f"{base}_no_timing", base=base,
                timing="none", gate="none",
                pool=BASES[base]["pool"], selection=BASES[base]["selection"],
                exit_rule=BASES[base]["exit_rule"], n_slots=10, horizon=20,
                seed=SEED, source="issue18_archive", elapsed_s=np.nan,
                ann_return=float(r["ann_return"]), max_dd=float(r["max_dd"]),
                sharpe=float(r["sharpe"]), calmar=float(r["calmar"]),
                turnover=float(r["turnover"]),
                cost_ratio=float(r["cost_ratio"]) if pd.notna(r["cost_ratio"]) else np.nan,
                utilization=float(r["utilization"]), coverage=float(r["coverage"]),
                final_equity=float(r["final_equity"]), n_trades=int(r["n_trades"]),
                entered=int(r["entered"]), total_signals=int(r["total_signals"]),
                expo_days=np.nan, expo_full_days=np.nan, expo_half_days=np.nan,
                expo_zero_days=np.nan, expo_full_frac=np.nan, expo_half_frac=np.nan,
                expo_zero_frac=np.nan, gate_blocked=np.nan, forced_exits=np.nan,
                forced_deferred=np.nan, timing_fallback_days=np.nan,
                yearly_returns=yearly))
    return rows


# ---------------------------------------------------------------- 报告辅助
def pct(x, digits=2):
    return "NaN" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x * 100:.{digits}f}"


def pp(x, digits=2):
    """百分点差值，带符号。"""
    return "NaN" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x * 100:+.{digits}f}"


def ratio(x, digits=3):
    return "NaN" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


METRIC_HEAD = ["配置", "年化%", "最大回撤%", "夏普", "卡玛", "年换手", "成本/毛利",
               "覆盖率%", "利用率%"]


def family_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in df.iterrows():
        rows.append([r["config"], pct(r["ann_return"]), pct(r["max_dd"]),
                     ratio(r["sharpe"]), ratio(r["calmar"]), ratio(r["turnover"]),
                     ratio(r["cost_ratio"]), pct(r["coverage"]), pct(r["utilization"])])
    return rows


def increment_rows(timed_df: pd.DataFrame, base_df: pd.DataFrame) -> list[list[str]]:
    """增量表：择时配置 vs 同底座无择时 —— 回撤降幅(pp) / 年化变化(pp) / 卡玛变化。"""
    rows = []
    for _, r in timed_df.iterrows():
        b = base_df[base_df["base"] == r["base"]]
        assert len(b) == 1
        b = b.iloc[0]
        dd_cut = abs(b["max_dd"]) - abs(r["max_dd"])     # >0 = 回撤变浅
        d_ann = r["ann_return"] - b["ann_return"]
        d_calmar = r["calmar"] - b["calmar"]
        rows.append([r["config"], r["base"],
                     pct(b["max_dd"]), pct(r["max_dd"]), pp(dd_cut),
                     pct(b["ann_return"]), pct(r["ann_return"]), pp(d_ann),
                     pp(d_calmar)])
    return rows


INCR_HEAD = ["配置", "底座", "底座回撤%", "择时回撤%", "回撤降幅(pp)",
             "底座年化%", "择时年化%", "年化变化(pp)", "卡玛变化(pp)"]


# ---------------------------------------------------------------- 主流程
def main() -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    t_all = time.time()
    log("RUN_EXPOSURE_TIMING START (issue #19): 16 配置 val + train_late + 前3 test")

    summary_rows: list[dict] = []

    # ---- 1. val 段 16 配置（落盘）----
    for base in BASES:
        for timing in TIMINGS:
            for gate in GATES:
                summary_rows.append(run_one(base, timing, gate, "val", persist=True))
    log("RUN_EXPOSURE_TIMING val 16/16 done")

    # ---- 2. 稳健性段 16 配置（仅汇总，不落盘 runs）----
    for base in BASES:
        for timing in TIMINGS:
            for gate in GATES:
                summary_rows.append(run_one(base, timing, gate, "train_late",
                                            persist=False))
    log("RUN_EXPOSURE_TIMING train_late 16/16 done（参考段，不落盘 runs）")

    # ---- 3. #18 归档基线引用（对照，不重跑）----
    summary_rows.extend(load_baselines_18())

    # ---- 4. 排名：val 卡玛降序，平局取年化高者，定前 3 名（仅 16 个择时配置）----
    timed_rows = [r for r in summary_rows if r["source"] == "run"]
    val_df = pd.DataFrame([r for r in timed_rows if r["segment"] == "val"])
    ranked = val_df.sort_values(["calmar", "ann_return"],
                                ascending=[False, False], na_position="last")
    top3 = ranked.head(3)
    top3_names = top3["config"].tolist()
    log("RUN_EXPOSURE_TIMING top3 by val calmar: " + ", ".join(
        f"{r.config}(calmar={r.calmar:.3f})" for r in top3.itertuples()))

    # ---- 5. 测试段一次性确认：仅前 3 名 ----
    for r in top3.itertuples():
        summary_rows.append(run_one(r.base, r.timing, r.gate, "test", persist=True))
    log("RUN_EXPOSURE_TIMING test top3 done")

    # ---- 6. exposure_timing_summary.csv ----
    all_years = sorted({y for r in summary_rows for y in r["yearly_returns"]})
    flat_rows = []
    for r in summary_rows:
        row = {k: v for k, v in r.items() if k != "yearly_returns"}
        for y in all_years:
            row[f"ret_{y}"] = r["yearly_returns"].get(y, np.nan)
        flat_rows.append(row)
    summary_df = pd.DataFrame(flat_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    log(f"RUN_EXPOSURE_TIMING summary saved: {SUMMARY_CSV} rows={len(summary_df)}")

    # ---- 7. exposure_timing_report.md ----
    write_report(summary_df, top3_names)
    log(f"RUN_EXPOSURE_TIMING report saved: {REPORT_MD}")
    log(f"RUN_EXPOSURE_TIMING DONE total_elapsed={time.time() - t_all:.1f}s")


def write_report(summary_df: pd.DataFrame, top3_names: list[str]) -> None:
    L: list[str] = []
    run_df = summary_df[summary_df["source"] == "run"].copy()
    base_df_all = summary_df[summary_df["source"] == "issue18_archive"].copy()
    val = run_df[run_df["segment"] == "val"].copy()
    tl = run_df[run_df["segment"] == "train_late"].copy()
    test = run_df[run_df["segment"] == "test"].copy()
    val = val.sort_values(["calmar", "ann_return"], ascending=[False, False])
    tl = tl.sort_values(["calmar", "ann_return"], ascending=[False, False])
    base_val = base_df_all[base_df_all["segment"] == "val"]
    base_test = base_df_all[base_df_all["segment"] == "test"]
    zz_test = ZZ500_REF["test"]

    L.append("# 池级敞口择时家族全量回测报告（issue #19）")
    L.append("")
    L.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}。")
    L.append("")
    L.append("## 1. 协议与口径")
    L.append("")
    L.append("协议来源：issue #19 预登记（池级敞口择时：压回撤的最后残余方向）。")
    L.append("引擎：v3_pipeline/scripts/strategy_engine_v2.py（v1 冻结引擎 + 择时覆盖层，校验 4/4 通过后冻结，见 validation_v2_report.md）。")
    L.append("底座：B1=backup_S3_E1_N10_H20、B2=backup_S1_E1_N10_H20（#18 val 卡玛前 2，避免用测试段信息选底座）。")
    L.append("择时信号源：中证500（000905.SH）日线收盘；t 日决策只用 ≤t-1 日收盘信息，引擎内置无泄漏硬断言。")
    L.append("T1 趋势200：t-1 收盘 > 200 日均线 → 全敞口，否则 0。")
    L.append("T2 趋势60：同上，60 日均线。")
    L.append("T3 回撤阶梯：t-1 收盘距 250 日最高收盘回撤 <10% → 全；10%~20% → 半；>20% → 0。")
    L.append("T4 波动闸门：t-1 的 20 日已实现波动 > 其过去 250 日的 90 分位 → 半，否则全。")
    L.append("G1 软闸门：只拦新入场，已有持仓自然退出。")
    L.append("G2 硬闸门：敞口降档时强制退出超限持仓（浮盈最低者优先，顺延跌停，遵守 T+1）。")
    L.append("敞口→槽位：当日可用槽位 = round_half_up(N × 敞口)；单仓预算恒为权益/N=10，不随敞口变。")
    L.append("切分：val 2019-01-01~2022-10-31（主迭代，16 配置落盘）；train 后段 2015-01-01~2018-12-31（稳健性参考，仅汇总）；test 2022-11-01~2026-05-31（仅前 3 名一次性确认，本项目测试段第三次也是最后一次使用）。")
    L.append("指标口径与 #18 完全一致（复用 run_strategy_family.compute_metrics）：扣成本年化、最大回撤、夏普（日频√252）、卡玛、年换手、成本占毛利比、逐年收益、资金利用率、覆盖率。")
    L.append("择时专项指标：敞口时间占比（全/半/0 各多少天）、闸门拦截笔数、G2 强制退出笔数、窗口不足降级天数。")
    L.append("无择时底座 B1/B2 各段数值直接引用 #18 归档 family_summary.csv（本报告标注 issue18_archive，不重跑）。")
    L.append("中证500 同段年化/回撤引用 #18 归档 strategy_family_report.md §2：val +9.30%/-31.57%、test +10.03%/-31.15%、train 后段 -6.38%/-65.20%。")
    L.append("排名口径：val 段卡玛降序，平局取年化高者，取前 3 名进测试段。")
    L.append("")
    L.append("## 2. val 段 16 配置对比（2019-01~2022-10，按卡玛降序；末两行为 #18 归档无择时底座对照）")
    L.append("")
    val_with_base = pd.concat([val, base_val])
    L.append(md_table(METRIC_HEAD, family_rows(val_with_base)))
    L.append("")
    L.append(f"对照：中证500 val 段年化 {pct(ZZ500_REF['val']['ann_return'])}%、最大回撤 {pct(ZZ500_REF['val']['max_dd'])}%（#18 归档）。")
    L.append("")
    L.append("## 3. 择时专项：敞口时间占比与闸门行为（val 段）")
    L.append("")
    expo_rows = []
    for _, r in val.iterrows():
        expo_rows.append([r["config"],
                          f"{int(r['expo_full_days'])}({pct(r['expo_full_frac'])}%)",
                          f"{int(r['expo_half_days'])}({pct(r['expo_half_frac'])}%)",
                          f"{int(r['expo_zero_days'])}({pct(r['expo_zero_frac'])}%)",
                          str(int(r["gate_blocked"])),
                          str(int(r["forced_exits"])),
                          str(int(r["forced_deferred"])),
                          str(int(r["timing_fallback_days"]))])
    L.append(md_table(["配置", "全敞口天数(占比)", "半敞口天数(占比)", "零敞口天数(占比)",
                       "闸门拦截笔数", "G2强制退出笔数", "G2顺延笔数", "降级天数"],
                      expo_rows))
    L.append("")
    L.append("## 4. train 后段稳健性参考（2015-2018，含 2015 股灾与 2018 熊市；仅参考，未参与排名）")
    L.append("")
    tl_with_base = pd.concat([tl, base_df_all[base_df_all["segment"] == "train_late"]])
    L.append(md_table(METRIC_HEAD, family_rows(tl_with_base)))
    L.append("")
    L.append(f"对照：中证500 train 后段年化 {pct(ZZ500_REF['train_late']['ann_return'])}%、最大回撤 {pct(ZZ500_REF['train_late']['max_dd'])}%（#18 归档）。")
    L.append("")
    L.append("## 5. 增量判定（一）：val 段择时 vs 同底座无择时")
    L.append("")
    L.append(md_table(INCR_HEAD, increment_rows(val, base_val)))
    L.append("")
    L.append("## 6. 前 3 名测试段一次性确认（2022-11-01~2026-05-31）")
    L.append("")
    L.append("val 卡玛前 3 名：" + "、".join(
        f"{r.config}（卡玛 {ratio(r.calmar)}）" for r in val.head(3).itertuples()) + "。")
    L.append("")
    test = test.sort_values(["calmar", "ann_return"], ascending=[False, False])
    test_rows = []
    for _, r in test.iterrows():
        ex_idx = r["ann_return"] - zz_test["ann_return"]
        test_rows.append([r["config"], pct(r["ann_return"]), pct(r["max_dd"]),
                          ratio(r["sharpe"]), ratio(r["calmar"]), ratio(r["turnover"]),
                          ratio(r["cost_ratio"]), pct(r["coverage"]),
                          pct(r["utilization"]), pp(ex_idx)])
    L.append(md_table(METRIC_HEAD + ["超额vs中证500(pp)"], test_rows))
    L.append("")
    L.append(f"对照：中证500 test 段年化 {pct(zz_test['ann_return'])}%、最大回撤 {pct(zz_test['max_dd'])}%（#18 归档）。")
    L.append("")
    L.append("无择时底座 test 段（#18 归档）：" + "；".join(
        f"{r.config} 年化 {pct(r.ann_return)}%、回撤 {pct(r.max_dd)}%、卡玛 {ratio(r.calmar)}"
        for r in base_test.itertuples()) + "。")
    L.append("")
    L.append("### 6.1 前 3 名测试段逐年收益（%）")
    L.append("")
    yr_cols = sorted([c for c in test.columns
                      if c.startswith("ret_") and not test[c].isna().all()])
    yrows = []
    for _, r in test.iterrows():
        yrows.append([r["config"]] + [pct(r[c]) if pd.notna(r[c]) else "-"
                                      for c in yr_cols])
    base_yrows = []
    for _, r in base_test.iterrows():
        base_yrows.append([r["config"]] + [pct(r[c]) if pd.notna(r[c]) else "-"
                                           for c in yr_cols])
    L.append(md_table(["配置"] + [c.replace("ret_", "") for c in yr_cols],
                      yrows + base_yrows))
    L.append("")
    L.append("## 7. 增量判定（二）：test 段择时 vs 同底座无择时（回答“择时买没买到回撤”）")
    L.append("")
    L.append(md_table(INCR_HEAD, increment_rows(test, base_test)))
    L.append("")
    L.append("## 8. 验收判定")
    L.append("")
    L.append("验收线（与 #18 同线）：测试段扣成本年化 > 中证500 年化（+10.03%）且最大回撤 ≤15% 且为正。")
    L.append("")
    verdict_rows = []
    n_pass = 0
    for _, r in test.iterrows():
        c1 = r["ann_return"] > zz_test["ann_return"]
        c2 = abs(r["max_dd"]) <= 0.15
        c3 = r["ann_return"] > 0
        ok = c1 and c2 and c3
        n_pass += int(ok)
        verdict_rows.append([r["config"], pct(r["ann_return"]),
                             pct(zz_test["ann_return"]), "是" if c1 else "否",
                             pct(r["max_dd"]), "是" if c2 else "否",
                             "是" if c3 else "否", "PASS" if ok else "FAIL"])
    L.append(md_table(["配置", "年化%", "中证500年化%", "年化>指数", "最大回撤%",
                       "回撤≤15%", "收益为正", "判定"], verdict_rows))
    L.append("")
    if n_pass == 0:
        L.append("结论：前 3 名无一通过验收线 —— 按预登记，择时方向证伪，背离狙击项目所有方向关闭，出项目总档。")
    elif n_pass == len(test):
        L.append("结论：前 3 名全部通过验收线 —— 有活口，与用户讨论实盘论证。")
    else:
        L.append(f"结论：{n_pass}/{len(test)} 通过验收线 —— 部分通过，预登记未定义此情形，需用户拍板。")
    L.append("")
    L.append("## 9. 产物清单")
    L.append("")
    L.append("- runs/val_<底座>_<timing>_<gate>/ × 16：equity_curve.parquet / trades.parquet / exposure.parquet / stats.json。")
    L.append("- runs/test_<配置>/ × 3：前 3 名测试段落盘（同四件套）。")
    L.append("- exposure_timing_summary.csv：16 配置 ×（val/train 后段）+ 前 3 名 test + 6 行 #18 归档基线引用，全指标。")
    L.append("- progress.log：逐配置心跳（追加）。")
    L.append("- validation_v2_report.md：引擎 v2 冻结前校验记录（4/4 通过）。")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
