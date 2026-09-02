"""strategy_engine_v2 独立校验脚本 —— issue #19 择时覆盖层冻结前验收。

四项校验（全过才冻结）：
 1. 回归断言（最重要）：timing=None 跑 B1(backup_S3_E1_N10_H20) 与 B2(backup_S1_E1_N10_H20)
    val 段（2019-01-01~2022-10-31, seed=42），资金曲线与交易明细同 #18 归档
    runs/val_backup_S3_E1_N10/ 与 runs/val_backup_S1_E1_N10/ 落盘 parquet 逐格一致（容差 1e-6）。
 2. 择时专项合成测试：
    (a) 指数满仓中途降为 0 —— G1 后无新入场、旧仓按原规则自然退出；
        G2 超限仓次日强制退出，且 T+1 不可卖与跌停顺延仍生效；
    (b) 槽位 round 行为（round_half_up，10×0.5=5，5×0.5=3）与 T3 半敞口入场上限；
    (c) 指数历史不足窗口时降级为全敞口并记 warning；
    (d) G2 强制退出优先级：持有浮盈最低者优先。
 3. 无泄漏断言：校验方用 pandas shift/rolling 独立重算 T1..T4 全规则敞口序列，
    与引擎 compute_exposure 逐日比对一致，且每日 max_src_date < date（<= t-1 信息）。
 4. 冒烟：B1+T1+G1 val 段跑通出曲线，敞口序列分布、强制退出笔数写日志并落盘。

产物：v3_pipeline/reports/exposure_timing/validation_v2_report.md（每句一行）
     v3_pipeline/reports/exposure_timing/smoke_B1_T1_G1/{equity_curve,trades,exposure}.parquet + stats.json
退出码：全部通过 0，任一失败 1。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy_engine as se  # noqa: E402
import strategy_engine_v2 as se2  # noqa: E402

REPO = "/home/karl/repos/personal/stock_qt_nd"
ART = os.path.join(REPO, "v3_pipeline/reports/exposure_timing")
ARCH = os.path.join(REPO, "v3_pipeline/reports/strategy_real_trading/runs")
LOG = os.path.join(ART, "progress.log")
REPORT = os.path.join(ART, "validation_v2_report.md")
VAL_START, VAL_END, SEED = "2019-01-01", "2022-10-31", 42
TOL = 1e-6

CHECKS: list[tuple[str, bool, str]] = []


def log(msg: str) -> None:
    se._log(LOG, msg)


def record(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    log(f"CHECK [{'PASS' if passed else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- 校验 1：回归断言（逐格对账）
def compare_frames(got: pd.DataFrame, want: pd.DataFrame, label: str) -> str:
    """逐格比对；任何不一致抛 AssertionError。返回最大绝对差描述。"""
    assert list(got.columns) == list(want.columns), \
        f"{label}: columns {list(got.columns)} != {list(want.columns)}"
    assert len(got) == len(want), f"{label}: rows {len(got)} != {len(want)}"
    got = got.reset_index(drop=True)
    want = want.reset_index(drop=True)
    max_diff = 0.0
    for col in want.columns:
        g, w = got[col], want[col]
        if pd.api.types.is_datetime64_any_dtype(w) or pd.api.types.is_datetime64_any_dtype(g):
            assert (pd.to_datetime(g) == pd.to_datetime(w)).all(), f"{label}.{col}: dates differ"
        elif pd.api.types.is_float_dtype(w) or pd.api.types.is_float_dtype(g):
            gn = g.to_numpy(dtype=float)
            wn = w.to_numpy(dtype=float)
            both_nan = np.isnan(gn) & np.isnan(wn)
            diff = np.where(both_nan, 0.0, np.abs(gn - wn))
            d = float(np.nanmax(diff)) if len(diff) else 0.0
            max_diff = max(max_diff, d)
            assert d <= TOL, f"{label}.{col}: max abs diff {d} > {TOL}"
        else:
            assert (g.astype(str) == w.astype(str)).all(), f"{label}.{col}: values differ"
    return f"rows={len(want)} max_abs_diff={max_diff:.3e}"


def check_regression() -> None:
    details = []
    for label, sel in (("B1", "S3"), ("B2", "S1")):
        arch_dir = os.path.join(ARCH, f"val_backup_{sel}_E1_N10")
        r = se2.run_config(pool="backup", selection=sel, exit_rule="E1", horizon=20,
                           n_slots=10, start=VAL_START, end=VAL_END, seed=SEED,
                           repo_root=REPO, log_path=LOG, timing=None)
        arch_eq = pd.read_parquet(os.path.join(arch_dir, "equity_curve.parquet"))
        arch_tr = pd.read_parquet(os.path.join(arch_dir, "trades.parquet"))
        d1 = compare_frames(r["equity"], arch_eq, f"{label}.equity")
        d2 = compare_frames(r["trades"], arch_tr, f"{label}.trades")
        arch_stats = json.load(open(os.path.join(arch_dir, "stats.json")))
        fe_diff = abs(r["stats"]["final_equity"] - arch_stats["final_equity"])
        assert fe_diff <= TOL, f"{label} final_equity diff {fe_diff}"
        assert r["stats"]["entered"] == arch_stats["entered"]
        details.append(f"{label}(backup_{sel}_E1_N10_H20): equity[{d1}] trades[{d2}] "
                       f"final_equity={r['stats']['final_equity']:.6f} "
                       f"(归档 {arch_stats['final_equity']:.6f})")
    record("regression_timing_none_vs_18_archive", True, "；".join(details))


# ---------------------------------------------------------------- 校验 2 素材：合成市场
CAL = list(pd.bdate_range("2020-01-02", periods=10))   # d1..d10
D = {i + 1: d for i, d in enumerate(CAL)}


def _mk_daily(close_overrides: dict[tuple[str, int], float] | None = None,
              codes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """生成日线：默认 open/high/low/close 全 10 元；可按 (code, 日序号) 覆盖收盘。"""
    close_overrides = close_overrides or {}
    daily = {}
    for code in (codes or ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"]):
        rows = []
        for i, day in enumerate(CAL, start=1):
            c = close_overrides.get((code, i), 10.0)
            rows.append((day, 10.0, max(10.2, c), min(9.8, c), c))
        daily[code] = pd.DataFrame(
            rows, columns=["trade_date", "open", "high", "low", "close"]
        ).set_index("trade_date").sort_index()
    return daily


def _mk_limits(dn_overrides: dict[tuple[str, int], float] | None = None
               ) -> dict[pd.Timestamp, pd.DataFrame]:
    """涨跌停表：默认涨停 100 / 跌停 0.01（无约束）；可按 (code, 日序号) 覆盖跌停。"""
    dn_overrides = dn_overrides or {}
    limits = {}
    codes = ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"]
    for i, day in enumerate(CAL, start=1):
        rows = [(c, 100.0, dn_overrides.get((c, i), 0.01)) for c in codes]
        limits[day] = pd.DataFrame(
            rows, columns=["ts_code", "up_limit", "down_limit"]).set_index("ts_code")
    return limits


def _mk_events(codes_day: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [(c, D[di], 0.05, -0.10, float(k)) for k, (c, di) in enumerate(codes_day)],
        columns=["ts_code", "event_date", "ATRN", "RET20", "sig_idx"],
    )


def _t1_crash_index() -> pd.Series:
    """T1 合成指数：d4(2020-01-07) 之前缓慢上行（收盘>200日均线 -> 全敞口），
    d4 起暴跌至 50（d5 起敞口=0）。"""
    days = pd.bdate_range("2018-12-03", "2020-01-15")
    crash = pd.Timestamp("2020-01-07")
    vals = [100.0 + 0.01 * i if d < crash else 50.0 for i, d in enumerate(days)]
    return pd.Series(vals, index=days)


def _t3_half_index() -> pd.Series:
    """T3 合成指数：250 日窗口内最高收盘 100，近期收盘 85.5 -> 回撤 14.5% -> 半敞口。"""
    days = pd.bdate_range("2018-12-03", "2020-01-15")
    vals = []
    for d in days:
        if d == pd.Timestamp("2019-06-03"):
            vals.append(100.0)
        elif d >= pd.Timestamp("2019-12-02"):
            vals.append(85.5)
        else:
            vals.append(90.0)
    return pd.Series(vals, index=days)


def check_synthetic() -> None:
    details = []

    # ---- (a) G1 软闸门：敞口 d5 起降 0，旧仓自然退出、新信号被拦 ----
    idx = _t1_crash_index()
    expo = se2.compute_exposure(CAL, idx, "T1")
    assert expo.loc[expo["date"] == D[4], "exposure"].iloc[0] == 1.0
    assert (expo.loc[expo["date"] >= D[5], "exposure"] == 0.0).all()
    md = se.MarketData(daily=_mk_daily(), limits=_mk_limits(), calendar=CAL)
    ev = _mk_events([(f"T0{k}", 1) for k in range(1, 6)] + [("T06", 5)])
    r = se2.run_backtest_timed(ev, md, expo, n_slots=5, selection="S1",
                               exit_rule="E1", horizon=6, gate="G1")
    tr = r["trades"]
    assert r["stats"]["entered"] == 5, f"G1 entered {r['stats']['entered']} != 5"
    assert set(tr["ts_code"]) == {"T01", "T02", "T03", "T04", "T05"}, "G1: T06 不应入场"
    assert (tr["exit_reason"] == "E1").all() and (tr["exit_date"] == D[7]).all(), \
        "G1: 旧仓应按原规则自然退出（E1@d7）"
    assert (tr["entry_date"] == D[2]).all()
    details.append("(a-G1) 降档后 0 笔新入场，5 仓全部 E1@d7 自然退出")

    # ---- (a') G2 硬闸门：超限仓次日强制退出；跌停顺延仍生效 ----
    daily = _mk_daily(close_overrides={("T05", 5): 9.0, ("T05", 6): 9.0})
    limits = _mk_limits(dn_overrides={("T05", 5): 9.0})   # d5 跌停 -> 顺延；d6 恢复
    md2 = se.MarketData(daily=daily, limits=limits, calendar=CAL)
    r2 = se2.run_backtest_timed(ev, md2, expo, n_slots=5, selection="S1",
                                exit_rule="E1", horizon=6, gate="G2")
    tr2 = r2["trades"].set_index("ts_code")
    assert r2["stats"]["entered"] == 5, "G2: T06 不应入场"
    for c in ("T01", "T02", "T03", "T04"):
        row = tr2.loc[c]
        assert row["exit_reason"] == "G2_forced" and row["exit_date"] == D[5], \
            f"G2: {c} 应在降档日 d5 强制退出，实际 {row['exit_reason']}@{row['exit_date']}"
        assert row["exit_date"] > row["entry_date"], "T+1 不可卖被破坏"
    t05 = tr2.loc["T05"]
    assert t05["exit_reason"] == "G2_forced" and t05["exit_date"] == D[6], \
        f"G2: T05 跌停应顺延至 d6，实际 {t05['exit_reason']}@{t05['exit_date']}"
    assert r2["stats"]["forced_exits"] == 5 and r2["stats"]["forced_deferred"] >= 1
    assert abs(t05["exit_raw_price"] - 9.0) < 1e-9
    details.append("(a-G2) 4 仓 d5 强制退出(G2_forced)，T05 跌停顺延 d6，"
                   f"forced_exits=5 forced_deferred={r2['stats']['forced_deferred']}")

    # ---- (b) 槽位 round：round_half_up 单元断言 + T3 半敞口入场上限 ----
    assert se2.round_slots(10, 0.5) == 5 and se2.round_slots(10, 1.0) == 10
    assert se2.round_slots(10, 0.0) == 0 and se2.round_slots(5, 0.5) == 3
    expo3 = se2.compute_exposure(CAL, _t3_half_index(), "T3")
    assert (expo3["exposure"] == 0.5).all(), "T3 合成指数应全程半敞口"
    md3 = se.MarketData(daily=_mk_daily(), limits=_mk_limits(), calendar=CAL)
    ev3 = _mk_events([(f"T0{k}", 1) for k in range(1, 9)])   # 8 信号抢 5 槽
    r3 = se2.run_backtest_timed(ev3, md3, expo3, n_slots=10, selection="S1",
                                exit_rule="E1", horizon=3, gate="G1")
    assert r3["stats"]["entered"] == 5, f"半敞口 N=10 应只入 5 仓，实际 {r3['stats']['entered']}"
    assert set(r3["trades"]["ts_code"]) == {"T01", "T02", "T03", "T04", "T05"}
    details.append("(b) round_slots(10,0.5)=5；T3 半敞口下 8 信号只入 sig_idx 前 5")

    # ---- (c) 历史不足降级：全敞口 + warning ----
    short_idx = pd.Series(
        [100.0 + 0.01 * i for i in range(100)],
        index=pd.bdate_range("2019-08-01", periods=100))
    expo_e = se2.compute_exposure(CAL, short_idx, "T1", log_path=LOG)
    assert (expo_e["exposure"] == 1.0).all() and expo_e["fallback"].all()
    r5 = se2.run_backtest_timed(_mk_events([("T01", 1), ("T02", 1)]), md3, expo_e,
                                n_slots=2, selection="S1", exit_rule="E1", horizon=3,
                                gate="G1", log_path=LOG)
    assert r5["stats"]["entered"] == 2, "降级全敞口应正常入场"
    assert r5["stats"]["timing_fallback_days"] == len(CAL)
    details.append("(c) 仅 100 日历史时 T1 全程降级全敞口，fallback=10 天，正常入场 2 仓")

    # ---- (d) G2 强制退出优先级：浮盈最低者优先 ----
    expo_d = pd.DataFrame([
        dict(date=d, exposure=(1.0 if i < 4 else 0.5),
             max_src_date=(D[i] if i >= 1 else pd.Timestamp("2020-01-01")), fallback=False)
        for i, d in enumerate(CAL)
    ])  # d1..d4 全敞口，d5 起半敞口 -> 槽位 5->3，超出 2 仓
    daily_d = _mk_daily(close_overrides={
        ("T01", 5): 9.0, ("T02", 5): 9.5, ("T03", 5): 10.5,
        ("T04", 5): 11.0, ("T05", 5): 10.0})
    md4 = se.MarketData(daily=daily_d, limits=_mk_limits(), calendar=CAL)
    ev4 = _mk_events([(f"T0{k}", 1) for k in range(1, 6)])
    r4 = se2.run_backtest_timed(ev4, md4, expo_d, n_slots=5, selection="S1",
                                exit_rule="E1", horizon=6, gate="G2")
    tr4 = r4["trades"].set_index("ts_code")
    for c in ("T01", "T02"):    # 浮盈 -10% / -5% 最低，优先强退
        assert tr4.loc[c, "exit_reason"] == "G2_forced" and tr4.loc[c, "exit_date"] == D[5], \
            f"G2 优先级错误：{c} 应 d5 强退"
    for c in ("T03", "T04", "T05"):
        assert tr4.loc[c, "exit_reason"] == "E1" and tr4.loc[c, "exit_date"] == D[7], \
            f"G2 优先级错误：{c} 应保留至 E1@d7"
    details.append("(d) 5->3 槽时浮盈最低 2 仓(T01,T02) d5 强退，其余保留至 E1")

    record("timing_synthetic_tests", True, "；".join(details))


# ---------------------------------------------------------------- 校验 3：无泄漏独立重放
def check_leakage() -> None:
    raw = pd.read_parquet(os.path.join(REPO, "stock_data/index/000905.SH.parquet"),
                          columns=["trade_date", "close"])
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    c = raw.set_index("trade_date")["close"].sort_index()
    cal = [pd.Timestamp(d) for d in c.loc[VAL_START:VAL_END].index]

    prev = c.shift(1)
    indep: dict[str, pd.Series] = {}
    sma200 = prev.rolling(200).mean()
    indep["T1"] = (prev > sma200).astype(float).where(sma200.notna(), 1.0)
    sma60 = prev.rolling(60).mean()
    indep["T2"] = (prev > sma60).astype(float).where(sma60.notna(), 1.0)
    dd = 1.0 - prev / prev.rolling(250).max()
    t3 = pd.Series(np.where(dd < 0.10, 1.0, np.where(dd <= 0.20, 0.5, 0.0)), index=c.index)
    indep["T3"] = t3.where(prev.rolling(250).max().notna(), 1.0)
    vol = c.pct_change().rolling(20).std(ddof=1) * np.sqrt(252.0)
    pvol = vol.shift(1)
    q90 = pvol.rolling(250).quantile(0.9)
    indep["T4"] = (pvol > q90).map({True: 0.5, False: 1.0}).where(q90.notna(), 1.0)

    details = []
    for rule in ("T1", "T2", "T3", "T4"):
        eng = se2.compute_exposure(cal, c, rule)
        eng_s = eng.set_index("date")["exposure"]
        ind_s = indep[rule].loc[eng_s.index]
        assert eng_s.equals(ind_s.astype(float)), \
            f"{rule}: 引擎敞口与独立重算不一致（{int((eng_s != ind_s).sum())} 天）"
        src = eng.set_index("date")["max_src_date"]
        assert (src < src.index).all(), f"{rule}: 存在 max_src_date >= 决策日"
        vc = eng_s.value_counts().to_dict()
        details.append(f"{rule} 分布{ {k: int(v) for k, v in sorted(vc.items())} }")
    record("leakage_exposure_replay", True,
           "四规则敞口与 pandas shift/rolling 独立重算逐日一致，max_src_date 全部 < 决策日；"
           + "；".join(details))


# ---------------------------------------------------------------- 校验 4：冒烟 B1+T1+G1
def check_smoke() -> None:
    out_dir = os.path.join(ART, "smoke_B1_T1_G1")
    r = se2.run_config(pool="backup", selection="S3", exit_rule="E1", horizon=20,
                       n_slots=10, start=VAL_START, end=VAL_END, seed=SEED,
                       repo_root=REPO, out_dir=out_dir, log_path=LOG,
                       timing="T1", gate="G1")
    eq, tr, st, expo = r["equity"], r["trades"], r["stats"], r["exposure"]
    assert len(eq) == 928 and eq["equity"].notna().all() and (eq["equity"] > 0).all()
    assert len(tr) > 0
    src = expo.set_index("date")["max_src_date"]
    assert (src < src.index).all()
    vc = expo["exposure"].value_counts().to_dict()
    log(f"SMOKE B1+T1+G1 exposure_dist={ {k: int(v) for k, v in sorted(vc.items())} } "
        f"forced_exits={st['forced_exits']} gate_blocked={st['dropped_timing_gate']} "
        f"fallback_days={st['timing_fallback_days']} entered={st['entered']} "
        f"trades={len(tr)} equity {eq['equity'].iloc[0]:.0f} -> {eq['equity'].iloc[-1]:.0f}")
    record("smoke_B1_T1_G1_val", True,
           f"敞口分布{ {k: int(v) for k, v in sorted(vc.items())} }，强制退出 {st['forced_exits']} 笔"
           f"（G1 应为 0），闸门拦截 {st['dropped_timing_gate']} 笔，入场 {st['entered']}，"
           f"曲线 {eq['equity'].iloc[0]:.0f} -> {eq['equity'].iloc[-1]:.0f}，产物落 {out_dir}")


# ---------------------------------------------------------------- 主流程
def main() -> int:
    t_start = time.time()
    os.makedirs(ART, exist_ok=True)
    log("validate_strategy_engine_v2 START")

    for name, fn in (
        ("regression", check_regression),
        ("synthetic", check_synthetic),
        ("leakage", check_leakage),
        ("smoke", check_smoke),
    ):
        try:
            fn()
        except AssertionError as e:
            record(name + "_error", False, str(e))
        except Exception as e:  # noqa: BLE001
            record(name + "_error", False, repr(e))
        log(f"heartbeat: stage {name} done ({time.time() - t_start:.1f}s elapsed)")

    all_pass = all(ok for _, ok, _ in CHECKS) and len(CHECKS) == 4
    lines = ["# strategy_engine_v2 校验报告（issue #19 择时覆盖层冻结验收）", ""]
    lines.append(f"校验时间：{time.strftime('%Y-%m-%d %H:%M:%S')}，总耗时 {time.time() - t_start:.1f} 秒。")
    lines.append(f"总体结论：{'全部通过，引擎 v2 冻结。' if all_pass else '存在失败项，禁止冻结。'}")
    lines.append("")
    for name, ok, detail in CHECKS:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}：{detail}")
    lines.append("")
    lines.append("## 口径备注")
    lines.append("- 择时信号源为中证500（000905.SH）收盘价，t 日敞口只用 trade_date <= t-1 的指数收盘，引擎内硬断言 max_src_date < t。")
    lines.append("- T4 已实现波动 = 日收益（close.pct_change）20 日滚动样本 std（ddof=1）× sqrt(252)，分位窗口为含 t-1 当日的 250 个波动值的 90 分位。")
    lines.append("- 窗口历史不足（T1<200、T2<60、T3<250、T4<270 个 <=t-1 收盘）降级为全敞口并记 warning（timing_fallback_days）。")
    lines.append("- 槽位 = round_half_up(N × 敞口) = floor(x+0.5)，10×0.5=5、5×0.5=3。")
    lines.append("- G1 只拦新入场（信号截取按入场日槽位、T+1 入场按当日槽位两处拦截），旧仓按原出场规则自然退出。")
    lines.append("- G2 每日若持仓数 > 当日槽位，超出部分按收盘价强制退出（exit_reason=G2_forced），持有浮盈最低者优先（平局 ts_code 升序），遵守 T+1 不可卖与跌停顺延（次日重检）。")
    lines.append("- 单仓预算恒为信号日 T 收盘权益 / N，不随敞口变化。")
    lines.append("- timing=None 时 v2 直接委托 v1 run_config，落盘与 v1 逐字节一致。")
    lines.append("- G2 强制退出不可能落在买入当日：降档日持仓数已达新槽位上限时新入场已被闸门拦截，T+1 分支为防御性保留。")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"validate_strategy_engine_v2 DONE all_pass={all_pass} "
        f"elapsed={time.time() - t_start:.1f}s report={REPORT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
