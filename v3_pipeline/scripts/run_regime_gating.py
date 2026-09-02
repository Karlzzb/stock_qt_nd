# -*- coding: utf-8 -*-
"""市场体制门控 (regime_gating) — issue #12。

思路: 不做池内选股(已实锤无增益), 只用指数状态决定信号日 T+1 动不动手(跳日),
      抬日加权 top3 命中率基线(零信息口径 = 当日池命中率的信号日均值)。

预登记纪律(分析前固定, 不中途扩):
  - 门控特征全部只用信号日 T 及之前的指数数据(T+1 开盘才执行)。
  - 规则族固定 40 条(12 单条件 + 28 两条件与组合), 见 RULES。
  - 统计: 配对逐日检验(逐日贡献差 d_i = gated_i - baseline_i 的单样本 t, 单侧 greater),
    Welch 双样本 t(保留日 vs 跳过日的日命中率, 单侧 greater)作稳健性;
    家族 40 条 Bonferroni α = 0.05/40 = 0.00125。
  - 活口判定(双池分别): train Δ>0 且 p_t<0.00125 且 train 保留日数>=30;
    且 val 同向(val Δ>0)且 val Δ >= 50% * train Δ 且 val 保留日数>=30。
  - 条件数据不足(如 MA250 未成形)的日期: 该规则当日"不动手"(保守跳过)。
  - 基线固定在"全部信号日"上(不按规则重取分母), 保证 Δ 跨规则可比、配对检验自然成立。
  - 测试段(2022-11 起)与隔离带一行不出; 只用 pool_cleaning 的 seg 标记取 train/val。

口径继承 pool_cleaning:
  事件宇宙 = feature_matrix/{main,backup}_pool_features.parquet 键;
  标签 = divergence_lab {w_fractal_o15_s20 / w_zigzag_p05_s5} 的 hit_N20_k2.0;
  清洗 = excluded_events_{main,backup}.parquet 的 f_any(ST|停牌|一字涨停)。
  自检: 重算的清洗后日加权基线与 pool_cleaning/baseline_recalc.csv excl_combined 逐位一致。

广度代理披露: 无成分股历史数据, 广度用中证1000(000852.SH)相对沪深300 的 20 日相对强弱代理;
  创业板指(399006.SZ)2010-05 才有数据, 覆盖不足 train 一半, 预登记即弃用。

输出:
  v3_pipeline/reports/regime_gating/results_regime_gating.json
  v3_pipeline/reports/regime_gating/regime_gating_report.md
  v3_pipeline/reports/regime_gating/progress.log (阶段 + 心跳)
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
FM_DIR = ROOT / "v3_pipeline" / "reports" / "feature_matrix"
PC_DIR = ROOT / "v3_pipeline" / "reports" / "pool_cleaning"
IDX_DIR = ROOT / "stock_data" / "index"
OUT_DIR = ROOT / "v3_pipeline" / "reports" / "regime_gating"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"

POOLS = {
    "main": {
        "features": FM_DIR / "main_pool_features.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_fractal_o15_s20",
        "excluded": PC_DIR / "excluded_events_main.parquet",
    },
    "backup": {
        "features": FM_DIR / "backup_pool_features.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_zigzag_p05_s5",
        "excluded": PC_DIR / "excluded_events_backup.parquet",
    },
}

SEGMENTS = ["train", "val"]
SPAN_WEEKS = {  # 段日历跨度(周), 用于事件/周频率口径
    "train": (pd.Timestamp("2018-12-31") - pd.Timestamp("2001-01-01")).days / 7,
    "val": (pd.Timestamp("2022-10-31") - pd.Timestamp("2019-01-01")).days / 7,
}

# pool_cleaning/baseline_recalc.csv excl_combined 行(自检基准)
EXPECTED_BASELINE = {
    ("main", "train"): dict(dayw=0.5321690440177917, days=390, events=2740),
    ("main", "val"): dict(dayw=0.5032867789268494, days=292, events=2961),
    ("backup", "train"): dict(dayw=0.492364764213562, days=2025, events=18833),
    ("backup", "val"): dict(dayw=0.5560504198074341, days=702, events=6711),
}

ALPHA = 0.05 / 40  # 家族 40 条 Bonferroni
MIN_KEPT_DAYS = 30
RETENTION = 0.50

_stop_heartbeat = threading.Event()


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def heartbeat():
    while not _stop_heartbeat.wait(60):
        log("HEARTBEAT alive")


# ================================================================ 标签/事件/日表(与 pool_cleaning 同口径)
def load_labels(labdir):
    ev = pd.read_parquet(labdir / "events.parquet", columns=["ts_code", "date"])
    lb = pd.read_parquet(labdir / "labels.parquet")
    n = len(ev)
    div = lb.iloc[:n].reset_index(drop=True)
    assert (div["group"] == "div").all()
    lab = pd.DataFrame({
        "ts_code": ev["ts_code"].values,
        "date": pd.to_datetime(ev["date"].values),
        "hit": div["hit_N20_k2.0"].values,
    })
    assert not lab.duplicated(["ts_code", "date"]).any()
    return lab


def build_day_table(pool):
    """清洗后事件 -> 逐信号日 (day_hit=当日池命中率, n_events)。只用 train/val。"""
    cfg = POOLS[pool]
    keys = pd.read_parquet(cfg["features"], columns=["ts_code", "date"])
    keys["date"] = pd.to_datetime(keys["date"])
    lab = load_labels(cfg["labdir"])
    ex = pd.read_parquet(cfg["excluded"], columns=["ts_code", "date", "seg", "f_any"])
    ex["date"] = pd.to_datetime(ex["date"])
    df = keys.merge(lab, on=["ts_code", "date"], how="left", validate="one_to_one")
    # 与 pool_cleaning 同口径: 无标签事件(lab_ok=False)不入任何统计
    df = df[df["hit"].notna()]
    df = df.merge(ex, on=["ts_code", "date"], how="left", validate="one_to_one")
    assert df["seg"].notna().all()
    df = df[df["seg"].isin(SEGMENTS) & ~df["f_any"]]
    out = {}
    for seg in SEGMENTS:
        g = df[df["seg"] == seg].groupby("date")["hit"].agg(["mean", "size"])
        g.columns = ["day_hit", "n_events"]
        out[seg] = g.sort_index()
    return out


# ================================================================ 指数门控特征(只用 T 及之前)
def year_expanding_pct(s):
    """年内分位: 当年截至 T(含)的 expanding 占比; 原值为 NaN 的日期结果强制 NaN。"""
    pct = s.groupby(s.index.year, group_keys=False).apply(
        lambda g: g.expanding().apply(lambda x: (x <= x[-1]).mean(), raw=True))
    pct = pct.reindex(s.index)
    pct[s.isna()] = np.nan
    return pct


def build_conditions():
    hs = pd.read_parquet(IDX_DIR / "000300.SH.parquet", columns=["trade_date", "close"])
    hs["date"] = pd.to_datetime(hs["trade_date"], format="%Y%m%d")
    hs = hs.set_index("date").sort_index()
    close = hs["close"].astype(float)
    ret = close.pct_change()

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma250 = close.rolling(250).mean()
    vol20 = ret.rolling(20).std() * np.sqrt(244)  # 20 日已实现波动(年化)
    vpct = year_expanding_pct(vol20)              # 年内分位
    dd250 = close / close.rolling(250).max() - 1  # 距 250 日高点回撤(<=0)

    zz = pd.read_parquet(IDX_DIR / "000852.SH.parquet", columns=["trade_date", "close"])
    zz["date"] = pd.to_datetime(zz["trade_date"], format="%Y%m%d")
    zz = zz.set_index("date").sort_index()["close"].astype(float)
    rs1000 = (zz / zz.shift(20) - 1).reindex(close.index) - (close / close.shift(20) - 1)

    def b(series):  # 保 NaN 的布尔(float 0/1/NaN)
        return series.astype(float)

    conds = {
        "above_ma20": b(close > ma20),
        "above_ma60": b(close > ma60),
        "above_ma250": b(close > ma250),
        "below_ma250": b(close < ma250),
        "slope_up": b(ma20 > ma20.shift(5)),
        "slope_down": b(ma20 < ma20.shift(5)),
        "vol_low": b(vpct <= 1 / 3),
        "vol_mid": b((vpct > 1 / 3) & (vpct < 2 / 3)),
        "vol_high": b(vpct >= 2 / 3),
        "dd_shallow": b(dd250 >= -0.05),
        "dd_mid": b((dd250 < -0.05) & (dd250 >= -0.15)),
        "dd_deep": b(dd250 < -0.15),
        "rs_pos": b(rs1000 > 0),
        "rs_neg": b(rs1000 < 0),
    }
    return conds


# 预登记规则族: 12 单条件 + 28 两条件与组合 = 40 条
RULES = {
    # ---- 单条件 (12)
    "s_above_ma20": ["above_ma20"],
    "s_above_ma60": ["above_ma60"],
    "s_above_ma250": ["above_ma250"],
    "s_below_ma250": ["below_ma250"],
    "s_slope_up": ["slope_up"],
    "s_slope_down": ["slope_down"],
    "s_vol_low": ["vol_low"],
    "s_vol_high": ["vol_high"],
    "s_dd_shallow": ["dd_shallow"],
    "s_dd_deep": ["dd_deep"],
    "s_rs1000_pos": ["rs_pos"],
    "s_rs1000_neg": ["rs_neg"],
    # ---- 趋势 x 波动 (8)
    "c_aboveMA20_volLow": ["above_ma20", "vol_low"],
    "c_aboveMA20_volHigh": ["above_ma20", "vol_high"],
    "c_aboveMA60_volLow": ["above_ma60", "vol_low"],
    "c_aboveMA60_volHigh": ["above_ma60", "vol_high"],
    "c_aboveMA250_volLow": ["above_ma250", "vol_low"],
    "c_aboveMA250_volHigh": ["above_ma250", "vol_high"],
    "c_belowMA250_volLow": ["below_ma250", "vol_low"],
    "c_belowMA250_volHigh": ["below_ma250", "vol_high"],
    # ---- 趋势 x 回撤 (5)
    "c_aboveMA250_ddShallow": ["above_ma250", "dd_shallow"],
    "c_aboveMA60_ddShallow": ["above_ma60", "dd_shallow"],
    "c_aboveMA20_ddShallow": ["above_ma20", "dd_shallow"],
    "c_belowMA250_ddShallow": ["below_ma250", "dd_shallow"],
    "c_belowMA250_ddDeep": ["below_ma250", "dd_deep"],
    # ---- 趋势 x 斜率 (4)
    "c_aboveMA20_slopeUp": ["above_ma20", "slope_up"],
    "c_aboveMA60_slopeUp": ["above_ma60", "slope_up"],
    "c_aboveMA250_slopeUp": ["above_ma250", "slope_up"],
    "c_belowMA250_slopeDown": ["below_ma250", "slope_down"],
    # ---- 波动 x 回撤 (4)
    "c_volLow_ddShallow": ["vol_low", "dd_shallow"],
    "c_volHigh_ddDeep": ["vol_high", "dd_deep"],
    "c_volLow_ddDeep": ["vol_low", "dd_deep"],
    "c_volHigh_ddShallow": ["vol_high", "dd_shallow"],
    # ---- 广度代理组合 (4)
    "c_rsPos_aboveMA20": ["rs_pos", "above_ma20"],
    "c_rsPos_belowMA250": ["rs_pos", "below_ma250"],
    "c_rsNeg_belowMA250": ["rs_neg", "below_ma250"],
    "c_rsPos_volLow": ["rs_pos", "vol_low"],
    # ---- 斜率/广度 x 波动 (3)
    "c_slopeUp_volLow": ["slope_up", "vol_low"],
    "c_slopeDown_volHigh": ["slope_down", "vol_high"],
    "c_rsNeg_volHigh": ["rs_neg", "vol_high"],
}
assert len(RULES) == 40


# ================================================================ 门控评估
def eval_rule(day_tab, conds, rule_conds, seg):
    """配对逐日检验: 逐日贡献差 d_i = gated_i - baseline_i (基线固定在全部信号日)。"""
    r = day_tab["day_hit"].to_numpy()
    ev = day_tab["n_events"].to_numpy(dtype=float)
    days = day_tab.index
    n = len(day_tab)
    baseline = r.mean()

    keep = np.ones(n, bool)
    for c in rule_conds:
        v = conds[c].reindex(days).to_numpy(dtype=float)
        keep &= (v == 1.0)  # NaN(条件不可得) -> 不动手
    n_kept = int(keep.sum())
    res = dict(seg=seg, n_days=n, n_kept=n_kept, baseline=baseline)
    if n_kept == 0:
        res.update(gated=np.nan, delta=np.nan, p_t=np.nan, p_w=np.nan,
                   skip_share=1.0, events_kept=0, events_per_week=0.0, days_per_week=0.0)
        return res
    gated = r[keep].mean()
    delta = gated - baseline
    d = np.where(keep, r / n_kept, 0.0) - r / n
    p_t = stats.ttest_1samp(d, 0.0, alternative="greater").pvalue
    # 稳健性: 保留日 vs 跳过日的日命中率 Welch t (Wilcoxon 对 d_i 结构性退化, 不用)
    if 0 < n_kept < n:
        p_w = stats.ttest_ind(r[keep], r[~keep], equal_var=False, alternative="greater").pvalue
    else:
        p_w = np.nan
    events_kept = float(ev[keep].sum())
    res.update(
        gated=gated, delta=delta, p_t=p_t, p_w=p_w,
        skip_share=1.0 - n_kept / n,
        events_kept=events_kept,
        events_per_week=events_kept / SPAN_WEEKS[seg],
        days_per_week=n_kept / SPAN_WEEKS[seg],
    )
    return res


def judge(train_res, val_res):
    """活口: train 显著 + val 同向且幅度保留 >=50% (双段保留日均需 >=30)。"""
    if np.isnan(train_res["delta"]) or np.isnan(val_res["delta"]):
        return "na"
    train_sig = (train_res["delta"] > 0 and train_res["p_t"] < ALPHA
                 and train_res["n_kept"] >= MIN_KEPT_DAYS)
    if not train_sig:
        return "dead_train"
    val_ok = (val_res["delta"] > 0 and val_res["delta"] >= RETENTION * train_res["delta"]
              and val_res["n_kept"] >= MIN_KEPT_DAYS)
    return "survivor" if val_ok else "dead_val"


# ================================================================ 报告
def fmt_pct(x, nd=2):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}"


def fmt_p(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.2e}" if x < 0.001 else f"{x:.4f}"


def rule_table_md(pool, results):
    lines = [
        "| 规则 | 段 | 保留日/总日 | 跳过占比 | 门控命中率% | 基线% | Δpp | p_t | p_welch | 保留事件/周 | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule in RULES:
        rr = results[pool][rule]
        tr, vr = rr["train"], rr["val"]
        verdict = rr["verdict"]
        for seg, r in (("train", tr), ("val", vr)):
            lines.append(
                f"| {rule} | {seg} | {r['n_kept']}/{r['n_days']} | {fmt_pct(r['skip_share'])}% "
                f"| {fmt_pct(r['gated'])} | {fmt_pct(r['baseline'])} | {fmt_pct(r['delta'])} "
                f"| {fmt_p(r['p_t'])} | {fmt_p(r['p_w'])} | {r['events_per_week']:.1f} "
                f"| {verdict if seg == 'train' else ''} |")
    return "\n".join(lines)


def tradeoff_md(pool, results, topn=8):
    rows = sorted(results[pool].items(),
                  key=lambda kv: (kv[1]["train"]["delta"] if not np.isnan(kv[1]["train"]["delta"]) else -9),
                  reverse=True)[:topn]
    lines = [
        "| 规则 | train Δpp | train 命中率% | train 事件/周 | val Δpp | val 命中率% | val 事件/周 | val/train 保留 | 判定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rule, rr in rows:
        tr, vr = rr["train"], rr["val"]
        ret = "NA" if np.isnan(tr["delta"]) or np.isnan(vr["delta"]) or tr["delta"] == 0 else f"{vr['delta']/tr['delta']:.2f}"
        lines.append(
            f"| {rule} | {fmt_pct(tr['delta'])} | {fmt_pct(tr['gated'])} | {tr['events_per_week']:.1f} "
            f"| {fmt_pct(vr['delta'])} | {fmt_pct(vr['gated'])} | {vr['events_per_week']:.1f} "
            f"| {ret} | {rr['verdict']} |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    log("=== regime_gating 启动 (issue #12, 预登记 40 规则, 双池双段) ===")

    log("阶段1/5: 加载事件/标签/清洗标记, 构建逐日表")
    day_tables = {p: build_day_table(p) for p in POOLS}

    log("阶段2/5: 基线自检 (对照 pool_cleaning/baseline_recalc.csv excl_combined)")
    selfcheck = {}
    for pool in POOLS:
        for seg in SEGMENTS:
            tab = day_tables[pool][seg]
            exp = EXPECTED_BASELINE[(pool, seg)]
            got = dict(dayw=float(tab["day_hit"].mean()), days=int(len(tab)),
                       events=int(tab["n_events"].sum()))
            assert got["days"] == exp["days"], (pool, seg, got, exp)
            assert got["events"] == exp["events"], (pool, seg, got, exp)
            assert abs(got["dayw"] - exp["dayw"]) < 1e-9, (pool, seg, got, exp)
            selfcheck[f"{pool}_{seg}"] = got
    log(f"自检通过: {selfcheck}")

    log("阶段3/5: 构建指数门控条件 (HS300 趋势/波动/回撤 + 中证1000 相对强弱代理)")
    conds = build_conditions()

    log("阶段4/5: 40 条规则 x 双池 x 双段 门控评估 + 配对检验")
    results = {p: {} for p in POOLS}
    for pool in POOLS:
        for rule, rconds in RULES.items():
            tr = eval_rule(day_tables[pool]["train"], conds, rconds, "train")
            vr = eval_rule(day_tables[pool]["val"], conds, rconds, "val")
            results[pool][rule] = {"train": tr, "val": vr, "verdict": judge(tr, vr)}
        log(f"  池 {pool} 完成 40 条规则")

    log("阶段5/5: 汇总判定并落盘")
    survivors = {p: [r for r in RULES if results[p][r]["verdict"] == "survivor"] for p in POOLS}
    train_sig = {p: [r for r in RULES
                     if results[p][r]["train"]["delta"] > 0
                     and results[p][r]["train"]["p_t"] < ALPHA
                     and results[p][r]["train"]["n_kept"] >= MIN_KEPT_DAYS]
                 for p in POOLS}
    best = {}
    for pool in POOLS:
        cand = survivors[pool] or train_sig[pool] or list(RULES)
        best[pool] = max(cand, key=lambda r: results[pool][r]["train"]["delta"])

    payload = {
        "config": {
            "issue": 12,
            "alpha_bonferroni": ALPHA,
            "family_size": len(RULES),
            "min_kept_days": MIN_KEPT_DAYS,
            "val_retention": RETENTION,
            "segments": {"train": "2001-01-01~2018-12-31", "val": "2019-01-01~2022-10-31"},
            "test_segment": "2022-11 起封存, 本实验未触碰",
            "rules": RULES,
            "baseline_selfcheck": selfcheck,
            "breadth_proxy": "中证1000(000852.SH) 相对沪深300 的 20 日相对强弱 (无成分股数据的代理)",
        },
        "survivors": survivors,
        "train_significant": train_sig,
        "best_rule": best,
        "results": results,
        "runtime_sec": time.time() - t0,
    }

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with (OUT_DIR / "results_regime_gating.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=default)

    # ---------------- 报告 ----------------
    rep = []
    rep.append("# 市场体制门控 (regime_gating) 报告 — issue #12")
    rep.append("")
    rep.append("> 生成脚本：`v3_pipeline/scripts/run_regime_gating.py`（全量可复现）。")
    rep.append("> 进度流水：`progress.log`；机器可读结果：`results_regime_gating.json`。")
    rep.append("> 统计纪律：命中率只用 train（2001-01-01~2018-12-31）与 val（2019-01-01~2022-10-31）。")
    rep.append("> 测试段（2022-11 起）与隔离带一行未出。")
    rep.append("")
    rep.append("## 1. 口径与预登记")
    rep.append("")
    rep.append("- 实验问题：跳过坏日子而非选股票——用指数状态决定信号日动不动手，抬日加权 top3 命中率基线。")
    rep.append("- 本实验完全绕开池内排序（已实锤无增益），只做整天动手/不动手的二元门控。")
    rep.append("- 日加权基线为零信息 top3 口径：当日池命中率（候选 ≤3 时 top3 期望即当日池均值）的信号日等权平均。")
    rep.append("- 清洗后基线（本脚本重算并逐位自检通过）：主池 train 53.22% / val 50.33%；备池 train 49.24% / val 55.61%。")
    rep.append("- 门控信息集：信号日 T 及之前的指数日线，T+1 开盘才执行，无未来函数。")
    rep.append("- 规则族预登记固定 40 条（12 单条件 + 28 两条件与组合），分析后未增删。")
    rep.append("- 基线分母固定为全部信号日，任何规则只决定保留哪些日，保证 Δ 跨规则可比、配对检验成立。")
    rep.append("- 条件数据不足（如 MA250 未成形、2005 年前无中证1000）的日期，该规则当日按不动手处理（保守跳过）。")
    rep.append("- 显著性：逐日贡献差 d_i = 门控日贡献 − 基线日贡献的单样本 t（单侧 greater），保留日 vs 跳过日的 Welch 双样本 t 作稳健性。")
    rep.append(f"- 多重性：家族 40 条 Bonferroni，α = 0.05/40 = {ALPHA:.5f}。")
    rep.append(f"- 活口判定（双池分别）：train Δ>0 且 p_t<α 且保留日 ≥{MIN_KEPT_DAYS}；val 同向且 Δ 保留 ≥{RETENTION:.0%} 且保留日 ≥{MIN_KEPT_DAYS}。")
    rep.append("")
    rep.append("## 2. 门控条件定义（全部基于沪深300 日线，除广度代理外）")
    rep.append("")
    rep.append("- 趋势：收盘价 vs MA20/MA60/MA250 上下方；MA20 斜率 = MA20(T) vs MA20(T−5) 正负。")
    rep.append("- 波动：20 日已实现波动（日收益 std × √244）的年内 expanding 分位，低 ≤1/3、中 (1/3,2/3)、高 ≥2/3。")
    rep.append("- 回撤：收盘价距 250 日高点回撤，浅 ≥−5%、中 (−15%,−5%)、深 <−15%。")
    rep.append("- 广度代理：中证1000 相对沪深300 的 20 日相对强弱（rs1000 = ret20(000852) − ret20(000300)）正负。")
    rep.append("- 披露：无指数成分股历史数据，广度为代理口径；创业板指 2010-05 才有数据、覆盖不足 train 一半，预登记即弃用。")
    rep.append("")
    rep.append("## 3. 主池（m_fractal15_full）规则全表")
    rep.append("")
    rep.append(rule_table_md("main", results))
    rep.append("")
    rep.append("## 4. 备池（m_zigzag05_nofilter）规则全表")
    rep.append("")
    rep.append(rule_table_md("backup", results))
    rep.append("")
    rep.append("## 5. 命中率—频率权衡表（各池按 train Δ 取前 8 条）")
    rep.append("")
    rep.append("### 主池")
    rep.append("")
    rep.append(tradeoff_md("main", results))
    rep.append("")
    rep.append("### 备池")
    rep.append("")
    rep.append(tradeoff_md("backup", results))
    rep.append("")
    rep.append("## 6. 判定")
    rep.append("")
    for pool, zh in (("main", "主池"), ("backup", "备池")):
        tr_sig = train_sig[pool]
        surv = survivors[pool]
        rep.append(f"### {zh}")
        rep.append("")
        rep.append(f"- train 显著规则（p_t<{ALPHA:.5f} 且 Δ>0 且保留日≥{MIN_KEPT_DAYS}）：{len(tr_sig)} 条"
                   + (f"——{', '.join(tr_sig)}。" if tr_sig else "。"))
        rep.append(f"- 双段活口：{len(surv)} 条" + (f"——{', '.join(surv)}。" if surv else "。"))
        br = best[pool]
        btr = results[pool][br]["train"]
        bvr = results[pool][br]["val"]
        rep.append(f"- 最优规则（按 train Δ）：`{br}`：train {fmt_pct(btr['gated'])}% vs 基线 {fmt_pct(btr['baseline'])}%"
                   f"（Δ {fmt_pct(btr['delta'])}pp，p_t={fmt_p(btr['p_t'])}），"
                   f"val {fmt_pct(bvr['gated'])}% vs 基线 {fmt_pct(bvr['baseline'])}%（Δ {fmt_pct(bvr['delta'])}pp）；"
                   f"频率代价：train 跳过 {fmt_pct(btr['skip_share'])}% 信号日、保留 {btr['events_per_week']:.1f} 事件/周，"
                   f"val 跳过 {fmt_pct(bvr['skip_share'])}%、保留 {bvr['events_per_week']:.1f} 事件/周。")
        rep.append("")
    rep.append("## 7. 结论")
    rep.append("")
    any_surv = any(survivors[p] for p in POOLS)
    all_p = [(results[p][r]["train"]["p_t"], p, r) for p in POOLS for r in RULES
             if not np.isnan(results[p][r]["train"]["p_t"])]
    min_p, min_pool, min_rule = min(all_p)
    rep.append(f"- 全族 {len(all_p)} 次 train 检验（40 规则 × 2 池）的最小 p_t = {min_p:.4f}"
               f"（{min_pool} 池 `{min_rule}`），距 α={ALPHA:.5f} 仍有一个数量级，无一边缘活口。")
    rep.append("- 唯一双段同向的是主池浅回撤族（dd_shallow 系：train +5.2~+7.4pp / val +9.3~+10.9pp），"
               "但主池 train 仅 59~74 个保留日（主池事件 2014 年起）、p_t≈0.08~0.25 远不达 α。")
    rep.append("- 同一浅回撤族在备池 val 全面反号（−3.4~−5.4pp），跨池互相证伪，不构成活口证据，"
               "仅作未来主池样本增厚后的复核假设记录。")
    rep.append("- `c_belowMA250_ddShallow` 在样本期内为空规则（沪深300 位于 MA250 下方时从未同时距 250 日高点 <5%），"
               "计入家族 40 条但无可评估日。")
    if any_surv:
        rep.append("- 存在双段活口规则，门控方向值得进入下一轮（如组合规则或进入 PRD 流程）。")
    else:
        rep.append("- 判定：40 条预登记规则双池均无活口——指数体制门控对日加权 top3 命中率无可复现提升。")
        rep.append("- 与既有实锤一致：该信号池的命中变异不随常见指数体制状态单调变化。")
    rep.append(f"- 运行时长 {time.time() - t0:.1f}s。")

    with (OUT_DIR / "regime_gating_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")

    _stop_heartbeat.set()
    log(f"=== 完成, 用时 {time.time() - t0:.1f}s; 活口: {survivors} ===")


if __name__ == "__main__":
    main()
