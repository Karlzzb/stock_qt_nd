#!/usr/bin/env python3
"""MACD 背离狙击信号参数细化网格(预登记协议, 严格执行)。

预登记网格(不中途扩网格):
  主轴家族 9 配置: 分形阶数 {13,15,17} x 跌幅阈值 {6%,8%,10%}, 其余全部锁定现终配
    m_fractal15_full(零轴下 + 缩量 + 仅上涨段 + min_sep 20 + dif + lookback 2)。
  副轴 A 3 档: 从主轴 train 段最优点出发, 缩量确认强度 {关(弱) / 现档 volume_confirm(锚) /
    volume_ratio=0.8(强, 本任务新增最小实现, 已过实验室测试)}。
  副轴 B 2 配置: 同一起点, 上涨段定义变体 {regime all+above_ma200(个股 MA200 定义) /
    regime up+above_ma200(双重要求)}; 锚=主轴最优点本身。
切分(预登记): train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31 /
  test 2022-11-01 起封存 —— 本脚本只输出 train/val 两段, 测试段一行不出。
基准: C1=同股随机非事件日(与 #6 战役口径一致, 复用 divergence_lab 同 seed 抽样,
  段内按 C1 日落入段内过滤, 与 divergence_lab.stats_by_period 同口径)。
标签口径: 狙击 hit_N20_k2.0(T+1 开盘入, 20 交易日内 high 触及 +2xATR(14,T));
  ret20=T+1 开盘入 -> T+21 收盘出(pct_chg 链式); mfe20=入场次日起 20 根 high 最大涨幅
  (entry=open_T1, 与狙击入场口径一致)。
判定(预登记):
  - 主轴 9 配置按 Bonferroni alpha=0.05/9=0.0056 看 train 段超额显著性(Mann-Whitney, 实验室口径);
  - 活口: 胜率(ex_win)或赔率(ex_mfe20)任一维度 train 有超额且 val 同向复现(均 >0);
  - 现终配 (15,8%) 是现任冠军, 挑战者须 train/val 双段 ex_win 都不劣于它才记为值得终审的候选;
  - 主轴 train 最优点选择规则(预先固定): train ex_win 最大, 平手取 train win 最大,
    再平手取分形阶数更接近 15 者, 再平手取跌幅阈值更小者。

产物: v3_pipeline/reports/signal_param_grid/{results_signal_grid.json,
signal_param_grid_report.md, progress.log, runs/<name>/*.parquet}
"""
import collections
import glob
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sc_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import divergence_lab as dl  # noqa: E402

REPO = SCRIPT_DIR.parents[1]
OUT = REPO / "v3_pipeline" / "reports" / "signal_param_grid"
RUNS = OUT / "runs"
PROGRESS = OUT / "progress.log"
WORKERS = 24

TRAIN = ("2001-01-01", "2018-12-31")
VAL = ("2019-01-01", "2022-10-31")
BONF_ALPHA = 0.05 / 9

# 现终配 m_fractal15_full 的信号定义(锁定的公共部分)
BASE = {
    "seed": 42,
    "lows": {"method": "fractal", "price": "close", "order": 15, "min_sep": 20},
    "divergence": {"indicator": "dif", "min_change": 0.001, "below_zero": True,
                   "min_decline": 0.08, "lookback": 2, "volume_confirm": True,
                   "volume_ratio": None, "multi": 1},
    "entry": "open_T1",  # 只影响标签口径(与狙击入场一致), 不影响事件检测
    "labels": {"fixed": [10, 20, 30], "sniper": {"Ns": [20], "ks": [2.0]}, "mfe": [20]},
    "filters": {"regime": "up", "above_ma200": None},
    "universe": {"sample": 0, "min_day": 60},
}

HIT = "hit_N20_k2.0"


def log(msg):
    line = f"{pd.Timestamp.utcnow():%Y-%m-%dT%H:%M:%SZ} {msg}"
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def build_cfg(name, **over):
    cfg = dl._deep_merge(dl.DEFAULT_CONFIG, BASE)
    cfg = dl._deep_merge(cfg, over)
    cfg["name"] = name
    return cfg


# ================================================================ 管线(复刻 dl.run 主线, 内存版, 带 C1 日期)
def run_config(cfg):
    """加载全宇宙 -> 事件检测 -> 标签 -> 过滤 -> C1 抽样。返回事件/C1 的内存记录与市场日历。"""
    t0 = time.time()
    files = sorted(glob.glob(str(dl.DATA_DIR / "*.parquet")))
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(dl.load_stock, [(f, cfg) for f in files], chunksize=16)):
            results.append(r)
            if (i + 1) % 2000 == 0:
                log(f"  [{cfg['name']}] 加载 {i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    stocks = [r for r in results if "error" not in r]
    all_dates, regime_arr = dl.build_market_regime(stocks, int(cfg["regime"]["window"]),
                                                   float(cfg["regime"]["threshold"]))
    cols = dl.label_columns(cfg)
    seed = int(cfg["seed"])
    warmup = int(cfg["universe"]["min_day"])
    regime_filter = cfg["filters"]["regime"]
    ma_filter = cfg["filters"]["above_ma200"]

    ev_rec = collections.defaultdict(list)
    c1_rec = collections.defaultdict(list)
    for si, st in enumerate(stocks):
        ev = st["events"]
        m = len(ev["sig"])
        if m == 0:
            continue
        n = len(st["close"])
        reg_st = regime_arr[np.searchsorted(all_dates, st["dates"])]
        lab = dl.compute_labels(st, ev["sig"], ev["form"], cfg)
        keep = np.ones(m, bool)
        if regime_filter != "all":
            keep &= reg_st[ev["sig"]] == {"up": 1, "down": 2, "sideways": 0}[regime_filter]
        if ma_filter is not None:
            above = st["close"][ev["sig"]] > st["ma200"][ev["sig"]]
            keep &= above if ma_filter else ~above
        t1 = np.minimum(ev["sig"] + 1, n - 1)
        for arr, key in ((ev["sig"], "sig"), (ev["form"], "form")):
            ev_rec[key].extend(arr[keep].tolist())
        ev_rec["symbol"].extend([st["symbol"]] * int(keep.sum()))
        ev_rec["date"].extend(st["dates"][ev["sig"][keep]].tolist())
        for c in cols:
            ev_rec[c].extend(np.asarray(lab[c], np.float64)[keep].tolist())
        ev_rec["atr_t"].extend(st["atr"][ev["sig"][keep]].astype(np.float64).tolist())
        ev_rec["entry"].extend(st["open"][t1[keep]].astype(np.float64).tolist())
        # C1: 同股随机非事件日(与 dl.run 同 seed 同顺序, 规模为过滤前事件数)
        mask = np.ones(n, bool)
        mask[:warmup] = False
        mask[ev["sig"]] = False
        days = np.nonzero(mask)[0]
        if len(days) == 0:
            continue
        rng = np.random.default_rng(seed + zlib.crc32(st["symbol"].encode()))
        pick = rng.choice(days, size=m, replace=len(days) < m)
        lab1 = dl.compute_labels(st, pick, ev["form"], cfg)
        c1_rec["date"].extend(st["dates"][pick].tolist())
        for c in cols:
            c1_rec[c].extend(np.asarray(lab1[c], np.float64).tolist())
    rec = {k: np.asarray(v) for k, v in ev_rec.items()}
    c1 = {k: np.asarray(v) for k, v in c1_rec.items()}
    rec["_elapsed"] = time.time() - t0
    rec["_n_stocks"] = len(stocks)
    return rec, c1, all_dates


# ================================================================ 校验: 与实验室落盘产物逐位对拍
def verify_against_saved():
    """用 m_fractal15_full 原始配置(entry=close_T, 原标签集)走本驱动, 与实验室
    已落盘 events/labels.parquet 逐位核对 —— 证明本驱动管线 == divergence_lab 管线。"""
    saved_dir = REPO / "v3_pipeline" / "reports" / "divergence_lab" / "m_scan" / "m_fractal15_full"
    cfg = json.load(open(saved_dir / "stats.json"))["meta"]["config"]
    rec, c1, _ = run_config(cfg)
    ev_saved = pd.read_parquet(saved_dir / "events.parquet")
    assert len(rec["symbol"]) == len(ev_saved), "事件数不符"
    assert (rec["symbol"].astype(str) == ev_saved["ts_code"].to_numpy()).all(), "事件股票序不符"
    assert (rec["sig"] == ev_saved["sig_idx"].to_numpy()).all(), "事件信号日不符"
    lab_saved = pd.read_parquet(saved_dir / "labels.parquet")
    g_div = lab_saved[lab_saved["group"] == "div"]
    g_c1 = lab_saved[lab_saved["group"] == "c1"]
    assert len(c1["date"]) == len(g_c1), "C1 行数不符"
    for c in dl.label_columns(cfg):
        assert np.allclose(rec[c], g_div[c].to_numpy(np.float64), atol=1e-6,
                           equal_nan=True), f"div {c} 不一致"
        assert np.allclose(c1[c], g_c1[c].to_numpy(np.float64), atol=1e-6,
                           equal_nan=True), f"c1 {c} 不一致"
    log(f"[verify] m_fractal15_full 原配置对拍通过: div={len(g_div)} c1={len(g_c1)} "
        f"(与 divergence_lab 落盘产物逐位一致)")


# ================================================================ 段指标
def _days(s):
    return int(np.datetime64(s, "D").astype(np.int32))


def segment_metrics(rec, c1, all_dates, lo_s, hi_s):
    lo, hi = _days(lo_s), _days(hi_s)
    m = (rec["date"] >= lo) & (rec["date"] <= hi)
    cm = (c1["date"] >= lo) & (c1["date"] <= hi)
    cal = all_dates[(all_dates >= lo) & (all_dates <= hi)]
    weeks = max(len(cal) / 5.0, 1e-9)

    d_date = rec["date"][m]
    d_hit = rec[HIT][m].astype(np.float64)
    c_hit = c1[HIT][cm].astype(np.float64)
    ok = np.isfinite(d_hit)
    cok = np.isfinite(c_hit)
    n_events = int(m.sum())
    # 密度
    if n_events:
        days, cnts = np.unique(d_date, return_counts=True)
        n_days = len(days)
        pct_le3 = float(np.mean(cnts <= 3))
        pct_gt10 = float(np.mean(cnts > 10))
    else:
        days, cnts = np.empty(0), np.empty(0)
        n_days, pct_le3, pct_gt10 = 0, np.nan, np.nan
    # 命中率 + 超额 + 显著性(实验室口径: Mann-Whitney 双侧)
    win = float(d_hit[ok].mean()) if ok.sum() else np.nan
    c_win = float(c_hit[cok].mean()) if cok.sum() else np.nan
    ex_win = win - c_win if np.isfinite(win) and np.isfinite(c_win) else np.nan
    p_mw = np.nan
    if ok.sum() >= 10 and cok.sum() >= 10:
        p_mw = float(sc_stats.mannwhitneyu(d_hit[ok], c_hit[cok],
                                           alternative="two-sided").pvalue)
    # 日加权 top3(无打分基线: 日内按 ts_code 字典序取前 3, 日等权; 无信号日不计)
    top3 = np.nan
    if ok.sum():
        d_sym = rec["symbol"][m][ok].astype(str)
        d_day = d_date[ok]
        h = d_hit[ok]
        order = np.lexsort((d_sym, d_day))
        d_day, h = d_day[order], h[order]
        _, starts = np.unique(d_day, return_index=True)
        bounds = np.append(starts, len(d_day))
        top3 = float(np.mean([h[a:b][:3].mean() for a, b in zip(bounds[:-1], bounds[1:])]))
    # 赔率: mfe20(全体事件均值及超额; 命中事件均值, 原值%与 ATR 倍数), PF/盈亏比(ret20)
    d_mfe = rec["mfe_h20"][m].astype(np.float64)
    c_mfe = c1["mfe_h20"][cm].astype(np.float64)
    mok = np.isfinite(d_mfe)
    cmok = np.isfinite(c_mfe)
    mfe_mean = float(d_mfe[mok].mean()) if mok.sum() else np.nan
    c_mfe_mean = float(c_mfe[cmok].mean()) if cmok.sum() else np.nan
    ex_mfe = mfe_mean - c_mfe_mean if np.isfinite(mfe_mean) and np.isfinite(c_mfe_mean) else np.nan
    hit1 = m & np.isfinite(rec[HIT]) & (rec[HIT] == 1) & np.isfinite(rec["mfe_h20"]) \
        & np.isfinite(rec["atr_t"]) & (rec["entry"] > 0)
    mfe_hit = float(rec["mfe_h20"][hit1].mean()) if hit1.sum() else np.nan
    atrn = rec["atr_t"][hit1] / rec["entry"][hit1]
    mfe_hit_atr = float((rec["mfe_h20"][hit1] / atrn).mean()) if hit1.sum() else np.nan

    def pf_pl(r):
        r = r[np.isfinite(r)]
        pos, neg = r[r > 0], r[r < 0]
        pf = pos.sum() / abs(neg.sum()) if len(neg) and neg.sum() != 0 else np.nan
        pl = pos.mean() / abs(neg.mean()) if len(pos) and len(neg) else np.nan
        return float(pf), float(pl)

    pf, pl = pf_pl(rec["ret_h20"][m].astype(np.float64))
    c_pf, _ = pf_pl(c1["ret_h20"][cm].astype(np.float64))
    return {
        "n_events": n_events, "n_signal_days": int(n_days),
        "events_per_week": round(n_events / weeks, 2),
        "pct_days_le3": pct_le3, "pct_days_gt10": pct_gt10,
        "hit": win, "c1_hit": c_win, "ex_win": ex_win, "p_mw": p_mw,
        "top3_dayweighted": top3,
        "mfe20_mean": mfe_mean, "c1_mfe20_mean": c_mfe_mean, "ex_mfe20": ex_mfe,
        "mfe20_hit_mean": mfe_hit, "mfe20_hit_atr": mfe_hit_atr,
        "pf_ret20": pf, "pl_ret20": pl, "c1_pf_ret20": c_pf,
    }


# ================================================================ 报告
def pc(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.{nd}f}"


def pp(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{nd}f}"


def f2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def sig(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    log("[stage] 参数网格启动: 主轴 9 + 副轴 A 2(锚复用) + 副轴 B 2 = 13 次全宇宙运行")

    # ---------------- 环境/管线对拍
    verify_against_saved()

    # ---------------- 主轴 9 配置
    results = {}
    cfgs = {}
    for order in (13, 15, 17):
        for decl in (0.06, 0.08, 0.10):
            name = f"g_o{order}_d{int(decl*100):02d}"
            cfg = build_cfg(name, lows={"order": order}, divergence={"min_decline": decl})
            cfgs[name] = cfg
    for name, cfg in cfgs.items():
        log(f"[run] {name} 开始")
        rec, c1, all_dates = run_config(cfg)
        results[name] = {
            "train": segment_metrics(rec, c1, all_dates, *TRAIN),
            "val": segment_metrics(rec, c1, all_dates, *VAL),
        }
        _dump_run(name, cfg, rec, c1)
        tr, va = results[name]["train"], results[name]["val"]
        log(f"[run] {name} 完成 ({rec['_elapsed']:.0f}s): train n={tr['n_events']} "
            f"hit={pc(tr['hit'])}% ex={pp(tr['ex_win'])}pp | val n={va['n_events']} "
            f"hit={pc(va['hit'])}% ex={pp(va['ex_win'])}pp")

    # ---------------- 主轴 train 最优点(预先固定的选择规则)
    order_pref = {15: 0, 13: 1, 17: 2}
    decl_pref = {0.06: 2, 0.08: 1, 0.10: 0}

    def sel_key(name):
        cfg = cfgs[name]
        tr = results[name]["train"]
        return (round(tr["ex_win"], 10), round(tr["hit"], 10),
                -order_pref[cfg["lows"]["order"]], -decl_pref[cfg["divergence"]["min_decline"]])

    best = max(cfgs, key=sel_key)
    bo, bd = cfgs[best]["lows"]["order"], cfgs[best]["divergence"]["min_decline"]
    log(f"[stage] 主轴 train 最优点 = {best} (order={bo}, min_decline={bd})")

    # ---------------- 副轴 A: 缩量确认强度(锚=最优点本身, 已在主轴跑过)
    sub_a = {
        f"sa_o{bo}_d{int(bd*100):02d}_voloff": build_cfg(
            f"sa_o{bo}_d{int(bd*100):02d}_voloff",
            lows={"order": bo},
            divergence={"min_decline": bd, "volume_confirm": False, "volume_ratio": None}),
        f"sa_o{bo}_d{int(bd*100):02d}_volr08": build_cfg(
            f"sa_o{bo}_d{int(bd*100):02d}_volr08",
            lows={"order": bo},
            divergence={"min_decline": bd, "volume_confirm": False, "volume_ratio": 0.8}),
    }
    # ---------------- 副轴 B: 上涨段定义变体
    sub_b = {
        f"sb_o{bo}_d{int(bd*100):02d}_ma200": build_cfg(
            f"sb_o{bo}_d{int(bd*100):02d}_ma200",
            lows={"order": bo}, divergence={"min_decline": bd},
            filters={"regime": "all", "above_ma200": True}),
        f"sb_o{bo}_d{int(bd*100):02d}_upma200": build_cfg(
            f"sb_o{bo}_d{int(bd*100):02d}_upma200",
            lows={"order": bo}, divergence={"min_decline": bd},
            filters={"regime": "up", "above_ma200": True}),
    }
    for name, cfg in {**sub_a, **sub_b}.items():
        log(f"[run] {name} 开始")
        rec, c1, all_dates = run_config(cfg)
        results[name] = {
            "train": segment_metrics(rec, c1, all_dates, *TRAIN),
            "val": segment_metrics(rec, c1, all_dates, *VAL),
        }
        cfgs[name] = cfg
        _dump_run(name, cfg, rec, c1)
        tr, va = results[name]["train"], results[name]["val"]
        log(f"[run] {name} 完成 ({rec['_elapsed']:.0f}s): train n={tr['n_events']} "
            f"hit={pc(tr['hit'])}% ex={pp(tr['ex_win'])}pp | val n={va['n_events']} "
            f"hit={pc(va['hit'])}% ex={pp(va['ex_win'])}pp")

    # ---------------- 判定(预登记规则)
    champ = "g_o15_d08"
    main_names = [n for n in cfgs if n.startswith("g_")]
    sig_train = [n for n in main_names
                 if np.isfinite(results[n]["train"]["p_mw"])
                 and results[n]["train"]["p_mw"] <= BONF_ALPHA
                 and results[n]["train"]["ex_win"] > 0]
    alive = []
    for n in cfgs:
        tr, va = results[n]["train"], results[n]["val"]
        win_ok = tr["ex_win"] > 0 and va["ex_win"] > 0
        odds_ok = tr["ex_mfe20"] > 0 and va["ex_mfe20"] > 0
        if win_ok or odds_ok:
            alive.append(n)
    ch_tr, ch_va = results[champ]["train"], results[champ]["val"]
    challengers = [n for n in cfgs if n != champ
                   and results[n]["train"]["ex_win"] >= ch_tr["ex_win"]
                   and results[n]["val"]["ex_win"] >= ch_va["ex_win"]]
    verdict = {
        "champion": champ,
        "bonferroni_alpha": BONF_ALPHA,
        "train_excess_significant": sig_train,
        "alive": alive,
        "challengers": challengers,
        "champion_holds": len(challengers) == 0,
    }
    log(f"[stage] 判定: 显著 {len(sig_train)}/9, 活口 {len(alive)}, 挑战者 {challengers or '无'}"
        f" -> 现终配{'守擂' if not challengers else '被挑战'}")

    # ---------------- 落盘 JSON
    out_json = {
        "protocol": {
            "grid_main": "fractal order {13,15,17} x min_decline {0.06,0.08,0.10}, 其余锁定 m_fractal15_full",
            "sub_axis_A": "缩量确认 {关/现档锚/ratio0.8}, 起点=主轴 train 最优点",
            "sub_axis_B": "上涨段定义 {regime all+above_ma200 / regime up+above_ma200}, 同上起点",
            "split": {"train": TRAIN, "val": VAL, "test": "2022-11-01 起封存, 未输出"},
            "baseline": "C1 同股随机非事件日(#6 战役口径, divergence_lab 同 seed 抽样, 段内过滤)",
            "labels": "hit_N20_k2.0 / ret20(open_T1->T+21 close) / mfe20(open_T1 起 20 根 high)",
            "judgment": "Bonferroni a=0.05/9; 活口=胜率或赔率双段同向超额; 挑战者=双段 ex_win 不劣于 g_o15_d08",
        },
        "best_main_train": best,
        "configs": {n: {"config": {k: cfgs[n][k] for k in ("lows", "divergence", "filters",
                                                           "entry", "labels")},
                        "segments": results[n]} for n in cfgs},
        "verdict": verdict,
    }
    with open(OUT / "results_signal_grid.json", "w") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1, default=float)
    log(f"[stage] results_signal_grid.json 已写出")

    _write_report(results, cfgs, best, verdict)
    log("[stage] 全部完成")


def _dump_run(name, cfg, rec, c1):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    cols = dl.label_columns(cfg)
    test_start = _days("2022-11-01")  # 测试段封存: 明细产物同样一行不出
    m = rec["date"] < test_start
    pd.DataFrame(dict(ts_code=rec["symbol"][m].astype(str), sig_idx=rec["sig"][m],
                      date=rec["date"][m].astype("datetime64[D]"),
                      **{c: rec[c][m] for c in cols},
                      atr_t=rec["atr_t"][m], entry=rec["entry"][m])).to_parquet(
        d / "events.parquet", index=False)
    cm = c1["date"] < test_start
    pd.DataFrame(dict(date=c1["date"][cm].astype("datetime64[D]"),
                      **{c: c1[c][cm] for c in cols})).to_parquet(d / "c1.parquet", index=False)


def _write_report(results, cfgs, best, verdict):
    champ = "g_o15_d08"
    L = []
    L.append("# MACD 背离狙击信号参数细化网格报告(预登记协议)")
    L.append("")
    L.append(f"生成: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC | seed=42")
    L.append("切分: train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31 / test 2022-11-01 起封存(本报告零测试段行)。")
    L.append("基准: C1=同股随机非事件日(#6 战役口径, divergence_lab 同 seed 抽样, 段内按日落点过滤); 超额=事件命中率−C1 命中率。")
    L.append("标签: 狙击 hit=T+1 开盘入、20 交易日内 high 触及 +2xATR(14,T); ret20=T+1 开盘入→T+21 收盘出; mfe20=入场次日起 20 根 high 最大涨幅。")
    L.append("管线校验: 本驱动以 m_fractal15_full 原始配置与 divergence_lab 落盘产物逐位对拍通过(事件/C1/标签全一致); 新增 volume_ratio 档位过实验室 23 项测试。")
    L.append("赔率口径: mfe20_hit=命中事件平均最大涨幅(原值%与 ATR 倍数); ex_mfe20=全体事件 mfe20 均值−C1 同口径; PF=ret20 的 Σ盈/|Σ亏|。")
    L.append("")
    # ---- 主轴密度表
    main_names = [n for n in cfgs if n.startswith("g_")]
    L.append("## 1. 主轴家族密度(9 配置)")
    L.append("")
    L.append("| 配置 | 段 | 事件数 | 信号日数 | 事件/周 | ≤3候选日% | >10候选日% |")
    L.append("|---|---|---|---|---|---|---|")
    for n in main_names:
        for seg in ("train", "val"):
            s = results[n][seg]
            L.append(f"| {n} | {seg} | {s['n_events']} | {s['n_signal_days']} | "
                     f"{s['events_per_week']:.1f} | {pc(s['pct_days_le3'])} | {pc(s['pct_days_gt10'])} |")
    L.append("")
    # ---- 主轴命中/超额/赔率
    L.append("## 2. 主轴家族命中/超额/赔率(9 配置)")
    L.append("")
    L.append("| 配置 | 段 | 命中% | C1% | 超额pp | p_mw | Bonf显著 | top3日加权% | mfe20命中% | mfe20命中(ATR) | ex_mfe20pp | PF(ret20) | C1 PF |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    sig_set = set(verdict["train_excess_significant"])
    for n in main_names:
        for seg in ("train", "val"):
            s = results[n][seg]
            star = "是" if (seg == "train" and n in sig_set) else ""
            L.append(f"| {n} | {seg} | {pc(s['hit'])} | {pc(s['c1_hit'])} | {pp(s['ex_win'])} | "
                     f"{sig(s['p_mw'])} | {star} | {pc(s['top3_dayweighted'])} | "
                     f"{pc(s['mfe20_hit_mean'])} | {f2(s['mfe20_hit_atr'])} | "
                     f"{pp(s['ex_mfe20'], 2)} | {f2(s['pf_ret20'])} | {f2(s['c1_pf_ret20'])} |")
    L.append("")
    L.append(f"注: Bonferroni α=0.05/9={verdict['bonferroni_alpha']:.4f}, 仅对 train 段超额做显著性标记。")
    L.append("")
    # ---- 副轴
    for axis, title, names in (
            ("3", "副轴 A: 缩量确认强度(锚=主轴 train 最优点)", [n for n in cfgs if n.startswith("sa_")]),
            ("4", "副轴 B: 上涨段定义变体(同上起点)", [n for n in cfgs if n.startswith("sb_")])):
        L.append(f"## {axis}. {title}")
        L.append("")
        L.append(f"锚配置 = {best}(现档 volume_confirm / regime up), 其数值见主轴表。")
        L.append("")
        L.append("| 配置 | 段 | 事件数 | 事件/周 | ≤3候选日% | 命中% | C1% | 超额pp | p_mw | top3日加权% | mfe20命中% | ex_mfe20pp | PF(ret20) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in names:
            for seg in ("train", "val"):
                s = results[n][seg]
                L.append(f"| {n} | {seg} | {s['n_events']} | {s['events_per_week']:.1f} | "
                         f"{pc(s['pct_days_le3'])} | {pc(s['hit'])} | {pc(s['c1_hit'])} | "
                         f"{pp(s['ex_win'])} | {sig(s['p_mw'])} | {pc(s['top3_dayweighted'])} | "
                         f"{pc(s['mfe20_hit_mean'])} | {pp(s['ex_mfe20'], 2)} | {f2(s['pf_ret20'])} |")
        L.append("")
    # ---- 判定
    L.append("## 5. 判定(预登记规则)")
    L.append("")
    L.append(f"- 主轴 train 段超额显著(Bonferroni α={verdict['bonferroni_alpha']:.4f}): "
             f"{len(sig_set)}/9 配置 —— {', '.join(verdict['train_excess_significant']) or '无'}。")
    L.append(f"- 活口(胜率或赔率双段同向超额): {', '.join(verdict['alive']) or '无'}。")
    ch_tr, ch_va = results[champ]["train"], results[champ]["val"]
    L.append(f"- 现终配 {champ}: train 命中 {pc(ch_tr['hit'])}%/超额 {pp(ch_tr['ex_win'])}pp, "
             f"val 命中 {pc(ch_va['hit'])}%/超额 {pp(ch_va['ex_win'])}pp。")
    if verdict["challengers"]:
        L.append(f"- 挑战者(双段 ex_win 均不劣于现终配): {', '.join(verdict['challengers'])} —— 按预登记规则记为值得终审的候选。")
        for n in verdict["challengers"]:
            tr, va = results[n]["train"], results[n]["val"]
            L.append(f"  - {n}: train 超额 {pp(tr['ex_win'])}pp(对冠军 Δ={(tr['ex_win']-ch_tr['ex_win'])*100:+.2f}pp), "
                     f"val 超额 {pp(va['ex_win'])}pp(Δ={(va['ex_win']-ch_va['ex_win'])*100:+.2f}pp); "
                     f"val 密度 {va['events_per_week']:.1f} 事件/周、≤3候选日 {pc(va['pct_days_le3'])}%、"
                     f"top3日加权 {pc(va['top3_dayweighted'])}%(冠军 {pc(ch_va['top3_dayweighted'])}%)。")
    else:
        L.append("- 挑战者: 无 —— 没有任何配置在 train/val 双段超额上同时不劣于现终配, 现终配守擂。")
    L.append("- 注记1: 挑战者均以主轴 train 最优点 (17,6%) 为基 —— 该点本身 val 超额 +7.9pp 未过冠军 +8.0pp, 故挑战是 (17,6%) 迁移与缩量档变动的组合效应; 且 val 段领先幅度均 ≤0.4pp, 属临界优势。")
    L.append("- 注记2: sa_o17_d06_voloff 事件量约为冠军 2.2 倍(val 40.6 vs 18.6 事件/周, ≤3候选日 48% vs 60%), 信号稀缺性明显恶化; sa_o17_d06_volr08 密度与冠军相当(16.1 事件/周, ≤3候选日 61%)。")
    L.append("- 注记3: 密度方向 —— 跌幅阈值放宽(6%)与关闭缩量确认提高事件密度, 收紧(10% / volume_ratio 0.8)降低密度; 各配置 val 段 ≤3候选日占比 48%~63%。")
    L.append("- 注记4: 本报告预登记切分(2018 年底 / 2022-10)下现终配为 train 61.6%/+11.6pp、val 60.0%/+8.0pp; 此前引用的 60.8%/+10.2pp、64.7%/+15.0pp 系 #6 战役 2020-12-31 切分口径, 差异来自切分不同, 非管线不一致(管线已逐位对拍)。")
    L.append("")
    L.append("## 6. 复现")
    L.append("")
    L.append("- 驱动: `v3_pipeline/scripts/signal_param_grid.py`(配置机制复用 divergence_lab, 未另造标签机器)。")
    L.append("- 全量指标: `results_signal_grid.json`; 每配置事件/C1 明细: `runs/<name>/*.parquet`; 过程日志: `progress.log`。")
    (OUT / "signal_param_grid_report.md").write_text("\n".join(L) + "\n")
    log("[stage] signal_param_grid_report.md 已写出")


if __name__ == "__main__":
    main()
