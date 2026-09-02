#!/usr/bin/env python3
"""分布审计(票据 #6 终审赔率口径补充): 胜率之外的赔率画像。

复用 divergence_lab 的快速管线: 同一 seed/配置 => 事件序列与 C1 抽样与已落盘产物一致
(脚本内置逐位校验 labels.parquet 的 ret/hit 列, 不一致直接报错)。
补算 MAE/MFE(w=10/20/40, 含到达时间)与 dynamic horizon(c=0.5/1/2)标签。

输出 5 节(探索期 <=explore_end 与 验证期分开, 全部对照 C1 同口径):
  S1 收益分布: r10/r30 均值/中位/p75/p90/p95 及 div-C1 超额;
  S2 盈亏结构: 平均盈利/平均亏损(盈亏比)、profit factor、单笔期望;
  S3 路径结构: MFE-MAE 联合画像(先涨后跌/先跌后涨、MFE>+2ATR 且 MAE>-1ATR 的干净狙击占比);
  S4 m_fractal15_full 动态 horizon c∈{0.5,1,2} 复核(分布口径, 对比固定 h30);
  S5 m_zigzag05_full 被三件套+上涨段过滤丢弃信号的复审。

输出: v3_pipeline/reports/divergence_lab/m_scan/distribution_audit.md
"""
import collections
import glob
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import divergence_lab as dl  # noqa: E402

REPO = SCRIPT_DIR.parents[1]
SCAN = REPO / "v3_pipeline" / "reports" / "divergence_lab" / "m_scan"
CONFIGS = ["m_fractal15_full", "m_zigzag05_full", "m_zigzag05_nofilter", "m_zigzag05_notrigconf"]
WS = (10, 20, 40)
DYN_CS = (0.5, 1.0, 2.0)
DYN_CAP = 60
WORKERS = 16


# ================================================================ 补充标签
def extra_labels(st, sig, form):
    """对信号日数组补算 MFE/MAE(含到达时间)与 dynamic horizon 收益。
    口径与 dl.compute_labels 对齐: entry=close_T 收盘, 窗口=入场次日起 w 根 high/low。"""
    n = len(st["close"])
    sig = np.asarray(sig, np.int64)
    m = len(sig)
    close, high, low, cf, atr = st["close"], st["high"], st["low"], st["cf"], st["atr"]
    out = {}
    if m == 0:
        return out
    e = close[sig].astype(np.float64)
    out["entry"] = e
    out["atr_t"] = atr[sig].astype(np.float64)
    for w in WS:
        mfe = np.full(m, np.nan)
        mae = np.full(m, np.nan)
        tmfe = np.full(m, np.nan)
        tmae = np.full(m, np.nan)
        if n >= w:
            H = sliding_window_view(high, w)
            L = sliding_window_view(low, w)
            ok = (sig + w <= n - 1) & (e > 0)
            i = np.nonzero(ok)[0]
            if len(i):
                Hw = H[sig[i] + 1]
                Lw = L[sig[i] + 1]
                mfe[i] = Hw.max(axis=1) / e[i] - 1.0
                mae[i] = Lw.min(axis=1) / e[i] - 1.0
                tmfe[i] = Hw.argmax(axis=1) + 1  # 入场后第几根首次触及窗口最高
                tmae[i] = Lw.argmin(axis=1) + 1
        out[f"mfe_h{w}"], out[f"mae_h{w}"] = mfe, mae
        out[f"tmfe_h{w}"], out[f"tmae_h{w}"] = tmfe, tmae
    for c in DYN_CS:
        hh = np.clip(np.rint(c * np.asarray(form, np.float64)), 1, DYN_CAP).astype(np.int64)
        r = np.full(m, np.nan)
        ok = sig + hh <= n - 1
        i = np.nonzero(ok)[0]
        r[i] = cf[sig[i] + hh[i]] / cf[sig[i]] - 1.0
        out[f"dyn_c{c}"] = r
        out[f"dynh_c{c}"] = hh.astype(np.float64)
    return out


# ================================================================ 复刻管线(无 C2, 全量事件保留)
def run_slim(cfg, workers=WORKERS):
    files = sorted(glob.glob(str(dl.DATA_DIR / "*.parquet")))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(dl.load_stock, [(f, cfg) for f in files], chunksize=16))
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
    lab_ev, lab_c1 = collections.defaultdict(list), collections.defaultdict(list)
    c1_count = 0
    for si, st in enumerate(stocks):
        ev = st["events"]
        m = len(ev["sig"])
        if m == 0:
            continue
        n = len(st["close"])
        reg_st = regime_arr[np.searchsorted(all_dates, st["dates"])]
        lab = dl.compute_labels(st, ev["sig"], ev["form"], cfg)
        ext = extra_labels(st, ev["sig"], ev["form"])
        keep = np.ones(m, bool)
        if regime_filter != "all":
            keep &= reg_st[ev["sig"]] == {"up": 1, "down": 2, "sideways": 0}[regime_filter]
        if ma_filter is not None:
            above = st["close"][ev["sig"]] > st["ma200"][ev["sig"]]
            keep &= above if ma_filter else ~above
        for c in cols:
            lab_ev[c].append(np.asarray(lab[c], np.float64))
        for c, a in ext.items():
            lab_ev[c].append(a)
        # C1: 同股随机非事件日(与检测出的全部事件 1:1 配对, 复刻 dl.run)
        mask = np.ones(n, bool)
        mask[:warmup] = False
        mask[ev["sig"]] = False
        days = np.nonzero(mask)[0]
        if len(days):
            rng = np.random.default_rng(seed + zlib.crc32(st["symbol"].encode()))
            pick = rng.choice(days, size=m, replace=len(days) < m)
            lab1 = dl.compute_labels(st, pick, ev["form"], cfg)
            ext1 = extra_labels(st, pick, ev["form"])
            for c in cols:
                lab_c1[c].append(np.asarray(lab1[c], np.float64))
            for c, a in ext1.items():
                lab_c1[c].append(a)
            c1_rec["si"].extend([si] * m)
            c1_rec["day"].extend(pick.tolist())
            c1_rec["date"].extend(st["dates"][pick].tolist())
            c1_row = c1_count + np.arange(m)
            c1_count += m
        else:
            c1_row = np.full(m, -1)
        ev_rec["si"].extend([si] * m)
        ev_rec["symbol"].extend([st["symbol"]] * m)
        ev_rec["sig"].extend(ev["sig"].tolist())
        ev_rec["form"].extend(ev["form"].tolist())
        ev_rec["date"].extend(st["dates"][ev["sig"]].tolist())
        ev_rec["keep"].extend(keep.tolist())
        ev_rec["c1_row"].extend(c1_row.tolist())
    ev = pd.DataFrame(ev_rec)
    c1 = pd.DataFrame(c1_rec)
    lab_ev = {c: np.concatenate(v) for c, v in lab_ev.items()}
    lab_c1 = {c: np.concatenate(v) for c, v in lab_c1.items()}
    return dict(cfg=cfg, ev=ev, c1=c1, lab_ev=lab_ev, lab_c1=lab_c1)


def verify(name, res):
    """与已落盘 labels.parquet 逐位核对(div=keep 后事件, c1=抽样)。"""
    saved = pd.read_parquet(SCAN / name / "labels.parquet")
    keep = res["ev"]["keep"].to_numpy(bool)
    g_div = saved[saved["group"] == "div"]
    g_c1 = saved[saved["group"] == "c1"]
    assert int(keep.sum()) == len(g_div), f"{name}: div 行数不符 {keep.sum()} vs {len(g_div)}"
    assert len(res["c1"]) == len(g_c1), f"{name}: c1 行数不符 {len(res['c1'])} vs {len(g_c1)}"
    for c in dl.label_columns(res["cfg"]):
        a, b = res["lab_ev"][c][keep], g_div[c].to_numpy(np.float64)
        assert np.allclose(a, b, atol=1e-6, equal_nan=True), f"{name} div {c} 不一致"
        a, b = res["lab_c1"][c], g_c1[c].to_numpy(np.float64)
        assert np.allclose(a, b, atol=1e-6, equal_nan=True), f"{name} c1 {c} 不一致"
    # 事件表核对
    ev_saved = pd.read_parquet(SCAN / name / "events.parquet")
    kept = res["ev"][res["ev"]["keep"]]
    assert (kept["symbol"].to_numpy() == ev_saved["ts_code"].to_numpy()).all()
    assert (kept["sig"].to_numpy() == ev_saved["sig_idx"].to_numpy()).all()
    print(f"  [{name}] 校验通过: div={len(g_div)} c1={len(g_c1)} (与落盘产物一致)", flush=True)


# ================================================================ 指标
def qstats(r):
    r = np.asarray(r, np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return None
    w = r[r > 0]
    l = r[r < 0]
    aw = w.mean() if len(w) else np.nan
    al = l.mean() if len(l) else np.nan
    return dict(n=n, win=len(w) / n, mean=r.mean(), p50=np.percentile(r, 50),
                p75=np.percentile(r, 75), p90=np.percentile(r, 90), p95=np.percentile(r, 95),
                avg_win=aw, avg_loss=al,
                pl=aw / abs(al) if len(w) and len(l) else np.nan,
                pf=w.sum() / abs(l.sum()) if len(w) and len(l) and l.sum() != 0 else np.nan)


def periods_split(dates, explore_end_days):
    d = np.asarray(dates)
    return d <= explore_end_days, d > explore_end_days


# ================================================================ 报告工具
def pc(x, nd=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.{nd}f}"


def f2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


def dist_rows(tag, d, c):
    """div/C1/Δ 三行(收益单位=%, 差值=pp)。"""
    rows = []
    for name, s in ((f"{tag} div", d), (f"{tag} C1", c)):
        rows.append([name, s["n"], pc(s["win"]), pc(s["mean"]), pc(s["p50"]), pc(s["p75"]),
                     pc(s["p90"]), pc(s["p95"]), pc(s["avg_win"]), pc(s["avg_loss"]),
                     f2(s["pl"]), f2(s["pf"])])
    rows.append([f"{tag} Δ", "—", pc(d["win"] - c["win"]), pc(d["mean"] - c["mean"]),
                 pc(d["p50"] - c["p50"]), pc(d["p75"] - c["p75"]), pc(d["p90"] - c["p90"]),
                 pc(d["p95"] - c["p95"]), pc(d["avg_win"] - c["avg_win"]),
                 pc(d["avg_loss"] - c["avg_loss"]), f2(d["pl"] - c["pl"]), f2(d["pf"] - c["pf"])])
    return rows


DIST_HEADERS = ["口径", "n", "胜率%", "均值%", "p50%", "p75%", "p90%", "p95%",
                "平均盈利%", "平均亏损%", "盈亏比", "PF"]


def main():
    explore_end_days = int(np.datetime64("2020-12-31", "D").astype(np.int32))
    results = {}
    for name in CONFIGS:
        print(f"[{name}] 重跑管线(补算 MAE/MFE/dyn) ...", flush=True)
        cfg = json.load(open(SCAN / name / "stats.json"))["meta"]["config"]
        res = run_slim(cfg)
        verify(name, res)
        results[name] = res

    L = []  # 报告行
    L.append("# 背离实验室分布审计(票据 #6 赔率口径补充)")
    L.append("")
    L.append("口径: 复用 divergence_lab 管线重算(事件与 C1 抽样同 seed, 已与落盘 labels.parquet 逐位核对一致);")
    L.append("entry=close_T, 收益为 pct_chg 链式累乘; 探索期=信号日<=2020-12-31, 验证期=2021+; 对照仅 C1(同股随机非事件日)。")
    L.append("MAE/MFE 窗口=入场次日起 w∈{10,20,40} 根 low/high; 干净狙击=MFE>+2×ATR(14,T) 且 MAE>−1×ATR(14,T);")
    L.append("收益数字单位 %(Δ=div−C1, 单位 pp); 盈亏比=平均盈利/|平均亏损|, PF=Σ盈/|Σ亏|。")
    L.append("")

    # ---------------- S1+S2 收益分布 + 盈亏结构
    L.append("## 1-2. 收益分布与盈亏结构(r10/r30, vs C1)")
    for name in CONFIGS:
        res = results[name]
        keep = res["ev"]["keep"].to_numpy(bool)
        d_dates = res["ev"]["date"].to_numpy()[keep]
        c_dates = res["c1"]["date"].to_numpy()
        dm, cm = periods_split(d_dates, explore_end_days)[0], periods_split(c_dates, explore_end_days)[0]
        dv, cv = periods_split(d_dates, explore_end_days)[1], periods_split(c_dates, explore_end_days)[1]
        L.append(f"### {name}")
        for tag, col in (("r10", "ret_h10"), ("r30", "ret_h30")):
            L.append(f"**{tag} 探索期(≤2020)**")
            rows = dist_rows(tag, qstats(res["lab_ev"][col][keep][dm]), qstats(res["lab_c1"][col][cm]))
            L += md_table(DIST_HEADERS, rows)
            L.append("")
            L.append(f"**{tag} 验证期(2021+)**")
            rows = dist_rows(tag, qstats(res["lab_ev"][col][keep][dv]), qstats(res["lab_c1"][col][cv]))
            L += md_table(DIST_HEADERS, rows)
            L.append("")

    # ---------------- S3 路径结构
    L.append("## 3. 路径结构(MFE/MAE 联合画像)")
    for name in CONFIGS:
        res = results[name]
        keep = res["ev"]["keep"].to_numpy(bool)
        d_dates = res["ev"]["date"].to_numpy()[keep]
        c_dates = res["c1"]["date"].to_numpy()
        L.append(f"### {name}")
        headers = ["指标", "div 探索", "C1 探索", "Δ", "div 验证", "C1 验证", "Δ"]
        rows = []
        dem, dvm = periods_split(d_dates, explore_end_days)
        cem, cvm = periods_split(c_dates, explore_end_days)

        def gv(col, mask, kept=True):
            a = res["lab_ev"][col][keep] if kept else res["lab_c1"][col]
            return a[mask]

        def share(fn, col_mfe, col_mae, dm_, cm_):
            """fn(mfe, mae, atr, entry, tmfe, tmae) -> bool 数组, 返回 div/C1 占比。"""
            out = []
            for src, m_ in (("ev", dm_), ("c1", cm_)):
                lab = res["lab_ev"] if src == "ev" else res["lab_c1"]
                w = col_mfe.split("_h")[1]
                if src == "c1":
                    mfe, mae, atr, ent = (lab[col_mfe], lab[col_mae], lab["atr_t"], lab["entry"])
                    tmfe, tmae = lab[f"tmfe_h{w}"], lab[f"tmae_h{w}"]
                else:
                    mfe, mae = lab[col_mfe][keep], lab[col_mae][keep]
                    atr, ent = lab["atr_t"][keep], lab["entry"][keep]
                    tmfe, tmae = lab[f"tmfe_h{w}"][keep], lab[f"tmae_h{w}"][keep]
                mfe, mae, atr, ent, tmfe, tmae = (x[m_] for x in (mfe, mae, atr, ent, tmfe, tmae))
                ok = np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(atr) & (ent > 0)
                out.append(float(np.mean(fn(mfe[ok], mae[ok], atr[ok], ent[ok],
                                            tmfe[ok], tmae[ok]))) if ok.sum() else np.nan)
            return out

        def stat_pair(col, reducer, dm_, cm_, nd=2, mult=100):
            a = gv(col, dm_)
            b = res["lab_c1"][col][cm_]
            va, vb = reducer(a[np.isfinite(a)]), reducer(b[np.isfinite(b)])
            return va * mult, vb * mult, (va - vb) * mult

        for w in WS:
            va, vb, dd = stat_pair(f"mfe_h{w}", np.mean, dem, cem)
            vav, vbv, ddv = stat_pair(f"mfe_h{w}", np.mean, dvm, cvm)
            rows.append([f"MFE{w} 均值%", f"{va:.2f}", f"{vb:.2f}", f"{dd:+.2f}",
                         f"{vav:.2f}", f"{vbv:.2f}", f"{ddv:+.2f}"])
        for w in WS:
            va, vb, dd = stat_pair(f"mae_h{w}", np.mean, dem, cem)
            vav, vbv, ddv = stat_pair(f"mae_h{w}", np.mean, dvm, cvm)
            rows.append([f"MAE{w} 均值%", f"{va:.2f}", f"{vb:.2f}", f"{dd:+.2f}",
                         f"{vav:.2f}", f"{vbv:.2f}", f"{ddv:+.2f}"])
        # MFE/|MAE| (w=40, 均值比)
        def mfe_mae_ratio(dm_):
            mfe = gv("mfe_h40", dm_)
            mae = gv("mae_h40", dm_)
            ok = np.isfinite(mfe) & np.isfinite(mae)
            return mfe[ok].mean() / abs(mae[ok].mean())
        def mfe_mae_ratio_c(cm_):
            mfe = res["lab_c1"]["mfe_h40"][cm_]
            mae = res["lab_c1"]["mae_h40"][cm_]
            ok = np.isfinite(mfe) & np.isfinite(mae)
            return mfe[ok].mean() / abs(mae[ok].mean())
        re_, rc = mfe_mae_ratio(dem), mfe_mae_ratio_c(cem)
        rv_, rvc = mfe_mae_ratio(dvm), mfe_mae_ratio_c(cvm)
        rows.append(["MFE40/|MAE40|", f2(re_), f2(rc), f"{re_-rc:+.2f}",
                     f2(rv_), f2(rvc), f"{rv_-rvc:+.2f}"])
        # 先涨后跌 / 先跌后涨 (w=40)
        up_first_e = share(lambda mfe, mae, atr, ent, tf, ta: tf < ta, "mfe_h40", "mae_h40", dem, cem)
        up_first_v = share(lambda mfe, mae, atr, ent, tf, ta: tf < ta, "mfe_h40", "mae_h40", dvm, cvm)
        dn_first_e = share(lambda mfe, mae, atr, ent, tf, ta: tf > ta, "mfe_h40", "mae_h40", dem, cem)
        dn_first_v = share(lambda mfe, mae, atr, ent, tf, ta: tf > ta, "mfe_h40", "mae_h40", dvm, cvm)
        rows.append(["先涨后跌占比%(w40)", pc(up_first_e[0]), pc(up_first_e[1]),
                     f"{(up_first_e[0]-up_first_e[1])*100:+.2f}",
                     pc(up_first_v[0]), pc(up_first_v[1]), f"{(up_first_v[0]-up_first_v[1])*100:+.2f}"])
        rows.append(["先跌后涨占比%(w40)", pc(dn_first_e[0]), pc(dn_first_e[1]),
                     f"{(dn_first_e[0]-dn_first_e[1])*100:+.2f}",
                     pc(dn_first_v[0]), pc(dn_first_v[1]), f"{(dn_first_v[0]-dn_first_v[1])*100:+.2f}"])
        # 干净狙击
        for w in (20, 40):
            cs_e = share(lambda mfe, mae, atr, ent, tf, ta: (mfe > 2 * atr / ent) & (mae > -atr / ent),
                         f"mfe_h{w}", f"mae_h{w}", dem, cem)
            cs_v = share(lambda mfe, mae, atr, ent, tf, ta: (mfe > 2 * atr / ent) & (mae > -atr / ent),
                         f"mfe_h{w}", f"mae_h{w}", dvm, cvm)
            rows.append([f"干净狙击占比%(w{w})", pc(cs_e[0]), pc(cs_e[1]),
                         f"{(cs_e[0]-cs_e[1])*100:+.2f}",
                         pc(cs_v[0]), pc(cs_v[1]), f"{(cs_v[0]-cs_v[1])*100:+.2f}"])
        L += md_table(headers, rows)
        L.append("")

    # ---------------- S4 动态 horizon 复核(m_fractal15_full)
    L.append("## 4. 动态 horizon 复核(m_fractal15_full, dyn c∈{0.5,1,2} vs 固定 h30)")
    name = "m_fractal15_full"
    res = results[name]
    keep = res["ev"]["keep"].to_numpy(bool)
    d_dates = res["ev"]["date"].to_numpy()[keep]
    c_dates = res["c1"]["date"].to_numpy()
    dem, dvm = periods_split(d_dates, explore_end_days)
    cem, cvm = periods_split(c_dates, explore_end_days)
    cols4 = [("dyn c=0.5", "dyn_c0.5", "dynh_c0.5"), ("dyn c=1.0", "dyn_c1.0", "dynh_c1.0"),
             ("dyn c=2.0", "dyn_c2.0", "dynh_c2.0"), ("固定 h10", "ret_h10", None),
             ("固定 h30", "ret_h30", None)]
    headers4 = ["标签", "平均h", "胜率%", "均值%", "盈亏比", "PF", "p90%", "p95%",
                "Δ均值", "ΔPF", "Δp90", "Δp95"]
    for pname, dm_, cm_ in (("探索期(≤2020)", dem, cem), ("验证期(2021+)", dvm, cvm)):
        L.append(f"**{pname}**")
        rows = []
        for tag, col, hcol in cols4:
            d = qstats(res["lab_ev"][col][keep][dm_])
            c = qstats(res["lab_c1"][col][cm_])
            avgh = np.nanmean(res["lab_ev"][hcol][keep][dm_]) if hcol else (
                "10" if col == "ret_h10" else "30")
            avgh = f"{avgh:.1f}" if isinstance(avgh, float) else avgh
            rows.append([tag, avgh, pc(d["win"]), pc(d["mean"]), f2(d["pl"]), f2(d["pf"]),
                         pc(d["p90"]), pc(d["p95"]),
                         f"{(d['mean']-c['mean'])*100:+.2f}", f"{d['pf']-c['pf']:+.2f}",
                         f"{(d['p90']-c['p90'])*100:+.2f}", f"{(d['p95']-c['p95'])*100:+.2f}"])
        L += md_table(headers4, rows)
        L.append("")

    # ---------------- S5 被过滤信号复审(m_zigzag05_full)
    L.append("## 5. 被过滤信号复审(m_zigzag05_full: 三件套+上涨段过滤丢弃的信号)")
    res_nt = results["m_zigzag05_notrigconf"]  # 检测条件最松(无三件套), 事件全集
    ev_nt = res_nt["ev"]
    keys_nt = list(zip(ev_nt["symbol"], ev_nt["sig"]))
    ev_full = pd.read_parquet(SCAN / "m_zigzag05_full" / "events.parquet")
    ev_nof = pd.read_parquet(SCAN / "m_zigzag05_nofilter" / "events.parquet")
    keys_full = set(zip(ev_full["ts_code"], ev_full["sig_idx"]))
    keys_trig = set(zip(ev_nof["ts_code"], ev_nof["sig_idx"]))
    in_full = np.array([k in keys_full for k in keys_nt])
    in_trig = np.array([k in keys_trig for k in keys_nt])
    assert in_full.sum() == len(keys_full), "full 事件未全部落入 notrigconf 全集"
    assert in_trig.sum() == len(keys_trig), "nofilter 事件未全部落入 notrigconf 全集"
    assert (in_full <= in_trig).all(), "full 非 nofilter 子集"
    dropped = ~in_full
    d_trigfail = ~in_trig            # 被三件套丢弃(任意市场阶段)
    d_regime = in_trig & ~in_full    # 过三件套但被上涨段过滤丢弃
    L.append(f"notrigconf 检测全集 {len(ev_nt)}; full 保留 {int(in_full.sum())}; "
             f"被丢弃 {int(dropped.sum())} = 三件套丢弃 {int(d_trigfail.sum())} "
             f"+ 仅上涨段过滤丢弃 {int(d_regime.sum())}。")
    L.append("对照 C1 为这些被丢信号的同股随机非事件日(逐信号配对, 与库口径一致)。")
    L.append("")
    c1_rows = ev_nt["c1_row"].to_numpy()
    nt_dates = ev_nt["date"].to_numpy()
    c1_dates_nt = res_nt["c1"]["date"].to_numpy()
    headers5 = ["子集", "n", "胜率%", "均值%", "盈亏比", "PF", "p90%", "p95%",
                "Δ均值", "ΔPF", "Δp90", "Δp95"]
    for col, tag in (("ret_h10", "r10"), ("ret_h30", "r30")):
        for pname, dm_all, cm_all in (("探索期(≤2020)", nt_dates <= explore_end_days,
                                       c1_dates_nt <= explore_end_days),
                                      ("验证期(2021+)", nt_dates > explore_end_days,
                                       c1_dates_nt > explore_end_days)):
            L.append(f"**{tag} {pname}**")
            rows = []
            subsets = [("full 保留(对照)", in_full), ("被丢弃-全部", dropped),
                       ("被丢弃-三件套", d_trigfail), ("被丢弃-上涨段", d_regime)]
            for sname, smask in subsets:
                m_ = smask & dm_all
                d = qstats(res_nt["lab_ev"][col][m_])
                cr = c1_rows[m_]
                cr = cr[cr >= 0]
                cm_ = cm_all[cr]
                c = qstats(res_nt["lab_c1"][col][cr][cm_])
                rows.append([sname, d["n"], pc(d["win"]), pc(d["mean"]), f2(d["pl"]), f2(d["pf"]),
                             pc(d["p90"]), pc(d["p95"]),
                             f"{(d['mean']-c['mean'])*100:+.2f}", f"{d['pf']-c['pf']:+.2f}",
                             f"{(d['p90']-c['p90'])*100:+.2f}", f"{(d['p95']-c['p95'])*100:+.2f}"])
            # 全样本 C1 参考行
            c = qstats(res_nt["lab_c1"][col][cm_all])
            rows.append(["C1 全样本(参考)", c["n"], pc(c["win"]), pc(c["mean"]), f2(c["pl"]),
                         f2(c["pf"]), pc(c["p90"]), pc(c["p95"]), "—", "—", "—", "—"])
            L += md_table(headers5, rows)
            L.append("")

    out_path = SCAN / "distribution_audit.md"
    out_path.write_text("\n".join(L) + "\n")
    print(f"报告已写出: {out_path}")

    # ---------------- 控制台摘要(供 final text)
    print("\n===== 摘要(验证期 r30) =====")
    for name in CONFIGS:
        res = results[name]
        keep = res["ev"]["keep"].to_numpy(bool)
        dvm_ = res["ev"]["date"].to_numpy()[keep] > explore_end_days
        cvm_ = res["c1"]["date"].to_numpy() > explore_end_days
        d = qstats(res["lab_ev"]["ret_h30"][keep][dvm_])
        c = qstats(res["lab_c1"]["ret_h30"][cvm_])
        mfe = res["lab_ev"]["mfe_h40"][keep][dvm_]
        mae = res["lab_ev"]["mae_h40"][keep][dvm_]
        atr = res["lab_ev"]["atr_t"][keep][dvm_]
        ent = res["lab_ev"]["entry"][keep][dvm_]
        ok = np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(atr) & (ent > 0)
        cs = np.mean((mfe[ok] > 2 * atr[ok] / ent[ok]) & (mae[ok] > -atr[ok] / ent[ok]))
        mfe_c = res["lab_c1"]["mfe_h40"][cvm_]
        mae_c = res["lab_c1"]["mae_h40"][cvm_]
        atr_c = res["lab_c1"]["atr_t"][cvm_]
        ent_c = res["lab_c1"]["entry"][cvm_]
        okc = np.isfinite(mfe_c) & np.isfinite(mae_c) & np.isfinite(atr_c) & (ent_c > 0)
        cs_c = np.mean((mfe_c[okc] > 2 * atr_c[okc] / ent_c[okc]) & (mae_c[okc] > -atr_c[okc] / ent_c[okc]))
        print(f"{name}: PL={d['pl']:.2f}(C1 {c['pl']:.2f}) PF={d['pf']:.2f}(C1 {c['pf']:.2f}) "
              f"p90Δ={(d['p90']-c['p90'])*100:+.2f}pp p95Δ={(d['p95']-c['p95'])*100:+.2f}pp "
              f"干净狙击={cs*100:.1f}%(C1 {cs_c*100:.1f}%)")


if __name__ == "__main__":
    main()
