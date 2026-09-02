# -*- coding: utf-8 -*-
"""验收口径重铸 (geometry_recast): ATRN 升序几何基线规则的收益类裁决 (issue #13)。

背景: topn_density 复核实锤"密集日(>10 候选) ATRN 升序 top3"无模型规则
命中率 +6.61pp(p=0.02), 但对命中率验收是循环论证(门槛∝ATR)。
本实验改用收益类口径裁决该规则是否构成可用交易规则(预登记设计):

规则组(全部无模型, T 日收盘前可算):
  R1: 密集日(>10 候选) ATRN 升序 top3 (原几何规则)
  R2: 密集日 ATRN 降序 top3 (反向对照)
  R3: 密集日随机 top3, 重复 200 次取分布 (蒙特卡洛对照)
  R4: 全池基线 (所有信号日全候选等权)

收益口径(T+1 开盘买入, 与 hit_N20_k2.0 标签同几何):
  主变体 TP    : 20 交易日内 high 触及 入场价+2*ATR14[T] 即按触及价止盈(触及日),
                 否则第 20 日收盘卖出;
  稳健变体 TPSL: 另加 low 触及 入场价-2*ATR14[T] 按触及价止损;
                 同日双边同触保守记止损(悲观假设)。
  触发判定用原始价(与标签机器一致), 收益用 pct_chg 累积因子 cf 做除权除息调整。
  费用: 佣金双边各 0.1%, 印花税卖出 0.05%; 滑点假设=0(精确按 开盘/触及价/收盘 成交)。

指标: 每笔期望收益(净)/PF/盈亏比/胜率 + 按日等权组合净值的最大回撤/夏普
      (持仓期间逐日盯市, 无持仓日记 0, 段界后未平仓部分不计)。
切分: train 2001-2018 / val 2019-01~2022-10 (隔离带同 race_rerun_v2), test 封存禁用。
池清洗: excluded_events_{main,backup}.parquet f_any==True 剔除, 应用于全部规则。
双池: main=m_fractal15_full / backup=m_zigzag05_nofilter。

判定(预登记):
  R1 须在 train/val 双段同时满足 PF(净)>1 且不劣于 R4 全池基线
  (期望净收益与 PF 均不劣于 R4)才算可用;
  R1 vs R3 蒙特卡洛分布的分位给出随机性排除证据;
  若 R1 双段都不如 R4 或 R3 中位数, 判"不可用"并关闭。

输出:
  v3_pipeline/reports/geometry_recast/results_geometry_recast.json
  v3_pipeline/reports/geometry_recast/geometry_recast_report.md
  v3_pipeline/reports/geometry_recast/progress.log (阶段+心跳, 追加)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
FM_DIR = ROOT / "v3_pipeline" / "reports" / "feature_matrix"
CLEAN_DIR = ROOT / "v3_pipeline" / "reports" / "pool_cleaning"
DAILY_DIR = ROOT / "stock_data" / "daily"
OUT_DIR = ROOT / "v3_pipeline" / "reports" / "geometry_recast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"

POOLS = {
    "main": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_fractal_o15_s20",
    "backup": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_zigzag_p05_s5",
}

TRAIN_LO, TRAIN_HI = pd.Timestamp("2001-01-01"), pd.Timestamp("2018-12-31")
VAL_LO, VAL_HI = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-10-31")
EMBARGO = [(pd.Timestamp("2018-11-19"), pd.Timestamp("2018-12-28")),
           (pd.Timestamp("2022-09-13"), pd.Timestamp("2022-10-31"))]

N_WIN = 20          # 持有窗口(交易日)
K_ATR = 2.0         # 止盈/止损 ATR 倍数
DENSE_MIN = 11      # 密集日: 候选数 > 10
TOP_N = 3
MC_REPS = 200
SEED = 42
FEE = 0.001         # 佣金单边
STAMP = 0.0005      # 印花税(卖出)
COST_DRAG = (1 - FEE) * (1 - FEE - STAMP) - 1.0  # 往返费用净拖累 ≈ -0.0024985


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 事件装配
def segment_mask(dates):
    in_emb = np.zeros(len(dates), bool)
    for lo, hi in EMBARGO:
        in_emb |= (dates >= lo) & (dates <= hi)
    seg = np.full(len(dates), "drop", dtype=object)
    seg[(dates >= TRAIN_LO) & (dates <= TRAIN_HI)] = "train"
    seg[(dates >= VAL_LO) & (dates <= VAL_HI)] = "val"
    seg[in_emb] = "embargo"
    return seg


def load_pool_events(pool):
    """特征矩阵宇宙(fm keys) + sig_idx + ATRN + hit 标签 + 清洗标记 + 段。"""
    fm = pd.read_parquet(FM_DIR / f"{pool}_pool_features.parquet",
                         columns=["ts_code", "date", "ATRN"])
    fm["date"] = pd.to_datetime(fm["date"])
    labdir = POOLS[pool]
    ev = pd.read_parquet(labdir / "events.parquet",
                         columns=["ts_code", "date", "sig_idx"])
    ev["date"] = pd.to_datetime(ev["date"])
    lb = pd.read_parquet(labdir / "labels.parquet",
                         columns=["group", "hit_N20_k2.0"])
    div = lb.iloc[:len(ev)].reset_index(drop=True)
    assert (div["group"] == "div").all()
    ev["hit"] = div["hit_N20_k2.0"].to_numpy()
    df = fm.merge(ev, on=["ts_code", "date"], how="left", validate="1:1")
    assert df["sig_idx"].notna().all()
    ex = pd.read_parquet(CLEAN_DIR / f"excluded_events_{pool}.parquet",
                         columns=["ts_code", "date", "f_any"])
    ex["date"] = pd.to_datetime(ex["date"])
    df = df.merge(ex, on=["ts_code", "date"], how="left", validate="1:1")
    assert df["f_any"].notna().all()
    df["seg"] = segment_mask(df["date"].to_numpy())
    df["sig_idx"] = df["sig_idx"].astype(np.int64)
    return df.sort_values(["date", "ts_code"]).reset_index(drop=True)


# ================================================================ 交易模拟
def simulate_stock(path):
    """读单股日线, 计算 talib ATR14 与 cf 除权调整因子(口径同 divergence_lab)。"""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    close = df["close"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    pct = df["pct_chg"].to_numpy(np.float64)
    cf = np.cumprod(1.0 + np.where(np.isfinite(pct), pct, 0.0) / 100.0)
    atr = talib.ATR(high, low, close, timeperiod=14)
    days = df["trade_date"].to_numpy("datetime64[D]").astype(np.int32)
    return {"open": open_, "high": high, "low": low, "close": close,
            "cf": cf, "atr": atr, "days": days}


def trade_gross(st, e, x, px):
    """交易总收益(毛): 入场日 open->close 日内因子 × close[e]->close[x] 全收益因子
    (cf 比值, 含除权除息) × 退出日 close->px 日内因子。
    x==e 时退化为 px/open[e]-1 (同日, 无需调整)。"""
    open_, close, cf = st["open"], st["close"], st["cf"]
    if x == e:
        return px / open_[e] - 1.0
    return (close[e] / open_[e]) * (cf[x] / cf[e]) * (px / close[x]) - 1.0


def build_marks(st, e, x, px):
    """逐日盯市 marks: 返回 (全局交易日历日值 int32 数组, 日收益数组), 费用落在退出日。

    e=入场日(个股局部索引), x=退出日, px=退出价。
    退出日因子 = px/pre_close[x] = px/close[x] * cf[x]/cf[x-1]。
    """
    open_, close, cf = st["open"], st["close"], st["cf"]
    if x == e:
        return (st["days"][e:e + 1].copy(),
                np.array([(px / open_[e]) * (1.0 + COST_DRAG) - 1.0], np.float64))
    rets = np.empty(x - e + 1, np.float64)
    rets[0] = close[e] / open_[e] - 1.0
    if x - e > 1:
        d = np.arange(e + 1, x)
        rets[1:-1] = cf[d] / cf[d - 1] - 1.0
    rets[-1] = (px / close[x] * (cf[x] / cf[x - 1])) * (1.0 + COST_DRAG) - 1.0
    return st["days"][e:x + 1].copy(), rets


def simulate_pool(ev_df, pool):
    """逐股模拟全部事件的 TP/TPSL 两变体交易。

    返回 (trades 列表, all_days 全局交易日历 int32)。
    trades[i] = dict(ts_code, date, seg, atrn, hit_label, tp_hit_sim,
                     tp=dict(gross,net,mpos,mret) | None,
                     tpsl=dict(gross,net,mpos,mret) | None)
    无效事件(标签 NaN 同条件)两变体均为 None。
    """
    stocks = sorted(ev_df["ts_code"].unique())
    log(f"阶段=交易模拟 池={pool} 股票数={len(stocks)} 事件数={len(ev_df)}")
    data, missing = {}, []
    t0 = time.time()
    for i, s in enumerate(stocks):
        p = DAILY_DIR / f"{s}.parquet"
        if p.exists():
            data[s] = simulate_stock(p)
        else:
            missing.append(s)
        if (i + 1) % 1000 == 0:
            log(f"心跳=股票载入 {i + 1}/{len(stocks)} 耗时{time.time() - t0:.0f}s")
    if missing:
        log(f"!!警告=缺行情股票 {len(missing)} 只: {missing[:5]}")
    all_days = np.unique(np.concatenate([d["days"] for d in data.values()]))

    trades = []
    atrn_anchor_done = False
    for i, s in enumerate(stocks):
        st = data.get(s)
        if st is None:
            continue
        sub = ev_df[ev_df["ts_code"] == s]
        n = len(st["close"])
        open_, high, low, close, cf, atr = (st["open"], st["high"], st["low"],
                                            st["close"], st["cf"], st["atr"])
        if not atrn_anchor_done:  # ATRN 口径锚定: fm 的 ATRN vs talib ATR14/close
            sig0 = sub["sig_idx"].to_numpy()
            ok0 = np.isfinite(atr[sig0]) & (close[sig0] > 0)
            diff = np.abs(atr[sig0[ok0]] / close[sig0[ok0]]
                          - sub["ATRN"].to_numpy()[ok0])
            log(f"锚定=ATRN口径 池={pool} 股={s} n={int(ok0.sum())} "
                f"max|fmATRN-ATR14/close|={diff.max():.2e}")
            atrn_anchor_done = True
        for r in sub.itertuples(index=False):
            sig = int(r.sig_idx)
            rec = {"ts_code": s, "date": r.date, "seg": r.seg,
                   "atrn": float(r.ATRN), "hit_label": r.hit,
                   "tp_hit_sim": np.nan, "tp": None, "tpsl": None}
            if sig + N_WIN <= n - 1 and open_[sig + 1] > 0 and np.isfinite(atr[sig]):
                e, x_end = sig + 1, sig + N_WIN
                entry = open_[e]
                target = entry + K_ATR * atr[sig]
                stop = entry - K_ATR * atr[sig]
                hi_w = high[e:x_end + 1]
                lo_w = low[e:x_end + 1]
                tp_first = int(np.argmax(hi_w >= target)) if (hi_w >= target).any() else -1
                rec["tp_hit_sim"] = float(tp_first >= 0)
                # ---- TP 变体
                if tp_first >= 0:
                    x, px = e + tp_first, target
                else:
                    x, px = x_end, close[x_end]
                gross = trade_gross(st, e, x, px)
                mpos, mret = build_marks(st, e, x, px)
                rec["tp"] = {"gross": gross,
                             "net": (1 + gross) * (1 + COST_DRAG) - 1.0,
                             "mpos": mpos, "mret": mret}
                # ---- TPSL 变体(同日双触保守记止损)
                x2, px2 = x_end, close[x_end]
                for d in range(x_end - e + 1):
                    tpd, sld = hi_w[d] >= target, lo_w[d] <= stop
                    if sld:                      # 止损优先(含同日双触)
                        x2, px2 = e + d, stop
                        break
                    if tpd:
                        x2, px2 = e + d, target
                        break
                gross2 = trade_gross(st, e, x2, px2)
                mpos2, mret2 = build_marks(st, e, x2, px2)
                rec["tpsl"] = {"gross": gross2,
                               "net": (1 + gross2) * (1 + COST_DRAG) - 1.0,
                               "mpos": mpos2, "mret": mret2}
            trades.append(rec)
        if (i + 1) % 500 == 0:
            log(f"心跳=交易模拟 {i + 1}/{len(stocks)} 股 "
                f"耗时{time.time() - t0:.0f}s")
    log(f"阶段=交易模拟完成 池={pool} 交易数={len(trades)} "
        f"耗时{time.time() - t0:.0f}s")
    return trades, all_days


# ================================================================ 指标
def per_trade_metrics(nets):
    nets = np.asarray(nets, np.float64)
    n = len(nets)
    if n == 0:
        return {"n_trades": 0}
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    pf = gp / gl if gl > 1e-12 else (np.inf if gp > 0 else np.nan)
    return {
        "n_trades": n,
        "mean_net": float(nets.mean()),
        "median_net": float(np.median(nets)),
        "win_rate": float(len(wins) / n),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "pl_ratio": (float(wins.mean() / np.abs(losses.mean()))
                     if len(wins) and len(losses) else np.nan),
        "pf": pf,
        "gross_profit": gp, "gross_loss": gl,
    }


def portfolio_metrics(sel, variant, all_days, seg_lo_day, seg_hi_day):
    """按日等权组合: 持仓逐日盯市, 当日收益=持仓交易 marks 均值, 无持仓=0。

    段界: 只统计 [seg_lo_day, seg_hi_day] 内的 marks (段界后未平仓部分不计)。
    """
    pos_chunks, ret_chunks = [], []
    for t in sel:
        v = t[variant]
        if v is None:
            continue
        m = (v["mpos"] >= seg_lo_day) & (v["mpos"] <= seg_hi_day)
        if m.any():
            pos_chunks.append(v["mpos"][m])
            ret_chunks.append(v["mret"][m])
    win_mask = (all_days >= seg_lo_day) & (all_days <= seg_hi_day)
    win_days = all_days[win_mask]
    n_days = int(len(win_days))
    if n_days == 0:
        return {"n_days": 0}
    daily = np.zeros(n_days, np.float64)
    if pos_chunks:
        pos = np.concatenate(pos_chunks)
        ret = np.concatenate(ret_chunks)
        lo = int(np.searchsorted(all_days, seg_lo_day))
        pidx = np.searchsorted(all_days, pos) - lo
        ok = (pidx >= 0) & (pidx < n_days)
        sums = np.zeros(n_days, np.float64)
        cnts = np.zeros(n_days, np.float64)
        np.add.at(sums, pidx[ok], ret[ok])
        np.add.at(cnts, pidx[ok], 1.0)
        np.divide(sums, cnts, out=daily, where=cnts > 0)
    nav = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1.0).min())
    mu, sd = float(daily.mean()), float(daily.std(ddof=1)) if n_days > 1 else 0.0
    sharpe = mu / sd * np.sqrt(252.0) if sd > 1e-12 else np.nan
    return {"n_days": n_days, "total_return": float(nav[-1] - 1.0),
            "max_drawdown": mdd, "sharpe": sharpe,
            "ann_return": float(nav[-1] ** (252.0 / n_days) - 1.0) if n_days else np.nan,
            "mean_daily": mu, "std_daily": sd}


def rule_metrics(sel, variant, all_days, seg_lo_day, seg_hi_day):
    nets = [t[variant]["net"] for t in sel if t[variant] is not None]
    out = per_trade_metrics(nets)
    out["portfolio"] = portfolio_metrics(sel, variant, all_days,
                                         seg_lo_day, seg_hi_day)
    return out


# ================================================================ 规则选股
def select_rules(trades, seg):
    """对指定段的事件做 R1/R2/R4 选股与 R3 蒙特卡洛。

    返回 (fixed=dict(rule->交易索引列表), dense_days_info, mc=dict(rep->索引列表))。
    """
    idx = [i for i, t in enumerate(trades)
           if t["seg"] == seg and t["tp"] is not None]
    # 按日分组
    by_day = {}
    for i in idx:
        by_day.setdefault(trades[i]["date"], []).append(i)
    days = sorted(by_day)
    dense = [d for d in days if len(by_day[d]) >= DENSE_MIN]
    r1, r2 = [], []
    for d in dense:
        rows = by_day[d]
        atrn = np.array([trades[i]["atrn"] for i in rows], np.float64)
        order = np.argsort(atrn, kind="stable")
        r1.extend(rows[j] for j in order[:TOP_N])
        r2.extend(rows[j] for j in order[::-1][:TOP_N])
    fixed = {"R1_atrn_asc_top3": r1, "R2_atrn_desc_top3": r2,
             "R4_pool_all": idx}
    mc = {}
    for rep in range(MC_REPS):
        rng = np.random.default_rng(SEED * 1000 + rep)
        sel = []
        for d in dense:
            rows = by_day[d]
            take = rng.choice(len(rows), size=min(TOP_N, len(rows)),
                              replace=False)
            sel.extend(rows[int(j)] for j in take)
        mc[rep] = sel
    info = {"n_days": len(days), "n_dense_days": len(dense),
            "n_events": len(idx)}
    return fixed, info, mc


# ================================================================ 主流程
def main():
    t_all = time.time()
    log("阶段=脚本启动 geometry_recast(收益类口径裁决 ATRN 升序几何规则, issue #13)")
    log(f"口径=N{N_WIN} k{K_ATR} 密集>{DENSE_MIN - 1} top{TOP_N} "
        f"MC={MC_REPS} 费用=双边{FEE}+印花税{STAMP} 滑点=0 seed={SEED}")
    results = {"seed": SEED, "n_win": N_WIN, "k_atr": K_ATR,
               "dense_min": DENSE_MIN, "top_n": TOP_N, "mc_reps": MC_REPS,
               "fee_one_side": FEE, "stamp_sell": STAMP,
               "cost_drag_roundtrip": COST_DRAG, "slippage": 0.0,
               "variants": ["tp", "tpsl"], "pools": {}}

    seg_days = {
        "train": (np.int32(TRAIN_LO.to_datetime64().astype("datetime64[D]").astype(np.int32)),
                  np.int32(TRAIN_HI.to_datetime64().astype("datetime64[D]").astype(np.int32))),
        "val": (np.int32(VAL_LO.to_datetime64().astype("datetime64[D]").astype(np.int32)),
                np.int32(VAL_HI.to_datetime64().astype("datetime64[D]").astype(np.int32))),
    }

    for pool in ("main", "backup"):
        log(f"阶段=事件装配 池={pool}")
        ev = load_pool_events(pool)
        n0 = len(ev)
        ev = ev.loc[~ev["f_any"]].reset_index(drop=True)  # 池清洗剔除
        log(f"池={pool} 清洗剔除 {n0 - len(ev)} 事件(f_any), 余 {len(ev)}")
        trades, all_days = simulate_pool(ev, pool)

        # ---- 锚定: 模拟 TP 触及 vs 标签 hit (train+val 有效事件应 ~100% 一致)
        lab, sim = [], []
        for t in trades:
            if t["seg"] in ("train", "val") and t["tp"] is not None \
                    and t["hit_label"] is not None and np.isfinite(t["hit_label"]):
                lab.append(float(t["hit_label"]))
                sim.append(t["tp_hit_sim"])
        lab, sim = np.array(lab), np.array(sim)
        match = float((lab == sim).mean()) if len(lab) else np.nan
        log(f"锚定=hit标签 池={pool} train+val 有效事件 n={len(lab)} "
            f"模拟触及==标签hit 一致率={match:.6f}")

        pool_res = {"n_events_after_clean": len(trades),
                    "hit_anchor": {"n": int(len(lab)), "match_rate": match},
                    "segments": {}}
        for seg in ("train", "val"):
            lo_d, hi_d = seg_days[seg]
            fixed, info, mc = select_rules(trades, seg)
            log(f"阶段=规则评估 池={pool} 段={seg} 信号日={info['n_days']} "
                f"密集日={info['n_dense_days']} 事件={info['n_events']}")
            seg_res = {"info": info, "variants": {}}
            for variant in ("tp", "tpsl"):
                vres = {"rules": {}}
                for rname, sel in fixed.items():
                    m = rule_metrics([trades[i] for i in sel], variant,
                                     all_days, lo_d, hi_d)
                    vres["rules"][rname] = m
                    log(f"池={pool} 段={seg} 变体={variant} 规则={rname} "
                        f"n={m.get('n_trades', 0)} "
                        f"E[net]={m.get('mean_net', np.nan):+.4f} "
                        f"PF={m.get('pf', np.nan):.3f} "
                        f"Sharpe={m['portfolio'].get('sharpe', np.nan)}")
                # ---- R3 蒙特卡洛
                t0 = time.time()
                mc_rows = []
                for rep, sel in mc.items():
                    m = rule_metrics([trades[i] for i in sel], variant,
                                     all_days, lo_d, hi_d)
                    mc_rows.append(m)
                    if (rep + 1) % 50 == 0:
                        log(f"心跳=MC 池={pool} 段={seg} 变体={variant} "
                            f"{rep + 1}/{MC_REPS} 耗时{time.time() - t0:.0f}s")
                def q(key, sub=None):
                    xs = [r[key] if sub is None else r["portfolio"][key]
                          for r in mc_rows]
                    xs = np.array([x for x in xs if np.isfinite(x)], np.float64)
                    if len(xs) == 0:
                        return {"median": np.nan, "q05": np.nan, "q95": np.nan}
                    return {"median": float(np.median(xs)),
                            "q05": float(np.quantile(xs, 0.05)),
                            "q95": float(np.quantile(xs, 0.95))}
                r1 = vres["rules"]["R1_atrn_asc_top3"]
                dist = {
                    "mean_net": q("mean_net"), "pf": q("pf"),
                    "sharpe": q("sharpe", "portfolio"),
                    "max_drawdown": q("max_drawdown", "portfolio"),
                }
                # R1 在 R3 分布中的分位(随机性排除证据)
                xs_net = np.array([r["mean_net"] for r in mc_rows], np.float64)
                xs_pf = np.array([r["pf"] for r in mc_rows
                                  if np.isfinite(r["pf"])], np.float64)
                dist["r1_percentile_mean_net"] = (
                    float((xs_net < r1["mean_net"]).mean())
                    if r1.get("n_trades") else np.nan)
                dist["r1_percentile_pf"] = (
                    float((xs_pf < r1["pf"]).mean())
                    if r1.get("n_trades") and len(xs_pf) else np.nan)
                vres["R3_random_top3_mc"] = dist
                seg_res["variants"][variant] = vres
            pool_res["segments"][seg] = seg_res
        results["pools"][pool] = pool_res

    with (OUT_DIR / "results_geometry_recast.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=数值落盘 耗时{time.time() - t_all:.0f}s -> results_geometry_recast.json")


if __name__ == "__main__":
    main()
