#!/usr/bin/env python3
"""阶段底信号当天识别赛(A 方案本体)—— 预登记 = 同目录 README.md(先于跑数落盘)。

研究问题:用事件日收盘(含)之前可知的信息,能否选出 H60 持有能赚钱的信号子集?
6 个预登记候选(R0~R5)+ 基线 ALL,只打一发;训练段拟合,验证段宣判。

口径要点(逐条对应 README):
  宇宙成员资格与交易模拟:引擎序列(load_market_data 返回,起点约 2025-11-03);
  特征 f1~f4 与规则 R1/R2/R3:stock_data/daily 全历史日线(同文件同口径,不被引擎
  60 自然日前向缓冲截断),事件日处截断,特征/规则函数只接收截断数组(结构防前视)。
  模拟逐条复刻 stage_ceiling(已独立复核):次日开盘买、涨停/无报价拒买、整手现金、
  第 60 个交易日收盘卖、跌停顺延、耗尽 incomplete、成本调冻结引擎原语。

产物:features.parquet / trades_recognizer.parquet / summary_recognizer.csv(28 行)/
verdict.json / report.md / progress.log(心跳)。
全程禁网络;除本目录外仓库只读。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline", "scripts"))

import strategy_engine as se  # noqa: E402  v1 冻结引擎(只读复用原语:加载/成本/常量)

EXP_DIR = os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026")
OUT_DIR = os.path.join(EXP_DIR, "stage_recognizer")
EVENTS = {
    "v1": os.path.join(EXP_DIR, "events_v1.parquet"),
    "v2": os.path.join(EXP_DIR, "events_v2.parquet"),
}
CEILING_TRADES = os.path.join(EXP_DIR, "stage_ceiling", "trades_ceiling.parquet")
LIMIT_DIR = os.path.join(REPO_ROOT, "stock_data", "stk_limit")
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

START, END = "2026-01-01", "2026-08-31"
BUDGET = 100_000.0
H = 60                        # 持有期(交易日,日历口径)
HIST_MIN = 120                # 事件日前全历史最少交易日数(特征需要)
TRAIN_END = pd.Timestamp("2026-04-01")   # 训练段 [<] / 验证段 [>=]
TOL = 1e-9                    # 价格比较容差
SAMPLE_SEED = 20260904        # 自检抽样种子(预登记)
REF_BASE_V1 = 2449            # 自检 3 参照数(v1 基准集)
CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
GRID_ROWS_ORDER = ["ALL", "R0", "R1", "R2", "R3", "R4", "R5"]
PASS_MIN_N = 30               # 判活线 1:功效下限
PASS_T = 2.0                  # 判活线 3:cluster_t 下限
PASS_EXCESS = 0.10            # 判活线 4:超 ALL 净笔均下限(10pp)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 数据结构
class StockBars:
    """单股行情对齐到交易日历的 numpy 数组(缺日 = NaN),仅用于模拟。"""

    __slots__ = ("open_", "high", "low", "close", "up_lim", "dn_lim")

    def __init__(self, n: int) -> None:
        self.open_ = np.full(n, np.nan)
        self.high = np.full(n, np.nan)
        self.low = np.full(n, np.nan)
        self.close = np.full(n, np.nan)
        self.up_lim = np.full(n, np.nan)   # NaN = 当日无约束(引擎同款缺省)
        self.dn_lim = np.full(n, np.nan)


def build_bars(md: se.MarketData, cal_dt: pd.DatetimeIndex, codes: list[str]) -> dict:
    """日线向量化对齐 + 涨跌停表一次性对齐成 DataFrame 再按列取(禁逐股逐日 to_dict)。"""
    n = len(cal_dt)
    bars: dict[str, StockBars] = {}
    for code in codes:
        sb = StockBars(n)
        df = md.daily.get(code)
        if df is not None:
            loc = cal_dt.get_indexer(df.index)
            mask = loc >= 0
            idx = loc[mask]
            sub = df.iloc[np.flatnonzero(mask)]
            sb.open_[idx] = sub["open"].to_numpy(dtype=float)
            sb.high[idx] = sub["high"].to_numpy(dtype=float)
            sb.low[idx] = sub["low"].to_numpy(dtype=float)
            sb.close[idx] = sub["close"].to_numpy(dtype=float)
        bars[code] = sb
    up_df = pd.DataFrame({d: lim["up_limit"] for d, lim in md.limits.items()}).T
    dn_df = pd.DataFrame({d: lim["down_limit"] for d, lim in md.limits.items()}).T
    up_df = up_df.reindex(cal_dt)
    dn_df = dn_df.reindex(cal_dt)
    for code in codes:
        if code in up_df.columns:
            bars[code].up_lim = up_df[code].to_numpy(dtype=float)
            bars[code].dn_lim = dn_df[code].to_numpy(dtype=float)
    return bars


# ---------------------------------------------------------------- 全历史序列(特征用)
class FullHist:
    """单股全历史日线(不复权,与引擎同源),特征与规则 R1/R2/R3 在此序列上事件日截断计算。"""

    __slots__ = ("dates", "open_", "high", "low", "close", "pos", "dn_lim")

    def __init__(self, df: pd.DataFrame) -> None:
        self.dates = pd.DatetimeIndex(df["trade_date"])
        self.open_ = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.pos = {d: i for i, d in enumerate(self.dates)}
        self.dn_lim = np.full(len(df), np.nan)


def load_full_hist(codes: list[str]) -> dict:
    out: dict[str, FullHist] = {}
    t0 = time.time()
    for i, code in enumerate(codes):
        p = os.path.join(REPO_ROOT, "stock_data", "daily", f"{code}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["trade_date", "open", "high", "low", "close"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        assert not df["trade_date"].duplicated().any(), f"{code} 全历史日期重复"
        out[code] = FullHist(df)
        if (i + 1) % 1000 == 0:
            log(f"heartbeat: 全历史加载 {i + 1}/{len(codes)} ({time.time() - t0:.1f}s)")
    log(f"[load] 全历史日线就绪:{len(out)} 股({time.time() - t0:.0f}s)")
    return out


def attach_full_dn_limits(fh: dict, min_date: pd.Timestamp) -> int:
    """R3 跌停收盘判定:全历史序列日期 -> 跌停价。一次性对齐成 DataFrame,禁逐股逐日。
    缺文件日 = 无约束(NaN)。返回缺文件日计数(披露)。"""
    all_dates = sorted({d for h in fh.values() for d in h.dates if d >= min_date})
    cols = {}
    missing = 0
    for d in all_dates:
        fp = os.path.join(LIMIT_DIR, d.strftime("%Y%m%d") + ".parquet")
        if not os.path.exists(fp):
            missing += 1
            continue
        lf = pd.read_parquet(fp)
        cols[d] = lf.set_index("ts_code")["down_limit"]
    dn_df = pd.DataFrame(cols).T   # index=日期, columns=ts_code
    for code, h in fh.items():
        in_idx = h.dates[h.dates >= min_date]
        if code in dn_df.columns:
            s = dn_df[code].reindex(in_idx)
            pos = np.flatnonzero(h.dates >= min_date)
            h.dn_lim[pos] = s.to_numpy(dtype=float)
    return missing


# ---------------------------------------------------------------- 特征(因果:只收截断数组)
def compute_features(c: np.ndarray, j_anchor: int, cross_dif: float,
                     dif_lift: float) -> dict:
    """c = 截断到事件日(含)的 close 数组,j = len(c)-1 = 事件日下标。
    只接收截断数组,从结构上杜绝前视。f5/f6 为事件表列(事件日收盘可知)。"""
    j = len(c) - 1
    close_ev = c[j]
    return dict(
        f1=close_ev / c[max(0, j - 119):j + 1].max() - 1.0,
        f2=close_ev / c[max(0, j - 59):j + 1].min() - 1.0,
        f3=close_ev / c[max(0, j - 119):j + 1].min() - 1.0,
        f4=close_ev / c[j_anchor] - 1.0,
        f5=float(cross_dif),
        f6=float(dif_lift),
    )


# ---------------------------------------------------------------- 规则候选(因果:只收截断数组)
def rule_r1(c: np.ndarray) -> bool:
    """R1 阶梯下跌排除:n_lower >= 4 -> True(拒绝)。"""
    j = len(c) - 1
    m = [c[ck - 19:ck + 1].min() for ck in (j - 100, j - 80, j - 60, j - 40, j - 20, j)]
    n_lower = sum(1 for k in range(5) if m[k + 1] < m[k])
    return n_lower >= 4


def rule_r2(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
    """R2 箱体收窄下破排除:收窄且下破 -> True(拒绝)。"""
    j = len(c) - 1
    close_ev = c[j]
    r_recent = (h[j - 19:j + 1].max() - l[j - 19:j + 1].min()) / close_ev
    r_prior = (h[j - 79:j - 59].max() - l[j - 79:j - 59].min()) / close_ev
    narrow = r_recent < 0.6 * r_prior
    breakdown = close_ev == c[j - 59:j + 1].min()
    return bool(narrow and breakdown)


def rule_r3(o: np.ndarray, c: np.ndarray, dn: np.ndarray) -> bool:
    """R3 利空冲击排除(价格代理,README 披露 1):过去 60 个交易日内存在跌停收盘
    或大幅跳空低开(open/前收 - 1 <= -5%)-> True(拒绝)。dn 缺值日 = 无约束。"""
    j = len(c) - 1
    for k in range(j - 59, j + 1):
        if np.isfinite(dn[k]) and c[k] <= dn[k] + TOL:
            return True
        if o[k] / c[k - 1] - 1.0 <= -0.05:
            return True
    return False


# ---------------------------------------------------------------- 逐笔模拟(复刻 stage_ceiling)
def simulate(sb: StockBars, di_ev: int, horizon: int, cal_arr: list) -> dict:
    """入场:事件日下一交易日开盘买(无报价/涨停拒买不递补,整手现金逐条复刻冻结引擎)。
    出场:入场日记第 1 日,第 horizon 个交易日(日历口径)收盘卖,跌停顺延,耗尽 incomplete。"""
    n_cal = len(cal_arr)
    if di_ev + 1 >= n_cal:
        return dict(status="dropped_no_next_day")
    di_e = di_ev + 1
    o = sb.open_[di_e]
    if not np.isfinite(o):
        return dict(status="dropped_no_quote")
    up = sb.up_lim[di_e]
    if np.isfinite(up) and o >= up - TOL:
        return dict(status="dropped_limitup")
    px = o * (1.0 + se.SLIPPAGE)
    sh = int(BUDGET / px / se.BOARD_LOT) * se.BOARD_LOT
    if sh < se.BOARD_LOT:
        sh = se.BOARD_LOT
    comm = se.buy_cost(sh, px)
    while sh > 0 and sh * px + comm > BUDGET + 1e-6:
        sh -= se.BOARD_LOT
        comm = se.buy_cost(sh, px) if sh > 0 else 0.0
    if sh <= 0:
        return dict(status="dropped_cash")
    base = dict(entry_date=cal_arr[di_e], entry_raw=o, entry_exec=px,
                shares=sh, buy_comm=comm)
    deferred = 0
    for di in range(di_e + horizon - 1, n_cal):
        c = sb.close[di]
        if not np.isfinite(c):
            continue  # 停牌日计入持有日但不评估
        dn = sb.dn_lim[di]
        if np.isfinite(dn) and c <= dn + TOL:
            deferred += 1
            continue  # 跌停顺延至下一非跌停收盘日
        xs = c * (1.0 - se.SLIPPAGE)
        xcomm, stamp = se.sell_costs(sh, xs, cal_arr[di])
        gross_ret = xs / px - 1.0
        net_pnl = sh * (xs - px) - comm - xcomm - stamp
        net_ret = net_pnl / (sh * px + comm)
        return dict(status="closed", exit_date=cal_arr[di], exit_raw=c, exit_exec=xs,
                    sell_comm=xcomm, stamp=stamp, gross_ret=gross_ret,
                    net_ret=net_ret, net_pnl=net_pnl,
                    held_days=di - di_e + 1, deferred_days=deferred, **base)
    return dict(status="incomplete", deferred_days=deferred, **base)


# ---------------------------------------------------------------- 聚类稳健 t(Liang-Zeger)
def cluster_t(x: np.ndarray, clusters: np.ndarray) -> float:
    """单样本均值的聚类稳健 t。预登记口径:score = x - x̄ 按簇求和 S_c;
    var(x̄) = [G/(G-1)] x Σ S_c² / n²;G<2 记 NaN。"""
    n = len(x)
    if n < 2:
        return np.nan
    xbar = x.mean()
    s = x - xbar
    df = pd.DataFrame({"s": s, "c": clusters})
    sums = df.groupby("c")["s"].sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan
    var = (g / (g - 1.0)) * float((sums ** 2).sum()) / (n * n)
    if var <= 0:
        return np.nan
    return float(xbar / np.sqrt(var))


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t_all = time.time()
    log(f"STAGE RECOGNIZER START window=[{START}..{END}] H={H} hist_min={HIST_MIN} "
        f"train_end={TRAIN_END.date()} | 候选={CANDIDATES} + 基线 ALL | "
        f"预计总时长约 3-5 分钟(数据加载 ~40s,模拟/特征秒级)")

    # ---------------- 载入事件与已复核分组标签 ----------------
    events = {}
    for name, path in EVENTS.items():
        ev = pd.read_parquet(path)
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        ev["anchor_date"] = pd.to_datetime(ev["anchor_date"])
        assert not ev.duplicated(["ts_code", "event_date"]).any(), f"{name} 事件键重复"
        assert ev[["cross_dif", "dif_lift"]].notna().all().all(), f"{name} f5/f6 缺值"
        events[name] = ev
        log(f"[load] {name}: 事件 {len(ev)} 行 / {ev['ts_code'].nunique()} 股")

    tc = pd.read_parquet(CEILING_TRADES)
    tc = tc[(tc["W"] == 60) & (tc["H"] == 60)].copy()
    tc["event_date"] = pd.to_datetime(tc["event_date"])
    labels = {(r.variant, r.ts_code, r.event_date): (r.group, r.status)
              for r in tc.itertuples(index=False)}
    log(f"[load] 分组标签(W60H60,已复核直接读):{len(tc)} 行;"
        f"near={int((tc['group'] == 'near').sum())} far={int((tc['group'] == 'far').sum())}")

    codes = sorted(set(events["v1"]["ts_code"]) | set(events["v2"]["ts_code"]))
    t0 = time.time()
    md = se.load_market_data(codes, START, END, REPO_ROOT, log_path=LOG_PATH)
    cal_arr = list(md.calendar)
    cal_dt = pd.DatetimeIndex(cal_arr)
    cal_index = {d: i for i, d in enumerate(cal_arr)}
    cal_set = set(cal_arr)
    assert md.limit_missing_dates == 0, "本预登记假设 stk_limit 窗口内零缺日"
    log(f"[load] 行情就绪:{len(md.daily)} 股 / 日历 {len(cal_arr)} 交易日 "
        f"({time.time() - t0:.0f}s)")

    bars = build_bars(md, cal_dt, codes)
    log("[align] 日历对齐数组与涨跌停矩阵就绪")

    # 引擎序列位置表(宇宙成员资格用)
    posmap: dict[str, dict] = {}
    serlen: dict[str, int] = {}
    for code, df in md.daily.items():
        posmap[code] = {d: i for i, d in enumerate(df.index)}
        serlen[code] = len(df)

    # 全历史序列(特征/规则用)
    fh = load_full_hist(codes)

    # ---------------- 宇宙构造(每变体) ----------------
    uni: dict[str, pd.DataFrame] = {}
    exclusions: dict[str, list] = {}
    base_events: dict[str, pd.DataFrame] = {}
    for name in ["v1", "v2"]:
        ev = events[name]
        rows, excl, base_rows = [], [], []
        for r in ev.itertuples(index=False):
            pm = posmap.get(r.ts_code)
            if pm is None or r.event_date not in cal_set:
                continue
            i_ev = pm.get(r.event_date, -1)
            if i_ev < 0:
                continue
            if i_ev >= serlen[r.ts_code] - H:
                continue  # 无 +60 交易日可测(截断),不进基准集
            base_rows.append(r)
            h = fh.get(r.ts_code)
            j_ev = h.pos.get(r.event_date, -1) if h else -1
            if j_ev < HIST_MIN:
                excl.append(dict(ts_code=r.ts_code, event_date=str(r.event_date.date()),
                                 reason=f"hist_lt_{HIST_MIN}", hist_days=int(j_ev)))
                continue
            j_anchor = h.pos.get(r.anchor_date, -1)
            if j_anchor < 0:
                excl.append(dict(ts_code=r.ts_code, event_date=str(r.event_date.date()),
                                 reason="anchor_out_full_hist", hist_days=int(j_ev)))
                continue
            rows.append(dict(ts_code=r.ts_code, event_date=r.event_date,
                             anchor_date=r.anchor_date, i_ev=i_ev, j_ev=j_ev,
                             j_anchor=j_anchor, cross_dif=r.cross_dif,
                             dif_lift=r.dif_lift))
        uni[name] = pd.DataFrame(rows)
        base_events[name] = base_rows
        exclusions[name] = excl
        log(f"[universe] {name}: 基准集 {len(base_rows)}(参照 {REF_BASE_V1 if name == 'v1' else '—'})"
            f" -> 剔除 {len(excl)}(逐笔见 verdict.json)-> 宇宙 {len(uni[name])}")

    # ---------------- R3 跌停价全历史对齐 ----------------
    min_dn_date = min(h.dates[u["j_ev"] - (H - 1)]
                      for name in uni for _, u in uni[name].iterrows()
                      for h in [fh[u["ts_code"]]])
    n_dn_missing = attach_full_dn_limits(fh, min_dn_date)
    log(f"[align] 全历史跌停价对齐就绪:覆盖起点 {min_dn_date.date()},"
        f"缺文件日 {n_dn_missing}(=无约束)")

    # ---------------- 特征 + 规则判定(每宇宙事件,因果截断) ----------------
    feat_rows: dict[str, list] = {"v1": [], "v2": []}
    for name in ["v1", "v2"]:
        t0 = time.time()
        for u in uni[name].itertuples(index=False):
            h = fh[u.ts_code]
            j = u.j_ev
            c_tr = h.close[:j + 1]
            o_tr = h.open_[:j + 1]
            h_tr = h.high[:j + 1]
            l_tr = h.low[:j + 1]
            dn_tr = h.dn_lim[:j + 1]
            feats = compute_features(c_tr, u.j_anchor, u.cross_dif, u.dif_lift)
            grp, _ = labels.get((name, u.ts_code, u.event_date), (None, "missing"))
            feat_rows[name].append(dict(
                ts_code=u.ts_code, event_date=u.event_date, anchor_date=u.anchor_date,
                j_ev=j, **feats,
                rej_R1=rule_r1(c_tr), rej_R2=rule_r2(o_tr, h_tr, l_tr, c_tr),
                rej_R3=rule_r3(o_tr, c_tr, dn_tr),
                group=grp if isinstance(grp, str) else "other",
                label_near=1 if grp == "near" else 0,
                segment="train" if u.event_date < TRAIN_END else "validation"))
        log(f"[features] {name}: {len(feat_rows[name])} 宇宙事件特征+规则判定完成"
            f"({time.time() - t0:.0f}s)")
    feats_df = {name: pd.DataFrame(feat_rows[name]) for name in ["v1", "v2"]}

    # ---------------- 训练段拟合:R0 模型 / R5 θ ----------------
    from sklearn.linear_model import LogisticRegression
    FCOLS = ["f1", "f2", "f3", "f4", "f5", "f6"]
    fit_info = {}
    for name in ["v1", "v2"]:
        ftr = feats_df[name]
        tr = ftr[ftr["segment"] == "train"]
        X = tr[FCOLS].to_numpy(dtype=float)
        y = tr["label_near"].to_numpy(dtype=int)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xs = (ftr[FCOLS].to_numpy(dtype=float) - mu) / sd
        if len(np.unique(y)) < 2:
            # 训练段单类(如无 near):模型不可拟合,R0 恒拒,原样披露不回炉
            p = np.zeros(len(ftr))
            clf_coef = {c: 0.0 for c in FCOLS}
            clf_intercept = 0.0
            log(f"[fit] {name}: 训练段标签单类,R0 退化为恒拒(披露)")
        else:
            clf = LogisticRegression(max_iter=2000)
            clf.fit((X - mu) / sd, y)
            p = clf.predict_proba(Xs)[:, 1]
            clf_coef = {c: float(v) for c, v in zip(FCOLS, clf.coef_[0])}
            clf_intercept = float(clf.intercept_[0])
        ftr["p_R0"] = p
        near_f1 = tr.loc[tr["label_near"] == 1, "f1"]
        theta = float(near_f1.median()) if len(near_f1) else np.nan
        fit_info[name] = dict(
            n_train=len(tr), n_train_near=int(y.sum()),
            r0_mu={c: float(v) for c, v in zip(FCOLS, mu)},
            r0_sd={c: float(v) for c, v in zip(FCOLS, sd)},
            r0_coef=clf_coef,
            r0_intercept=clf_intercept,
            r5_theta=theta)
        log(f"[fit] {name}: 训练段 {len(tr)} 事件(near {int(y.sum())})"
            f" | R0 拟合完成 | R5 θ(仅训练段 near 的 f1 中位数)= {theta:+.6f}")

    # ---------------- 候选选择布尔列 ----------------
    sel: dict[str, pd.DataFrame] = {}
    for name in ["v1", "v2"]:
        ftr = feats_df[name]
        s = pd.DataFrame(index=ftr.index)
        s["ALL"] = True
        s["R0"] = ftr["p_R0"] >= 0.5
        s["R1"] = ~ftr["rej_R1"]
        s["R2"] = ~ftr["rej_R2"]
        s["R3"] = ~ftr["rej_R3"]
        s["R4"] = s["R0"] & s["R1"] & s["R2"] & s["R3"]
        theta = fit_info[name]["r5_theta"]
        s["R5"] = (ftr["f1"] <= theta) if np.isfinite(theta) else False
        sel[name] = s
        log(f"[select] {name}: 各候选接受数 "
            + " ".join(f"{c}={int(s[c].sum())}" for c in GRID_ROWS_ORDER))

    # ---------------- 模拟:基准集每事件 H60 一次(⊇ 宇宙 ∪ near/far) ----------------
    sims: dict[str, pd.DataFrame] = {}
    for name in ["v1", "v2"]:
        t0 = time.time()
        recs = []
        for r in base_events[name]:
            di_ev = cal_index[r.event_date]
            res = simulate(bars[r.ts_code], di_ev, H, cal_arr)
            recs.append(dict(ts_code=r.ts_code, event_date=r.event_date, **res))
        sims[name] = pd.DataFrame(recs)
        vc = sims[name]["status"].value_counts().to_dict()
        log(f"[simulate] {name}: 基准集 {len(recs)} 事件 H60 模拟完成"
            f"({time.time() - t0:.0f}s)状态分布={vc}")

    # ---------------- 自检 2:模拟对拍(near/far 逐笔 vs stage_ceiling) ----------------
    check2 = {}
    check2_ok = True
    for name in ["v1", "v2"]:
        nf = tc[(tc["variant"] == name) & (tc["group"].isin(["near", "far"]))]
        mine = sims[name].merge(
            nf[["ts_code", "event_date", "status", "net_ret"]],
            on=["ts_code", "event_date"], how="inner", suffixes=("", "_ref"))
        assert len(mine) == len(nf), f"{name} 对拍键不齐:{len(mine)} vs {len(nf)}"
        st_mine = mine["status"].value_counts().to_dict()
        st_ref = mine["status_ref"].value_counts().to_dict()
        dist_ok = st_mine == st_ref
        pair_status_ok = bool((mine["status"] == mine["status_ref"]).all())
        both_closed = mine[(mine["status"] == "closed") & (mine["status_ref"] == "closed")]
        max_diff = float((both_closed["net_ret"] - both_closed["net_ret_ref"]).abs().max()) \
            if len(both_closed) else 0.0
        ok = dist_ok and pair_status_ok and max_diff <= 1e-9
        check2_ok &= ok
        check2[name] = dict(n_pairs=len(mine), status_mine=st_mine, status_ref=st_ref,
                            pair_status_ok=pair_status_ok, max_abs_net_ret_diff=max_diff,
                            ok=bool(ok))
        log(f"[check2] {name}: 对拍 {len(mine)} 笔,逐笔状态一致={pair_status_ok},"
            f"状态分布一致={dist_ok},净收益最大绝对差={max_diff:.2e} "
            f"-> {'PASS' if ok else 'FAIL'}")

    # ---------------- 汇总 28 行 ----------------
    sum_rows: list[dict] = []
    for name in ["v1", "v2"]:
        ftr = feats_df[name].merge(
            sims[name], on=["ts_code", "event_date"], how="left", validate="one_to_one")
        assert ftr["status"].notna().all(), f"{name} 宇宙事件必有模拟结果"
        for cand in GRID_ROWS_ORDER:
            for seg in ["train", "validation"]:
                m_seg = ftr["segment"] == seg
                m_sel = sel[name][cand].to_numpy() & m_seg.to_numpy()
                g = ftr[m_sel]
                st = g["status"].value_counts().to_dict()
                cl = g[g["status"] == "closed"]
                ret = cl["net_ret"].to_numpy(dtype=float)
                n_closed = len(cl)
                ct = cluster_t(ret, cl["entry_date"].astype(str).to_numpy()) \
                    if n_closed else np.nan
                near_share = float(g["label_near"].mean()) if (seg == "train" and len(g)) \
                    else np.nan
                sum_rows.append(dict(
                    variant=name, candidate=cand, segment=seg,
                    n_universe=int(m_seg.sum()), n_selected=int(m_sel.sum()),
                    n_closed=n_closed,
                    n_dropped_limitup=int(st.get("dropped_limitup", 0)),
                    n_dropped_no_quote=int(st.get("dropped_no_quote", 0)),
                    n_dropped_cash=int(st.get("dropped_cash", 0)),
                    n_incomplete=int(st.get("incomplete", 0)),
                    net_mean=float(ret.mean()) if n_closed else np.nan,
                    net_median=float(np.median(ret)) if n_closed else np.nan,
                    win_rate=float((ret > 0).mean()) if n_closed else np.nan,
                    cluster_t=ct, near_share_selected=near_share))
    summary = pd.DataFrame(sum_rows)
    sum_path = os.path.join(OUT_DIR, "summary_recognizer.csv")
    summary.to_csv(sum_path, index=False, float_format="%.6f")
    log(f"[dump] summary_recognizer.csv {len(summary)} 行 -> {sum_path}")

    # ---------------- 自检 3:计数对齐 ----------------
    c3_base = {name: len(base_events[name]) for name in ["v1", "v2"]}
    base_diff_v1 = c3_base["v1"] - REF_BASE_V1
    c3_base_ok = abs(base_diff_v1) <= 2
    c3_sel_ok = bool((summary["n_selected"] <= summary["n_universe"]).all())
    c3_all_ok = bool((summary.loc[summary["candidate"] == "ALL", "n_selected"]
                      == summary.loc[summary["candidate"] == "ALL", "n_universe"]).all())
    check3_ok = c3_base_ok and c3_sel_ok and c3_all_ok
    log(f"[check3] 基准集 v1={c3_base['v1']}(参照 {REF_BASE_V1},差 {base_diff_v1})"
        f" v2={c3_base['v2']} | n_selected<=n_universe={c3_sel_ok}"
        f" | ALL 行 n_selected==n_universe={c3_all_ok}"
        f" -> {'PASS' if check3_ok else 'FAIL'}"
        f" | 剔除逐笔: v1 {len(exclusions['v1'])} 笔 / v2 {len(exclusions['v2'])} 笔")

    # ---------------- 自检 1:特征因果断言(抽 5 事件手工重算 f1~f4) ----------------
    rng = np.random.default_rng(SAMPLE_SEED)
    uni_all = pd.concat([uni[name].assign(variant=name) for name in ["v1", "v2"]],
                        ignore_index=True)
    pick1 = uni_all.iloc[rng.choice(len(uni_all), size=5, replace=False)]
    c1_rows, check1_ok = [], True
    for r in pick1.itertuples(index=False):
        h = fh[r.ts_code]
        j = h.pos[r.event_date]
        c_tr = h.close[:j + 1]          # 只取 <= 事件日数据(手工切片)
        man = dict(
            f1=c_tr[-1] / c_tr[-120:].max() - 1.0,
            f2=c_tr[-1] / c_tr[-60:].min() - 1.0,
            f3=c_tr[-1] / c_tr[-120:].min() - 1.0,
            f4=c_tr[-1] / h.close[r.j_anchor] - 1.0)
        pipe = feats_df[r.variant]
        prow = pipe[(pipe["ts_code"] == r.ts_code)
                    & (pipe["event_date"] == r.event_date)].iloc[0]
        diffs = {k: abs(man[k] - float(prow[k])) for k in man}
        ok = all(v <= 1e-12 for v in diffs.values())
        check1_ok &= ok
        c1_rows.append(dict(variant=r.variant, ts_code=r.ts_code,
                            event_date=str(r.event_date.date()),
                            max_abs_diff=float(max(diffs.values())), ok=bool(ok)))
        log(f"[check1-sample] {r.variant} {r.ts_code} ev={r.event_date.date()} "
            f"max|diff|={max(diffs.values()):.2e} -> {'PASS' if ok else 'FAIL'}")
    log(f"[check1] 特征因果断言(5 事件手工重算 f1~f4,容差 1e-12;"
        f"特征函数只接收截断数组): {'PASS' if check1_ok else 'FAIL'}")

    # ---------------- 自检 4:交易因果 ----------------
    entered = pd.concat([sims[name].assign(variant=name) for name in ["v1", "v2"]],
                        ignore_index=True)
    entered = entered[entered["status"].isin({"closed", "incomplete"})]
    n_causal_viol = int((entered["entry_date"] <= entered["event_date"]).sum())
    check4_ok = n_causal_viol == 0
    log(f"[check4] 交易因果:entry_date <= event_date 违反 {n_causal_viol} 笔"
        f" -> {'PASS' if check4_ok else 'FAIL'}")

    # ---------------- 自检 5:随机 3 笔完整生命周期 ----------------
    pool_closed = entered[entered["status"] == "closed"]
    pick5 = pool_closed.iloc[rng.choice(len(pool_closed), size=3, replace=False)]
    samples5 = []
    for r in pick5.itertuples(index=False):
        samples5.append(dict(
            variant=r.variant, ts_code=r.ts_code,
            event_date=str(r.event_date.date()),
            entry_date=str(r.entry_date.date()), entry_raw=float(r.entry_raw),
            entry_exec=float(r.entry_exec), shares=int(r.shares),
            exit_date=str(r.exit_date.date()), exit_raw=float(r.exit_raw),
            exit_exec=float(r.exit_exec), gross_ret=float(r.gross_ret),
            buy_comm=float(r.buy_comm), sell_comm=float(r.sell_comm),
            stamp=float(r.stamp), net_pnl=float(r.net_pnl),
            net_ret=float(r.net_ret), held_days=int(r.held_days),
            deferred_days=int(r.deferred_days)))
        log(f"[check5-sample] {r.variant} {r.ts_code} ev={r.event_date.date()} "
            f"入{r.entry_date.date()}@{r.entry_raw:.3f} 出{r.exit_date.date()}@{r.exit_raw:.3f} "
            f"净{r.net_ret:+.4f} 持有{r.held_days} 顺延{r.deferred_days}")

    # ---------------- 宣判(验证段,两变体同过四线才算过) ----------------
    verdicts = {}
    for cand in CANDIDATES:
        per_variant = {}
        for name in ["v1", "v2"]:
            row = summary[(summary["variant"] == name) & (summary["candidate"] == cand)
                          & (summary["segment"] == "validation")].iloc[0]
            all_row = summary[(summary["variant"] == name)
                              & (summary["candidate"] == "ALL")
                              & (summary["segment"] == "validation")].iloc[0]
            c_n = row["n_selected"] >= PASS_MIN_N
            c_pos = np.isfinite(row["net_mean"]) and row["net_mean"] > 0
            c_t = np.isfinite(row["cluster_t"]) and row["cluster_t"] >= PASS_T
            excess = (row["net_mean"] - all_row["net_mean"]) \
                if np.isfinite(row["net_mean"]) and np.isfinite(all_row["net_mean"]) \
                else np.nan
            c_ex = np.isfinite(excess) and excess >= PASS_EXCESS
            per_variant[name] = dict(
                n_selected=int(row["n_selected"]),
                net_mean=float(row["net_mean"]) if np.isfinite(row["net_mean"]) else None,
                cluster_t=float(row["cluster_t"]) if np.isfinite(row["cluster_t"]) else None,
                all_net_mean=float(all_row["net_mean"])
                if np.isfinite(all_row["net_mean"]) else None,
                excess_vs_all=float(excess) if np.isfinite(excess) else None,
                crit_power=bool(c_n), crit_positive=bool(c_pos),
                crit_t=bool(c_t), crit_excess=bool(c_ex),
                passed=bool(c_n and c_pos and c_t and c_ex))
        passed_all = all(v["passed"] for v in per_variant.values())
        verdicts[cand] = dict(per_variant=per_variant, passed=bool(passed_all),
                              verdict="过线(两变体验证段四线同满)" if passed_all
                              else "不过线")
        pv = " ".join(f"{name}:{'过' if v['passed'] else '否'}"
                      f"(n={v['n_selected']},净={v['net_mean']},t={v['cluster_t']},"
                      f"超ALL={v['excess_vs_all']})"
                      for name, v in per_variant.items())
        log(f"[verdict] {cand}: {pv} -> {verdicts[cand]['verdict']}")
    n_pass = sum(1 for v in verdicts.values() if v["passed"])
    overall = ("当天可识别,A 活。过线候选:" + ",".join(c for c in CANDIDATES
                                                      if verdicts[c]["passed"])
               if n_pass else "当天不可识别,A 封档(6 候选验证段全部不过线)")
    log(f"[verdict] 总评:{overall}")

    # ---------------- 落盘:features / trades ----------------
    f_out = pd.concat([feats_df[name].assign(variant=name) for name in ["v1", "v2"]],
                      ignore_index=True)
    f_out = f_out[["variant", "ts_code", "event_date", "anchor_date", "f1", "f2", "f3",
                   "f4", "f5", "f6", "group", "label_near", "segment", "p_R0",
                   "rej_R1", "rej_R2", "rej_R3"]]
    f_path = os.path.join(OUT_DIR, "features.parquet")
    f_out.to_parquet(f_path, index=False)
    log(f"[dump] features.parquet {len(f_out)} 行 -> {f_path}")

    t_out = pd.concat(
        [feats_df[name][["ts_code", "event_date", "segment"]]
         .merge(sims[name], on=["ts_code", "event_date"], how="left",
                validate="one_to_one")
         .assign(variant=name, **{f"sel_{c}": sel[name][c].to_numpy()
                                  for c in GRID_ROWS_ORDER})
         for name in ["v1", "v2"]], ignore_index=True)
    t_path = os.path.join(OUT_DIR, "trades_recognizer.parquet")
    t_out.to_parquet(t_path, index=False)
    log(f"[dump] trades_recognizer.parquet {len(t_out)} 行 -> {t_path}")

    # ---------------- verdict.json ----------------
    checks = dict(
        check1_feature_causality=dict(ok=bool(check1_ok), seed=SAMPLE_SEED,
                                      samples=c1_rows,
                                      note="特征/规则函数只接收截断到事件日的数组"),
        check2_sim_replay=dict(ok=bool(check2_ok), detail=check2),
        check3_counts=dict(ok=bool(check3_ok), base_counts=c3_base,
                           ref_base_v1=REF_BASE_V1, base_diff_v1=int(base_diff_v1),
                           exclusions=exclusions,
                           sel_le_universe=bool(c3_sel_ok), all_eq_universe=bool(c3_all_ok)),
        check4_trade_causality=dict(ok=bool(check4_ok), n_violations=n_causal_viol),
        check5_samples=dict(ok=True, seed=SAMPLE_SEED, samples=samples5),
    )
    all_ok = all(v["ok"] for v in checks.values())
    vout = dict(
        experiment="阶段底信号当天识别赛(A 方案本体)",
        prereadme="README.md 先于任何跑数落盘",
        disclosure=[
            "R3 为价格行为代理(跌停收盘/跳空低开),非真新闻利空数据(token 过期不可得)",
            "特征用全历史日线(引擎序列 60 日前向缓冲不足以供 120 根窗口,README 披露 2)",
            "near/far 标签为事后信息,仅训练段拟合与对照披露,不进特征不进验证宣判",
        ],
        fit_info=fit_info,
        candidates=verdicts,
        pass_criteria="验证段两变体同满:n_selected>=30 且 净笔均>0 且 cluster_t>=2 且 "
                      "净笔均>=同段ALL+10pp",
        n_candidates_passed=n_pass,
        overall=overall,
        checks=checks,
        checks_all_pass=bool(all_ok),
        duration_sec=round(time.time() - t_all, 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    vpath = os.path.join(OUT_DIR, "verdict.json")
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(vout, f, ensure_ascii=False, indent=2, default=str)
    log(f"[dump] verdict.json -> {vpath}")

    # ---------------- report.md ----------------
    write_report(summary, verdicts, fit_info, checks, all_ok, overall, exclusions,
                 c3_base)
    log(f"[dump] report.md -> {os.path.join(OUT_DIR, 'report.md')}")
    log(f"STAGE RECOGNIZER DONE ({time.time() - t_all:.0f}s) "
        f"自检总评={'ALL PASS' if all_ok else 'HAS FAIL'}")

    pd.set_option("display.width", 300)
    print("\n===== SUMMARY(配置为行,28 格全出数) =====")
    print(summary.to_string(index=False))
    print("\n===== 验证段宣判 =====")
    for cand in CANDIDATES:
        v = verdicts[cand]
        print(f"{cand}: {v['verdict']} | "
              + " ".join(f"{n2} n={pv['n_selected']} 净={pv['net_mean']} "
                         f"t={pv['cluster_t']} 超ALL={pv['excess_vs_all']}"
                         for n2, pv in v["per_variant"].items()))
    print(overall)


def write_report(summary, verdicts, fit_info, checks, all_ok, overall, exclusions,
                 c3_base):
    L = []
    L.append("# 阶段底信号当天识别赛报告(预登记一发,2026-09-04)")
    L.append("")
    L.append("> **披露 1**:R3 是价格行为代理(跌停收盘/跳空低开),不是真新闻利空数据(token 过期不可得)。")
    L.append("> **披露 2**:特征 f1~f4 与规则 R1/R2/R3 用全历史日线(引擎序列 60 自然日前向缓冲不足以供 120 根窗口);宇宙与模拟用引擎序列。")
    L.append("> **披露 3**:near/far 标签为事后信息,仅训练段拟合(R0 标签、R5 θ)与对照披露;验证段宣判不含任何标签信息。")
    L.append("")
    L.append(f"**总评:{overall}**")
    L.append("")
    L.append("## 1. 验证段宣判(预登记判活线:两变体同满 n>=30 / 净笔均>0 / cluster_t>=2 / 超ALL+10pp)")
    L.append("")
    L.append("| 候选 | v1: n / 净笔均 / cluster_t / 超ALL | v1 四线 | v2: n / 净笔均 / cluster_t / 超ALL | v2 四线 | 宣判 |")
    L.append("|---|---|---|---|---|")
    for cand, v in verdicts.items():
        cells = []
        for name in ["v1", "v2"]:
            pv = v["per_variant"][name]
            fmt = lambda x, p="+.4f": (format(x, p) if x is not None else "NaN")  # noqa: E731
            cells.append(f"{pv['n_selected']} / {fmt(pv['net_mean'])} / "
                         f"{fmt(pv['cluster_t'], '.3f')} / {fmt(pv['excess_vs_all'])}")
            crit = "".join("1" if k else "0" for k in
                           (pv["crit_power"], pv["crit_positive"], pv["crit_t"],
                            pv["crit_excess"]))
            cells.append(crit)
        L.append(f"| {cand} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {v['verdict']} |")
    L.append("")
    L.append("四线编码顺序:功效n>=30 / 净笔均>0 / cluster_t>=2 / 超ALL+10pp。")
    L.append("")
    L.append("## 2. 训练段拟合产物(仅训练段,冻结)")
    L.append("")
    for name, fi in fit_info.items():
        L.append(f"- {name}: 训练段 {fi['n_train']} 事件(near {fi['n_train_near']});"
                 f"R5 θ = {fi['r5_theta']:+.6f};R0 系数 = "
                 + json.dumps(fi["r0_coef"], ensure_ascii=False)
                 + f",截距 {fi['r0_intercept']:+.6f}")
    L.append("")
    L.append("## 3. 全格子表(配置为行,28 格全出数)")
    L.append("")
    L.append("| 变体 | 候选 | 段 | n_universe | n_selected | n_closed | drop涨停 | drop无报价 | drop现金 | n_incomplete | 净笔均 | 净中位 | 胜率 | cluster_t | near占比(仅train) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary.itertuples(index=False):
        fmt = lambda x, p="+.4f": (format(x, p) if np.isfinite(x) else "NaN")  # noqa: E731
        L.append(f"| {r.variant} | {r.candidate} | {r.segment} | {r.n_universe} | "
                 f"{r.n_selected} | {r.n_closed} | {r.n_dropped_limitup} | "
                 f"{r.n_dropped_no_quote} | {r.n_dropped_cash} | {r.n_incomplete} | "
                 f"{fmt(r.net_mean)} | {fmt(r.net_median)} | {fmt(r.win_rate, '.3f')} | "
                 f"{fmt(r.cluster_t, '.3f')} | {fmt(r.near_share_selected, '.3f')} |")
    L.append("")
    L.append("## 4. 自检结果")
    L.append("")
    c1 = checks["check1_feature_causality"]
    L.append(f"1. 特征因果断言(种子 {c1['seed']},5 事件手工切片重算 f1~f4,容差 1e-12;"
             "特征/规则函数只接收截断到事件日的数组):"
             f"{'PASS' if c1['ok'] else 'FAIL'}。")
    for s in c1["samples"]:
        L.append(f"   - {s['variant']} {s['ts_code']} {s['event_date']}: "
                 f"max|diff|={s['max_abs_diff']:.2e} {'PASS' if s['ok'] else 'FAIL'}")
    c2 = checks["check2_sim_replay"]["detail"]
    for name in ["v1", "v2"]:
        d = c2[name]
        L.append(f"2. 模拟对拍 {name}:near/far {d['n_pairs']} 笔,"
                 f"逐笔状态一致={d['pair_status_ok']},净收益最大绝对差="
                 f"{d['max_abs_net_ret_diff']:.2e}(<=1e-9),状态分布 {d['status_mine']}"
                 f" -> {'PASS' if d['ok'] else 'FAIL'}。")
    c3 = checks["check3_counts"]
    L.append(f"3. 计数对齐:基准集 v1={c3['base_counts']['v1']}(参照 2449,差 "
             f"{c3['base_diff_v1']})v2={c3['base_counts']['v2']};"
             f"n_selected<=n_universe 全行成立={c3['sel_le_universe']};"
             f"ALL 行 n_selected==n_universe={c3['all_eq_universe']}"
             f" -> {'PASS' if c3['ok'] else 'FAIL'}。")
    L.append("   剔除逐笔(基准集 -> 宇宙):")
    for name in ["v1", "v2"]:
        for e in exclusions[name]:
            L.append(f"   - {name} {e['ts_code']} {e['event_date']}: {e['reason']}"
                     f"(hist_days={e['hist_days']})")
    c4 = checks["check4_trade_causality"]
    L.append(f"4. 交易因果:entry_date > event_date 违反 {c4['n_violations']} 笔"
             f" -> {'PASS' if c4['ok'] else 'FAIL'}。")
    c5 = checks["check5_samples"]
    L.append(f"5. 随机 3 笔成交完整生命周期(种子 {c5['seed']},供人工抽查):")
    L.append("")
    L.append("| 变体 | 代码 | 事件日 | 入场日 | 入场价(原始/执行) | 股数 | 出场日 | 出场价(原始/执行) | 毛收益 | 买佣 | 卖佣 | 印花税 | 净盈亏 | 净收益 | 持有日 | 顺延日 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in c5["samples"]:
        L.append(f"| {s['variant']} | {s['ts_code']} | {s['event_date']} | {s['entry_date']} | "
                 f"{s['entry_raw']:.3f}/{s['entry_exec']:.3f} | {s['shares']} | {s['exit_date']} | "
                 f"{s['exit_raw']:.3f}/{s['exit_exec']:.3f} | {s['gross_ret']:+.4f} | "
                 f"{s['buy_comm']:.2f} | {s['sell_comm']:.2f} | {s['stamp']:.2f} | "
                 f"{s['net_pnl']:+.2f} | {s['net_ret']:+.4f} | {s['held_days']} | "
                 f"{s['deferred_days']} |")
    L.append("")
    L.append(f"**自检总评:{'ALL PASS' if all_ok else 'HAS FAIL'}**")
    L.append("")
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
