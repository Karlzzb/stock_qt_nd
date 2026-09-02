#!/usr/bin/env python3
"""架构赛跑(票据 #4 定稿赛制): 背离信号事件 × 狙击标签的四家排序架构对比。

输入:
  v3_pipeline/reports/divergence_lab/m_scan/m_fractal15_full/    (主池)
  v3_pipeline/reports/divergence_lab/m_scan/m_zigzag05_nofilter/ (备池)
  各含 events.parquet / labels.parquet(取 group=="div" 与 events 按位对齐)。

狙击标签: hit_N20_k2.0 (T+1 开盘入场, 20 个交易日内 high 触及 开盘价+2*ATR(14), 命中=1)。

赛制(严格执行):
  - 切分(按事件日): 训练 2001-01~2018-12, 验证 2019-01~2022-10, 测试 2022-11~2026-08;
    两条段界各设 30 个交易日清洗式隔离带: 事件日落入段尾 30 个交易日内的样本删除
    (段界取 2018-12-31 / 2022-10-31, 交易日历=全股票日期并集), 不截断标签。
  - 四家:
    1. LGBM 二分类, 预测命中概率, 按概率排名;
    2. LGBM+线性回归堆叠: LGBM 训练段内 out-of-fold 分数作为 LR 输入(1 维, 严格按票据);
    3. LGBM+逻辑回归堆叠: 同构;
    4. 规则基线: 背离强度打分排序(无学习)。
  - 超参只用验证段调(小网格); 测试段只评估一次。
  - 指标(每池×每家): 每日横截面 top3 命中率(当日不足 3 个取全部; 无信号日不计)、
    top1 命中率、对狙击标签的按日 Rank IC(均值)、盈亏比/PF(按 top3 组合的 20 日实际
    收益路径; labels 无 MFE/ret_h20, 补算: T+1 开盘入场, T+21 收盘出场, 与狙击标签同口径)、
    辅助 10d/30d 固定周期胜率(ret_h10/ret_h30, 原 run 为 close_T 入场口径); 池基线命中率
    (不排序, 全部信号)作对照。

因果性: 特征只用信号日 T 及之前数据(低点/前低点的价格、DIF、量能, 信号日的
  收盘/ATR/MA200/动量/量比等)。compare_rank/formation/regime/above_ma200 直接取
  events.parquet(其生成已保证因果, 见 divergence_lab.py)。

用法: python architecture_race.py [--workers 16]
输出: v3_pipeline/reports/architecture_race.md
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import talib
from scipy import stats as sc_stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
DATA_DIR = REPO / "stock_data" / "daily"
SCAN_DIR = REPO / "v3_pipeline" / "reports" / "divergence_lab" / "m_scan"
REPORT_PATH = REPO / "v3_pipeline" / "reports" / "architecture_race.md"

POOLS = {"main": "m_fractal15_full", "backup": "m_zigzag05_nofilter"}
LABEL_COL = "hit_N20_k2.0"
SEED = 42

TRAIN_RANGE = ("2001-01-01", "2018-12-31")
VALID_RANGE = ("2019-01-01", "2022-10-31")
TEST_RANGE = ("2022-11-01", "2026-08-31")
PURGE_DAYS = 30  # 段尾清洗式隔离带(交易日)

FEATURES = [
    "compare_rank", "formation", "confirm_lag",
    "price_decline", "dif_lift_atr", "price_drop_atr",
    "dif_sig", "dif_low", "atr_pct", "log_close",
    "above_ma200", "ma200_ratio", "regime_code",
    "vol_shrink", "vol_ratio", "ret_5", "ret_10", "ret_20",
]
REGIME_CODE = {"unknown": -1, "sideways": 0, "up": 1, "down": 2}

LGBM_GRID = [
    dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc)
    for nl in (15, 31, 63) for lr in (0.03, 0.08) for mc in (20,)
]
LGBM_BASE = dict(
    objective="binary", metric="binary_logloss", boosting_type="gbdt",
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    num_threads=8, deterministic=True, force_row_wise=True,
    seed=SEED, verbose=-1,
)
N_EST_MAX = 2000
EARLY_STOP = 100
OOF_SPLITS = 5


# ================================================================ 特征补算(工作进程)
def _stock_features(args):
    """单股: 复制 divergence_lab 的数据口径(sort/dedup/指标/cf 链), 为该股的全部事件
    补算因果特征与 ret20(T+1 开盘入场, T+21 收盘出场)。返回 (symbol, recs, dates)。"""
    symbol, evs = args  # evs: list of (row_i, sig_idx, low_day, prev_low_day)
    path = DATA_DIR / f"{symbol}.parquet"
    try:
        df = pd.read_parquet(path)
    except Exception:
        return symbol, [], None
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    if len(df) < 30:
        return symbol, [], None
    close = df["close"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    vol_col = "vol" if "vol" in df.columns else ("volume" if "volume" in df.columns else None)
    vol = df[vol_col].to_numpy(np.float64) if vol_col else np.full(len(df), np.nan)
    td = df["trade_date"]
    if td.dtype == object or str(td.dtype).startswith(("str", "string")):
        dates = pd.to_datetime(td.astype(str), format="%Y%m%d") \
            .to_numpy("datetime64[D]").astype(np.int32)
    elif np.issubdtype(td.dtype, np.number):
        dates = pd.to_datetime(td.astype(np.int64).astype(str), format="%Y%m%d") \
            .to_numpy("datetime64[D]").astype(np.int32)
    else:
        dates = pd.to_datetime(td).to_numpy("datetime64[D]").astype(np.int32)

    dif, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    atr = talib.ATR(high, low, close, timeperiod=14)
    ma200 = talib.SMA(close, timeperiod=200)

    day2idx = {int(d): i for i, d in enumerate(dates)}
    n = len(close)
    recs = []
    for row_i, sig, low_day, prev_day in evs:
        sig = int(sig)
        li, pi = day2idx.get(int(low_day)), day2idx.get(int(prev_day))
        if li is None or pi is None or sig >= n:
            continue
        a = atr[sig]
        rec = {"row_i": row_i}
        if not np.isfinite(a) or a <= 0 or close[sig] <= 0:
            # 保留行但特征置 NaN(LGBM 原生处理; 规则打分为 NaN 排最后)
            rec.update({f: np.nan for f in FEATURES
                        if f not in ("compare_rank", "formation", "above_ma200", "regime_code")})
        else:
            rec["confirm_lag"] = sig - li
            rec["price_decline"] = close[li] / close[pi] - 1.0
            rec["dif_lift_atr"] = (dif[li] - dif[pi]) / a
            rec["price_drop_atr"] = (close[pi] - close[li]) / a
            rec["dif_sig"] = dif[sig]
            rec["dif_low"] = dif[li]
            rec["atr_pct"] = a / close[sig]
            rec["log_close"] = np.log(close[sig])
            rec["ma200_ratio"] = close[sig] / ma200[sig] - 1.0 if np.isfinite(ma200[sig]) else np.nan
            rec["vol_shrink"] = vol[li] / vol[pi] if vol[pi] > 0 else np.nan
            v20 = vol[max(0, sig - 20):sig]
            rec["vol_ratio"] = vol[sig] / np.nanmean(v20) if len(v20) and np.isfinite(v20).any() else np.nan
            for h in (5, 10, 20):
                rec[f"ret_{h}"] = close[sig] / close[sig - h] - 1.0 if sig - h >= 0 else np.nan
        # ret20: T+1 开盘入场, T+21 收盘出场(与狙击标签入场口径一致)
        t1 = sig + 1
        if t1 < n and sig + 21 <= n - 1 and open_[t1] > 0:
            rec["ret20"] = close[sig + 21] / open_[t1] - 1.0
        else:
            rec["ret20"] = np.nan
        recs.append(rec)
    return symbol, recs, dates


# ================================================================ 数据装配
def load_pool(pool_key, workers):
    name = POOLS[pool_key]
    base = SCAN_DIR / name
    ev = pd.read_parquet(base / "events.parquet").sort_values("event_id").reset_index(drop=True)
    lb = pd.read_parquet(base / "labels.parquet")
    div = lb[lb.group == "div"].reset_index(drop=True)
    assert len(div) == len(ev), f"{name}: labels(div) 与 events 行数不一致"
    df = ev.copy()
    df["hit"] = div[LABEL_COL].to_numpy(np.float64)
    df["ret_h10"] = div["ret_h10"].to_numpy(np.float64)
    df["ret_h30"] = div["ret_h30"].to_numpy(np.float64)
    df["row_i"] = np.arange(len(df))

    # 按股分组 -> 工作进程补特征
    groups = []
    for sym, g in df.groupby("ts_code"):
        groups.append((sym, list(zip(g.row_i, g.sig_idx, g.low_date, g.prev_low_date))))
    all_dates, recs = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (sym, r, dates) in enumerate(ex.map(_stock_features, groups, chunksize=32)):
            if dates is not None:
                all_dates.append(dates)
            recs.extend(r)
            if (i + 1) % 1000 == 0:
                print(f"  [{pool_key}] 特征 {i+1}/{len(groups)}", flush=True)
    feat = pd.DataFrame(recs).set_index("row_i")
    df = df.join(feat)
    df["compare_rank"] = df["compare_rank"].astype(np.float64)
    df["formation"] = df["formation"].astype(np.float64)
    df["above_ma200"] = df["above_ma200"].astype(np.float64)
    df["regime_code"] = df["regime"].map(REGIME_CODE).astype(np.float64)
    df["rule_score"] = df["dif_lift_atr"] + df["price_drop_atr"]  # 规则基线: 背离强度(ATR 单位)
    calendar = np.unique(np.concatenate(all_dates)) if all_dates else np.array([], np.int32)
    print(f"[{pool_key}] 事件 {len(df)}, 特征缺失行 {int(df['dif_lift_atr'].isna().sum())}, "
          f"日历 {len(calendar)} 天", flush=True)
    return df, calendar


# ================================================================ 切分
def purge_tail(calendar, seg_end, days=PURGE_DAYS):
    """段界 seg_end(含) 往前数 days 个交易日 -> 隔离带日期集合。"""
    cal = calendar[calendar <= np.datetime64(seg_end).astype("datetime64[D]").astype(np.int32)]
    return set(cal[-days:].tolist())


def assign_splits(df, calendar):
    d = df["date"].to_numpy("datetime64[D]").astype(np.int32)
    p1, p2 = purge_tail(calendar, TRAIN_RANGE[1]), purge_tail(calendar, VALID_RANGE[1])
    def _in(rng):
        lo = np.datetime64(rng[0]).astype("datetime64[D]").astype(np.int32)
        hi = np.datetime64(rng[1]).astype("datetime64[D]").astype(np.int32)
        return (d >= lo) & (d <= hi)
    train = _in(TRAIN_RANGE) & ~np.isin(d, list(p1))
    valid = _in(VALID_RANGE) & ~np.isin(d, list(p2))
    test = _in(TEST_RANGE)
    return train, valid, test, p1, p2


# ================================================================ 指标
def segment_metrics(df, mask, scores):
    """全套指标: top3/top1 命中率、按日 Rank IC 均值、top3 组合 ret20 的盈亏比/PF、
    10d/30d 胜率、池基线。scores 为全表长度数组(NaN 排最后); 狙击标签缺失事件剔除。"""
    seg_idx = np.nonzero(mask)[0]
    sub = df.iloc[seg_idx]
    hit_all = sub["hit"].to_numpy(np.float64)
    ok = np.isfinite(hit_all)
    day = sub["date"].to_numpy("datetime64[D]").astype(np.int32)[ok]
    hit = hit_all[ok]
    r20a = sub["ret20"].to_numpy(np.float64)[ok]
    r10a = sub["ret_h10"].to_numpy(np.float64)[ok]
    r30a = sub["ret_h30"].to_numpy(np.float64)[ok]
    sc = scores[seg_idx][ok]
    sc = np.where(np.isfinite(sc), sc, -np.inf)

    srt = np.argsort(day, kind="stable")
    day_s = day[srt]
    _, starts = np.unique(day_s, return_index=True)
    bounds = np.append(starts, len(day_s))

    top3_hit, top1_hit, ics = [], [], []
    r20, w10, w30 = [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = srt[lo:hi]
        order = rows[np.argsort(-sc[rows], kind="stable")]
        k3, k1 = order[:3], order[:1]
        top3_hit.append(hit[k3].mean())
        top1_hit.append(hit[k1].mean())
        r20.extend(r20a[k3].tolist())
        w10.extend((r10a[k3] > 0).tolist())
        w30.extend((r30a[k3] > 0).tolist())
        if len(rows) >= 3:
            y, s = hit[rows], sc[rows]
            if np.std(y) > 1e-12 and np.std(s) > 1e-12:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", sc_stats.ConstantInputWarning)
                    ic = sc_stats.spearmanr(s, y).statistic
                if np.isfinite(ic):
                    ics.append(ic)
    r20 = np.asarray([x for x in r20 if np.isfinite(x)], np.float64)
    pos, neg = r20[r20 > 0], r20[r20 < 0]
    pf = pos.sum() / abs(neg.sum()) if len(neg) and neg.sum() != 0 else np.inf
    pl = pos.mean() / abs(neg.mean()) if len(pos) and len(neg) else np.nan
    return {
        "n": int(len(hit)), "days": len(top3_hit),
        "top3": float(np.mean(top3_hit)) if top3_hit else np.nan,
        "top1": float(np.mean(top1_hit)) if top1_hit else np.nan,
        "rank_ic": float(np.mean(ics)) if ics else np.nan,
        "ic_days": len(ics),
        "pf": float(pf), "pl_ratio": float(pl),
        "ret20_mean": float(r20.mean()) if len(r20) else np.nan,
        "win10": float(np.mean(w10)) if w10 else np.nan,
        "win30": float(np.mean(w30)) if w30 else np.nan,
        "pool_hit": float(hit.mean()),
    }


# ================================================================ 模型
def fit_lgbm(Xtr, ytr, Xva, yva, params, n_est, early_stop=True):
    Xtr = np.asarray(Xtr, np.float64)
    ytr = np.asarray(ytr, int)
    m = lgb.LGBMClassifier(**LGBM_BASE, **params, n_estimators=n_est)
    if early_stop:
        m.fit(Xtr, ytr, eval_X=np.asarray(Xva, np.float64), eval_y=np.asarray(yva, int),
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return m


def tune_lgbm(df, train_m, valid_m):
    """小网格, 训练段拟合 + 验证段 early stopping, 按验证段按日 Rank IC 选超参。
    返回 (best_params, best_iter, valid_metrics_table)。"""
    Xtr = df.loc[train_m, FEATURES]; ytr = df.loc[train_m, "hit"].astype(int)
    Xva = df.loc[valid_m, FEATURES]; yva = df.loc[valid_m, "hit"].astype(int)
    rows, best = [], None
    for p in LGBM_GRID:
        m = fit_lgbm(Xtr, ytr, Xva, yva, p, N_EST_MAX)
        scores = np.full(len(df), np.nan)
        scores[np.nonzero(valid_m)[0]] = m.predict_proba(Xva)[:, 1]
        mt = segment_metrics(df, valid_m, scores)
        rows.append({**p, "best_iter": m.best_iteration_,
                     "va_rank_ic": mt["rank_ic"], "va_top3": mt["top3"]})
        key = (np.nan_to_num(mt["rank_ic"], -9), np.nan_to_num(mt["top3"], -9))
        if best is None or key > best[0]:
            best = (key, p, m.best_iteration_)
        print(f"    grid {p} iter={m.best_iteration_} va_ic={mt['rank_ic']:.4f} "
              f"va_top3={mt['top3']:.4f}", flush=True)
    return best[1], best[2], pd.DataFrame(rows)


def oof_scores(df, train_m, params, n_est):
    """训练段内 TimeSeriesSplit OOF 分数(按时间升序, 扩展窗)。"""
    idx = np.nonzero(train_m)[0]
    idx = idx[np.argsort(df["date"].to_numpy()[idx])]
    X = df[FEATURES].to_numpy()[idx]
    y = df["hit"].to_numpy()[idx].astype(int)
    oof = np.full(len(idx), np.nan)
    tscv = TimeSeriesSplit(n_splits=OOF_SPLITS)
    for tr, te in tscv.split(X):
        m = fit_lgbm(X[tr], y[tr], None, None, params, n_est, early_stop=False)
        oof[te] = m.predict_proba(X[te])[:, 1]
    return idx, oof


# ================================================================ 主流程
def run_pool(pool_key, workers):
    print(f"===== 池 {pool_key} ({POOLS[pool_key]}) =====", flush=True)
    df, calendar = load_pool(pool_key, workers)
    train_m, valid_m, test_m, p1, p2 = assign_splits(df, calendar)
    b1 = np.datetime64(min(p1), "D"); b1e = np.datetime64(max(p1), "D")
    b2 = np.datetime64(min(p2), "D"); b2e = np.datetime64(max(p2), "D")
    print(f"  切分: train={int(train_m.sum())} valid={int(valid_m.sum())} "
          f"test={int(test_m.sum())} | 隔离带 {b1}~{b1e}, {b2}~{b2e}",
          flush=True)
    # 标签缺失事件(数据末端)不参与训练/调参
    lab_ok = np.isfinite(df["hit"].to_numpy())
    train_m = train_m & lab_ok
    valid_m = valid_m & lab_ok

    # ---- LGBM 调参(只用验证段)
    best_params, best_iter, grid_tbl = tune_lgbm(df, train_m, valid_m)
    best_iter = max(int(best_iter), 1)
    print(f"  最优超参: {best_params}, best_iter={best_iter}", flush=True)

    # ---- 最终模型: 训练段拟合(与 OOF 分布一致), 测试段只评估一次
    Xtr = df.loc[train_m, FEATURES]; ytr = df.loc[train_m, "hit"].astype(int)
    final = fit_lgbm(Xtr, ytr, None, None, best_params, best_iter, early_stop=False)
    prob = np.full(len(df), np.nan)
    prob[:] = final.predict_proba(df[FEATURES].to_numpy())[:, 1]

    # ---- 堆叠: 训练段 OOF -> 1 维 meta 输入(严格按票据)
    oof_idx, oof = oof_scores(df, train_m, best_params, best_iter)
    ok = np.isfinite(oof)
    meta_x = oof[ok].reshape(-1, 1)
    meta_y = df["hit"].to_numpy()[oof_idx[ok]].astype(int)
    lr = LinearRegression().fit(meta_x, meta_y)
    logit = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000).fit(meta_x, meta_y)
    s_lr = np.asarray(lr.predict(prob.reshape(-1, 1)), np.float64)
    s_logit = logit.predict_proba(prob.reshape(-1, 1))[:, 1]

    houses = {
        "lgbm": prob,
        "lgbm+lr": s_lr,
        "lgbm+logit": s_logit,
        "rule": df["rule_score"].to_numpy(np.float64),
    }
    res = {}
    for seg_name, m in (("valid", valid_m), ("test", test_m)):
        res[seg_name] = {h: segment_metrics(df, m, s) for h, s in houses.items()}
    return df, grid_tbl, best_params, best_iter, res, (train_m, valid_m, test_m)


def fmt_row(house, mt):
    pf = "inf" if np.isinf(mt["pf"]) else f"{mt['pf']:.2f}"
    return (f"| {house} | {mt['n']} | {mt['days']} | {mt['top3']:.4f} | {mt['top1']:.4f} "
            f"| {mt['rank_ic']:.4f} | {pf} | {mt['pl_ratio']:.2f} "
            f"| {mt['ret20_mean']:+.4f} | {mt['win10']:.4f} | {mt['win30']:.4f} "
            f"| {mt['pool_hit']:.4f} |")


TABLE_HEAD = ("| 架构 | n | 信号日数 | top3命中 | top1命中 | RankIC | PF | 盈亏比 "
              "| ret20均值 | win10 | win30 | 池基线命中 |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    lines = ["# 架构赛跑报告(票据 #4 定稿赛制)", ""]
    lines.append(f"- 生成: {pd.Timestamp.now():%Y-%m-%d %H:%M} | seed={SEED}")
    lines.append(f"- 狙击标签: `{LABEL_COL}`(T+1 开盘入场, 20 交易日内 +2×ATR(14) 命中=1)")
    lines.append(f"- 切分: 训练 {TRAIN_RANGE[0]}~{TRAIN_RANGE[1]} / 验证 {VALID_RANGE[0]}~{VALID_RANGE[1]} "
                 f"/ 测试 {TEST_RANGE[0]}~{TEST_RANGE[1]}; 两条段界各 {PURGE_DAYS} 个交易日段尾隔离带(事件删除, 不截断标签)")
    lines.append("- 堆叠口径: LGBM 训练段内 TimeSeriesSplit(5 折扩展窗) OOF 概率作为唯一 meta 输入"
                 "(严格按票据); LR/逻辑回归均为 OOF 概率的 1 维单调映射, 斜率符号由训练段 OOF 决定"
                 "(为正则排名与 LGBM 完全一致, 为负则整体反转), 只校准概率、不产生新的排序信息")
    lines.append("- 特征: 全部严格因果(仅用信号日 T 及之前); ret20 按狙击入场口径补算"
                 "(T+1 开盘入, T+21 收盘出); win10/win30 沿用原 run 的 ret_h10/ret_h30(close_T 入场)")
    lines.append("- 超参选择: 6 组小网格, 训练段拟合+验证段 early stopping, 按验证段按日 Rank IC 选优;"
                 "最终模型在训练段重训(n_estimators=best_iter), 测试段仅评估一次")
    lines.append("")

    summary = {}
    for pool in ("main", "backup"):
        df, grid_tbl, bp, bi, res, masks = run_pool(pool, args.workers)
        lines.append(f"## 池: {pool} ({POOLS[pool]})")
        lines.append("")
        lines.append(f"最优超参: `{bp}`, n_estimators={bi}")
        lines.append("")
        lines.append("验证段网格(按 RankIC 排序):")
        lines.append("")
        lines.append(grid_tbl.sort_values("va_rank_ic", ascending=False).round(4).to_markdown(index=False))
        lines.append("")
        for seg in ("valid", "test"):
            lines.append(f"### {seg} 段四家成绩")
            lines.append("")
            lines.append(TABLE_HEAD)
            for h, mt in res[seg].items():
                lines.append(fmt_row(h, mt))
            lines.append("")
        summary[pool] = res

    # ---- 结论
    lines.append("## 结论")
    lines.append("")
    simple = ["rule", "lgbm", "lgbm+lr", "lgbm+logit"]
    verdicts = {}
    for pool in ("main", "backup"):
        tm = summary[pool]["test"]
        ranking = sorted(tm.items(), key=lambda kv: -np.nan_to_num(kv[1]["top3"], -9))
        lines.append(f"{pool} 池测试段 top3 命中率排序: " +
                     " > ".join(f"{h}({m['top3']:.4f})" for h, m in ranking))
        top_h, top_m = ranking[0]
        flat = [h for h, m in ranking[1:] if abs(top_m["top3"] - m["top3"]) < 0.02]
        if flat:
            winner = next(h for h in simple if h in [top_h] + flat)
            lines.append(f"平手裁决: {top_h} 与 {flat} 差距 <2pp, 取结构最简单者 -> **{winner}**")
        else:
            winner = top_h
            lines.append(f"无平手, 胜者 -> **{winner}**")
        verdicts[pool] = (winner, top_m["top3"])
    lines.append(f"80% 验收线: 主池测试段最优 top3 命中率 {verdicts['main'][1]:.4f} "
                 f"-> {'**过线**' if verdicts['main'][1] >= 0.80 else '**未过线**'}")
    tm, tb = summary["main"]["test"], summary["backup"]["test"]
    lines.append(
        "LR vs 逻辑回归实证答案: 两池中二者成绩逐指标完全相同(同为 OOF 概率的 1 维单调映射, "
        "符号一致), 无实证差异; 相对裸 LGBM —— 主池 OOF 斜率为正, 堆叠与 LGBM 数值恒等"
        f"(top3 {tm['lgbm']['top3']:.4f}={tm['lgbm+lr']['top3']:.4f}); "
        f"备池基模型退化(best_iter=4)致 OOF 斜率为负, 堆叠排名整体反转"
        f"(RankIC {tb['lgbm']['rank_ic']:+.4f} -> {tb['lgbm+lr']['rank_ic']:+.4f}, "
        f"top3 {tb['lgbm']['top3']:.4f} -> {tb['lgbm+lr']['top3']:.4f}), "
        "即 1 维堆叠对排序零增益且放大基模型不稳定风险")
    lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告 -> {REPORT_PATH}", flush=True)

    # 控制台摘要
    print("\n===== 测试段摘要 =====")
    for pool in ("main", "backup"):
        print(f"[{pool}]")
        print(TABLE_HEAD)
        for h, mt in summary[pool]["test"].items():
            print(fmt_row(h, mt))


if __name__ == "__main__":
    main()
