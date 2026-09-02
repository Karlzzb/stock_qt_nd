# -*- coding: utf-8 -*-
"""换目标假设验证 (target_switch, issue #7): 反弹幅度的事前预测作为训练目标。

相对 run_race_rerun_v2.py 的唯一改动 = 训练标签; 数据装配/分组/早停/评估协议一字不变:
  标签: 四个候选目标, 全部按 hit 标签同口径重算 (T+1 开盘入场):
    1. mfe20        = max(high[T+1..T+20]) / open[T+1] - 1        (20 交易日窗口最大涨幅)
    2. mfe20_atrn   = mfe20 / ATRN(信号日, 特征矩阵现成列)         (波动中性化)
    3. ret_h10      = close[T+1]/open[T+1] * cf[T+11]/cf[T+1] - 1  (10 日收益, 复权)
    4. ret_h10_atrn = ret_h10 / ATRN
  目标变换: 逐日截面 rank_pct -> 31 桶整数 gain (0..30, lambdarank 默认 label_gain 上限) 喂 lambdarank;
    早停 feval 的日 Rank IC 用连续目标值(秩相关对单调变换不变, 避免桶内并列损失粒度)。
  特征: C4 名单 = ATRN + B档28(已去 RET20_CSR) + 边界带31, 共 60 个;
    注: VOL15/VOL20/AMP20 本就不在 B28/band31 中, "剔除"为空操作(报告注明)。
    每个目标配两个特征变体: 含 ATRN(60) / 剔 ATRN(59, 防 /ATRN 目标与同分母特征的机械化相关)。
  早停/选参: 验证段日 Rank IC(对连续目标), 网格与 v2 相同 6 组, tiebreak = 验证段 top3(hit)。
    已知局限(继承自翻案局): 验证段既早停又评估, 报告明示。
  评估(与翻案局一字不变): 模型分数排序的验证段日加权 top3 狙击命中率(hit_N20_k2.0),
    对照日加权零信息基线(独立重算, 预期 主池 49.97% / 备池 55.73%);
    辅助: 日 Rank IC(对 hit)、逐年表、逐日配对 Wilcoxon。
  多重比较: 主池 8 组合(4 目标 x 2 特征变体) Bonferroni + BH-FDR; 16 组合全家族作敏感性。

纪律: 只用训练段(2001-2018)与验证段(2019-01~2022-10, 含 30 交易日段界隔离带);
  2022-11 之后事件不入任何统计(标签前视窗口可越界, 与 hit 标签自身口径一致)。

输出:
  v3_pipeline/reports/target_switch/results.json        全部数值
  v3_pipeline/reports/target_switch/daily_curves.json   逐日 top3 序列
  v3_pipeline/reports/target_switch/targets_{pool}.parquet  事件级四目标(可复跑核对)
  v3_pipeline/reports/target_switch/progress.log        进度(追加)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(ROOT / "v3_pipeline" / "scripts"))
import run_race_rerun_v2 as rv2  # noqa: E402  复用数据装配/评估/早停代码

OUT_DIR = ROOT / "v3_pipeline" / "reports" / "target_switch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"
DATA_DIR = ROOT / "stock_data" / "daily"

N_BINS = 31  # gain 分桶数: 逐日 rank_pct -> 0..30 整数 gain (lambdarank 默认 label_gain 上限 30)
MFE_W = 20   # mfe 窗口(交易日), 与 hit 标签 N=20 一致
RET_H = 10   # ret_h10  horizon

TARGETS = ["mfe20", "mfe20_atrn", "ret_h10", "ret_h10_atrn"]
FEAT_VARIANTS = {"withATRN": True, "noATRN": False}


def log(msg):
    line = f"{msg} {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 目标重算(T+1 开盘口径)
def _stock_arrays(path):
    """复刻 divergence_lab.load_stock 的数组构造(dropna close/dedup/sort/cf)。"""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    close = df["close"].to_numpy(np.float64)
    if "pct_chg" in df.columns:
        pct = df["pct_chg"].to_numpy(np.float64)
    else:
        pct = np.concatenate([[np.nan], close[1:] / close[:-1] * 100.0 - 100.0])
    pct_safe = np.where(np.isfinite(pct), pct, 0.0)
    cf = np.cumprod(1.0 + pct_safe / 100.0)
    return {
        "dates": pd.to_datetime(df["trade_date"], format="%Y%m%d").to_numpy(),
        "open": df["open"].to_numpy(np.float64),
        "high": df["high"].to_numpy(np.float64),
        "close": close,
        "cf": cf,
    }


def compute_targets(pool, keys):
    """对特征矩阵键集合计算四个候选目标。返回 (ts_code, date, mfe20, ret_h10)。

    口径(与 hit_N20_k2.0 同入场): entry = open[T+1];
    mfe20 = max(high[T+1..T+20])/open[T+1] - 1;
    ret_h10 = close[T+1]/open[T+1] * cf[T+1+10]/cf[T+1] - 1 (与 compute_labels
    entry_open 分支逐式一致, 复权经 cf 比值)。
    """
    cfg = rv2.POOLS[pool]
    ev = pd.read_parquet(cfg["labdir"] / "events.parquet",
                         columns=["ts_code", "date", "sig_idx"])
    ev["date"] = pd.to_datetime(ev["date"])
    m = keys.merge(ev, on=["ts_code", "date"], how="left", validate="1:1")
    assert m["sig_idx"].notna().all(), f"{pool} 有键未匹配到 events"
    m["sig_idx"] = m["sig_idx"].astype(np.int64)

    mfe = np.full(len(m), np.nan)
    ret = np.full(len(m), np.nan)
    n_sig_checked = 0
    for ts, sub in m.groupby("ts_code", sort=False):
        path = DATA_DIR / f"{ts}.parquet"
        st = _stock_arrays(path)
        n = len(st["close"])
        sig = sub["sig_idx"].to_numpy()
        # 对齐校验: sig_idx 指向的日期必须等于事件日期
        d_evt = sub["date"].to_numpy()
        assert (st["dates"][sig] == d_evt).all(), f"{pool}/{ts} sig_idx 日期不符"
        n_sig_checked += len(sig)
        t1 = sig + 1
        ok_base = (t1 <= n - 1) & (st["open"][np.minimum(t1, n - 1)] > 0)
        H = sliding_window_view(st["high"], MFE_W) if n >= MFE_W else None
        pos = sub.index.to_numpy()
        # mfe20: 窗口 high[T+1 .. T+20]
        ok = ok_base & (sig + MFE_W <= n - 1)
        if ok.any() and H is not None:
            i = np.nonzero(ok)[0]
            mfe[pos[i]] = H[t1[i]].max(axis=1) / st["open"][t1[i]] - 1.0
        # ret_h10: T+1 开盘 -> T+1+10 收盘(复权)
        ok = ok_base & (sig + 1 + RET_H <= n - 1)
        if ok.any():
            i = np.nonzero(ok)[0]
            ret[pos[i]] = (st["close"][t1[i]] / st["open"][t1[i]]
                           * (st["cf"][t1[i] + RET_H] / st["cf"][t1[i]]) - 1.0)
    log(f"池={pool} 目标重算完成 事件={len(m)} sig_idx校验={n_sig_checked} "
        f"mfe20缺失={int(np.isnan(mfe).sum())} ret_h10缺失={int(np.isnan(ret).sum())}")
    out = m[["ts_code", "date"]].copy()
    out["mfe20"] = mfe
    out["ret_h10"] = ret
    return out


def attach_targets(df, tgt):
    """合并目标并按 ATRN 派生波动中性化变体; 返回 df(新增 4 列)与缺失统计。"""
    df = df.merge(tgt, on=["ts_code", "date"], how="left", validate="1:1")
    atrn = df["ATRN"].to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["mfe20_atrn"] = np.where(atrn > 0, df["mfe20"] / atrn, np.nan)
        df["ret_h10_atrn"] = np.where(atrn > 0, df["ret_h10"] / atrn, np.nan)
    stats = {t: int(df[t].isna().sum()) for t in TARGETS}
    return df, stats


def sanity_checks(df, pool):
    """口径 sanity: hit=1 的 mfe20 应系统性高于 hit=0;
    新 ret_h10(T+1 开盘) 与 labels.parquet 旧 ret_h10(close_T) 应高相关但不相同。"""
    res = {}
    hit = df["hit"].to_numpy()
    for hval in (0.0, 1.0):
        m = (hit == hval) & df["mfe20"].notna().to_numpy()
        res[f"mfe20_hit{int(hval)}_mean"] = float(df.loc[m, "mfe20"].mean())
    labdir = rv2.POOLS[pool]["labdir"]
    ev = pd.read_parquet(labdir / "events.parquet", columns=["ts_code", "date"])
    lb = pd.read_parquet(labdir / "labels.parquet")
    div = lb.iloc[:len(ev)].reset_index(drop=True)
    assert (div["group"] == "div").all()
    old = pd.DataFrame({"ts_code": ev["ts_code"].values,
                        "date": pd.to_datetime(ev["date"].values),
                        "ret_h10_closeT": div["ret_h10"].values})
    mm = df[["ts_code", "date", "ret_h10"]].merge(old, on=["ts_code", "date"],
                                                  how="left", validate="1:1")
    a = mm["ret_h10"].to_numpy(np.float64)
    b = mm["ret_h10_closeT"].to_numpy(np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    res["ret_h10_new_vs_oldCloseT_corr"] = float(np.corrcoef(a[fin], b[fin])[0, 1])
    res["ret_h10_new_vs_oldCloseT_meandiff"] = float(np.mean(a[fin] - b[fin]))
    log(f"池={pool} sanity: {res}")
    return res


# ================================================================ 单日截面 gain 分桶
def bucket_gains(df, mask, target):
    """逐日截面 rank_pct -> 0..30 整数 gain; 仅对 mask 内且目标非 NaN 的行。
    返回 (gains float64 全表 NaN 初始化, 连续目标值 y_cont)。"""
    gains = np.full(len(df), np.nan)
    y = df[target].to_numpy(np.float64)
    ok = mask & np.isfinite(y)
    sub = pd.DataFrame({"date": df.loc[ok, "date"].to_numpy(), "y": y[ok]})
    rp = sub.groupby("date")["y"].rank(pct=True, method="average").to_numpy()
    g = np.clip(np.floor(rp * N_BINS), 0, N_BINS - 1).astype(np.float64)
    gains[np.nonzero(ok)[0]] = g
    return gains


# ================================================================ 单配置训练(骨架与 v2 同)
def run_target_config(df, train_m, val_m, feats, target, tag):
    y_cont = df[target].to_numpy(np.float64)
    train_ok = train_m & np.isfinite(y_cont)
    val_ok = val_m & np.isfinite(y_cont)

    gtr_all = bucket_gains(df, train_m, target)
    gva_all = bucket_gains(df, val_m, target)

    X = df[feats].to_numpy(np.float64)
    tr_idx, tr_bounds, gtr = rv2.sort_by_day(df, train_ok)
    va_idx, va_bounds, gva = rv2.sort_by_day(df, val_ok)
    Xtr = X[tr_idx]
    ytr = gtr_all[tr_idx].astype(int)
    Xva = X[va_idx]
    yva_gain = gva_all[va_idx].astype(int)
    yva_cont = y_cont[va_idx]

    ic_calc = rv2.DayRankIC(yva_cont, va_bounds)
    feval = rv2.make_feval(ic_calc)

    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr,
                         feature_name=list(feats), free_raw_data=True)
    dval = lgb.Dataset(Xva, label=yva_gain, group=gva, reference=dtrain,
                       free_raw_data=True)

    day_val_full = df["date"].to_numpy()
    hit = df["hit"].to_numpy(np.float64)

    best = None
    for p in rv2.LGBM_GRID:
        params = dict(rv2.LGBM_BASE, **p)
        mdl = lgb.train(params, dtrain, num_boost_round=rv2.N_EST_MAX,
                        valid_sets=[dval], feval=feval,
                        callbacks=[lgb.early_stopping(rv2.EARLY_STOP, verbose=False)])
        bi = max(int(mdl.best_iteration), 1)
        sc = mdl.predict(Xva, num_iteration=bi)
        # 选参 key: (验证段日RankIC(对连续目标), tiebreak=验证段top3(hit)) —— 与 v2 结构一致
        key_ic = float(ic_calc(sc))
        sc_full = np.full(len(df), -np.inf)
        sc_full[va_idx] = sc
        t3, _, _, _ = rv2.topk_metrics(day_val_full[va_idx],
                                       hit[va_idx], sc_full[va_idx])
        key = (key_ic, float(t3.mean()))
        if best is None or key > best[0]:
            best = (key, p, bi)
    best_params, best_iter = best[1], best[2]

    final = lgb.train(dict(rv2.LGBM_BASE, **best_params), dtrain,
                      num_boost_round=best_iter)
    prob = final.predict(X, num_iteration=best_iter)
    mt = rv2.eval_segment(df, val_m, prob)  # 评估: 全验证段 vs hit(与翻案局一字不变)

    gain = final.feature_importance(importance_type="gain")
    imp = dict(zip(feats, [float(g) for g in gain]))
    atrn_gain = imp.get("ATRN", 0.0)
    incr = {f: (g / atrn_gain if atrn_gain > 0 else np.nan) for f, g in imp.items()}
    out = {
        "target": target, "n_features": len(feats),
        "n_train_used": int(train_ok.sum()), "n_val_label_ok": int(val_ok.sum()),
        "best_params": best_params, "best_iter": best_iter,
        "val": {k: v for k, v in mt.items() if not k.startswith("_")},
        "gain_importance": imp, "gain_vs_atrn": incr,
        "_daily": mt["_daily"],
    }
    log(f"配置={tag} val_top3={mt['top3']:.4f} val_rankIC_hit={mt['rank_ic']:.4f} "
        f"iter={best_iter} 完成")
    return out


def main():
    t0 = time.time()
    log("阶段=脚本启动 target_switch(换目标: 反弹幅度事前预测, lambdarank gain=目标逐日秩31桶)")
    b28, band = rv2.feature_lists()
    feats_full = ["ATRN"] + b28 + band
    feats_noatrn = b28 + band
    log(f"阶段=名单载入 含ATRN={len(feats_full)} 剔ATRN={len(feats_noatrn)} "
        f"(VOL15/VOL20/AMP20 本不在名单: "
        f"{[f for f in ('VOL15','VOL20','AMP20') if f in feats_full] or '确认缺席'})")

    results = {"seed": rv2.SEED, "grid": rv2.LGBM_GRID, "objective": "lambdarank",
               "gain_bins": N_BINS,
               "early_stop_metric": "val daily RankIC (vs continuous target)",
               "targets": TARGETS, "pools": {}}
    daily_curves = {}
    for pool in ("main", "backup"):
        log(f"阶段=池载入 {pool}")
        df, train_m, val_m = rv2.load_pool(pool)
        keys = df[["ts_code", "date"]].copy()
        tgt = compute_targets(pool, keys)
        tgt.to_parquet(OUT_DIR / f"targets_{pool}.parquet", index=False)
        df, na_stats = attach_targets(df, tgt)
        log(f"池={pool} 目标缺失统计 {na_stats}")
        sanity_checks(df, pool)
        base = rv2.independent_baseline(pool, df)

        pool_res = {
            "n_train": int(train_m.sum()), "n_val": int(val_m.sum()),
            "target_na": na_stats,
            "baseline": {k: v for k, v in base.items() if k != "daily_series"},
            "configs": {},
        }
        for target in TARGETS:
            for vname, with_atrn in FEAT_VARIANTS.items():
                feats = feats_full if with_atrn else feats_noatrn
                tag = f"{pool}/{target}__{vname}"
                r = run_target_config(df, train_m, val_m, feats, target, tag)
                daily_curves[tag] = {
                    "days": [str(d)[:10] for d in r["_daily"][0]],
                    "top3_daily": [float(v) for v in r["_daily"][1]],
                }
                st = rv2.wilcoxon_vs_baseline(r["_daily"][0], r["_daily"][1],
                                              base["daily_series"])
                r["baseline_day_weighted"] = base["day_weighted"]
                r["excess_pp"] = (r["val"]["top3"] - base["day_weighted"]) * 100.0
                r["stats"] = st
                r.pop("_daily")
                pool_res["configs"][f"{target}__{vname}"] = r
        results["pools"][pool] = pool_res

    # ================================================================ 多重比较校正
    main_cfgs = results["pools"]["main"]["configs"]
    fam8 = {k: v["stats"]["wilcoxon_p"] for k, v in main_cfgs.items()}
    all_cfgs = {f"{p}/{k}": v["stats"]["wilcoxon_p"]
                for p in ("main", "backup")
                for k, v in results["pools"][p]["configs"].items()}

    def bonferroni(p, m):
        return min(p * m, 1.0) if np.isfinite(p) else np.nan

    def bh_fdr(pvals):
        items = [(k, p) for k, p in pvals.items() if np.isfinite(p)]
        items.sort(key=lambda x: x[1])
        m = len(items)
        adj, prev = {}, 1.0
        for rank, (k, p) in reversed(list(enumerate(items, 1))):
            prev = min(prev, p * m / rank)
            adj[k] = prev
        return adj

    results["multiple_comparison"] = {
        "family_main8": {k: {"p": p, "bonf": bonferroni(p, 8),
                             "bh": bh_fdr(fam8).get(k)}
                         for k, p in fam8.items()},
        "family_all16": {k: {"p": p, "bonf": bonferroni(p, 16),
                             "bh": bh_fdr(all_cfgs).get(k)}
                         for k, p in all_cfgs.items()},
    }

    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    with (OUT_DIR / "daily_curves.json").open("w", encoding="utf-8") as f:
        json.dump(daily_curves, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=全部完成 耗时{time.time()-t0:.0f}s -> results.json")


if __name__ == "__main__":
    main()
