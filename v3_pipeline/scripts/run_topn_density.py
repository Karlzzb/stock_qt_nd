# -*- coding: utf-8 -*-
"""topN 扫描 × 密度分层实验 (topn_density)。

回答两个未测问题(只用训练段 2001-2018 + 验证段 2019-01~2022-10, 严禁测试段):
  1. topN 扫描(N=1/3/5/10/全选)下模型与零信息基线的日加权命中率曲线;
  2. 按当日候选数分层(稀疏≤3 / 中等4-10 / 扎堆11-50 / 爆发>50)后,
     模型 top3 在高密度子集里是否显著超过同层基线("口径稀释"假说检验)。

训练协议与 race_rerun_v2 一字不变(直接 import 其函数):
  lambdarank 按日分组、验证段日 Rank IC 早停、同 6 组超参网格按
  (RankIC, top3) 选参、最终模型训练段重训 n_estimators=best_iter。

模型配置(每池 3 个, 翻案局三个代表):
  C1_atrn_only : ["ATRN"] 原始值(几何对照; 单特征 rank/raw 日序相同);
  C3_raw       : ["ATRN"] + B档28 原始值(上轮最高 +2.08pp);
  C4_rank      : ["ATRN"] + B档28 + 边界带31 全部逐日截面秩变换。

评估扩展(验证段):
  A. topN 扫描: N∈{1,3,5,10,全选}, 日加权命中率;
     基线=同 N 日加权零信息基线(每日随机 N 个的期望=当日池命中率, 按日平均,
     因此对全部 N 为同一条线; topAll 模型≡基线作恒等校验)。
  B. 密度分层: 四层 + 合并层(>10), 每层内模型 top3 vs 同层基线,
     逐日配对 Wilcoxon(剔零差, n<10 记 NaN), 口径照 race_rerun_v2 复核规范。
  C. 十分位单调性: 仅 >10 候选的日子, 日内核分数降序十分位,
     事件加权命中率曲线 + 十分位序与命中率的 spearman。

输出:
  v3_pipeline/reports/topn_density/results_topn_density.json  全部数值
  v3_pipeline/reports/topn_density/progress.log               进度(追加)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sc_stats

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(ROOT / "v3_pipeline" / "scripts"))

import run_race_rerun_v2 as rv2  # noqa: E402  复用其数据装配/训练/日截面评估

OUT_DIR = ROOT / "v3_pipeline" / "reports" / "topn_density"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"

TOPN_LIST = [1, 3, 5, 10]
STRATA = [  # (名称, 下界含, 上界含; None=不限)
    ("sparse_<=3", 1, 3),
    ("medium_4_10", 4, 10),
    ("crowded_11_50", 11, 50),
    ("explosive_>50", 51, None),
    ("dense_>10", 11, None),  # 核心判定层: 扎堆+爆发合并
]


def log(msg):
    line = f"{msg} {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 逐日评估表
def per_day_table(df, val_m, scores):
    """每个验证段信号日一行: 候选数/池命中率/模型 topN 命中率(N=1,3,5,10,全选)。"""
    idx = np.nonzero(val_m)[0]
    sub = df.iloc[idx]
    day = sub["date"].to_numpy()
    hit = sub["hit"].to_numpy(np.float64)
    sc = np.where(np.isfinite(scores[idx]), scores[idx], -np.inf)
    srt, bounds = rv2.daily_groups(day)
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        r = srt[lo:hi]
        order = r[np.argsort(-sc[r], kind="stable")]
        d = {"date": str(day[r[0]])[:10], "n_cand": int(len(r)),
             "pool_hit": float(hit[r].mean())}
        for n in TOPN_LIST:
            d[f"top{n}"] = float(hit[order[:n]].mean())
        d["topAll"] = float(hit[r].mean())
        rows.append(d)
    return pd.DataFrame(rows)


def topn_scan(day_df):
    """topN 曲线: 模型日加权命中率 vs 零信息基线(当日池命中率按日平均, 与 N 无关)。"""
    base = float(day_df["pool_hit"].mean())
    out = {"baseline_day_weighted": base, "n_days": int(len(day_df))}
    for n in TOPN_LIST + ["All"]:
        col = f"top{n}"
        m = float(day_df[col].mean())
        out[col] = {"model": m, "excess_pp": (m - base) * 100.0}
    return out


# ================================================================ 密度分层
def wilcoxon_paired(model_daily, base_daily):
    """逐日配对 Wilcoxon, 口径照 race_rerun_v2 复核规范: 剔零差(|d|<=1e-12), 双侧, n<10 记 NaN。"""
    diff = np.asarray(model_daily, np.float64) - np.asarray(base_daily, np.float64)
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) < 10:
        return np.nan, int(len(nz))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = float(sc_stats.wilcoxon(nz).pvalue)
    return p, int(len(nz))


def density_strata(day_df):
    """按当日候选数分层; 每层: 模型 top3 vs 同层基线 + Wilcoxon。"""
    out = {}
    for name, lo, hi in STRATA:
        m = day_df["n_cand"] >= lo
        if hi is not None:
            m &= day_df["n_cand"] <= hi
        sub = day_df.loc[m]
        if len(sub) == 0:
            out[name] = {"n_days": 0}
            continue
        model = float(sub["top3"].mean())
        base = float(sub["pool_hit"].mean())
        p, npair = wilcoxon_paired(sub["top3"].to_numpy(),
                                   sub["pool_hit"].to_numpy())
        out[name] = {
            "n_days": int(len(sub)),
            "model_top3": model, "baseline": base,
            "excess_pp": (model - base) * 100.0,
            "wilcoxon_p": p, "n_pairs": npair,
        }
    return out


# ================================================================ 十分位单调性(仅 >10 候选日)
def decile_monotonicity(df, val_m, scores):
    idx = np.nonzero(val_m)[0]
    sub = df.iloc[idx]
    day = sub["date"].to_numpy()
    hit = sub["hit"].to_numpy(np.float64)
    sc = np.where(np.isfinite(scores[idx]), scores[idx], -np.inf)
    srt, bounds = rv2.daily_groups(day)
    dec_hits = [[] for _ in range(10)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        r = srt[lo:hi]
        n = len(r)
        if n <= 10:
            continue
        order = r[np.argsort(-sc[r], kind="stable")]
        dec = np.minimum(np.arange(n) * 10 // n, 9)  # 0=最高分位
        for k in range(10):
            sel = dec == k
            if sel.any():
                dec_hits[k].append(hit[order[sel]].mean())
    curve = {f"D{k+1}": {"hit": float(np.mean(v)), "n_days": len(v)}
             for k, v in enumerate(dec_hits)}
    xs = [k + 1 for k in range(10) if dec_hits[k]]
    ys = [curve[f"D{k+1}"]["hit"] for k in range(10) if dec_hits[k]]
    rho = float(sc_stats.spearmanr(xs, ys).statistic) if len(xs) >= 3 else np.nan
    return {"curve": curve, "decile_vs_hit_spearman": rho,
            "note": "日内核十分位, 命中率=逐日分位命中率的事件日平均(日加权); D1=最高分位"}


# ================================================================ 训练(与 rv2.run_config 逐行同构, 多返回分数)
def run_config_with_scores(df, train_m, val_m, feats, tag):
    X = df[feats].to_numpy(np.float64)
    y = df["hit"].to_numpy(np.float64)
    tr_idx, tr_bounds, gtr = rv2.sort_by_day(df, train_m)
    va_idx, va_bounds, gva = rv2.sort_by_day(df, val_m)
    Xtr, ytr = X[tr_idx], y[tr_idx].astype(int)
    Xva, yva = X[va_idx], y[va_idx].astype(int)
    ic_calc = rv2.DayRankIC(yva, va_bounds)
    feval = rv2.make_feval(ic_calc)

    import lightgbm as lgb
    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr,
                         feature_name=list(feats), free_raw_data=True)
    dval = lgb.Dataset(Xva, label=yva, group=gva, reference=dtrain,
                       free_raw_data=True)
    best = None
    for p in rv2.LGBM_GRID:
        params = dict(rv2.LGBM_BASE, **p)
        m = lgb.train(params, dtrain, num_boost_round=rv2.N_EST_MAX,
                      valid_sets=[dval], feval=feval,
                      callbacks=[lgb.early_stopping(rv2.EARLY_STOP,
                                                    verbose=False)])
        bi = max(int(m.best_iteration), 1)
        sc = m.predict(Xva, num_iteration=bi)
        t3, _, ics, _ = rv2.topk_metrics(df["date"].to_numpy()[va_idx],
                                         yva.astype(np.float64), sc)
        key = (float(ics.mean()) if len(ics) else -9.0, float(t3.mean()))
        if best is None or key > best[0]:
            best = (key, p, bi)
    best_params, best_iter = best[1], best[2]
    final = lgb.train(dict(rv2.LGBM_BASE, **best_params), dtrain,
                      num_boost_round=best_iter)
    prob = final.predict(X, num_iteration=best_iter)
    return prob, best_params, best_iter


# ================================================================ 主流程
def main():
    t0 = time.time()
    log("阶段=脚本启动 topn_density(topN扫描+密度分层; 训练协议=race_rerun_v2)")
    b28, band = rv2.feature_lists()
    log(f"阶段=名单载入 B档去重后{len(b28)} 边界带{len(band)}")

    configs = {
        "C1_atrn_only": (["ATRN"], "raw"),
        "C3_raw": (["ATRN"] + b28, "raw"),
        "C4_rank": (["ATRN"] + b28 + band, "rank"),
    }

    results = {"seed": rv2.SEED, "protocol": "race_rerun_v2 identical training",
               "topn_list": TOPN_LIST, "strata": [s[0] for s in STRATA],
               "pools": {}}
    for pool in ("main", "backup"):
        log(f"阶段=池载入 {pool}")
        df, train_m, val_m = rv2.load_pool(pool)
        base = rv2.independent_baseline(pool, df)  # 内含基线独立重算核对
        keep_m = train_m | val_m
        all_feats = sorted(set(sum([c[0] for c in configs.values()], [])))
        log(f"阶段=秩变换 {pool} 特征数={len(all_feats)}")
        rank_map = rv2.add_daily_rank(df, all_feats, keep_m)
        pool_res = {"n_train": int(train_m.sum()), "n_val": int(val_m.sum()),
                    "baseline_day_weighted": base["day_weighted"],
                    "configs": {}}
        for cname, (feats, mode) in configs.items():
            rfeats = [rank_map[f] for f in feats] if mode == "rank" else list(feats)
            missing = [f for f in rfeats if f not in df.columns]
            assert not missing, f"{pool} 缺特征 {missing}"
            prob, bp, bi = run_config_with_scores(
                df, train_m, val_m, rfeats, f"{pool}/{cname}")
            day_df = per_day_table(df, val_m, prob)
            scan = topn_scan(day_df)
            strat = density_strata(day_df)
            deci = decile_monotonicity(df, val_m, prob)
            pool_res["configs"][cname] = {
                "n_features": len(rfeats), "mode": mode,
                "best_params": bp, "best_iter": bi,
                "topn_scan": scan, "density_strata": strat,
                "decile_monotonicity": deci,
                "per_day": day_df.to_dict("records"),
            }
            d = strat.get("dense_>10", {})
            log(f"池x配置完成 {pool}/{cname} iter={bi} "
                f"top3超额={scan['top3']['excess_pp']:+.2f}pp "
                f"dense>10超额={d.get('excess_pp', float('nan')):+.2f}pp "
                f"p={d.get('wilcoxon_p', float('nan'))}")
        results["pools"][pool] = pool_res

    with (OUT_DIR / "results_topn_density.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=全部完成 耗时{time.time()-t0:.0f}s -> results_topn_density.json")


if __name__ == "__main__":
    main()
