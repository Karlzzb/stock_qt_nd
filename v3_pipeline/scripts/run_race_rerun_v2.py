# -*- coding: utf-8 -*-
"""终审赛跑翻案局 (race_rerun_v2): 与评估指标同构的训练协议。

相对 run_race_rerun.py 的训练侧改动(评估口径一字不变):
  1. LGBM lambdarank 目标(native API), 按"事件日"分组(每交易日一个 query group),
     标签 = hit_N20_k2.0;
  2. 早停与迭代数选择按验证段日 Rank IC(自定义 feval, 不用 logloss/ndcg);
  3. 特征逐日截面秩变换(rank_pct, 按日分组)为主变体, 不做秩变换为对照变体;
  4. ATRN 强制留在全部配置特征集中(几何控制);
  5. 可选: C3 加样本权重变体(逆日频 1/当日训练事件数, 日等权)。

配置矩阵(每池): C1=ATRN 仅; C2=ATRN+排雷4; C3=ATRN+B档28(剔除 RET20_CSR);
C4=ATRN+B档28+边界带31。每配置: rank(主) + raw(对照)。

评估协议(与 run_race_rerun.py 完全一致):
  - 每日横截面 top3 命中率(当日不足 3 个取全部, 无信号日不计), 日加权;
  - 切分 train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31,
    隔离带 2018-11-19~2018-12-28 与 2022-09-13~2022-10-31 事件丢弃,
    2022-11 之后事件不入任何统计;
  - 对照基线: 日加权零信息基线(当日池命中率的信号日均值), 从标签独立重算核对
    (预期 主池 49.97% / 备池 55.73%);
  - 超参网格与 run_race_rerun.py 相同 6 组, 按验证段日 Rank IC 选参(top3 tiebreak),
    最终模型训练段重训 n_estimators=best_iter。

辅助指标: 验证段日 Rank IC、逐年 top3 对逐年日加权基线、逐日配对 Wilcoxon p、
增量增益重要性前 15(相对 ATRN)。

输出:
  v3_pipeline/reports/race_rerun_v2/results_v2.json   全部数值
  v3_pipeline/reports/race_rerun_v2/progress.log      进度(追加)
"""
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sc_stats

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
FM_DIR = ROOT / "v3_pipeline" / "reports" / "feature_matrix"
SCR_DIR = ROOT / "v3_pipeline" / "reports" / "feature_screening"
OUT_DIR = ROOT / "v3_pipeline" / "reports" / "race_rerun_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"

POOLS = {
    "main": {
        "features": FM_DIR / "main_pool_features.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_fractal_o15_s20",
    },
    "backup": {
        "features": FM_DIR / "backup_pool_features.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_zigzag_p05_s5",
    },
}

TRAIN_LO, TRAIN_HI = pd.Timestamp("2001-01-01"), pd.Timestamp("2018-12-31")
VAL_LO, VAL_HI = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-10-31")
EMBARGO = [(pd.Timestamp("2018-11-19"), pd.Timestamp("2018-12-28")),
           (pd.Timestamp("2022-09-13"), pd.Timestamp("2022-10-31"))]

SEED = 42
LGBM_BASE = dict(
    objective="lambdarank", metric="None", boosting_type="gbdt",
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    num_threads=8, deterministic=True, force_row_wise=True,
    seed=SEED, verbose=-1,
)
LGBM_GRID = [
    dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc)
    for nl in (15, 31, 63) for lr in (0.03, 0.08) for mc in (20,)
]
N_EST_MAX = 2000
EARLY_STOP = 100

PLUS4 = ["JUMPFREQ60", "PRICE_IMPACT", "VOL_CONTRACTION60", "BBW_PCTILE250"]

EXPECTED_DAYW_BASELINE = {"main": 0.4997, "backup": 0.5573}


def log(msg):
    line = f"{msg} {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 数据装配(与筛查/上轮同口径)
def load_labels(labdir):
    """独立标签链: 直接读 events+labels, 返回 (ts_code, date, hit)。"""
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


def segment_mask(dates):
    in_emb = np.zeros(len(dates), bool)
    for lo, hi in EMBARGO:
        in_emb |= (dates >= lo) & (dates <= hi)
    seg = np.full(len(dates), "drop", dtype=object)
    seg[(dates >= TRAIN_LO) & (dates <= TRAIN_HI)] = "train"
    seg[(dates >= VAL_LO) & (dates <= VAL_HI)] = "val"
    seg[in_emb] = "embargo"
    return seg


def load_pool(name):
    cfg = POOLS[name]
    fm = pd.read_parquet(cfg["features"])
    fm = fm.replace([np.inf, -np.inf], np.nan)
    lab = load_labels(cfg["labdir"])
    fm["date"] = pd.to_datetime(fm["date"])
    df = fm.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    df["seg"] = segment_mask(df["date"].to_numpy())
    lab_ok = df["hit"].notna().to_numpy()
    train_m = (df["seg"] == "train").to_numpy() & lab_ok
    val_m = (df["seg"] == "val").to_numpy() & lab_ok
    return df, train_m, val_m


def independent_baseline(name, df):
    """日加权零信息基线: 从标签独立重算(不经特征矩阵 merge 结果),
    只借用特征矩阵的 (ts_code,date) 键集合限定宇宙。"""
    cfg = POOLS[name]
    lab = load_labels(cfg["labdir"])
    keys = df[["ts_code", "date"]].copy()
    m = keys.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    seg = segment_mask(m["date"].to_numpy())
    ok = (seg == "val") & m["hit"].notna().to_numpy()
    sub = m.loc[ok]
    daily = sub.groupby("date")["hit"].mean()
    dayw = float(daily.mean())
    exp = EXPECTED_DAYW_BASELINE[name]
    status = "一致" if abs(dayw - exp) < 5e-4 else "!!不符!!"
    log(f"池={name} 基线独立重算: 日加权={dayw:.4f} 事件加权={sub['hit'].mean():.4f} "
        f"信号日={len(daily)} (预期日加权≈{exp}, {status})")
    yearly = {int(y): float(daily[pd.DatetimeIndex(daily.index).year == y].mean())
              for y in sorted(set(pd.DatetimeIndex(daily.index).year))}
    return {
        "day_weighted": dayw,
        "event_weighted": float(sub["hit"].mean()),
        "n_events": int(len(sub)),
        "n_days": int(len(daily)),
        "yearly_day_weighted": yearly,
        "daily_series": {str(d)[:10]: float(v) for d, v in daily.items()},
    }


# ================================================================ 特征名单
def feature_lists():
    ic = pd.read_csv(SCR_DIR / "per_feature_ic.csv")
    b_tier = ic.loc[ic.decision == "B", "feature"].tolist()
    assert len(b_tier) == 29
    b28 = [f for f in b_tier if f != "RET20_CSR"]  # RET20 截面秩别名去重
    band = ic.loc[(ic.decision == "淘汰")
                  & ic.main_val_icir.abs().between(0.05, 0.1, inclusive="left"),
                  "feature"].tolist()
    assert len(band) == 31
    return b28, band


# ================================================================ 指标(与 #4 赛跑/上轮 segment_metrics 同口径, 逐字复用)
def daily_groups(day):
    srt = np.argsort(day, kind="stable")
    day_s = day[srt]
    _, starts = np.unique(day_s, return_index=True)
    bounds = np.append(starts, len(day_s))
    return srt, bounds


def topk_metrics(day, hit, sc):
    """每日横截面 top3/top1 命中率与按日 Rank IC。sc 为分数(大者优先)。"""
    srt, bounds = daily_groups(day)
    top3_hit, top1_hit, ics, days = [], [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = srt[lo:hi]
        order = rows[np.argsort(-sc[rows], kind="stable")]
        top3_hit.append(hit[order[:3]].mean())
        top1_hit.append(hit[order[:1]].mean())
        days.append(day[rows[0]])
        if len(rows) >= 3:
            y, s = hit[rows], sc[rows]
            if np.std(y) > 1e-12 and np.std(s) > 1e-12:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", sc_stats.ConstantInputWarning)
                    ic = sc_stats.spearmanr(s, y).statistic
                if np.isfinite(ic):
                    ics.append(ic)
    return (np.asarray(top3_hit), np.asarray(top1_hit), np.asarray(ics),
            np.asarray(days))


def eval_segment(df, mask, scores):
    idx = np.nonzero(mask)[0]
    sub = df.iloc[idx]
    day = sub["date"].to_numpy()
    hit = sub["hit"].to_numpy(np.float64)
    sc = np.where(np.isfinite(scores[idx]), scores[idx], -np.inf)
    t3, t1, ics, days = topk_metrics(day, hit, sc)
    years = pd.DatetimeIndex(days).year
    yearly = {int(y): float(t3[years == y].mean()) for y in sorted(set(years))}
    return {
        "n": int(len(hit)), "days": int(len(t3)),
        "top3": float(t3.mean()), "top1": float(t1.mean()),
        "rank_ic": float(ics.mean()) if len(ics) else np.nan,
        "pool_hit": float(hit.mean()),
        "yearly_top3": yearly,
        "_daily": (days, t3),
    }


# ================================================================ 快速日 Rank IC(feval 用)
def _rank_avg(x):
    """平均秩(处理并列), 从 1 开始。"""
    n = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


class DayRankIC:
    """按日 spearman(score, hit) 均值; 行序须与构造时一致(已按日排序)。"""

    def __init__(self, y_sorted, bounds):
        self.groups = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            yy = np.asarray(y_sorted[lo:hi], np.float64)
            if len(yy) < 3 or np.std(yy) <= 1e-12:
                continue
            ry = _rank_avg(yy)
            cy = ry - ry.mean()
            ny = float(np.sqrt(cy @ cy))
            self.groups.append((slice(lo, hi), cy, ny))

    def __call__(self, preds):
        tot, cnt = 0.0, 0
        for sl, cy, ny in self.groups:
            s = np.asarray(preds[sl], np.float64)
            if np.std(s) <= 1e-12:
                continue
            rs = _rank_avg(s)
            cs = rs - rs.mean()
            ns = float(np.sqrt(cs @ cs))
            if ns <= 0:
                continue
            ic = float(cs @ cy) / (ns * ny)
            if np.isfinite(ic):
                tot += ic
                cnt += 1
        return tot / cnt if cnt else 0.0


def make_feval(ic_calc):
    def feval(preds, dataset):
        return [("day_rank_ic", float(ic_calc(preds)), True)]
    return feval


# ================================================================ 秩变换
def add_daily_rank(df, feats, keep_m):
    """逐日截面秩变换(rank_pct); 只在 train/val 行上计算, 输出新列 <f>__DR。"""
    out = {}
    sub = df.loc[keep_m, ["date"] + feats]
    ranked = sub.groupby("date")[feats].rank(pct=True)
    for f in feats:
        col = f + "__DR"
        df[col] = np.nan
        df.loc[keep_m, col] = ranked[f].to_numpy()
        out[f] = col
    return out


# ================================================================ 模型
def sort_by_day(df, mask):
    idx = np.nonzero(mask)[0]
    order = np.argsort(df["date"].to_numpy()[idx], kind="stable")
    idx = idx[order]
    day = df["date"].to_numpy()[idx]
    _, starts = np.unique(day, return_index=True)
    bounds = np.append(starts, len(day))
    group = np.diff(bounds).tolist()
    return idx, bounds, group


def run_config(df, train_m, val_m, feats, tag, use_weight=False):
    X = df[feats].to_numpy(np.float64)
    y = df["hit"].to_numpy(np.float64)
    tr_idx, tr_bounds, gtr = sort_by_day(df, train_m)
    va_idx, va_bounds, gva = sort_by_day(df, val_m)
    Xtr, ytr = X[tr_idx], y[tr_idx].astype(int)
    Xva, yva = X[va_idx], y[va_idx].astype(int)
    wtr = None
    if use_weight:
        cnt = np.repeat(np.diff(tr_bounds), np.diff(tr_bounds))
        wtr = (1.0 / cnt).astype(np.float64)
    ic_calc = DayRankIC(yva, va_bounds)
    feval = make_feval(ic_calc)

    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, weight=wtr,
                         feature_name=list(feats), free_raw_data=True)
    dval = lgb.Dataset(Xva, label=yva, group=gva, reference=dtrain,
                       free_raw_data=True)

    best = None
    for p in LGBM_GRID:
        params = dict(LGBM_BASE, **p)
        m = lgb.train(params, dtrain, num_boost_round=N_EST_MAX,
                      valid_sets=[dval], feval=feval,
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        bi = max(int(m.best_iteration), 1)
        sc = m.predict(Xva, num_iteration=bi)
        t3, _, ics, _ = topk_metrics(df["date"].to_numpy()[va_idx],
                                     yva.astype(np.float64), sc)
        key = (float(ics.mean()) if len(ics) else -9.0, float(t3.mean()))
        if best is None or key > best[0]:
            best = (key, p, bi)
    best_params, best_iter = best[1], best[2]

    final = lgb.train(dict(LGBM_BASE, **best_params), dtrain,
                      num_boost_round=best_iter)
    prob = np.full(len(df), np.nan)
    prob[:] = final.predict(X, num_iteration=best_iter)
    mt = eval_segment(df, val_m, prob)

    gain = final.feature_importance(importance_type="gain")
    imp = dict(zip(feats, [float(g) for g in gain]))
    atrn_key = next((f for f in feats if f == "ATRN" or f.startswith("ATRN__")), None)
    atrn_gain = imp.get(atrn_key, 0.0) if atrn_key else 0.0
    incr = {f: (g / atrn_gain if atrn_gain > 0 else np.nan) for f, g in imp.items()}
    out = {
        "n_features": len(feats),
        "features": feats, "best_params": best_params, "best_iter": best_iter,
        "val": {k: v for k, v in mt.items() if not k.startswith("_")},
        "gain_importance": imp, "gain_vs_atrn": incr,
        "_daily": mt["_daily"],
    }
    log(f"配置={tag} val_top3={mt['top3']:.4f} val_rankIC={mt['rank_ic']:.4f} "
        f"iter={best_iter} 完成")
    return out


def wilcoxon_vs_baseline(days, t3, baseline_daily):
    """逐日配对: 模型 top3 命中率 vs 当日池命中率(日加权零信息基线)。"""
    b = np.array([baseline_daily[str(d)[:10]] for d in days], np.float64)
    diff = t3 - b
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) < 10:
        return {"wilcoxon_p": np.nan, "ttest_p": np.nan, "n_pairs": int(len(nz))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wp = float(sc_stats.wilcoxon(nz).pvalue)
        tp = float(sc_stats.ttest_rel(t3, b).pvalue)
    return {"wilcoxon_p": wp, "ttest_p": tp, "n_pairs": int(len(nz))}


def main():
    t0 = time.time()
    log("阶段=脚本启动 race_rerun_v2(lambdarank+日RankIC早停+截面秩变换)")
    b28, band = feature_lists()
    log(f"阶段=名单载入 B档去重后{len(b28)} 边界带{len(band)}")

    configs = {
        "C1_atrn_only": ["ATRN"],
        "C2_atrn_plus4": ["ATRN"] + PLUS4,
        "C3_atrn_plus_b28": ["ATRN"] + b28,
        "C4_atrn_plus_full": ["ATRN"] + b28 + band,
    }

    results = {"seed": SEED, "grid": LGBM_GRID, "objective": "lambdarank",
               "early_stop_metric": "val daily RankIC", "pools": {}}
    daily_curves = {}
    for pool in ("main", "backup"):
        log(f"阶段=池载入 {pool}")
        df, train_m, val_m = load_pool(pool)
        base = independent_baseline(pool, df)
        keep_m = train_m | val_m
        all_feats = sorted(set(sum(configs.values(), [])))
        log(f"阶段=秩变换 {pool} 特征数={len(all_feats)}")
        rank_map = add_daily_rank(df, all_feats, keep_m)
        pool_res = {
            "n_train": int(train_m.sum()), "n_val": int(val_m.sum()),
            "baseline": {k: v for k, v in base.items() if k != "daily_series"},
            "configs": {},
        }
        runs = {}
        for cname, feats in configs.items():
            runs[cname + "__rank"] = [rank_map[f] for f in feats]
            runs[cname + "__raw"] = list(feats)
        # 可选: C3 样本权重变体(逆日频, 日等权), 仅秩变换变体
        weighted = {"C3_atrn_plus_b28__rank__w"}
        for rname, rfeats in runs.items():
            missing = [f for f in rfeats if f not in df.columns]
            assert not missing, f"{pool} 缺特征 {missing}"
            r = run_config(df, train_m, val_m, rfeats, f"{pool}/{rname}")
            daily_curves[f"{pool}/{rname}"] = {
                "days": [str(d)[:10] for d in r["_daily"][0]],
                "top3_daily": [float(v) for v in r["_daily"][1]],
            }
            st = wilcoxon_vs_baseline(r["_daily"][0], r["_daily"][1],
                                      base["daily_series"])
            base_dw = base["day_weighted"]
            r["baseline_day_weighted"] = base_dw
            r["excess_pp"] = (r["val"]["top3"] - base_dw) * 100.0
            r["stats"] = st
            r.pop("_daily")
            pool_res["configs"][rname] = r
        # 样本权重变体
        for cname in ("C3_atrn_plus_b28",):
            rfeats = [rank_map[f] for f in configs[cname]]
            r = run_config(df, train_m, val_m, rfeats,
                           f"{pool}/{cname}__rank__w", use_weight=True)
            daily_curves[f"{pool}/{cname}__rank__w"] = {
                "days": [str(d)[:10] for d in r["_daily"][0]],
                "top3_daily": [float(v) for v in r["_daily"][1]],
            }
            st = wilcoxon_vs_baseline(r["_daily"][0], r["_daily"][1],
                                      base["daily_series"])
            r["baseline_day_weighted"] = base["day_weighted"]
            r["excess_pp"] = (r["val"]["top3"] - base["day_weighted"]) * 100.0
            r["stats"] = st
            r.pop("_daily")
            pool_res["configs"][cname + "__rank__w"] = r
        results["pools"][pool] = pool_res

    with (OUT_DIR / "results_v2.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    with (OUT_DIR / "daily_curves.json").open("w", encoding="utf-8") as f:
        json.dump(daily_curves, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=全部完成 耗时{time.time()-t0:.0f}s -> results_v2.json")


if __name__ == "__main__":
    main()
