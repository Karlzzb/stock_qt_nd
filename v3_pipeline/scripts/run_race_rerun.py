# -*- coding: utf-8 -*-
"""终审赛跑重跑: 剥离标签几何后, 幸存特征能否超过信号池裸命中率基线。

协议(与 #4 赛跑 architecture_race.py 一致):
  - LGBM 二分类, 预测 hit_N20_k2.0, 6 组小网格, 训练段拟合 + 验证段 early stopping,
    按验证段按日 Rank IC 选超参(top3 命中率 tiebreak), 最终模型在训练段重训
    (n_estimators=best_iter), 只在验证段评估(本实验严禁测试段)。
  - 主指标: 每日横截面 top3 命中率(当日不足 3 个取全部, 无信号日不计)。
  - 切分与隔离带与 run_feature_screening.py 完全一致:
    train 2001-01-01~2018-12-31, val 2019-01-01~2022-10-31,
    隔离带 2018-11-19~2018-12-28 与 2022-09-13~2022-10-31 内事件丢弃,
    2022-11 之后事件不入任何统计。

配置(每池各跑一遍, ATRN 强制在全部配置特征集中):
  1. atrn_only        仅 ATRN(几何裸奔基线)
  2. atrn_plus4       ATRN + JUMPFREQ60/PRICE_IMPACT/VOL_CONTRACTION60/BBW_PCTILE250
  3. atrn_plus_b28    ATRN + B 档 29 个剔除 RET20_CSR(RET20 截面秩别名)
  4. atrn_plus_full   配置 3 + 31 个边界带特征(|val ICIR|∈[0.05,0.1), 交互感知复筛)

输出:
  v3_pipeline/reports/race_rerun/results.json      全部数值
  v3_pipeline/reports/race_rerun/progress.log      进度(追加)
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
OUT_DIR = ROOT / "v3_pipeline" / "reports" / "race_rerun"
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
    objective="binary", metric="binary_logloss", boosting_type="gbdt",
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


def log(msg):
    line = f"{msg} {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 数据装配(与筛查脚本同口径)
def load_pool(name):
    cfg = POOLS[name]
    fm = pd.read_parquet(cfg["features"])
    fm = fm.replace([np.inf, -np.inf], np.nan)
    ev = pd.read_parquet(cfg["labdir"] / "events.parquet", columns=["ts_code", "date"])
    lb = pd.read_parquet(cfg["labdir"] / "labels.parquet")
    n = len(ev)
    div = lb.iloc[:n].reset_index(drop=True)
    assert (div["group"] == "div").all()
    lab = pd.DataFrame({
        "ts_code": ev["ts_code"].values,
        "date": pd.to_datetime(ev["date"].values),
        "hit": div["hit_N20_k2.0"].values,
    })
    assert not lab.duplicated(["ts_code", "date"]).any()
    fm["date"] = pd.to_datetime(fm["date"])
    df = fm.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")

    in_emb = np.zeros(len(df), bool)
    for lo, hi in EMBARGO:
        in_emb |= (df["date"] >= lo) & (df["date"] <= hi)
    seg = np.full(len(df), "drop", dtype=object)
    seg[(df["date"] >= TRAIN_LO) & (df["date"] <= TRAIN_HI)] = "train"
    seg[(df["date"] >= VAL_LO) & (df["date"] <= VAL_HI)] = "val"
    seg[in_emb] = "embargo"
    df["seg"] = seg
    lab_ok = df["hit"].notna().to_numpy()
    train_m = (df["seg"] == "train").to_numpy() & lab_ok
    val_m = (df["seg"] == "val").to_numpy() & lab_ok
    return df, train_m, val_m


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


# ================================================================ 指标(与 #4 赛跑 segment_metrics 同口径)
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
    # 逐年 top3
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


# ================================================================ 模型
def fit_lgbm(Xtr, ytr, Xva, yva, params, n_est, early_stop=True):
    m = lgb.LGBMClassifier(**LGBM_BASE, **params, n_estimators=n_est)
    if early_stop:
        m.fit(np.asarray(Xtr, np.float64), np.asarray(ytr, int),
              eval_X=np.asarray(Xva, np.float64), eval_y=np.asarray(yva, int),
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    else:
        m.fit(np.asarray(Xtr, np.float64), np.asarray(ytr, int))
    return m


def run_config(df, train_m, val_m, feats, tag):
    Xtr = df.loc[train_m, feats]; ytr = df.loc[train_m, "hit"].astype(int)
    Xva = df.loc[val_m, feats]; yva = df.loc[val_m, "hit"].astype(int)
    best = None
    for p in LGBM_GRID:
        m = fit_lgbm(Xtr, ytr, Xva, yva, p, N_EST_MAX)
        sc = np.full(len(df), np.nan)
        sc[np.nonzero(val_m)[0]] = m.predict_proba(Xva)[:, 1]
        mt = eval_segment(df, val_m, sc)
        key = (np.nan_to_num(mt["rank_ic"], -9), np.nan_to_num(mt["top3"], -9))
        if best is None or key > best[0]:
            best = (key, p, m.best_iteration_)
    best_params, best_iter = best[1], max(int(best[2]), 1)
    final = fit_lgbm(Xtr, ytr, None, None, best_params, best_iter, early_stop=False)
    prob = np.full(len(df), np.nan)
    prob[:] = final.predict_proba(df[feats].to_numpy())[:, 1]
    mt = eval_segment(df, val_m, prob)
    # 增益重要性 + 相对 ATRN 增量
    gain = final.booster_.feature_importance(importance_type="gain")
    imp = dict(zip(feats, [float(g) for g in gain]))
    atrn_gain = imp.get("ATRN", 0.0)
    incr = {f: (g / atrn_gain if atrn_gain > 0 else np.nan) for f, g in imp.items()}
    out = {
        "features": feats, "best_params": best_params, "best_iter": best_iter,
        "val": {k: v for k, v in mt.items() if not k.startswith("_")},
        "gain_importance": imp, "gain_vs_atrn": incr,
        "_daily": mt["_daily"],
    }
    log(f"配置={tag} val_top3={mt['top3']:.4f} val_rankIC={mt['rank_ic']:.4f} "
        f"iter={best_iter} 完成")
    return out


def main():
    t0 = time.time()
    log("阶段=脚本启动")
    b28, band = feature_lists()
    log(f"阶段=名单载入 B档去重后{len(b28)} 边界带{len(band)}")

    configs = {
        "1_atrn_only": ["ATRN"],
        "2_atrn_plus4": ["ATRN"] + PLUS4,
        "3_atrn_plus_b28": ["ATRN"] + b28,
        "4_atrn_plus_full": ["ATRN"] + b28 + band,
    }

    results = {"seed": SEED, "grid": LGBM_GRID, "pools": {}}
    daily_curves = {}
    for pool in ("main", "backup"):
        log(f"阶段=池载入 {pool}")
        df, train_m, val_m = load_pool(pool)
        log(f"池={pool} train={int(train_m.sum())} val={int(val_m.sum())} "
            f"val裸命中率={df.loc[val_m, 'hit'].mean():.4f}")
        pool_res = {
            "n_train": int(train_m.sum()), "n_val": int(val_m.sum()),
            "val_pool_hit": float(df.loc[val_m, "hit"].mean()),
            "configs": {},
        }
        for cname, feats in configs.items():
            missing = [f for f in feats if f not in df.columns]
            assert not missing, f"{pool} 缺特征 {missing}"
            r = run_config(df, train_m, val_m, feats, f"{pool}/{cname}")
            daily_curves[f"{pool}/{cname}"] = {
                "days": [str(d)[:10] for d in r.pop("_daily")[0]],
                "top3": r["val"]["top3"],
            }
            pool_res["configs"][cname] = r
        results["pools"][pool] = pool_res

    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=全部完成 耗时{time.time()-t0:.0f}s -> results.json")


if __name__ == "__main__":
    main()
