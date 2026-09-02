# -*- coding: utf-8 -*-
"""稳定性三关筛（后两关）：单因子有效性 + 分年稳定 + 共线性 + 双池复现。

纪律：全程只用 2001-01-01 ~ 2022-10-31 的事件（含 30 交易日段界隔离带剔除），
2022-11 之后数据一律不读入统计。
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/home/karl/repos/personal/stock_qt_nd"
FM_DIR = f"{ROOT}/v3_pipeline/reports/feature_matrix"
OUT_DIR = f"{ROOT}/v3_pipeline/reports/feature_screening"
os.makedirs(OUT_DIR, exist_ok=True)

POOLS = {
    "main": {
        "features": f"{FM_DIR}/main_pool_features.parquet",
        "labdir": f"{ROOT}/v3_pipeline/reports/divergence_lab/w_fractal_o15_s20",
    },
    "backup": {
        "features": f"{FM_DIR}/backup_pool_features.parquet",
        "labdir": f"{ROOT}/v3_pipeline/reports/divergence_lab/w_zigzag_p05_s5",
    },
}

TRAIN_LO, TRAIN_HI = pd.Timestamp("2001-01-01"), pd.Timestamp("2018-12-31")
VAL_LO, VAL_HI = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-10-31")
EMBARGO = [(pd.Timestamp("2018-11-19"), pd.Timestamp("2018-12-28")),
           (pd.Timestamp("2022-09-13"), pd.Timestamp("2022-10-31"))]

GEO_CONTROLS = ["ATRN", "VOL20", "AMP20"]
MIN_DAY_N = 5      # 每日横截面最少事件数
MIN_YEAR_DAYS = 10  # 年度 IC 有效最少天数

fdict = pd.read_csv(f"{FM_DIR}/feature_dictionary.csv")
FEATS = fdict["column"].tolist()
assert len(FEATS) == 179, len(FEATS)
FAMILY = dict(zip(fdict["column"], fdict["family"]))
EVENT_ONLY = dict(zip(fdict["column"], fdict["event_only"]))


def load_pool(name):
    cfg = POOLS[name]
    fm = pd.read_parquet(cfg["features"])
    fm = fm.replace([np.inf, -np.inf], np.nan)
    ev = pd.read_parquet(cfg["labdir"] + "/events.parquet", columns=["ts_code", "date"])
    lb = pd.read_parquet(cfg["labdir"] + "/labels.parquet")
    n = len(ev)
    div = lb.iloc[:n].reset_index(drop=True)
    assert (div["group"] == "div").all()
    lab = pd.DataFrame({
        "ts_code": ev["ts_code"].values,
        "date": pd.to_datetime(ev["date"].values),
        "hit": div["hit_N20_k2.0"].values,
        "ret_h10": div["ret_h10"].values,
        "ret_h30": div["ret_h30"].values,
    })
    assert not lab.duplicated(["ts_code", "date"]).any()
    fm["date"] = pd.to_datetime(fm["date"])
    df = fm.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    assert df["ts_code"].notna().all()
    # 标签本身存在 NaN（前向窗不足/停牌等），逐标签在 IC 计算时自然剔除，这里只统计
    df.attrs["n_hit_nan"] = int(df["hit"].isna().sum())
    # 段切分 + 隔离带
    in_emb = np.zeros(len(df), bool)
    for lo, hi in EMBARGO:
        in_emb |= (df["date"] >= lo) & (df["date"] <= hi)
    df["seg"] = "drop"
    df.loc[(df["date"] >= TRAIN_LO) & (df["date"] <= TRAIN_HI), "seg"] = "train"
    df.loc[(df["date"] >= VAL_LO) & (df["date"] <= VAL_HI), "seg"] = "val"
    df.loc[in_emb, "seg"] = "embargo"
    return df


def daily_rank_ic(df, feat_cols, ycol, min_n=5):
    """每日横截面 Rank IC，返回 dates × feats 的 DataFrame。"""
    cnt = df.groupby("date").size()
    ok_dates = cnt[cnt >= min_n].index
    sub = df.loc[df["date"].isin(ok_dates), ["date", ycol] + feat_cols]
    g = sub.groupby("date")
    ranks = g[feat_cols + [ycol]].rank()
    keyed = ranks.groupby(sub["date"].values)
    centered = ranks - keyed.transform("mean")
    cy = centered[ycol]
    C = centered[feat_cols]
    num = C.mul(cy, axis=0).groupby(sub["date"].values).sum()
    denx = C.pow(2).groupby(sub["date"].values).sum()
    deny = cy.pow(2).groupby(sub["date"].values).sum()
    ic = num.div(np.sqrt(denx.mul(deny, axis=0)))
    return ic.sort_index()


def agg(ic_daily, lo, hi):
    seg = ic_daily.loc[(ic_daily.index >= lo) & (ic_daily.index <= hi)]
    out = pd.DataFrame({
        "ic_mean": seg.mean(),
        "ic_std": seg.std(),
        "n_days": seg.count(),
    })
    out["icir"] = out["ic_mean"] / out["ic_std"]
    out["pos_day_share"] = (seg > 0).sum() / out["n_days"]
    return out


def yearly_ic(ic_daily):
    df = ic_daily.copy()
    df["year"] = df.index.year
    yrs = {}
    days = {}
    for y, sub in df.groupby("year"):
        sub = sub.drop(columns="year")
        yrs[y] = sub.mean()
        days[y] = sub.count()
    return pd.DataFrame(yrs), pd.DataFrame(days)


results = {}
meta = {}
for pool in ["main", "backup"]:
    print(f"=== pool {pool} ===", flush=True)
    df = load_pool(pool)
    use = df[df["seg"].isin(["train", "val"])].copy()
    meta[pool] = {
        "n_events_total": int(len(df)),
        "n_train": int((use["seg"] == "train").sum()),
        "n_val": int((use["seg"] == "val").sum()),
        "n_embargo_dropped": int((df["seg"] == "embargo").sum()),
        "n_pre2001_dropped": int(((df["seg"] == "drop") & (df["date"] < TRAIN_LO)).sum()),
        "n_post_2022_10_untouched": int((df["date"] > VAL_HI).sum()),
        "hit_rate_train": float(use.loc[use["seg"] == "train", "hit"].mean()),
        "hit_rate_val": float(use.loc[use["seg"] == "val", "hit"].mean()),
        "n_hit_nan_train": int(use.loc[use["seg"] == "train", "hit"].isna().sum()),
        "n_hit_nan_val": int(use.loc[use["seg"] == "val", "hit"].isna().sum()),
    }
    ic_hit = daily_rank_ic(use, FEATS, "hit")
    ic_ret = daily_rank_ic(use, FEATS, "ret_h10")
    y_hit, y_days = yearly_ic(ic_hit)
    res = {
        "train_hit": agg(ic_hit, TRAIN_LO, TRAIN_HI),
        "val_hit": agg(ic_hit, VAL_LO, VAL_HI),
        "train_ret": agg(ic_ret, TRAIN_LO, TRAIN_HI),
        "val_ret": agg(ic_ret, VAL_LO, VAL_HI),
        "yearly_hit": y_hit,
        "yearly_days": y_days,
    }
    meta[pool]["n_days_train"] = int(ic_hit.loc[(ic_hit.index >= TRAIN_LO) & (ic_hit.index <= TRAIN_HI)].shape[0])
    meta[pool]["n_days_val"] = int(ic_hit.loc[(ic_hit.index >= VAL_LO) & (ic_hit.index <= VAL_HI)].shape[0])
    results[pool] = res
    # 对齐 sanity：hit=1 的 ret_h10 应显著高于 hit=0
    a = use.loc[use["hit"] == 1, "ret_h10"].mean()
    b = use.loc[use["hit"] == 0, "ret_h10"].mean()
    meta[pool]["ret_h10_hit1"] = float(a)
    meta[pool]["ret_h10_hit0"] = float(b)
    print(json.dumps(meta[pool], indent=1), flush=True)

# ---------------- 汇总 per-feature 表 ----------------
rows = []
YEARS = list(range(2001, 2023))
for f in FEATS:
    r = {"feature": f, "family": FAMILY[f], "event_only": bool(EVENT_ONLY[f])}
    for pool in ["main", "backup"]:
        R = results[pool]
        for seg in ["train", "val"]:
            H = R[f"{seg}_hit"].loc[f]
            T = R[f"{seg}_ret"].loc[f]
            r[f"{pool}_{seg}_ic"] = H["ic_mean"]
            r[f"{pool}_{seg}_icir"] = H["icir"]
            r[f"{pool}_{seg}_pos_day"] = H["pos_day_share"]
            r[f"{pool}_{seg}_days"] = int(H["n_days"])
            r[f"{pool}_{seg}_ic_ret10"] = T["ic_mean"]
            r[f"{pool}_{seg}_icir_ret10"] = T["icir"]
        yh = R["yearly_hit"].loc[f]
        yd = R["yearly_days"].loc[f]
        for y in YEARS:
            r[f"y{y}"] = yh.get(y, np.nan)
        # 年度稳定性（方向以 main-train IC 符号为准）
        valid = [y for y in YEARS if pd.notna(yh.get(y)) and yd.get(y, 0) >= MIN_YEAR_DAYS]
        r["n_valid_years"] = len(valid)
        if pool == "main":
            tr = r["main_train_ic"]
            d = np.sign(tr) if tr != 0 else 1.0
            agree = sum(1 for y in valid if np.sign(yh[y]) == d)
            r["dir_consistent_year_share"] = agree / len(valid) if valid else np.nan
            r["pos_year_share"] = sum(1 for y in valid if yh[y] > 0) / len(valid) if valid else np.nan
    rows.append(r)

perf = pd.DataFrame(rows)


def gates(r, pool):
    tr, va = r[f"{pool}_train_ic"], r[f"{pool}_val_ic"]
    g1 = np.sign(tr) == np.sign(va) and tr != 0
    if pool == "main":
        share = r["dir_consistent_year_share"]
    else:
        # 备池用自身 train 方向
        d = np.sign(tr) if tr != 0 else 1.0
        yh = results[pool]["yearly_hit"].loc[r["feature"]]
        yd = results[pool]["yearly_days"].loc[r["feature"]]
        valid = [y for y in YEARS if pd.notna(yh.get(y)) and yd.get(y, 0) >= MIN_YEAR_DAYS]
        share = sum(1 for y in valid if np.sign(yh[y]) == d) / len(valid) if valid else np.nan
    g2 = pd.notna(share) and share >= 0.55
    g3 = abs(r[f"{pool}_val_icir"]) >= 0.1
    fails = []
    if not g1:
        fails.append("符号相反")
    if not g2:
        fails.append(f"年度一致率{share:.2f}<0.55" if pd.notna(share) else "无有效年度")
    if not g3:
        fails.append(f"|val ICIR|={r[f'{pool}_val_icir']:.3f}<0.1")
    return g1 and g2 and g3, ";".join(fails)


perf["main_pass"], perf["main_fail_reason"] = zip(*perf.apply(lambda r: gates(r, "main"), axis=1))
perf["backup_pass"], perf["backup_fail_reason"] = zip(*perf.apply(lambda r: gates(r, "backup"), axis=1))
perf["is_geo_control"] = perf["feature"].isin(GEO_CONTROLS)

# ---------------- 共线性关（train 段 spearman，主池幸存者范围内） ----------------
dfm = load_pool("main")
use_tr = dfm[dfm["seg"] == "train"]
survivors = perf.loc[perf["main_pass"], "feature"].tolist()
corr = use_tr[survivors].corr(method="spearman")

adj = (corr.abs() > 0.85).copy()
adj = pd.DataFrame(np.where(np.eye(len(adj), dtype=bool), False, adj.values),
                   index=adj.index, columns=adj.columns)
from scipy.sparse.csgraph import connected_components
n_comp, labels = connected_components(adj.values, directed=False)
clusters = []
seen = set()
for c in range(n_comp):
    members = [survivors[i] for i in np.where(labels == c)[0]]
    if len(members) > 1:
        clusters.append(members)
        seen.update(members)

val_icir = perf.set_index("feature")["main_val_icir"]
cluster_rows = []
drop_by_coll = {}
for i, mem in enumerate(clusters, 1):
    non_geo = [m for m in mem if m not in GEO_CONTROLS]
    pool_candidates = non_geo if non_geo else mem
    keep = max(pool_candidates, key=lambda m: abs(val_icir[m]))
    for m in mem:
        cluster_rows.append({
            "cluster_id": i, "feature": m, "family": FAMILY[m],
            "val_icir": val_icir[m], "kept": m == keep,
            "is_geo_control": m in GEO_CONTROLS,
            "max_abs_rho": adj.loc[m, mem].drop(index=m).abs().max(),
        })
        if m != keep and m not in GEO_CONTROLS:
            drop_by_coll[m] = f"共线性簇#{i}败给{keep}"
        if m in GEO_CONTROLS and m != keep:
            cluster_rows[-1]["kept"] = True  # 几何控制变量不淘汰
            cluster_rows[-1]["note"] = "几何控制变量，保留为控制"
coll_df = pd.DataFrame(cluster_rows)

# ---------------- 定档 ----------------
decisions, reasons = [], []
for _, r in perf.iterrows():
    f = r["feature"]
    if r["is_geo_control"]:
        decisions.append("几何控制变量")
        reasons.append("ATRN/VOL20/AMP20 与标签可达性机械相关，建模作控制变量，不按 IC 判优劣")
    elif not r["main_pass"]:
        decisions.append("淘汰")
        reasons.append("主池三关未过: " + r["main_fail_reason"])
    elif f in drop_by_coll:
        decisions.append("淘汰")
        reasons.append(drop_by_coll[f])
    elif r["backup_pass"]:
        decisions.append("A")
        reasons.append("双池双段三关皆过，共线性幸存")
    else:
        decisions.append("B")
        reasons.append("主池过、备池未过: " + r["backup_fail_reason"])
perf["decision"] = decisions
perf["decision_reason"] = reasons

perf.to_csv(f"{OUT_DIR}/per_feature_ic.csv", index=False, float_format="%.5f")
coll_df.to_csv(f"{OUT_DIR}/collinearity_clusters.csv", index=False, float_format="%.5f")
with open(f"{OUT_DIR}/run_meta.json", "w") as fh:
    json.dump(meta, fh, indent=1)

# ---------------- 逐年 IC 热图 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ym = results["main"]["yearly_hit"]
ym = ym.reindex(columns=YEARS)
order = perf.sort_values(["family", "main_val_icir"])["feature"].tolist()
M = ym.loc[order].values
fig, ax = plt.subplots(figsize=(16, 0.16 * len(order) + 2))
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(YEARS))); ax.set_xticklabels(YEARS, rotation=90, fontsize=7)
ax.set_yticks(range(len(order)))
TAG = {"A": "A", "B": "B", "淘汰": "X", "几何控制变量": "geo"}
ax.set_yticklabels([f"{f} [{TAG[perf.set_index('feature').loc[f,'decision']]}]" for f in order], fontsize=4.5)
ax.set_title("Yearly mean daily Rank IC (main pool, label=hit_N20_k2.0)")
fig.colorbar(im, ax=ax, shrink=0.3, label="Rank IC")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/yearly_ic_heatmap.png", dpi=140)
print("saved heatmap")

# 汇总打印
print(perf["decision"].value_counts())
print("\nA 档:", perf.loc[perf["decision"] == "A", "feature"].tolist())
print("\nB 档:", perf.loc[perf["decision"] == "B", "feature"].tolist())
print("\n几何控制:", perf.loc[perf["decision"] == "几何控制变量", "feature"].tolist())
print("\n共线性簇数:", len(clusters))
