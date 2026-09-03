#!/usr/bin/env python3
"""T7 复核项1: 层1~层3 自写重算（不调 feature_selection 的任何选择函数）。

允许复用的基础设施: feature_master（主表口径/相关矩阵）、train_eval_pipeline /
label_race（#25/#26 已复核管线）。选择逻辑（排序/符号/聚类）全部本文件自写。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
import feature_master as fm  # noqa: E402
import label_race as lr  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402

RACE = REPO / "v3_pipeline" / "reports" / "label_race"
OUT = REPO / "v3_pipeline" / "reports" / "feature_selection"
LABEL = "hit_N20_k2.0"
PARAMS = lr.grid_params(lr.GRID[13])     # nl=31, md=50, lr=0.05, ff=0.8


def load_frames():
    master = pd.read_parquet(RACE / "master_merged.parquet")
    parts = []
    for p in ("main", "backup"):
        t = pd.read_parquet(RACE / f"labels_race_{p}.parquet",
                            columns=["ts_code", "date", LABEL])
        t["pool"] = p
        parts.append(t)
    df = master.merge(pd.concat(parts, ignore_index=True),
                      on=["pool", "ts_code", "date"], how="left", validate="one_to_one")
    df = df[df["seg"].isin(["train", "val", "test"])].reset_index(drop=True)
    return df, tep.model_feature_columns(master)


def main():
    df, feat_cols = load_frames()
    sub = df[df[LABEL].notna()]
    train = sub[sub["seg"] == "train"].sort_values(["date", "ts_code", "event_id"],
                                                   kind="mergesort")
    val = sub[sub["seg"] == "val"].sort_values(["date", "ts_code", "event_id"],
                                               kind="mergesort")

    # 基模终模（自走 tep 公开函数）
    oof, best_iters = tep.time_series_oof(
        train[feat_cols], train[LABEL].to_numpy(np.float64),
        pd.to_datetime(train["date"]).to_numpy(), params=PARAMS)
    n_round = max(int(round(float(np.mean(best_iters)))), 1)
    booster = tep.fit_final_model(train[feat_cols], train[LABEL].to_numpy(np.float64),
                                  num_boost_round=n_round, params=PARAMS)
    print(f"基模: best_iters={best_iters}, 终模 {n_round} 轮")

    # ---- 层1 自写: mean|SHAP| 排序, 0 剔除, 平局特征名升序
    raw = booster.predict(val[feat_cols], pred_contrib=True)
    shap = np.asarray(raw[:, :-1], dtype=np.float64)
    imp = np.abs(shap).mean(axis=0)
    my = pd.DataFrame({"feature": feat_cols, "importance": imp})
    my["kept"] = my["importance"] > 0.0
    my = my.sort_values(["importance", "feature"], ascending=[False, True],
                        kind="mergesort").reset_index(drop=True)
    my["rank"] = np.arange(len(my))
    disk = pd.read_csv(OUT / "layer1_shap_importance.csv",
                       dtype={"importance": str})
    disk["importance"] = disk["importance"].map(float)   # float() 精确解析, 容差 0
    assert len(disk) == len(my) == len(feat_cols)
    assert (disk["feature"] == my["feature"]).all(), "层1 排序序不一致"
    assert np.array_equal(disk["importance"].to_numpy(), my["importance"].to_numpy()), \
        "层1 importance 逐位不一致"
    assert (disk["kept"] == my["kept"]).all() and (disk["rank"] == my["rank"]).all()
    kept1 = my[my["kept"]]["feature"].tolist()
    print(f"层1 重算逐位一致: {len(feat_cols)} -> {len(kept1)} (剔 {(~my['kept']).sum()})")

    # ---- 层2 自写: 分年度特征值~SHAP Pearson 符号
    years = pd.to_datetime(val["date"]).dt.year.to_numpy()
    pos = {c: i for i, c in enumerate(feat_cols)}
    rows = []
    for feat in kept1:
        j = pos[feat]
        x = val[feat].to_numpy(np.float64)
        s = shap[:, j]
        rec = {"feature": feat}
        signs = []
        for y in (2019, 2020, 2021, 2022):
            m = years == y
            xx, ss = x[m], s[m]
            ok = np.isfinite(xx) & np.isfinite(ss)
            if ok.sum() < 30 or xx[ok].std() == 0.0 or ss[ok].std() == 0.0:
                sg = 0
            else:
                r = float(np.corrcoef(xx[ok], ss[ok])[0, 1])
                sg = 0 if r == 0.0 else (1 if r > 0 else -1)
            rec[f"sign_{y}"] = sg
            signs.append(sg)
        rec["consistent"] = all(sg == signs[0] for sg in signs) and signs[0] != 0
        rows.append(rec)
    my2 = pd.DataFrame(rows)
    disk2 = pd.read_csv(OUT / "layer2_yearly_signs.csv")
    assert len(disk2) == len(my2)
    assert (disk2["feature"] == my2["feature"]).all()
    for y in (2019, 2020, 2021, 2022):
        assert (disk2[f"sign_{y}"] == my2[f"sign_{y}"]).all(), f"层2 sign_{y} 不一致"
    assert (disk2["consistent"] == my2["consistent"]).all()
    surv2 = my2[my2["consistent"]]["feature"].tolist()
    print(f"层2 重算逐位一致: {len(kept1)} -> {len(surv2)}")

    # ---- 层3 自写贪婪聚类（相关矩阵用 fm.pairwise_corr 基础设施）
    rank_of = {r["feature"]: int(r["rank"]) for _, r in my[my["kept"]].iterrows()}
    order = sorted(surv2, key=lambda c: (rank_of[c], c))
    tv = sub[sub["seg"].isin(["train", "val"])]
    corr = fm.pairwise_corr(tv[order].to_numpy(np.float64))
    pos_o = {c: i for i, c in enumerate(order)}
    rep_of, reps = {}, []
    for feat in order:
        home = None
        for rep in reps:
            r = corr[pos_o[feat], pos_o[rep]]
            if np.isfinite(r) and abs(r) >= 0.9:
                home = rep
                break
        rep_of[feat] = home if home is not None else feat
        if home is None:
            reps.append(feat)
    disk3 = pd.read_csv(OUT / "layer3_clusters.csv")
    assert len(disk3) == len(order)
    d3 = disk3.set_index("feature")
    for feat in order:
        assert d3.loc[feat, "representative"] == rep_of[feat], f"层3 {feat} 簇属不一致"
        assert bool(d3.loc[feat, "is_representative"]) == (rep_of[feat] == feat)
        assert int(d3.loc[feat, "rank"]) == rank_of[feat]
        if rep_of[feat] != feat:
            assert abs(d3.loc[feat, "corr_with_rep"]
                       - corr[pos_o[feat], pos_o[rep_of[feat]]]) < 1e-15
    print(f"层3 重算逐条一致: {len(surv2)} -> {len(reps)} 代表")
    print("项1 总判: PASS")


if __name__ == "__main__":
    main()
