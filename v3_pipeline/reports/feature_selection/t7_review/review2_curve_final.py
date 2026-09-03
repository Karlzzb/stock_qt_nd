#!/usr/bin/env python3
"""T7 复核项2: 层4 拐点自写重算 + K=25/K=10 独立重跑逐位对比 + 终选清单与定版产物核查。"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
import label_race as lr  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402

RACE = REPO / "v3_pipeline" / "reports" / "label_race"
OUT = REPO / "v3_pipeline" / "reports" / "feature_selection"
LABEL = "hit_N20_k2.0"
PARAMS = lr.grid_params(lr.GRID[13])


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


def elbow_selfwrite(curve: pd.DataFrame) -> int:
    ks = curve["k"].to_numpy(np.int64)
    ps = curve["val_precision_at_5_dayavg"].to_numpy(np.float64)
    xs = np.log2(ks.astype(np.float64))
    vx, vy = xs[-1] - xs[0], ps[-1] - ps[0]
    norm = float(np.hypot(vx, vy))
    dists = np.abs(vx * (ps - ps[0]) - vy * (xs - xs[0])) / norm
    tol = 1e-12 * max(1.0, float(dists.max()))
    tied = np.flatnonzero(dists >= dists.max() - tol)
    return int(ks[tied[0]])


def main():
    # ---- 拐点重算（自写几何, 字符串读 CSV + float() 精确解析）
    curve = pd.read_csv(OUT / "layer4_curve.csv", dtype=str)
    ks = [int(v) for v in curve["k"]]
    assert ks == sorted(ks) and len(set(ks)) == len(ks)
    with (OUT / "layer4_elbow.json").open(encoding="utf-8") as f:
        ej = json.load(f)
    assert ej["ladder"] == ks, "阶梯落盘与曲线行不一致"
    curve_f = pd.DataFrame({"k": ks,
                            "val_precision_at_5_dayavg":
                                [float(v) for v in curve["val_precision_at_5_dayavg"]]})
    k_star = elbow_selfwrite(curve_f)
    assert k_star == ej["k_star"], f"拐点重算 {k_star} != 落盘 {ej['k_star']}"
    print(f"层4 拐点重算一致: K*={k_star}")

    # ---- 终选清单核查: final_features == 层3代表按层1名次前 K*
    rank = pd.read_csv(OUT / "layer1_shap_importance.csv")
    kept_rank = {r["feature"]: int(r["rank"])
                 for _, r in rank[rank["kept"]].iterrows()}
    clusters = pd.read_csv(OUT / "layer3_clusters.csv")
    reps = clusters[clusters["is_representative"]].sort_values("rank")["feature"].tolist()
    with (OUT / "final_features.json").open(encoding="utf-8") as f:
        final = json.load(f)
    feat_final = [r["feature"] for r in final["features"]]
    assert feat_final == reps[:k_star], "终选清单 != 层3代表前 K*（自写推导）"
    assert [r["rank"] for r in final["features"]] == [kept_rank[c] for c in feat_final]
    assert final["n_features"] == len(feat_final) == k_star
    assert all(r["cn_name"] for r in final["features"]), "终选特征缺中文全名"
    print(f"终选清单核查一致: {len(feat_final)} 特征, 名次 "
          f"{[kept_rank[c] for c in feat_final][:5]}...")

    # ---- K=25 与 K=10 独立重跑（自走管线）, 与曲线行逐位对比
    df, _ = load_frames()
    sub = df[df[LABEL].notna()]
    train = sub[sub["seg"] == "train"]
    val = sub[sub["seg"] == "val"]
    for k in (k_star, 10):
        row, art = lr.run_single_config(train, val, reps[:k], LABEL, PARAMS)
        crow = curve[curve["k"] == str(k)].iloc[0]
        for col in ("val_average_precision", "val_precision_at_5_dayavg",
                    "val_precision_at_5_eventavg", "train_oof_average_precision",
                    "train_oof_precision_at_5_dayavg"):
            assert float(crow[col]) == row[col], f"K={k} {col} 逐位不一致"
        assert int(crow["final_num_boost_round"]) == row["final_num_boost_round"]
        print(f"K={k} 独立重跑逐位一致 (dayavg={row['val_precision_at_5_dayavg']}, "
              f"轮数 {row['final_num_boost_round']})")
        if k == k_star:
            art_star, row_star = art, row

    # ---- 定版产物核查: model.txt / calibrator.json / scores
    booster = lgb.Booster(model_file=str(OUT / "model.txt"))
    assert booster.num_trees() == row_star["final_num_boost_round"]
    assert booster.feature_name() == feat_final, "model.txt 特征名与终选清单不一致"
    with (OUT / "calibrator.json").open(encoding="utf-8") as f:
        calib = json.load(f)
    train_sorted = train.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    y_tr = train_sorted[LABEL].to_numpy(np.float64)
    cal = tep.SquaredLogitCalibrator().fit(art_star["oof"][art_star["oof_mask"]],
                                           y_tr[art_star["oof_mask"]])
    assert cal.coef_[0] == calib["coef_p"] and cal.coef_[1] == calib["coef_p2"]
    assert cal.intercept_ == calib["intercept"]
    print(f"定版模型核查: {booster.num_trees()} 轮, 校准系数重拟合逐位一致")

    scores = pd.read_parquet(OUT / "scores_final.parquet")
    assert list(scores.columns) == ["pool", "ts_code", "date", "event_id", "seg",
                                    "prob", "y"]
    assert not scores["event_id"].duplicated().any()
    assert scores.loc[scores["seg"] == "test", "y"].isna().all(), "test 段 y 非 NaN!"
    p = scores["prob"]
    assert ((p.dropna() >= 0) & (p.dropna() <= 1)).all()
    # val 段抽验: 用 model.txt + 校准层重算 500 行逐位对比
    val_rows = df[df["seg"] == "val"].head(500)
    reprob = cal.predict(booster.predict(val_rows[feat_final]))
    got = scores[scores["seg"] == "val"]["prob"].head(500).to_numpy()
    assert np.array_equal(reprob, got), "val 分数抽验逐位不一致"
    seg_counts = scores["seg"].value_counts().to_dict()
    print(f"分数序列核查: {len(scores)} 行 {seg_counts}, val 前 500 行逐位一致")
    print("项2 总判: PASS")


if __name__ == "__main__":
    main()
