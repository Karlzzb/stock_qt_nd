#!/usr/bin/env python3
"""T7 特征精选与模型定版驱动（issue #27）：SHAP 五层精选 + 定版分数序列。

预登记正文见 issue #27 置顶评论（2026-09-03T04:07:40Z，先于本赛任何结果落盘）。
对象: 合并池（#26 正赛）x 当选标签 hit_N20_k2.0 x 当选配置（由 summary 读出断言）。
流程:
  层1 验证段 SHAP 重要性排序（零重要性剔除）
  层2 分年度符号一致性（漂移剔除）
  层3 相关簇去重（|corr|>=0.9 留代表）
  层4 拐点定容（K 阶梯全管线重训, log2(K)-头部五名精确率曲线垂距最远点）
  层5 独立复核（复核脚本随产物归档于 t7_review/, 记录见 t7_review_record.md）
定版: K* 那一跑产物即定版模型（model.txt / calibrator.json）,
分数序列 scores_final.parquet 覆盖 train(折外)/val/test 三段, test 段 y 恒 NaN。
复现断言（不可旁路）: 主跑与 --repro-check 复跑共用同一 run_chain 实现,
全新进程全链（层1~层4 重导 + 定版打分）重算后与落盘产物逐位一致才生效。

输出（v3_pipeline/reports/feature_selection/）: 见 issue #27 预登记清单。
用法: python v3_pipeline/scripts/run_feature_selection.py
      python v3_pipeline/scripts/run_feature_selection.py --repro-check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_selection as fsel  # noqa: E402
import label_race as lr  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402

MASTER_DIR = REPO / "v3_pipeline" / "reports" / "feature_master"
RACE_DIR = REPO / "v3_pipeline" / "reports" / "label_race"
CALENDAR_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "feature_selection"
PROGRESS = OUT_DIR / "progress.log"
POOL = "merged"          # #26 正赛（兜底已触发）; 主池对照不做精选
PREREG_TS = "2026-09-03T04:07:40Z"   # issue #27 置顶评论 createdAt
ELBOW_RULE = ("x=log2(K), y=val_precision_at_5_dayavg, 首末连线垂距最远, "
              "平局取小 K")
SORT_KEYS = ["date", "ts_code", "event_id"]

CURVE_METRIC_COLS = ("final_num_boost_round", "val_average_precision",
                     "val_precision_at_5_dayavg", "val_precision_at_5_eventavg",
                     "train_oof_average_precision", "train_oof_precision_at_5_dayavg")


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_calendar() -> np.ndarray:
    d = pd.read_parquet(CALENDAR_PATH, columns=["trade_date"])
    return np.sort(pd.to_datetime(d["trade_date"].astype(str)).unique())


def load_model_frames() -> tuple[pd.DataFrame, list[str]]:
    """合并池主表 ⋈ 标签表; 返回 (全段帧, 特征列)。

    全段帧含 seg in (train, val, test) 全部行（标签 NaN 行保留——分数序列要打分）;
    特征列走 feature_master 权威口径 + 泄漏排除断言。
    """
    master = pd.read_parquet(RACE_DIR / "master_merged.parquet")
    parts = []
    for p in ("main", "backup"):
        t = pd.read_parquet(RACE_DIR / f"labels_race_{p}.parquet",
                            columns=["ts_code", "date", lr.SNIPER_LABEL])
        t["pool"] = p
        parts.append(t)
    labels = pd.concat(parts, ignore_index=True)
    df = master.merge(labels, on=["pool", "ts_code", "date"], how="left",
                      validate="one_to_one")
    df = df[df["seg"].isin(["train", "val", "test"])].reset_index(drop=True)
    feat_cols = tep.model_feature_columns(master)
    return df, feat_cols


def load_summary() -> pd.DataFrame:
    """summary_merged.csv 以字符串口径读入（规避 pandas 快解析器 double 解析偏差,
    float() 精确解析, 见 #26 复核记录）。"""
    return pd.read_csv(RACE_DIR / "summary_merged.csv", dtype=str)


def resolve_winner_from_disk() -> tuple[str, int, dict]:
    with (RACE_DIR / "adjudication_merged.json").open(encoding="utf-8") as f:
        adj = json.load(f)
    return fsel.resolve_winner(load_summary(), adj)


def build_scores(df: pd.DataFrame, feat_cols: list[str], label_col: str,
                 oof: np.ndarray, oof_event_ids: np.ndarray,
                 calibrator: tep.SquaredLogitCalibrator, booster) -> pd.DataFrame:
    """三段分数: train=折外概率(无折外记 NaN), val/test=终模+校准; test 段 y=NaN。

    oof 与 oof_event_ids 须同序对齐（label_race.run_single_config 口径:
    train 按 (date,ts_code,event_id) mergesort 后的行序）。
    """
    assert len(np.unique(oof_event_ids)) == len(oof_event_ids), "折外键不唯一"
    keys = df[["pool", "ts_code", "date", "event_id", "seg"]].copy()
    prob = np.full(len(df), np.nan)
    # train 折外概率按 event_id 对齐（首块与标签 NaN 行无折外, 保持 NaN）
    oof_map = pd.Series(oof, index=oof_event_ids)
    is_train = (df["seg"] == "train").to_numpy()
    prob[is_train] = df.loc[is_train, "event_id"].map(oof_map).to_numpy()
    # val/test 终模+校准
    vt_mask = df["seg"].isin(["val", "test"]).to_numpy()
    prob[vt_mask] = calibrator.predict(booster.predict(df.loc[vt_mask, feat_cols]))
    y = df[label_col].to_numpy(dtype=np.float64)
    y[df["seg"].eq("test").to_numpy()] = np.nan
    return fsel.assemble_scores(keys, prob, y)


# ------------------------------------------------------------- 五层全链（主跑/复跑同一实现）
def run_chain(df: pd.DataFrame, feat_cols: list[str], label: str,
              params: dict, logger, t0: float) -> dict:
    """层1~层4 精选 + 定版打分全链。返回全部中间表与定版产物（不落盘）。"""
    sub = df[df[label].notna()]
    train = sub[sub["seg"] == "train"]
    val = sub[sub["seg"] == "val"]
    train_sorted = train.sort_values(SORT_KEYS, kind="mergesort")
    val_sorted = val.sort_values(SORT_KEYS, kind="mergesort")
    logger(f"[链] 标签非 NaN: train {len(train)} / val {len(val)}")

    # 基模（全特征）+ T6 折外/指标锚定
    row_base, art_base = lr.run_single_config(train, val, feat_cols, label, params)
    stored_oof = pd.read_parquet(RACE_DIR / f"oof_merged_{label}.parquet")
    summary_row = load_summary()
    summary_row = summary_row[summary_row["candidate"] == label].iloc[0]
    stored_col = f"config_{int(summary_row['config_id'])}"
    assert stored_col in stored_oof.columns, f"T6 折外产物缺列 {stored_col}"
    assert (stored_oof["event_id"].to_numpy()
            == train_sorted["event_id"].to_numpy()).all(), "T6 折外 event_id 序不一致"
    assert np.array_equal(stored_oof[stored_col].to_numpy(), art_base["oof"],
                          equal_nan=True), "基模 OOF 与 T6 落盘折外逐位不一致"
    assert float(summary_row["val_precision_at_5_dayavg"]) == \
        row_base["val_precision_at_5_dayavg"], "基模 val 指标与 T6 汇总不一致"
    logger(f"[链] 基模复锚通过: val 头部五名精确率日加权="
           f"{row_base['val_precision_at_5_dayavg']:.4f}")
    booster_base = tep.fit_final_model(
        train_sorted[feat_cols], train_sorted[label].to_numpy(dtype=np.float64),
        num_boost_round=tep.final_num_boost_round(art_base["best_iters"]),
        params=params)

    # 层1 重要性排序
    shap_val = fsel.shap_values(booster_base, val_sorted[feat_cols])
    rank_table = fsel.layer1_rank(shap_val, feat_cols)
    kept1 = rank_table[rank_table["kept"]]["feature"].tolist()
    logger(f"[层1] {len(feat_cols)} -> {len(kept1)} "
           f"(零重要性剔 {int((~rank_table['kept']).sum())})")

    # 层2 分年度符号一致性
    pos_in_feat = {c: i for i, c in enumerate(feat_cols)}
    shap_kept = shap_val[:, [pos_in_feat[c] for c in kept1]]  # 列序对齐 kept1
    years = pd.to_datetime(val_sorted["date"]).dt.year.to_numpy()
    signs = fsel.layer2_yearly_signs(val_sorted[kept1], shap_kept, years, kept1)
    surv2 = signs[signs["consistent"]]["feature"].tolist()
    logger(f"[层2] {len(kept1)} -> {len(surv2)} (漂移剔 {len(kept1) - len(surv2)})")

    # 层3 相关簇去重
    rank_of = {r["feature"]: int(r["rank"])
               for _, r in rank_table[rank_table["kept"]].iterrows()}
    tv = sub[sub["seg"].isin(["train", "val"])]
    reps, clusters, _ = fsel.layer3_clusters(tv[surv2], surv2, rank_of)
    logger(f"[层3] {len(surv2)} -> {len(reps)} (簇吸收 {len(surv2) - len(reps)}, "
           f"阈值 {fsel.CORR_CLUSTER_THRESHOLD})")

    # 层4 拐点定容（k_ladder 内含 N<5 不开路断言, 预登记口径）
    ladder = fsel.k_ladder(len(reps))
    logger(f"[层4] K 阶梯: {ladder}")
    curve_rows, runs = [], {}
    for k in ladder:
        topk = reps[:k]
        row_k, art_k = lr.run_single_config(train, val, topk, label, params)
        curve_rows.append({"k": k, **{c: row_k[c] for c in CURVE_METRIC_COLS}})
        runs[k] = (row_k, art_k, topk)
        logger(f"[心跳] 层4 K={k}: val 头部五名精确率日加权="
               f"{row_k['val_precision_at_5_dayavg']:.4f} "
               f"(耗时 {round(time.time() - t0)}s)")
    curve = pd.DataFrame(curve_rows).sort_values("k").reset_index(drop=True)
    elbow = fsel.find_elbow(curve)
    k_star = elbow["k_star"]
    logger(f"[层4] 拐点 K*={k_star} (N_surviving={len(reps)})")

    # 定版模型 + 三段分数
    row_star, art_star, feat_star = runs[k_star]
    n_round_star = tep.final_num_boost_round(art_star["best_iters"])
    booster_star = tep.fit_final_model(
        train_sorted[feat_star], train_sorted[label].to_numpy(dtype=np.float64),
        num_boost_round=n_round_star, params=params)
    calibrator_star = tep.SquaredLogitCalibrator().fit(
        art_star["oof"][art_star["oof_mask"]],
        train_sorted[label].to_numpy(dtype=np.float64)[art_star["oof_mask"]])
    scores = build_scores(df, feat_star, label, art_star["oof"],
                          train_sorted["event_id"].to_numpy(),
                          calibrator_star, booster_star)
    logger(f"[链] 定版: {len(feat_star)} 特征, 终模 {n_round_star} 轮, "
           f"分数 {len(scores)} 行")
    return {"rank_table": rank_table, "signs": signs, "clusters": clusters,
            "curve": curve, "curve_rows": curve_rows, "elbow": elbow,
            "ladder": ladder, "k_star": k_star, "feat_star": feat_star,
            "rank_of": rank_of, "kept1": kept1, "surv2": surv2, "reps": reps,
            "row_base": row_base, "row_star": row_star,
            "n_round_star": n_round_star, "booster_star": booster_star,
            "calibrator_star": calibrator_star, "scores": scores}


# ------------------------------------------------------------- 复现校验（全新进程, 全链重导）
def _read_float_exact(path: Path, str_cols: tuple) -> pd.DataFrame:
    """CSV 字符串读入 + float() 精确解析指定列（容差 0 口径, 见 #26 复核记录）。"""
    t = pd.read_csv(path, dtype={c: str for c in str_cols})
    for c in str_cols:
        t[c] = t[c].map(float)
    return t


def verify() -> None:
    """--repro-check: 全链重算, 与落盘产物逐位对比（任一处不一致即非零退出）。"""
    label, config_id, params = resolve_winner_from_disk()
    with (OUT_DIR / "final_features.json").open(encoding="utf-8") as f:
        final = json.load(f)
    assert final["label"] == label, "落盘特征清单标签与裁决不一致"
    assert final["config_id"] == config_id, "落盘特征清单 config_id 与裁决不一致"
    for k in lr.GRID_KEYS:
        assert float(final["params"][k]) == float(params[k]), \
            f"落盘特征清单超参 {k} 与网格参数不一致"
    df, feat_cols = load_model_frames()
    chain = run_chain(df, feat_cols, label, params, print, time.time())

    # 层1~层4 落盘表逐位
    d1 = _read_float_exact(OUT_DIR / "layer1_shap_importance.csv", ("importance",))
    r1 = chain["rank_table"]
    assert (d1["feature"] == r1["feature"]).all() and \
        (d1["rank"] == r1["rank"]).all() and (d1["kept"] == r1["kept"]).all()
    assert np.array_equal(d1["importance"].to_numpy(), r1["importance"].to_numpy())
    d2 = pd.read_csv(OUT_DIR / "layer2_yearly_signs.csv")
    pd.testing.assert_frame_equal(d2, chain["signs"], check_dtype=False)
    d3 = _read_float_exact(OUT_DIR / "layer3_clusters.csv", ("corr_with_rep",))
    r3 = chain["clusters"]
    assert (d3[["feature", "representative", "rank"]]
            == r3[["feature", "representative", "rank"]]).all().all()
    assert (d3["is_representative"] == r3["is_representative"]).all()
    assert np.array_equal(d3["corr_with_rep"].to_numpy(),
                          r3["corr_with_rep"].to_numpy())
    d4 = _read_float_exact(OUT_DIR / "layer4_curve.csv",
                           tuple(c for c in CURVE_METRIC_COLS
                                 if c != "final_num_boost_round"))
    assert (d4["k"] == chain["curve"]["k"]).all() and \
        (d4["final_num_boost_round"] == chain["curve"]["final_num_boost_round"]).all()
    for c in CURVE_METRIC_COLS:
        if c == "final_num_boost_round":
            continue
        assert np.array_equal(d4[c].to_numpy(), chain["curve"][c].to_numpy()), \
            f"层4 曲线 {c} 逐位不一致"
    with (OUT_DIR / "layer4_elbow.json").open(encoding="utf-8") as f:
        ej = json.load(f)
    assert ej["k_star"] == chain["k_star"] and ej["ladder"] == chain["ladder"]
    assert ej["distances"] == chain["elbow"]["distances"]
    assert [r["feature"] for r in final["features"]] == chain["feat_star"]
    print("[repro-check] 层1~层4 落盘表与拐点逐位一致（全链重导）")

    # 定版模型 / 校准层 / 分数序列逐位
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        chain["booster_star"].save_model(tmp.name)
        assert Path(tmp.name).read_bytes() == (OUT_DIR / "model.txt").read_bytes(), \
            "model.txt 逐字节不一致"
    with (OUT_DIR / "calibrator.json").open(encoding="utf-8") as f:
        calib = json.load(f)
    cal = chain["calibrator_star"]
    assert calib["coef_p"] == float(cal.coef_[0]) and \
        calib["coef_p2"] == float(cal.coef_[1]) and \
        calib["intercept"] == cal.intercept_
    stored = pd.read_parquet(OUT_DIR / "scores_final.parquet")
    scores = chain["scores"]
    assert list(stored.columns) == list(scores.columns), "分数列结构不一致"
    assert stored["event_id"].tolist() == scores["event_id"].tolist(), "event_id 序不一致"
    for col in ("prob", "y"):
        assert np.array_equal(stored[col].to_numpy(dtype=np.float64),
                              scores[col].to_numpy(dtype=np.float64), equal_nan=True), \
            f"分数列 {col} 逐位不一致"
    print("[repro-check] PASS: 全链重导 + 定版产物（模型/校准层/三段分数）逐位一致")


# ------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-check", action="store_true")
    args = ap.parse_args()
    if args.repro_check:
        verify()
        return

    t0 = time.time()
    log("[T7 特征精选] 开工: 合并池 x hit_N20_k2.0, 五层流程")

    # ---------------------------------------------------------- 1. 数据与当选配置
    df, feat_cols = load_model_frames()
    cal = load_calendar()
    tep.assert_segment_integrity(df[["date", "seg"]], cal)
    seg_counts = {k: int(v) for k, v in df["seg"].value_counts().items()}
    label, config_id, params = resolve_winner_from_disk()
    log(f"[阶段1] 全段帧 {len(df)} 行, 特征 {len(feat_cols)} 列; "
        f"当选 {label} config_id={config_id}; 段计数 {seg_counts}")

    # ---------------------------------------------------------- 2. 五层全链
    chain = run_chain(df, feat_cols, label, params, log, t0)

    # ---------------------------------------------------------- 3. 产物落盘
    chain["rank_table"].to_csv(OUT_DIR / "layer1_shap_importance.csv", index=False)
    chain["signs"].to_csv(OUT_DIR / "layer2_yearly_signs.csv", index=False)
    chain["clusters"].to_csv(OUT_DIR / "layer3_clusters.csv", index=False)
    chain["curve"].to_csv(OUT_DIR / "layer4_curve.csv", index=False)
    with (OUT_DIR / "layer4_elbow.json").open("w", encoding="utf-8") as f:
        json.dump({**chain["elbow"], "ladder": chain["ladder"], "rule": ELBOW_RULE,
                   "curve": chain["curve_rows"]}, f, ensure_ascii=False, indent=2)
    dictionary = pd.read_csv(MASTER_DIR / "master_dictionary.csv")
    cn_of = dict(zip(dictionary["column"], dictionary["cn_name"]))
    feat_star = chain["feat_star"]
    missing_cn = [c for c in feat_star if not cn_of.get(c)]
    assert not missing_cn, f"终选特征缺中文全名: {missing_cn[:5]}"
    final_doc = {
        "issue": 27, "pool": POOL, "label": label, "config_id": config_id,
        "params": {k: params[k] for k in lr.GRID_KEYS},
        "n_features": len(feat_star),
        "features": [{"feature": c, "rank": chain["rank_of"][c],
                      "cn_name": cn_of[c]} for c in feat_star],
    }
    with (OUT_DIR / "final_features.json").open("w", encoding="utf-8") as f:
        json.dump(final_doc, f, ensure_ascii=False, indent=2)
    chain["booster_star"].save_model(str(OUT_DIR / "model.txt"))
    cal = chain["calibrator_star"]
    with (OUT_DIR / "calibrator.json").open("w", encoding="utf-8") as f:
        json.dump({"form": "logistic(p, p^2)", "coef_p": float(cal.coef_[0]),
                   "coef_p2": float(cal.coef_[1]), "intercept": cal.intercept_},
                  f, indent=2)
    chain["scores"].to_parquet(OUT_DIR / "scores_final.parquet", index=False)
    n_test = int((chain["scores"]["seg"] == "test").sum())
    log(f"[阶段3] 产物落盘完成: 终选 {len(feat_star)} 特征, 分数 {n_test} 行 test")

    # ---------------------------------------------------------- 4. 复现断言（全新进程全链重导, 不可旁路）
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--repro-check"], capture_output=True, text=True)
    log(f"[阶段4] 复现断言: {'PASS' if r.returncode == 0 else 'FAIL'}")
    if r.returncode != 0:
        log(f"[阶段4] repro-check 输出:\n{r.stdout}\n{r.stderr}")
    assert r.returncode == 0, "复现断言失败"

    # ---------------------------------------------------------- 5. 台账
    row_star, row_base = chain["row_star"], chain["row_base"]
    results = {
        "issue": 27, "pool": POOL, "label": label, "config_id": config_id,
        "preregistration_ts": PREREG_TS,
        "n_features_total": len(feat_cols),
        "layers": {
            "layer1": {"in": len(feat_cols), "out": len(chain["kept1"]),
                       "removed_zero_importance":
                           len(feat_cols) - len(chain["kept1"])},
            "layer2": {"in": len(chain["kept1"]), "out": len(chain["surv2"]),
                       "removed_drift": len(chain["kept1"]) - len(chain["surv2"])},
            "layer3": {"in": len(chain["surv2"]), "out": len(chain["reps"]),
                       "cluster_absorbed": len(chain["surv2"]) - len(chain["reps"]),
                       "threshold": fsel.CORR_CLUSTER_THRESHOLD},
            "layer4": {"ladder": chain["ladder"], "k_star": chain["k_star"],
                       "rule": ELBOW_RULE},
        },
        "final": {
            "n_features": len(feat_star),
            "final_num_boost_round": chain["n_round_star"],
            "val_average_precision": row_star["val_average_precision"],
            "val_precision_at_5_dayavg": row_star["val_precision_at_5_dayavg"],
            "base_all_features_val_precision_at_5_dayavg":
                row_base["val_precision_at_5_dayavg"],
        },
        "seg_counts": seg_counts,
        "assertions": {
            "segment_embargo": "PASS (assert_segment_integrity)",
            "leakage_exclusion": "PASS (model_feature_columns 排除模式零命中)",
            "base_oof_anchor_t6": "PASS (基模 OOF 与 T6 落盘折外逐位一致)",
            "base_val_metric_anchor_t6": "PASS (基模 val 指标与 T6 汇总逐位一致)",
            "scores_repro_fresh_process":
                "PASS (--repro-check 全新进程全链重导, 层表/拐点/模型/分数逐位一致)",
            "test_zero_metric": (f"PASS (test {n_test} 行仅出分数, y 全 NaN, "
                                 "零指标零逐行统计)"),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with (OUT_DIR / "selection_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[完成] 耗时 {results['elapsed_sec']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
