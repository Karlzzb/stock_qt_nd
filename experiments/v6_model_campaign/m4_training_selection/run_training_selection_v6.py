#!/usr/bin/env python3
"""M4 训练管线与 SHAP 五层精选驱动(战役 #47 M4,issue #51;预登记 = 同目录 README.md,冻结)。

对象:ALL 全池 96,577 事件 × 当选标签 net_ret_60d(次日开盘入场净收益幅度回归标签,
60 个交易日视野)× 当选配置 config_id=24(断言读出,禁止手抄,见 resolve_winner_v6)。
回归头管线(M3 §五.3 钉死):无校准层,排序与指标一律用预测原值。

流程(预登记 §四/§五):
  基模(全 1,997 特征,五折折外 → 终模均值轮数)+ 对 M3 落盘折外 config_24 列逐位锚定
  层1 验证段 SHAP 重要性排序(零重要性剔除)
  层2 分年度符号一致性(年份清单 v6 适配 = 2020/2021/2022/2023,漂移剔除)
  层3 相关簇去重(|corr|>=0.9 留代表)
  层4 拐点定容(K_LADDER 原序,每 K 完整基模管线,评估指标 = M3 主指标
      验证段头部五名净笔均日加权,回归头预测原值排序;find_elbow 原样,列名适配)
  定版 = K* 那一跑管线产物;三段分数序列(96,577 行,event_id/ts_code/date/seg/score):
      train = 五折折外原值(首块与标签 NaN 行 NaN),
      val/test/embargo/pre2001 = 定版终模预测原值;seg 照标;test 零指标零逐行统计。
复现断言(不可旁路):主跑后全新进程 --repro-check 全链重导,层表/拐点/终选清单/
  model.txt 字节/分数数组逐位一致;分数双跑 md5 比对一致记入台账。

输出(experiments/v6_model_campaign/m4_training_selection/):见 README §八。
用法:
  python run_training_selection_v6.py --smoke        基模冒烟(对账 M3 config_24)
  python run_training_selection_v6.py                全量(五层精选 + 定版 + 复现断言)
  python run_training_selection_v6.py --repro-check  全链重导逐位校验(由主跑子进程调用)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
sys.path.insert(0, str(REPO / "experiments" / "v6_model_campaign" / "m3_label_race"))

import feature_selection as fsel  # noqa: E402  v5 五层精选纯函数层(原样复用)
import label_race as lr  # noqa: E402  36 组预登记网格原序
import train_eval_pipeline as tep  # noqa: E402  训练原语四件套
from run_label_race_v6 import top5_net  # noqa: E402  M3 主指标实现(与裁决链同型)

MASTER_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_v6.parquet"
DICT_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_dictionary_v6.csv"
RACE_DIR = REPO / "experiments" / "v6_model_campaign" / "m3_label_race"
LABELS_PATH = RACE_DIR / "labels_v6.parquet"
CALENDAR_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"
OUT_DIR = SCRIPT_DIR
PROGRESS = OUT_DIR / "progress.log"

LABEL = "net_ret_60d"
LABEL_CN = "次日开盘入场净收益幅度回归标签(60 个交易日视野)"
YEAR_LIST = (2020, 2021, 2022, 2023)   # 层2 年份清单 v6 适配(v6 val 段覆盖年份)
SORT_KEYS = ["date", "ts_code", "event_id"]
ELBOW_RULE = ("x=log2(K), y=验证段头部五名净笔均(日加权)(M3 主指标,回归头预测原值排序), "
              "首末连线垂距最远, 平局取小 K, 末点最远取末点")
CURVE_FLOAT_COLS = ("val_top5_net_dayavg", "val_top5_net_cluster_t",
                    "val_top5_hit_eventavg", "train_oof_top5_net_dayavg",
                    "train_oof_top5_net_cluster_t", "train_oof_top5_hit_eventavg")


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [run_training_selection_v6] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_calendar() -> np.ndarray:
    d = pd.read_parquet(CALENDAR_PATH, columns=["trade_date"])
    return np.sort(pd.to_datetime(d["trade_date"].astype(str)).unique())


def _read_csv_exact(path: Path, float_cols: tuple) -> pd.DataFrame:
    """M3 落盘 CSV 字符串读入 + float() 精确解析(容差 0 口径;规避 pandas 快解析器
    double 解析偏差,v5 #26 复核记录同款;adjudication_v6.json 的 winner 主指标系
    M3 汇总时经快解析器回读所得,与精确解析值差 1 ulp,交叉断言用容差 1e-15,
    修订记录第 1 条)。"""
    t = pd.read_csv(path, dtype={c: str for c in float_cols})
    for c in float_cols:
        t[c] = t[c].map(float)
    return t


def load_cn_names() -> dict:
    """特征中文全称词典(命名全称纪律;status=='kept' 恰 1,997 行)。"""
    d = pd.read_csv(DICT_PATH)
    kept = d[d["status"] == "kept"]
    assert len(kept) == 1997, f"词典 kept 行数 {len(kept)} != 1997"
    assert kept["cn_name"].notna().all(), "词典 kept 行存在空中文全称"
    return dict(zip(kept["column"], kept["cn_name"]))


# ---------------------------------------------------------------- 当选解析(断言守护)
def resolve_winner_v6() -> tuple[str, int, dict, dict]:
    """从 M3 落盘产物三重读出当选标签与配置并互证(预登记 §三,不接受手抄)。

    返回 (label, config_id, grid超参, 完整 LightGBM 参数)。
    """
    with (RACE_DIR / "adjudication_v6.json").open(encoding="utf-8") as f:
        adj = json.load(f)
    winner = adj.get("winner")
    assert winner == LABEL, f"裁决 winner={winner} 非预登记当选标签 {LABEL}"
    assert adj.get("double_constraint_passed") is True, "M3 双约束未过"
    summary = _read_csv_exact(RACE_DIR / "summary_v6.csv", tuple(lr.GRID_KEYS))
    row = summary[summary["candidate"] == winner]
    assert len(row) == 1, f"summary 中 {winner} 行数 {len(row)} != 1"
    row = row.iloc[0]
    assert row["head"] == "regression", "当选头型非回归(本阶段口径 = 回归头无校准)"
    config_id = int(row["config_id"])
    expect = lr.GRID[config_id]
    got = {k: row[k] for k in lr.GRID_KEYS}
    for k, v in expect.items():
        assert float(got[k]) == float(v), \
            f"summary 第 {config_id} 行超参 {k}={got[k]} 与预登记网格 {v} 不符"
    metrics = _read_csv_exact(RACE_DIR / f"metrics_v6_{winner}.csv",
                              ("val_top5_net_dayavg", "val_top5_net_cluster_t"))
    mrow = metrics[metrics["config_id"] == config_id]
    assert len(mrow) == 1, f"metrics 中 config_id={config_id} 行数 {len(mrow)} != 1"
    mrow = mrow.iloc[0]
    assert abs(float(mrow["val_top5_net_dayavg"])
               - float(adj["winner_val_top5_net_dayavg"])) <= 1e-15, \
        "metrics 行主指标与裁决书偏差超 1 ulp 容差(1e-15)"
    assert abs(float(mrow["val_top5_net_cluster_t"])
               - float(adj["winner_val_top5_net_cluster_t"])) <= 1e-15, \
        "metrics 行 cluster_t 与裁决书偏差超 1 ulp 容差(1e-15)"
    params = dict(tep.DEFAULT_LGBM_PARAMS)
    params.update(expect)
    params["objective"] = "regression"   # M3 预登记 §五.2 钉死:回归头
    params["metric"] = "rmse"
    return winner, config_id, expect, params


# ---------------------------------------------------------------- 数据装载
def load_frames() -> tuple[pd.DataFrame, list[str]]:
    """主表 ⋈ 标签(event_id 一对一);返回 (全量帧含全 96,577 行, 特征列)。

    帧列 = 元数据 + entry_date + net_ret_60d + 1,997 特征;选择动作只在 train/val,
    test 行只供定版打分,不携任何指标。
    """
    master = pd.read_parquet(MASTER_PATH)
    feat_cols = tep.model_feature_columns(master)
    assert len(feat_cols) == 1997, f"特征列数 {len(feat_cols)} != 1997"
    assert not any(c.startswith("label") for c in feat_cols), "特征集混入 label_ 列"
    labels = pd.read_parquet(LABELS_PATH,
                             columns=["event_id", "ts_code", "date", "seg",
                                      "entry_date", LABEL])
    j = master[["event_id", "ts_code", "date", "seg"]].merge(
        labels[["event_id", "ts_code", "date", "seg"]], on="event_id",
        suffixes=("", "_lab"), validate="one_to_one")
    assert len(j) == len(master), "主表与标签表 event_id 非一一对应"
    assert (j["ts_code"] == j["ts_code_lab"]).all() and \
           (j["date"] == j["date_lab"]).all() and \
           (j["seg"] == j["seg_lab"]).all(), "event_id 两侧键/段不一致"
    df = master.merge(labels[["event_id", "entry_date", LABEL]],
                      on="event_id", how="left", validate="one_to_one")
    return df, feat_cols


# ---------------------------------------------------------------- 基模管线(回归头,无校准)
def base_pipeline(train_sorted: pd.DataFrame, val_sorted: pd.DataFrame,
                  feat_cols: list[str], params: dict) -> dict:
    """完整基模管线:五折时间序列折外 → 终模(均值轮数,下限 1)→ val 终模预测原值。

    train_sorted/val_sorted 须已按 (date, ts_code, event_id) mergesort 且标签非 NaN。
    """
    X_tr = train_sorted[feat_cols]
    y_tr = train_sorted[LABEL].to_numpy(dtype=np.float64)
    dates_tr = pd.to_datetime(train_sorted["date"]).to_numpy()
    oof, best_iters = tep.time_series_oof(X_tr, y_tr, dates_tr, params=params)
    n_round = tep.final_num_boost_round(best_iters)
    booster = tep.fit_final_model(X_tr, y_tr, num_boost_round=n_round, params=params)
    val_score = booster.predict(val_sorted[feat_cols])
    return {"oof": oof, "oof_mask": np.isfinite(oof), "best_iters": best_iters,
            "n_round": n_round, "booster": booster, "val_score": val_score}


def metric_rows(art: dict, train_sorted: pd.DataFrame,
                val_sorted: pd.DataFrame) -> dict:
    """train_oof 与 val 主指标行(M3 主指标同型:头部五名净笔均日加权 + cluster_t)。"""
    out: dict = {}
    val_ev = val_sorted[["date", "ts_code", "event_id", "entry_date", LABEL]].copy()
    val_ev["net_ret"] = val_ev[LABEL].to_numpy(np.float64)
    val_ev["score"] = art["val_score"]
    m_val = top5_net(val_ev)
    tr_ev = train_sorted.loc[art["oof_mask"],
                             ["date", "ts_code", "event_id", "entry_date", LABEL]].copy()
    tr_ev["net_ret"] = tr_ev[LABEL].to_numpy(np.float64)
    tr_ev["score"] = art["oof"][art["oof_mask"]]
    m_tr = top5_net(tr_ev)
    for tag, m in (("val", m_val), ("train_oof", m_tr)):
        out[f"{tag}_top5_net_dayavg"] = m["top5_net_dayavg"]
        out[f"{tag}_top5_net_cluster_t"] = m["top5_net_cluster_t"]
        out[f"{tag}_top5_hit_eventavg"] = m["top5_hit_eventavg"]
    return out


# ---------------------------------------------------------------- 五层全链(主跑/复跑同一实现)
def run_chain(df: pd.DataFrame, feat_cols: list[str], params: dict,
              cn_of: dict, logger, t0: float) -> dict:
    """基模锚定 + 层1~层4 + 定版打分全链。返回全部中间表与定版产物(不落盘)。"""
    sub = df[df[LABEL].notna()]
    train = sub[sub["seg"] == "train"]
    val = sub[sub["seg"] == "val"]
    train_sorted = train.sort_values(SORT_KEYS, kind="mergesort")
    val_sorted = val.sort_values(SORT_KEYS, kind="mergesort")
    assert train["seg"].eq("train").all() and val["seg"].eq("val").all()
    logger(f"[链] 标签非 NaN: train {len(train)} / val {len(val)}")

    # ---- 基模(全特征)+ 对 M3 落盘折外逐位锚定(不可旁路)
    art_base = base_pipeline(train_sorted, val_sorted, feat_cols, params)
    stored_oof = pd.read_parquet(RACE_DIR / f"oof_v6_{LABEL}.parquet")
    stored_col = f"config_{RESOLVE[1]}"
    assert stored_col in stored_oof.columns, f"M3 折外产物缺列 {stored_col}"
    assert (stored_oof["event_id"].to_numpy()
            == train_sorted["event_id"].to_numpy()).all(), "M3 折外 event_id 序不一致"
    assert np.array_equal(stored_oof[stored_col].to_numpy(), art_base["oof"],
                          equal_nan=True), "基模 OOF 与 M3 落盘折外逐位不一致"
    metrics = _read_csv_exact(RACE_DIR / f"metrics_v6_{LABEL}.csv",
                              ("val_top5_net_dayavg", "val_top5_net_cluster_t"))
    mrow = metrics[metrics["config_id"] == RESOLVE[1]].iloc[0]
    row_base = metric_rows(art_base, train_sorted, val_sorted)
    assert float(mrow["val_top5_net_dayavg"]) == row_base["val_top5_net_dayavg"], \
        "基模 val 主指标与 M3 metrics 不精确相等"
    assert float(mrow["val_top5_net_cluster_t"]) == row_base["val_top5_net_cluster_t"], \
        "基模 val cluster_t 与 M3 metrics 不精确相等"
    logger(f"[链] 基模复锚通过: val 头部五名净笔均(日加权)="
           f"{row_base['val_top5_net_dayavg']:.6f}, cluster_t="
           f"{row_base['val_top5_net_cluster_t']:.3f}, 终模 {art_base['n_round']} 轮"
           f"(耗时 {round(time.time() - t0)}s)")

    # ---- 层1 重要性排序(SHAP 行集 = val 段标签非 NaN 事件)
    shap_val = fsel.shap_values(art_base["booster"], val_sorted[feat_cols])
    rank_table = fsel.layer1_rank(shap_val, feat_cols)
    rank_table.insert(1, "cn_name", rank_table["feature"].map(cn_of))
    assert rank_table["cn_name"].notna().all(), "层1 表存在缺中文全称特征"
    kept1 = rank_table[rank_table["kept"]]["feature"].tolist()
    logger(f"[层1] {len(feat_cols)} -> {len(kept1)} "
           f"(零重要性剔 {int((~rank_table['kept']).sum())}; 耗时 {round(time.time() - t0)}s)")

    # ---- 层2 分年度符号一致性(v6 年份清单 2020~2023)
    pos_in_feat = {c: i for i, c in enumerate(feat_cols)}
    shap_kept = shap_val[:, [pos_in_feat[c] for c in kept1]]
    years = pd.to_datetime(val_sorted["date"]).dt.year.to_numpy()
    signs = fsel.layer2_yearly_signs(val_sorted[kept1], shap_kept, years, kept1,
                                     year_list=YEAR_LIST)
    signs.insert(1, "cn_name", signs["feature"].map(cn_of))
    assert signs["cn_name"].notna().all(), "层2 表存在缺中文全称特征"
    surv2 = signs[signs["consistent"]]["feature"].tolist()
    logger(f"[层2] {len(kept1)} -> {len(surv2)} (漂移剔 {len(kept1) - len(surv2)}; "
           f"耗时 {round(time.time() - t0)}s)")

    # ---- 层3 相关簇去重(train+val 标签非 NaN 行)
    rank_of = {r["feature"]: int(r["rank"])
               for _, r in rank_table[rank_table["kept"]].iterrows()}
    tv = sub[sub["seg"].isin(["train", "val"])]
    reps, clusters, _ = fsel.layer3_clusters(tv[surv2], surv2, rank_of)
    clusters.insert(1, "cn_name", clusters["feature"].map(cn_of))
    assert clusters["cn_name"].notna().all(), "层3 表存在缺中文全称特征"
    logger(f"[层3] {len(surv2)} -> {len(reps)} (簇吸收 {len(surv2) - len(reps)}, "
           f"阈值 {fsel.CORR_CLUSTER_THRESHOLD}; 耗时 {round(time.time() - t0)}s)")

    # ---- 层4 拐点定容(评估指标 = M3 主指标,每 K 完整基模管线)
    ladder = fsel.k_ladder(len(reps))
    logger(f"[层4] K 阶梯: {ladder}")
    curve_rows, runs = [], {}
    for k in ladder:
        topk = reps[:k]
        art_k = base_pipeline(train_sorted, val_sorted, topk, params)
        row_k = metric_rows(art_k, train_sorted, val_sorted)
        curve_rows.append({"k": k, "final_num_boost_round": art_k["n_round"],
                           "best_iters": ",".join(str(i) for i in art_k["best_iters"]),
                           **row_k})
        runs[k] = (art_k, topk)
        logger(f"[心跳] 层4 K={k}: val 头部五名净笔均(日加权)="
               f"{row_k['val_top5_net_dayavg']:.6f}, cluster_t="
               f"{row_k['val_top5_net_cluster_t']:.3f}, 终模 {art_k['n_round']} 轮 "
               f"(累计 {round(time.time() - t0)}s)")
    curve = pd.DataFrame(curve_rows).sort_values("k").reset_index(drop=True)
    # find_elbow 原样复用:列名适配(纯改名挂载到该函数约定接口,算法零改动,预登记 §四)
    elbow = fsel.find_elbow(
        curve.rename(columns={"val_top5_net_dayavg": "val_precision_at_5_dayavg"}))
    k_star = elbow["k_star"]
    logger(f"[层4] 拐点 K*={k_star} (N_surviving={len(reps)}; "
           f"耗时 {round(time.time() - t0)}s)")

    # ---- 定版模型 + 三段分数(预登记 §五)
    art_star, feat_star = runs[k_star]
    scores = assemble_scores_v6(df, train_sorted, art_star, feat_star)
    logger(f"[链] 定版: K*={k_star} 特征, 终模 {art_star['n_round']} 轮, "
           f"分数 {len(scores)} 行(耗时 {round(time.time() - t0)}s)")
    return {"rank_table": rank_table, "signs": signs, "clusters": clusters,
            "curve": curve, "curve_rows": curve_rows, "elbow": elbow,
            "ladder": ladder, "k_star": k_star, "feat_star": feat_star,
            "rank_of": rank_of, "kept1": kept1, "surv2": surv2, "reps": reps,
            "row_base": row_base, "art_base": art_base,
            "art_star": art_star, "scores": scores}


def assemble_scores_v6(df: pd.DataFrame, train_sorted: pd.DataFrame,
                       art_star: dict, feat_star: list[str]) -> pd.DataFrame:
    """三段分数序列(96,577 行,event_id/ts_code/date/seg/score 恰五列,无标签列)。

    train 行 = 五折折外原值(首块与标签 NaN 行 NaN);val/test/embargo/pre2001 行 =
    定版终模预测原值;seg 照标主表原值;行序 = 主表行序。
    """
    keys = df[["event_id", "ts_code", "date", "seg"]].copy()
    assert not keys["event_id"].duplicated().any(), "event_id 不唯一"
    oof = art_star["oof"]
    assert len(np.unique(train_sorted["event_id"])) == len(train_sorted), "折外键不唯一"
    oof_map = pd.Series(oof, index=train_sorted["event_id"].to_numpy())
    score = np.full(len(df), np.nan)
    is_train = (df["seg"] == "train").to_numpy()
    score[is_train] = df.loc[is_train, "event_id"].map(oof_map).to_numpy()
    rest = ~is_train
    score[rest] = art_star["booster"].predict(df.loc[rest, feat_star])
    out = keys.reset_index(drop=True)
    out["score"] = score
    assert list(out.columns) == ["event_id", "ts_code", "date", "seg", "score"]
    return out


# ---------------------------------------------------------------- 复现校验(全新进程, 全链重导)
def _read_float_exact(path: Path, str_cols: tuple) -> pd.DataFrame:
    """CSV 字符串读入 + float() 精确解析指定列(容差 0 口径)。"""
    t = pd.read_csv(path, dtype={c: str for c in str_cols})
    for c in str_cols:
        t[c] = t[c].map(float)
    return t


def verify() -> None:
    """--repro-check: 全链重算,与落盘产物逐位对比 + 分数双跑 md5(非零退出即失败)。"""
    label, config_id, _, params = resolve_winner_v6()
    assert label == LABEL and config_id == RESOLVE[1]
    with (OUT_DIR / "final_features.json").open(encoding="utf-8") as f:
        final = json.load(f)
    assert final["label"] == label and final["config_id"] == config_id
    df, feat_cols = load_frames()
    cn_of = load_cn_names()
    chain = run_chain(df, feat_cols, params, cn_of, print, time.time())

    # 层1~层4 落盘表逐位
    d1 = _read_float_exact(OUT_DIR / "layer1_shap_importance.csv", ("importance",))
    r1 = chain["rank_table"]
    assert (d1["feature"] == r1["feature"]).all() and \
        (d1["cn_name"] == r1["cn_name"]).all() and \
        (d1["rank"] == r1["rank"]).all() and (d1["kept"] == r1["kept"]).all()
    assert np.array_equal(d1["importance"].to_numpy(), r1["importance"].to_numpy())
    d2 = pd.read_csv(OUT_DIR / "layer2_yearly_signs.csv")
    pd.testing.assert_frame_equal(d2, chain["signs"], check_dtype=False)
    d3 = _read_float_exact(OUT_DIR / "layer3_clusters.csv", ("corr_with_rep",))
    r3 = chain["clusters"]
    assert (d3[["feature", "cn_name", "representative", "rank"]]
            == r3[["feature", "cn_name", "representative", "rank"]]).all().all()
    assert (d3["is_representative"] == r3["is_representative"]).all()
    assert np.array_equal(d3["corr_with_rep"].to_numpy(),
                          r3["corr_with_rep"].to_numpy())
    d4 = _read_float_exact(OUT_DIR / "layer4_curve.csv", CURVE_FLOAT_COLS)
    assert (d4["k"] == chain["curve"]["k"]).all() and \
        (d4["final_num_boost_round"] == chain["curve"]["final_num_boost_round"]).all() and \
        (d4["best_iters"] == chain["curve"]["best_iters"]).all()
    for c in CURVE_FLOAT_COLS:
        assert np.array_equal(d4[c].to_numpy(), chain["curve"][c].to_numpy()), \
            f"层4 曲线 {c} 逐位不一致"
    with (OUT_DIR / "layer4_elbow.json").open(encoding="utf-8") as f:
        ej = json.load(f)
    assert ej["k_star"] == chain["k_star"] and ej["ladder"] == chain["ladder"]
    assert ej["distances"] == chain["elbow"]["distances"]
    assert [r["feature"] for r in final["features"]] == chain["feat_star"]
    print("[repro-check] 层1~层4 落盘表与拐点逐位一致(全链重导)")

    # 定版模型 / 基模 / 分数序列逐位 + 双跑 md5
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        chain["art_star"]["booster"].save_model(tmp.name)
        assert Path(tmp.name).read_bytes() == (OUT_DIR / "model.txt").read_bytes(), \
            "model.txt 逐字节不一致"
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        chain["art_base"]["booster"].save_model(tmp.name)
        assert Path(tmp.name).read_bytes() == (OUT_DIR / "base_model.txt").read_bytes(), \
            "base_model.txt 逐字节不一致"
    stored = pd.read_parquet(OUT_DIR / "scores_v6.parquet")
    scores = chain["scores"]
    assert list(stored.columns) == list(scores.columns), "分数列结构不一致"
    assert stored["event_id"].tolist() == scores["event_id"].tolist(), "event_id 序不一致"
    assert np.array_equal(stored["score"].to_numpy(dtype=np.float64),
                          scores["score"].to_numpy(dtype=np.float64), equal_nan=True), \
        "分数列 score 逐位不一致"
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scores.to_parquet(tmp_path, index=False)
        md_a, md_b = _md5(OUT_DIR / "scores_v6.parquet"), _md5(tmp_path)
        assert md_a == md_b, f"分数双跑 md5 不一致: {md_a} vs {md_b}"
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"[repro-check] PASS: 全链重导 + 定版产物(model/base_model/三段分数)逐位一致; "
          f"scores_md5={md_a}")


# ---------------------------------------------------------------- 主流程
RESOLVE: tuple = ()   # (label, config_id, grid, params),main/verify 入口处解析


def main() -> None:
    global RESOLVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="基模冒烟:当选配置全特征管线计时 + 对 M3 config_24 逐位对账")
    ap.add_argument("--repro-check", action="store_true")
    args = ap.parse_args()
    RESOLVE = resolve_winner_v6()
    label, config_id, grid, params = RESOLVE
    if args.repro_check:
        verify()
        return

    t0 = time.time()
    log(f"开工: 当选 {label}({LABEL_CN})config_id={config_id} {grid}"
        f"(objective=regression, 无校准层), smoke={args.smoke}")

    # ---------------------------------------------------------- 1. 数据与段断言
    df, feat_cols = load_frames()
    cal = load_calendar()
    tep.assert_segment_integrity(df[["date", "seg"]], cal)
    seg_counts = {k: int(v) for k, v in df["seg"].value_counts().items()}
    n_test = seg_counts.get("test", 0)
    assert n_test > 0, "test 段应在场(本阶段只出分数不出指标)"
    log(f"[阶段1] 全量帧 {len(df)} 行, 特征 {len(feat_cols)} 列; 段计数 {seg_counts}")

    sub = df[df[LABEL].notna()]
    train_sorted = sub[sub["seg"] == "train"].sort_values(SORT_KEYS, kind="mergesort")
    val_sorted = sub[sub["seg"] == "val"].sort_values(SORT_KEYS, kind="mergesort")

    if args.smoke:
        # ---------------------------------------------------------- 基模冒烟 + 对账
        ts = time.time()
        art_base = base_pipeline(train_sorted, val_sorted, feat_cols, params)
        smoke_sec = round(time.time() - ts, 1)
        stored_oof = pd.read_parquet(RACE_DIR / f"oof_v6_{label}.parquet")
        stored_col = f"config_{config_id}"
        same_keys = (stored_oof["event_id"].to_numpy()
                     == train_sorted["event_id"].to_numpy()).all()
        same_oof = bool(same_keys) and np.array_equal(
            stored_oof[stored_col].to_numpy(), art_base["oof"], equal_nan=True)
        row_base = metric_rows(art_base, train_sorted, val_sorted)
        metrics = _read_csv_exact(RACE_DIR / f"metrics_v6_{label}.csv",
                                  ("val_top5_net_dayavg",))
        mrow = metrics[metrics["config_id"] == config_id].iloc[0]
        same_val = float(mrow["val_top5_net_dayavg"]) == row_base["val_top5_net_dayavg"]
        log(f"[冒烟] 基模单跑 {smoke_sec}s;折外对 M3 {stored_col} 逐位一致={same_oof};"
            f"val 头部五名净笔均(日加权)={row_base['val_top5_net_dayavg']:.6f} "
            f"与 M3 精确相等={same_val}, cluster_t={row_base['val_top5_net_cluster_t']:.3f}")
        with (OUT_DIR / "smoke_results.json").open("w", encoding="utf-8") as f:
            json.dump({"label": label, "config_id": config_id, "grid": grid,
                       "elapsed_sec": smoke_sec,
                       "oof_bitwise_vs_m3": same_oof,
                       "val_metric_exact_vs_m3": same_val,
                       "final_num_boost_round": art_base["n_round"],
                       "best_iters": art_base["best_iters"],
                       "row_base": row_base,
                       "n_train": int(len(train_sorted)),
                       "n_val": int(len(val_sorted))},
                      f, ensure_ascii=False, indent=2)
        assert same_oof and same_val, "冒烟对账失败(折外或 val 指标与 M3 不一致)"
        return

    # ---------------------------------------------------------- 2. 五层全链
    cn_of = load_cn_names()
    chain = run_chain(df, feat_cols, params, cn_of, log, t0)

    # ---------------------------------------------------------- 3. 产物落盘
    chain["rank_table"].to_csv(OUT_DIR / "layer1_shap_importance.csv", index=False)
    chain["signs"].to_csv(OUT_DIR / "layer2_yearly_signs.csv", index=False)
    chain["clusters"].to_csv(OUT_DIR / "layer3_clusters.csv", index=False)
    chain["curve"].to_csv(OUT_DIR / "layer4_curve.csv", index=False)
    with (OUT_DIR / "layer4_elbow.json").open("w", encoding="utf-8") as f:
        json.dump({**chain["elbow"], "ladder": chain["ladder"], "rule": ELBOW_RULE,
                   "metric": "验证段头部五名净笔均(日加权)(M3 主指标,回归头预测原值排序)",
                   "curve": chain["curve_rows"]}, f, ensure_ascii=False, indent=2)
    feat_star = chain["feat_star"]
    final_doc = {
        "issue": 51, "part_of": 47, "label": label, "label_cn": LABEL_CN,
        "config_id": config_id,
        "params": {k: grid[k] for k in lr.GRID_KEYS},
        "objective": "regression", "calibration": "无(回归头,预测原值排序)",
        "n_features": len(feat_star),
        "features": [{"feature": c, "rank": chain["rank_of"][c],
                      "cn_name": cn_of[c]} for c in feat_star],
    }
    with (OUT_DIR / "final_features.json").open("w", encoding="utf-8") as f:
        json.dump(final_doc, f, ensure_ascii=False, indent=2)
    chain["art_star"]["booster"].save_model(str(OUT_DIR / "model.txt"))
    chain["art_base"]["booster"].save_model(str(OUT_DIR / "base_model.txt"))
    chain["scores"].to_parquet(OUT_DIR / "scores_v6.parquet", index=False)
    log(f"[阶段3] 产物落盘完成: 终选 {len(feat_star)} 特征, 分数 {len(chain['scores'])} 行")

    # ---------------------------------------------------------- 4. 复现断言(全新进程全链重导, 不可旁路)
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--repro-check"], capture_output=True, text=True)
    log(f"[阶段4] 复现断言: {'PASS' if r.returncode == 0 else 'FAIL'}")
    for ln in r.stdout.strip().splitlines():
        log(f"[阶段4] repro: {ln}")
    if r.returncode != 0:
        log(f"[阶段4] repro-check stderr:\n{r.stderr}")
    assert r.returncode == 0, "复现断言失败"
    scores_md5 = next((ln.split("scores_md5=")[1] for ln in r.stdout.splitlines()
                       if "scores_md5=" in ln), None)
    assert scores_md5, "repro-check 未回报 scores md5"

    # ---------------------------------------------------------- 5. 台账
    row_base = chain["row_base"]
    row_star = chain["curve"].set_index("k").loc[chain["k_star"]]
    results = {
        "issue": 51, "part_of": 47, "label": label, "label_cn": LABEL_CN,
        "config_id": config_id, "grid": grid,
        "objective": "regression", "calibration": "无(回归头,预测原值排序)",
        "n_features_total": len(feat_cols),
        "layers": {
            "layer1": {"in": len(feat_cols), "out": len(chain["kept1"]),
                       "removed_zero_importance":
                           len(feat_cols) - len(chain["kept1"])},
            "layer2": {"in": len(chain["kept1"]), "out": len(chain["surv2"]),
                       "removed_drift": len(chain["kept1"]) - len(chain["surv2"]),
                       "year_list": list(YEAR_LIST)},
            "layer3": {"in": len(chain["surv2"]), "out": len(chain["reps"]),
                       "cluster_absorbed": len(chain["surv2"]) - len(chain["reps"]),
                       "threshold": fsel.CORR_CLUSTER_THRESHOLD},
            "layer4": {"ladder": chain["ladder"], "k_star": chain["k_star"],
                       "rule": ELBOW_RULE},
        },
        "final": {
            "n_features": len(feat_star),
            "final_num_boost_round": int(row_star["final_num_boost_round"]),
            "val_top5_net_dayavg": float(row_star["val_top5_net_dayavg"]),
            "val_top5_net_cluster_t": float(row_star["val_top5_net_cluster_t"]),
            "base_all_features_val_top5_net_dayavg":
                row_base["val_top5_net_dayavg"],
            "base_all_features_val_top5_net_cluster_t":
                row_base["val_top5_net_cluster_t"],
        },
        "seg_counts": seg_counts,
        "scores_md5": scores_md5,
        "assertions": {
            "segment_embargo": "PASS (assert_segment_integrity)",
            "leakage_exclusion": "PASS (model_feature_columns 1,997 列零命中)",
            "base_oof_anchor_m3": f"PASS (基模 OOF 与 M3 落盘 config_{config_id} 列逐位一致)",
            "base_val_metric_anchor_m3": "PASS (基模 val 主指标/cluster_t 与 M3 metrics 精确相等)",
            "scores_repro_fresh_process":
                "PASS (--repro-check 全新进程全链重导, 层表/拐点/终选/模型字节/分数逐位一致)",
            "scores_double_run_md5": f"PASS (双跑 md5 一致: {scores_md5})",
            "test_zero_metric": (f"PASS (test {n_test} 行仅出分数, 分数表无标签列, "
                                 "零指标零逐行统计)"),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with (OUT_DIR / "selection_results_v6.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[完成] 耗时 {results['elapsed_sec']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
