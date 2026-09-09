#!/usr/bin/env python3
"""M3 标签赛全量裁决驱动(战役 #47 M3,issue #50;预登记 = 同目录 README.md,冻结)。

8 候选(net_pos_{10,20,25,60}d 二分类 + net_ret_{10,20,25,60}d 回归,
候选序 = H 升序、同 H 二分类先)× 36 组预登记网格(label_race.GRID 原序),
单池正赛(ALL 全池 96,577 事件,无主备结构)。

每配置管线(预登记 §五 十条):
  训练段五折时间序列折外(按事件日分块、折界不落日内)
  → 二分类头:逻辑回归校准层 [p, p²];回归头:无校准,排序直接用预测原值
  → 全训练段终模(轮数 = 五折 best_iteration 均值取整,下限 1)
  → 指标表行 = train_oof(折外口径)与 val(终模口径),配置为行。
  二分类头 objective=binary/metric=average_precision(#25 口径);
  回归头 objective=regression/metric=rmse(预登记 §五.2 钉死)。

主裁决指标 = 验证段头部五名净笔均(日加权):每事件日按分数排序
(分数降序, ts_code 升序, event_id 升序 确定性裁决平局)取 top-min(5, 当日事件数),
所取事件按该候选自身 H 口径真实净收益求日截面均值,再按日等权平均;
附 cluster_t(按入场日聚类的 Liang-Zeger 稳健 t,#31 §2.4 口径,
在全部所取事件上合并计算,聚类 = 标签表 entry_date)。
每候选选配置:主指标最高 → cluster_t 较高 → 网格序靠前。
总裁决:八候选主指标最高者当选 → cluster_t → 候选序;
双约束:主指标 ≥ 八候选中位数 且 val cluster_t ≥ 2,否则无当选(阴性结论落盘)。
死锁条款:任一候选训练段有效标签事件数 < 3000 → 不裁决、回用户拍板。

纪律:训练只用 seg=='train' 且标签非 NaN 的行;val 指标只用 seg=='val';
embargo/pre2001 不作建模型行;test 段零触碰(只在场断言);
断点续跑(每配置完成即落盘 metrics_v6_{candidate}.csv,重跑跳过已完成);
八候选当选配置 OOF 复跑逐位一致断言(不可旁路,以落盘折外分数为对照基准)。

输出(experiments/v6_model_campaign/m3_label_race/):
  metrics_v6_{candidate}.csv   每候选指标表(配置为行,36 行)
  oof_v6_{candidate}.parquet   36 配置折外分数(复现性断言对照基准)
  summary_v6.csv               八候选当选配置汇总(裁决输入)
  adjudication_v6.json         总裁决结果(含双约束核验)
  race_results_v6.json         口径/断言/台账/剔除类别计数
  progress.log                 心跳

用法:
  python run_label_race_v6.py [--workers 3]           全量(断点续跑)
  python run_label_race_v6.py --smoke                 冒烟:1 候选 × 1 配置计时
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import label_race as lr  # noqa: E402  复用 36 组预登记网格(原序)
import train_eval_pipeline as tep  # noqa: E402  五折 OOF/校准层/终模/指标

MASTER_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_v6.parquet"
LABELS_PATH = SCRIPT_DIR / "labels_v6.parquet"
CALENDAR_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"
OUT_DIR = SCRIPT_DIR
PROGRESS = OUT_DIR / "progress.log"

H_LIST = (10, 20, 25, 60)
MIN_TRAIN_EVENTS = 3000  # 死锁条款阈值(#43 §3.3 第 1 条,纯样本量保险丝)
TOP_K = 5

# 八候选(候选序 = H 升序、同 H 二分类先;破平用)
CANDIDATES: list[dict] = []
for _H in H_LIST:
    CANDIDATES.append({
        "name": f"net_pos_{_H}d", "head": "binary", "H": _H,
        "cn": f"次日开盘入场净收益大于零二分类标签({_H} 个交易日视野)"})
    CANDIDATES.append({
        "name": f"net_ret_{_H}d", "head": "regression", "H": _H,
        "cn": f"次日开盘入场净收益幅度回归标签({_H} 个交易日视野)"})
CANDIDATE_ORDER = {c["name"]: i for i, c in enumerate(CANDIDATES)}

_globals: dict = {}


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [run_label_race_v6] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_calendar() -> np.ndarray:
    d = pd.read_parquet(CALENDAR_PATH, columns=["trade_date"])
    return np.sort(pd.to_datetime(d["trade_date"].astype(str)).unique())


def candidate_label_series(df: pd.DataFrame, cand: dict) -> pd.Series:
    """候选标签:二分类 = 1{净收益>0}(NaN 保留);回归 = 净收益原值。"""
    ret = df[f"net_ret_{cand['H']}d"]
    if cand["head"] == "binary":
        return pd.Series(np.where(ret.isna(), np.nan, (ret > 0).to_numpy(np.float64)),
                         index=df.index)
    return ret.astype(np.float64)


def grid_params_v6(overrides: dict, head: str) -> dict:
    """单组网格参数 -> 完整 LightGBM 参数(回归头 objective/metric 按预登记改)。"""
    params = dict(tep.DEFAULT_LGBM_PARAMS)
    params.update(overrides)
    if head == "regression":
        params["objective"] = "regression"
        params["metric"] = "rmse"
    return params


# ---------------------------------------------------------------- 主指标与 cluster_t
def cluster_t_lz(x: np.ndarray, clusters: np.ndarray) -> float:
    """按入场日聚类的 Liang-Zeger 稳健 t(#31 §2.4 逐字口径)。
    x̄ = 净收益均值;S_c = 聚类 c 内 (x_i − x̄) 之和;
    var(x̄) = G/(G−1) × ΣS_c² / n²;G < 2 记 NaN;var ≤ 0 记 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return np.nan
    xbar = x.mean()
    df = pd.DataFrame({"s": x - xbar, "c": clusters})
    sums = df.groupby("c")["s"].sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan
    var = (g / (g - 1.0)) * float((sums ** 2).sum()) / (n * n)
    if var <= 0:
        return np.nan
    return float(xbar / np.sqrt(var))


def top5_net(df: pd.DataFrame, score_col: str = "score", k: int = TOP_K) -> dict:
    """头部五名净笔均(日加权)附 cluster_t(预登记 §五.4 主裁决指标)。

    每事件日按 (分数降序, ts_code 升序, event_id 升序) 取 top-min(k, 当日事件数),
    日截面净收益均值再按日等权平均;cluster_t 在全部所取事件上按 entry_date 聚类。
    同出 top-5 命中率(所取事件净收益>0 占比,事件加权/日加权,回归头辅助指标)。
    """
    need = {"date", "ts_code", "event_id", score_col, "net_ret", "entry_date"}
    assert need <= set(df.columns), f"top5_net 缺列: {need - set(df.columns)}"
    ranked = df.sort_values(["date", score_col, "ts_code", "event_id"],
                            ascending=[True, False, True, True], kind="mergesort")
    pick = ranked.groupby("date", sort=True).head(k)
    day_mean = pick.groupby("date")["net_ret"].mean()
    day_hit = pick.groupby("date").apply(
        lambda g: float((g["net_ret"] > 0).mean()), include_groups=False)
    return {
        "n_days": int(day_mean.size),
        "n_selected": int(len(pick)),
        "top5_net_dayavg": float(day_mean.mean()),
        "top5_net_cluster_t": cluster_t_lz(pick["net_ret"].to_numpy(),
                                           pick["entry_date"].to_numpy()),
        "top5_hit_eventavg": float((pick["net_ret"] > 0).mean()),
        "top5_hit_dayavg": float(day_hit.mean()),
    }


# ---------------------------------------------------------------- 单配置管线
def run_single_config_v6(train: pd.DataFrame, val: pd.DataFrame,
                         feat_cols: list[str], cand: dict,
                         params: dict) -> tuple[dict, dict]:
    """单候选单配置:五折 OOF → (二分类:校准层 [p,p²]) → 终模 → train_oof/val 指标行。

    返回 (row, artifacts);artifacts 含 oof(与 train 按 (date,ts_code,event_id)
    mergesort 后的行序对齐)、best_iters,供落盘与复跑断言。test 段不得传入。
    """
    assert train["seg"].eq("train").all() and val["seg"].eq("val").all(), \
        "run_single_config_v6 只接受 train/val 段"
    train = train.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    X_tr = train[feat_cols]
    y_tr = train["_y"].to_numpy(dtype=np.float64)
    dates_tr = pd.to_datetime(train["date"]).to_numpy()

    oof, best_iters = tep.time_series_oof(X_tr, y_tr, dates_tr, params=params)
    oof_mask = np.isfinite(oof)
    n_round = tep.final_num_boost_round(best_iters)
    booster = tep.fit_final_model(X_tr, y_tr, num_boost_round=n_round, params=params)

    row = {"best_iters": ",".join(str(i) for i in best_iters),
           "final_num_boost_round": n_round}
    if cand["head"] == "binary":
        calibrator = tep.SquaredLogitCalibrator().fit(oof[oof_mask], y_tr[oof_mask])
        train_score = calibrator.predict(oof[oof_mask])
        val_score = calibrator.predict(booster.predict(val[feat_cols]))
        row.update({"calib_coef_p": float(calibrator.coef_[0]),
                    "calib_coef_p2": float(calibrator.coef_[1]),
                    "calib_intercept": float(calibrator.intercept_)})
    else:
        calibrator = None
        train_score = oof[oof_mask]
        val_score = booster.predict(val[feat_cols])
        row.update({"calib_coef_p": np.nan, "calib_coef_p2": np.nan,
                    "calib_intercept": np.nan})

    train_ev = train.loc[oof_mask,
                         ["date", "ts_code", "event_id", "seg",
                          "net_ret", "entry_date"]].copy()
    train_ev["y"] = y_tr[oof_mask]
    train_ev["score"] = train_score
    val_ev = val[["date", "ts_code", "event_id", "seg",
                  "net_ret", "entry_date"]].copy()
    val_ev["y"] = val["_y"].to_numpy(dtype=np.float64)
    val_ev["score"] = val_score

    for tag, ev in (("train_oof", train_ev), ("val", val_ev)):
        main = top5_net(ev)
        row[f"{tag}_n_events"] = int(len(ev))
        row[f"{tag}_n_days"] = main["n_days"]
        row[f"{tag}_n_selected"] = main["n_selected"]
        row[f"{tag}_top5_net_dayavg"] = main["top5_net_dayavg"]
        row[f"{tag}_top5_net_cluster_t"] = main["top5_net_cluster_t"]
        row[f"{tag}_top5_hit_eventavg"] = main["top5_hit_eventavg"]
        row[f"{tag}_top5_hit_dayavg"] = main["top5_hit_dayavg"]
        if cand["head"] == "binary":
            aux = tep.evaluate_segment(
                ev.rename(columns={"score": "prob"}), prob_col="prob",
                label_col="y", k=TOP_K)
            row[f"{tag}_base_rate"] = aux["base_rate"]
            row[f"{tag}_average_precision"] = aux["average_precision"]
            row[f"{tag}_precision_at_5_dayavg"] = aux["precision_at_5_dayavg"]
            row[f"{tag}_precision_at_5_eventavg"] = aux["precision_at_5_eventavg"]
        else:
            row[f"{tag}_base_rate"] = float(ev["y"].mean())  # 回归头:标签均值(净收益)
            row[f"{tag}_average_precision"] = np.nan
            row[f"{tag}_precision_at_5_dayavg"] = np.nan
            row[f"{tag}_precision_at_5_eventavg"] = np.nan

    artifacts = {"oof": oof, "oof_mask": oof_mask, "best_iters": best_iters}
    return row, artifacts


# ---------------------------------------------------------------- 选配置与裁决(纯函数)
def select_best_config_v6(metrics: pd.DataFrame) -> int:
    """每候选选配置:主指标最高 → cluster_t 较高 → 网格序靠前(预登记 §五.5)。"""
    need = {"config_id", "val_top5_net_dayavg", "val_top5_net_cluster_t"}
    assert need <= set(metrics.columns), f"指标表缺列: {need - set(metrics.columns)}"
    assert len(metrics) > 0, "指标表为空"
    order = metrics.sort_values(
        ["val_top5_net_dayavg", "val_top5_net_cluster_t", "config_id"],
        ascending=[False, False, True], kind="mergesort", na_position="last")
    return int(metrics.index.get_loc(order.index[0]))


def adjudicate_v6(summary: pd.DataFrame) -> dict:
    """总裁决(预登记 §五.6/7,无自由裁量):八候选各取当选配置后,
    主指标最高者当选(平局 cluster_t → 候选序靠前);
    双约束:主指标 ≥ 八候选中位数 且 val cluster_t ≥ 2,否则无当选。"""
    need = {"candidate", "val_top5_net_dayavg", "val_top5_net_cluster_t"}
    assert need <= set(summary.columns), f"裁决表缺列: {need - set(summary.columns)}"
    canon = [c["name"] for c in CANDIDATES]
    assert list(summary["candidate"]) == canon, "裁决表候选集与预登记八候选不一致"
    s = summary.copy()
    s["_ord"] = s["candidate"].map(CANDIDATE_ORDER)
    s = s.sort_values(
        ["val_top5_net_dayavg", "val_top5_net_cluster_t", "_ord"],
        ascending=[False, False, True], kind="mergesort", na_position="last")
    top = s.iloc[0]
    median_main = float(s["val_top5_net_dayavg"].median())
    c1 = bool(top["val_top5_net_dayavg"] >= median_main)
    ct = float(top["val_top5_net_cluster_t"])
    c2 = bool(np.isfinite(ct) and ct >= 2.0)
    return {
        "winner": str(top["candidate"]) if (c1 and c2) else None,
        "winner_val_top5_net_dayavg": float(top["val_top5_net_dayavg"]),
        "winner_val_top5_net_cluster_t": ct,
        "median_val_top5_net_dayavg": median_main,
        "constraint_main_ge_median": c1,
        "constraint_cluster_t_ge_2": c2,
        "double_constraint_passed": bool(c1 and c2),
    }


# ---------------------------------------------------------------- worker 侧
def _worker_init() -> None:
    """每 worker 载一次主表元数据+特征与标签表,合并出建模型行集(主进程已过断言)。"""
    sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
    import train_eval_pipeline as _tep
    master = pd.read_parquet(MASTER_PATH)
    feat_cols = _tep.model_feature_columns(master)
    meta_cols = ["event_id", "ts_code", "date", "seg"]
    lab_cols = ["event_id", "entry_date"] + [f"net_ret_{H}d" for H in H_LIST]
    labels = pd.read_parquet(LABELS_PATH, columns=lab_cols)
    df = master[meta_cols + feat_cols].merge(labels, on="event_id", how="left",
                                             validate="one_to_one")
    model_df = df[df["seg"].isin(["train", "val"])].reset_index(drop=True)
    _globals.clear()
    _globals.update({"tep": _tep, "model_df": model_df, "feat_cols": feat_cols})


def _run_one(cand_name: str, config_id: int) -> tuple:
    """单候选单配置跑训(worker 内)。返回 (candidate, config_id, 指标行, event_ids, oof)。"""
    tep_, model_df, feat_cols = _globals["tep"], _globals["model_df"], _globals["feat_cols"]
    cand = next(c for c in CANDIDATES if c["name"] == cand_name)
    cfg = lr.GRID[config_id]
    df = model_df[["event_id", "ts_code", "date", "seg", "entry_date",
                   f"net_ret_{cand['H']}d"] + feat_cols].copy()
    df["net_ret"] = df[f"net_ret_{cand['H']}d"].to_numpy(dtype=np.float64)
    df["_y"] = candidate_label_series(df, cand).to_numpy()
    df = df[df["_y"].notna()]
    train = df[df["seg"] == "train"]
    val = df[df["seg"] == "val"]
    row, art = run_single_config_v6(train, val, feat_cols, cand,
                                    grid_params_v6(cfg, cand["head"]))
    row = {"candidate": cand_name, "config_id": config_id, **cfg, **row}
    train_sorted = train.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    return (cand_name, config_id, row,
            train_sorted["event_id"].to_numpy(), art["oof"])


def _rerun_oof(cand_name: str, config_id: int) -> tuple[str, bool, str]:
    """当选配置复现性断言(不可旁路):重算折外分数,与落盘产物逐位比对。"""
    tep_, model_df, feat_cols = _globals["tep"], _globals["model_df"], _globals["feat_cols"]
    cand = next(c for c in CANDIDATES if c["name"] == cand_name)
    cfg = lr.GRID[config_id]
    df = model_df[["event_id", "ts_code", "date", "seg",
                   f"net_ret_{cand['H']}d"] + feat_cols].copy()
    df["_y"] = candidate_label_series(df, cand).to_numpy()
    sub = df[(df["seg"] == "train") & df["_y"].notna()]
    sub = sub.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    X = sub[feat_cols]
    y = sub["_y"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(sub["date"]).to_numpy()
    oof_new, _ = tep_.time_series_oof(X, y, dates,
                                      params=grid_params_v6(cfg, cand["head"]))
    stored = pd.read_parquet(OUT_DIR / f"oof_v6_{cand_name}.parquet")
    same_keys = (stored["event_id"].to_numpy() == sub["event_id"].to_numpy()).all()
    ok = bool(same_keys) and np.array_equal(
        oof_new, stored[f"config_{config_id}"].to_numpy(), equal_nan=True)
    return cand_name, ok, "stored_vs_rerun"


# ---------------------------------------------------------------- 主进程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟:1 候选(net_pos_10d) × 1 配置(config_id=0)计时")
    args = ap.parse_args()
    t0 = time.time()
    cands = CANDIDATES
    log(f"开工: {len(cands)} 候选 × {len(lr.GRID)} 配置,workers={args.workers},"
        f"smoke={args.smoke}")

    # ---------------------------------------------------------- 1. 主表/标签/段断言
    master = pd.read_parquet(MASTER_PATH)
    feat_cols = tep.model_feature_columns(master)
    assert len(feat_cols) == 1997, f"特征列数 {len(feat_cols)} != 1997"
    assert not any(c.startswith("label") for c in feat_cols), "特征集混入 label_ 列"
    log(f"[阶段1] 主表 {len(master)} 行 × {len(master.columns)} 列,特征 {len(feat_cols)} 列")
    cal = load_calendar()
    tep.assert_segment_integrity(master[["date", "seg"]], cal)
    seg_counts = {k: int(v) for k, v in master["seg"].value_counts().items()}
    log(f"[阶段1] 段界与隔离带断言通过: {seg_counts}")

    labels = pd.read_parquet(LABELS_PATH)
    mkey = master[["event_id", "ts_code", "date"]]
    lkey = labels[["event_id", "ts_code", "date"]]
    j = mkey.merge(lkey, on="event_id", how="inner", suffixes=("", "_lab"))
    assert len(j) == len(master), "主表与标签表 event_id 非一一对应"
    assert (j["ts_code"] == j["ts_code_lab"]).all() and \
           (j["date"] == j["date_lab"]).all(), "event_id 两侧键不一致"
    df_all = master[["event_id", "ts_code", "date", "seg"]].merge(
        labels.drop(columns=["ts_code", "date", "seg", "event_row"]),
        on="event_id", how="left", validate="one_to_one")
    test_present = int((df_all["seg"] == "test").sum())
    assert test_present > 0, "test 段应在场(本赛零触碰)"

    # 死锁条款(预登记 §五.1):任一候选训练段有效标签事件数 < 3000 → 不裁决
    deadlock: dict[str, int] = {}
    train_eff: dict[str, int] = {}
    for cand in cands:
        y = candidate_label_series(df_all, cand)
        n_tr = int(((df_all["seg"] == "train") & y.notna()).sum())
        train_eff[cand["name"]] = n_tr
        if n_tr < MIN_TRAIN_EVENTS:
            deadlock[cand["name"]] = n_tr
    log(f"[阶段1] 训练段有效标签事件数: {train_eff}")
    if deadlock:
        log(f"[死锁] 训练段有效标签事件数 < {MIN_TRAIN_EVENTS}: {deadlock} -> 不裁决,回用户拍板")
        with (OUT_DIR / "race_results_v6.json").open("w", encoding="utf-8") as f:
            json.dump({"issue": 50, "deadlock": deadlock,
                       "train_effective_events": train_eff,
                       "rule": f"训练段有效标签事件数 < {MIN_TRAIN_EVENTS} -> 本赛季不裁决、回用户拍板"},
                      f, ensure_ascii=False, indent=2)
        return

    if args.smoke:
        # ---------------------------------------------------------- 冒烟:1 候选 × 1 配置计时
        cand = cands[0]  # net_pos_10d
        log(f"[冒烟] {cand['name']} × config_id=0 {lr.GRID[0]}")
        ts = time.time()
        with ProcessPoolExecutor(max_workers=1, initializer=_worker_init) as ex:
            _, _, row, _, _ = ex.submit(_run_one, cand["name"], 0).result()
        smoke_sec = round(time.time() - ts, 1)
        log(f"[冒烟] 单配置耗时 {smoke_sec}s;val 头部五名净笔均(日加权)="
            f"{row['val_top5_net_dayavg']:.6f},cluster_t={row['val_top5_net_cluster_t']:.3f}")
        with (OUT_DIR / "smoke_results.json").open("w", encoding="utf-8") as f:
            json.dump({"candidate": cand["name"], "config_id": 0, "grid": lr.GRID[0],
                       "elapsed_sec": smoke_sec, "row": row,
                       "train_effective_events": train_eff}, f,
                      ensure_ascii=False, indent=2, default=str)
        return

    # ---------------------------------------------------------- 2. 网格跑训(断点续跑)
    pending: dict[str, list[int]] = {}
    done_rows: dict[str, dict[int, dict]] = {}
    for cand in cands:
        csv = OUT_DIR / f"metrics_v6_{cand['name']}.csv"
        if csv.exists():
            t = pd.read_csv(csv)
            have = sorted(int(i) for i in t["config_id"])
            done_rows[cand["name"]] = {int(r["config_id"]): r.to_dict()
                                       for _, r in t.iterrows()}
            rest = [i for i in range(len(lr.GRID)) if i not in have]
            if rest:
                pending[cand["name"]] = rest
                log(f"[阶段2] {cand['name']}: 已完成 {len(have)}/36,续跑 {len(rest)}")
            else:
                log(f"[阶段2] {cand['name']}: 36 配置已完成,跳过(断点续跑)")
        else:
            pending[cand["name"]] = list(range(len(lr.GRID)))
            done_rows[cand["name"]] = {}
    rows_buf: dict[str, dict[int, dict]] = {c["name"]: dict(done_rows[c["name"]])
                                            for c in cands}
    oof_buf: dict[str, dict[int, np.ndarray]] = {c: {} for c in pending}
    eid_buf: dict[str, np.ndarray] = {}
    n_tasks = sum(len(v) for v in pending.values())
    log(f"[阶段2] 待跑 {len(pending)} 候选 / {n_tasks} 配置任务")
    done = 0
    if n_tasks:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as ex:
            futs = {ex.submit(_run_one, c, i): (c, i)
                    for c, ids in pending.items() for i in ids}
            for fut in as_completed(futs):
                c, i = futs[fut]
                try:
                    _, _, row, eids, oof = fut.result()
                except Exception:
                    log(f"[失败] {c} config_id={i}\n{traceback.format_exc()}")
                    raise
                rows_buf[c][i] = row
                oof_buf[c][i] = oof
                if c in eid_buf:
                    assert (eid_buf[c] == eids).all(), f"{c} 配置间训练行序不一致"
                else:
                    eid_buf[c] = eids
                # 每配置完成即落盘(断点续跑)
                table = pd.DataFrame([rows_buf[c][k] for k in sorted(rows_buf[c])])
                table.to_csv(OUT_DIR / f"metrics_v6_{c}.csv", index=False)
                done += 1
                log(f"[心跳] 配置任务 {done}/{n_tasks} 完成: {c} config_id={i} "
                    f"val_top5_net_dayavg={row['val_top5_net_dayavg']:.6f} "
                    f"cluster_t={row['val_top5_net_cluster_t']:.3f} "
                    f"(累计 {round(time.time() - t0)}s)")
                if len(rows_buf[c]) == len(lr.GRID):
                    oof_df = pd.DataFrame(
                        {f"config_{k}": oof_buf[c][k] for k in range(len(lr.GRID))})
                    oof_df.insert(0, "event_id", eid_buf[c])
                    oof_df.to_parquet(OUT_DIR / f"oof_v6_{c}.parquet", index=False)
                    del oof_buf[c]
                    log(f"[阶段2] {c}: 36 配置指标表与折外分数落盘")

    # ---------------------------------------------------------- 3. 选配置 + 汇总
    summary_rows = []
    for cand in cands:
        name = cand["name"]
        table = pd.read_csv(OUT_DIR / f"metrics_v6_{name}.csv")
        assert len(table) == len(lr.GRID), f"{name} 指标表行数 {len(table)} != {len(lr.GRID)}"
        best = table.iloc[select_best_config_v6(table)]
        summary_rows.append({
            "candidate": name, "label_cn": cand["cn"], "head": cand["head"], "H": cand["H"],
            **{k: best[k] for k in (
                "config_id", "num_leaves", "min_data_in_leaf", "learning_rate",
                "feature_fraction", "final_num_boost_round",
                "val_n_events", "val_n_days", "val_n_selected",
                "val_top5_net_dayavg", "val_top5_net_cluster_t",
                "val_top5_hit_eventavg", "val_top5_hit_dayavg",
                "val_base_rate", "val_average_precision",
                "val_precision_at_5_dayavg", "val_precision_at_5_eventavg",
                "train_oof_n_events", "train_oof_n_days", "train_oof_n_selected",
                "train_oof_top5_net_dayavg", "train_oof_top5_net_cluster_t",
                "train_oof_average_precision", "train_oof_precision_at_5_dayavg")}})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "summary_v6.csv", index=False)
    log("[阶段3] 八候选当选配置汇总落盘 summary_v6.csv")

    # ---------------------------------------------------------- 4. 当选配置复现性断言(不可旁路)
    repro_tasks = [(r["candidate"], int(r["config_id"])) for r in summary_rows]
    repro: dict[str, bool] = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init) as ex:
        futs = {ex.submit(_rerun_oof, c, i): c for c, i in repro_tasks}
        for fut in as_completed(futs):
            c, ok, mode = fut.result()
            repro[c] = ok
            log(f"[阶段4] {c} 当选配置 OOF 复现断言[{mode}]: "
                f"{'逐位一致' if ok else '不一致!'}")
    assert all(repro.values()), \
        f"复现性断言失败: {[c for c, ok in repro.items() if not ok]}"
    log("[阶段4] 八候选当选配置 OOF 复现断言全部逐位一致")

    # ---------------------------------------------------------- 5. 裁决(独立复核前为待核状态)
    adj = adjudicate_v6(summary)
    adj["status"] = "provisional_pending_independent_review"
    with (OUT_DIR / "adjudication_v6.json").open("w", encoding="utf-8") as f:
        json.dump(adj, f, ensure_ascii=False, indent=2)
    log(f"[阶段5] 裁决(待独立复核): winner={adj['winner']} "
        f"(主指标={adj['winner_val_top5_net_dayavg']:.6f}, "
        f"cluster_t={adj['winner_val_top5_net_cluster_t']:.3f} vs 双约束 "
        f"中位数 {adj['median_val_top5_net_dayavg']:.6f} / cluster_t≥2; "
        f"{'双约束通过' if adj['double_constraint_passed'] else '双约束未过——无当选'})")

    # ---------------------------------------------------------- 6. 台账
    results = {
        "issue": 50, "part_of": 47,
        "n_candidates": len(cands), "grid_size": len(lr.GRID), "grid": lr.GRID,
        "candidates": cands,
        "n_features": len(feat_cols), "seg_counts": seg_counts,
        "train_effective_events": train_eff,
        "adjudication": adj,
        "assertions": {
            "segment_embargo": "PASS (assert_segment_integrity)",
            "leakage_exclusion": "PASS (model_feature_columns 1,997 列零命中)",
            "winner_oof_reproducible": "PASS (8 候选当选配置 OOF 复现断言逐位一致; 模式 stored_vs_rerun)",
            "test_untouched": f"PASS (test {test_present} 行在场, 零指标零逐行统计)",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with (OUT_DIR / "race_results_v6.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[完成] 耗时 {results['elapsed_sec']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
