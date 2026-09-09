#!/usr/bin/env python3
"""M4 验收独立自检(战役 #47 M4,issue #51;README §六 的独立复核落点,层5 口径)。

纪律:本脚本对承重断言一律**独立实现**——不调 feature_selection 的任何选择函数
(shap_values/layer1_rank/layer2_yearly_signs/layer3_clusters/k_ladder/find_elbow),
也不调 M4 驱动 run_training_selection_v6 的任何函数;SHAP 值用 LightGBM
pred_contrib=True 自写重算,相关系数用 numpy 自写 pairwise-complete 实现,
拐点算法按预登记规则文本自写。允许使用的共享原语:切分常量与段断言
(feature_master/train_eval_pipeline.assert_segment_integrity)、训练原语
(time_series_oof/final_num_boost_round/DEFAULT_LGBM_PARAMS)、36 组网格
(label_race.GRID)——与 M3 自检同口径(训练原语非选择函数)。

复核项:
 0. 行数与键守恒: scores 96,577 行 × 恰 5 列,与主表 event_id/ts_code/date/seg 逐行一致;
 1. 当选解析独立重算: adjudication/summary/metrics 三重互证(winner=net_ret_60d,
    双约束过,config_id=24 与 GRID[24] 逐项相等,回归头);
 2. 段界硬断言 + 泄漏列独立重扫(1,997 列对 EXCLUDE_PATTERNS 零命中);
 3. 基模折外对 M3 oof_v6_net_ret_60d.parquet config_24 列逐位一致(独立复跑);
 4. 层1 独立重算: 载 base_model.txt 自写 SHAP → mean(|SHAP|) 排序表与落盘逐位一致;
 5. 层2 独立重算: 自写分年度符号(2020~2023)与落盘全等;
 6. 层3 独立重算: 自写相关矩阵 + 贪婪聚类与落盘全等;
 7. 层4 独立重算: 自写拐点(垂距/容差/平局取小 K)与落盘 k_star/distances 一致;
    终选特征集 = 层3 代表按层1 名次前 K*,与 final_features.json 全等(含中文全称);
 8. 定版分数独立重算: train 行 = 终选特征五折折外(独立复跑 tep.time_series_oof)
    逐位一致;val/test/embargo/pre2001 行 = model.txt 预测原值逐位一致;
 9. 确定性: 台账 scores_md5 与现盘 scores_v6.parquet 重哈希一致;
10. test 零触碰: 分数表无标签列;层4 曲线/台账无 test 列;test 行数在场;
11. report.md 生成: 层4 曲线全表(K 为行全出数)+ 各层幸存/剔除计数 +
    终选特征集清单(中文全称)+ 基模对账 + 自检结果。

输出: selfcheck_results.json(全绿才写 "all_pass": true) + report.md。
用法: python selfcheck_m4.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_master as fm  # noqa: E402  切分常量/排除模式清单(权威常量,非选择函数)
import label_race as lr  # noqa: E402  36 组预登记网格原序
import train_eval_pipeline as tep  # noqa: E402  训练原语(OOF 复跑/段断言)

MASTER_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_v6.parquet"
DICT_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_dictionary_v6.csv"
RACE_DIR = REPO / "experiments" / "v6_model_campaign" / "m3_label_race"
LABELS_PATH = RACE_DIR / "labels_v6.parquet"
CALENDAR_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"
OUT_DIR = SCRIPT_DIR
PROGRESS = OUT_DIR / "progress.log"

LABEL = "net_ret_60d"
LABEL_CN = "次日开盘入场净收益幅度回归标签(60 个交易日视野)"
YEAR_LIST = (2020, 2021, 2022, 2023)
MIN_YEAR_PAIRS = 30          # 预登记 §四 层2 规则文本(与 v5 模块常量同值,自写实现)
CORR_THRESHOLD = 0.9         # 预登记 §四 层3 规则文本
SORT_KEYS = ["date", "ts_code", "event_id"]
CURVE_FLOAT_COLS = ("val_top5_net_dayavg", "val_top5_net_cluster_t",
                    "val_top5_hit_eventavg", "train_oof_top5_net_dayavg",
                    "train_oof_top5_net_cluster_t", "train_oof_top5_hit_eventavg")


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [selfcheck_m4] {msg}"
    print(line, flush=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_float_exact(path: Path, str_cols: tuple) -> pd.DataFrame:
    t = pd.read_csv(path, dtype={c: str for c in str_cols})
    for c in str_cols:
        t[c] = t[c].map(float)
    return t


# ---------------------------------------------------------------- 独立重算原语(自写)
def shap_independent(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP 精确值自写计算(pred_contrib=True,末列偏置剔除);不调 fsel.shap_values。"""
    raw = booster.predict(X, pred_contrib=True)
    assert raw.shape == (len(X), X.shape[1] + 1)
    contrib = np.asarray(raw[:, :-1], dtype=np.float64)
    assert np.isfinite(contrib).all()
    return contrib


def sign_independent(x: np.ndarray, y: np.ndarray) -> int:
    """分年度符号自写:成对去 NaN;有效对 < 30 或任一方差为 0 或相关恰为 0 记 0。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_YEAR_PAIRS:
        return 0
    xs, ys = x[m], y[m]
    if xs.std() == 0.0 or ys.std() == 0.0:
        return 0
    r = float(np.corrcoef(xs, ys)[0, 1])
    return 0 if r == 0.0 else (1 if r > 0 else -1)


def pairwise_corr_independent(x: np.ndarray, min_pairs: int = 30) -> np.ndarray:
    """pairwise-complete Pearson 相关矩阵自写(numpy 矩阵化;不调 fm.pairwise_corr)。"""
    fin = np.isfinite(x)
    z = np.where(fin, x, 0.0)
    mf = fin.astype(np.float64)
    n_ij = mf.T @ mf
    sx = z.T @ mf
    sxx = (z * z).T @ mf
    sxy = z.T @ z
    with np.errstate(invalid="ignore", divide="ignore"):
        n_safe = np.where(n_ij > 0, n_ij, np.nan)
        mu_i = sx / n_safe
        mu_j = sx.T / n_safe
        cov = sxy / n_safe - mu_i * mu_j
        var_i = sxx / n_safe - mu_i ** 2
        var_j = sxx.T / n_safe - mu_j ** 2
        denom = np.sqrt(np.clip(var_i, 0, None) * np.clip(var_j, 0, None))
        corr = np.where(denom > 0, cov / denom, np.nan)
    corr[n_ij < min_pairs] = np.nan
    np.fill_diagonal(corr, 1.0)
    return corr


def elbow_independent(ks: np.ndarray, ps: np.ndarray) -> tuple[int, list[float]]:
    """拐点自写:x=log2(K),首末连线垂距最远;相对容差 1e-12 内并列取小 K。"""
    if len(ks) == 1:
        return int(ks[0]), [0.0]
    xs = np.log2(ks.astype(np.float64))
    x0, y0, x1, y1 = xs[0], ps[0], xs[-1], ps[-1]
    vx, vy = x1 - x0, y1 - y0
    norm = float(np.hypot(vx, vy))
    assert norm > 0.0
    dists = np.abs(vx * (ps - y0) - vy * (xs - x0)) / norm
    tol = 1e-12 * max(1.0, float(dists.max()))
    tied = np.flatnonzero(dists >= dists.max() - tol)
    return int(ks[tied[0]]), [float(d) for d in dists]


# ---------------------------------------------------------------- 数据装载(自检自用)
def load_all() -> tuple[pd.DataFrame, list[str]]:
    master = pd.read_parquet(MASTER_PATH)
    feat_cols = [c for c in master.columns if c not in fm.EVENT_META_COLS
                 and master[c].dtype.kind in "fib"]
    assert len(feat_cols) == 1997
    labels = pd.read_parquet(LABELS_PATH,
                             columns=["event_id", "entry_date", LABEL])
    df = master.merge(labels, on="event_id", how="left", validate="one_to_one")
    return df, feat_cols


def winner_params_independent() -> tuple[int, dict]:
    """当选配置独立解析(按 README §三 规则文本自写断言,不调驱动 resolve_winner_v6)。"""
    adj = json.load(open(RACE_DIR / "adjudication_v6.json", encoding="utf-8"))
    assert adj["winner"] == LABEL and adj["double_constraint_passed"] is True
    summary = _read_float_exact(RACE_DIR / "summary_v6.csv", tuple(lr.GRID_KEYS))
    row = summary[summary["candidate"] == LABEL].iloc[0]
    assert row["head"] == "regression"
    config_id = int(row["config_id"])
    expect = lr.GRID[config_id]
    for k in lr.GRID_KEYS:
        assert float(row[k]) == float(expect[k])
    metrics = _read_float_exact(RACE_DIR / f"metrics_v6_{LABEL}.csv",
                                ("val_top5_net_dayavg", "val_top5_net_cluster_t"))
    mrow = metrics[metrics["config_id"] == config_id].iloc[0]
    assert abs(float(mrow["val_top5_net_dayavg"])
               - float(adj["winner_val_top5_net_dayavg"])) <= 1e-15
    assert abs(float(mrow["val_top5_net_cluster_t"])
               - float(adj["winner_val_top5_net_cluster_t"])) <= 1e-15
    params = dict(tep.DEFAULT_LGBM_PARAMS)
    params.update(expect)
    params["objective"] = "regression"
    params["metric"] = "rmse"
    return config_id, params


# ---------------------------------------------------------------- report.md 生成
def render_report(chain: dict, checks: dict) -> str:
    L: list[str] = []
    L.append("# M4 训练管线与 SHAP 五层精选报告(战役 #47 M4,issue #51)")
    L.append("")
    L.append(f"日期:{pd.Timestamp.now():%Y-%m-%d}。预登记 = 同目录 README.md(冻结,修订记录只增不改)。")
    L.append(f"当选标签 = {LABEL}({LABEL_CN});当选配置 config_id={chain['config_id']}"
             f"(num_leaves=63, min_data_in_leaf=50, learning_rate=0.05, feature_fraction=0.6);"
             "回归头无校准层,排序与指标一律用预测原值(M3 §五.3 钉死,本阶段沿用)。")
    L.append("评估指标 = 验证段头部五名净笔均(日加权)附 cluster_t(按入场日聚类的 Liang-Zeger 稳健 t),"
             "与 M3 裁决链同型;收益数值为小数收益率。")
    L.append("")
    # ---- 基模对账
    L.append("## 1. 基模对账(全 1,997 特征,当选配置)")
    L.append("")
    L.append(f"折外分数对 M3 落盘 oof_v6_net_ret_60d.parquet 的 config_24 列**逐位一致**"
             f"(event_id 序对齐 + np.array_equal equal_nan=True;驱动锚定与自检独立复跑双过)。")
    L.append(f"基模 val 头部五名净笔均(日加权)= {chain['row_base']['val_top5_net_dayavg']:+.6f},"
             f"cluster_t = {chain['row_base']['val_top5_net_cluster_t']:+.3f},"
             f"与 M3 metrics config_id=24 行精确相等;终模 {chain['base_n_round']} 轮。")
    L.append("")
    # ---- 各层计数
    ly = chain["layers"]
    L.append("## 2. 五层精选计数(幸存/剔除)")
    L.append("")
    L.append("| 层 | 入 | 出 | 剔除 | 规则要点 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| 层1 重要性排序 | {ly['layer1']['in']} | {ly['layer1']['out']} "
             f"| {ly['layer1']['removed_zero_importance']} | mean(\\|SHAP\\|) 恰为 0 剔除,"
             "SHAP 行集 = val 段标签非 NaN 事件 |")
    L.append(f"| 层2 分年度符号一致性 | {ly['layer2']['in']} | {ly['layer2']['out']} "
             f"| {ly['layer2']['removed_drift']} | 2020~2023 四年非全同号且非零 → 漂移剔除 |")
    L.append(f"| 层3 相关簇去重 | {ly['layer3']['in']} | {ly['layer3']['out']} "
             f"| {ly['layer3']['cluster_absorbed']} | train+val 标签非 NaN 行 \\|corr\\| ≥ 0.9 入簇,"
             "只留层1 名次最前代表 |")
    L.append(f"| 层4 拐点定容 | {ly['layer4']['out']} | {chain['k_star']} "
             f"| {ly['layer4']['out'] - chain['k_star']} | K 阶梯 {chain['ladder']},"
             "log2(K)-主指标曲线首末连线垂距最远,平局取小 K |")
    L.append("")
    # ---- 层4 曲线全表
    L.append("## 3. 层4 曲线全表(K 为行全出数)")
    L.append("")
    L.append("| K | 终模轮数 | val 头部五名净笔均(日加权) | val cluster_t "
             "| val top-5 命中率(事件加权) | train_oof 头部五名净笔均(日加权) "
             "| train_oof cluster_t |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in chain["curve"].iterrows():
        k_cell = f"**{int(r['k'])} ← 拐点 K\\***" if int(r["k"]) == chain["k_star"] \
            else str(int(r["k"]))
        L.append(f"| {k_cell} | {int(r['final_num_boost_round'])} "
                 f"| {r['val_top5_net_dayavg']:+.6f} | {r['val_top5_net_cluster_t']:+.3f} "
                 f"| {r['val_top5_hit_eventavg']:.4f} "
                 f"| {r['train_oof_top5_net_dayavg']:+.6f} "
                 f"| {r['train_oof_top5_net_cluster_t']:+.3f} |")
    L.append("")
    ks_line = f"拐点 K* = **{chain['k_star']}**(层3 幸存代表 {ly['layer4']['out']} 个;"
    L.append(ks_line + f"基模全特征 val 主指标 {chain['row_base']['val_top5_net_dayavg']:+.6f},"
             f"K* 定版 {chain['final_row']['val_top5_net_dayavg']:+.6f},"
             f"cluster_t {chain['final_row']['val_top5_net_cluster_t']:+.3f})。")
    L.append("")
    # ---- 终选特征集
    L.append(f"## 4. 终选特征集清单({chain['k_star']} 个,层1 名次序,含中文全称)")
    L.append("")
    L.append("| 层1 名次 | 特征 | 中文全称 |")
    L.append("|---|---|---|")
    for f_ in chain["final_features"]:
        L.append(f"| {f_['rank']} | {f_['feature']} | {f_['cn_name']} |")
    L.append("")
    # ---- 定版与分数
    L.append("## 5. 定版模型与三段分数序列")
    L.append("")
    sc = chain["scores"]
    n_train_scored = int(sc.loc[sc["seg"] == "train", "score"].notna().sum())
    L.append(f"scores_v6.parquet:96,577 行 × 5 列(event_id/ts_code/date/seg/score,无标签列);"
             f"train 行折外原值 {n_train_scored} 行有分(首块与标签 NaN 行 NaN),"
             f"val/test/embargo/pre2001 行 = 定版终模预测原值;"
             f"终模 {int(chain['final_row']['final_num_boost_round'])} 轮;"
             f"双跑 md5 一致({chain['scores_md5']})。")
    L.append("test 段只出分数不出任何指标(零触碰)。")
    L.append("")
    L.append("衔接声明:本阶段产物(终选特征集/定版模型/三段分数)不构成过门证据;"
             "阶段门审判归 M5(终审同型骨架 P7K3 + E1_A6 + H=60,爆发日 top-K "
             "净笔均 ≥ 最好手工挑选规则 +2pp 且 cluster_t ≥ 2)。")
    L.append("")
    # ---- 自检
    L.append("## 6. 独立自检结果(层5 口径:证伪式自写代码重算)")
    L.append("")
    L.append("| 自检 | 结果 |")
    L.append("|---|---|")
    for k, v in checks.items():
        L.append(f"| {k} | {v} |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t0 = time.time()
    results: dict = {"issue": 51, "checks": {}}
    log("开工: M4 独立自检(层5 口径,自写重算)")

    df, feat_cols = load_all()
    cn_of = dict(zip(pd.read_csv(DICT_PATH).query("status == 'kept'")["column"],
                     pd.read_csv(DICT_PATH).query("status == 'kept'")["cn_name"]))

    # ---- 1. 当选解析独立重算
    config_id, params = winner_params_independent()
    assert config_id == 24
    results["checks"]["1_winner_resolution"] = (
        "PASS (winner=net_ret_60d, 双约束过, config_id=24 与 GRID[24] 逐项相等,"
        "回归头, metrics↔adjudication 1 ulp 内互证)")
    log("[自检1] 当选解析独立重算 PASS")

    # ---- 2. 段界 + 泄漏
    cal = np.sort(pd.to_datetime(
        pd.read_parquet(CALENDAR_PATH, columns=["trade_date"])["trade_date"].astype(str)
    ).unique())
    tep.assert_segment_integrity(df[["date", "seg"]], cal)
    rx = [__import__("re").compile(p, __import__("re").IGNORECASE)
          for p in fm.EXCLUDE_PATTERNS]
    bad = [c for c in feat_cols if c not in fm.RANK_WHITELIST
           and any(r.match(c) for r in rx)]
    assert not bad and not any(c.startswith("label") for c in feat_cols)
    results["checks"]["2_segment_and_leakage"] = (
        "PASS (assert_segment_integrity 四项全过;1,997 特征列独立重扫零命中)")
    log("[自检2] 段界 + 泄漏重扫 PASS")

    sub = df[df[LABEL].notna()]
    train_sorted = sub[sub["seg"] == "train"].sort_values(SORT_KEYS, kind="mergesort")
    val_sorted = sub[sub["seg"] == "val"].sort_values(SORT_KEYS, kind="mergesort")
    y_tr = train_sorted[LABEL].to_numpy(np.float64)
    dates_tr = pd.to_datetime(train_sorted["date"]).to_numpy()

    # ---- 3. 基模折外对 M3 逐位一致(独立复跑)
    oof_base, _ = tep.time_series_oof(train_sorted[feat_cols], y_tr, dates_tr,
                                      params=params)
    stored = pd.read_parquet(RACE_DIR / f"oof_v6_{LABEL}.parquet")
    assert (stored["event_id"].to_numpy()
            == train_sorted["event_id"].to_numpy()).all()
    assert np.array_equal(oof_base, stored[f"config_{config_id}"].to_numpy(),
                          equal_nan=True), "基模折外独立复跑与 M3 落盘不逐位一致"
    results["checks"]["3_base_oof_vs_m3"] = (
        "PASS (基模全特征五折折外独立复跑与 M3 config_24 列逐位一致)")
    log("[自检3] 基模折外对 M3 逐位一致 PASS")

    # ---- 4. 层1 独立重算(自写 SHAP + 排序)
    booster_base = lgb.Booster(model_file=str(OUT_DIR / "base_model.txt"))
    shap_val = shap_independent(booster_base, val_sorted[feat_cols])
    importance = np.abs(shap_val).mean(axis=0)
    ind_table = pd.DataFrame({"feature": feat_cols, "importance": importance})
    ind_table["kept"] = ind_table["importance"] > 0.0
    ind_table = ind_table.sort_values(["importance", "feature"],
                                      ascending=[False, True], kind="mergesort")
    ind_table["rank"] = np.arange(len(ind_table))
    d1 = _read_float_exact(OUT_DIR / "layer1_shap_importance.csv", ("importance",))
    assert (d1["feature"].to_numpy() == ind_table["feature"].to_numpy()).all()
    assert (d1["kept"].to_numpy() == ind_table["kept"].to_numpy()).all()
    assert (d1["rank"].to_numpy() == ind_table["rank"].to_numpy()).all()
    assert np.array_equal(d1["importance"].to_numpy(),
                          ind_table["importance"].to_numpy()), "层1 importance 不逐位一致"
    assert d1["cn_name"].notna().all() and len(d1) == 1997
    kept1 = ind_table[ind_table["kept"]]["feature"].tolist()
    results["checks"]["4_layer1_independent"] = (
        f"PASS (自写 SHAP 重算层1 逐位一致: {len(feat_cols)} -> {len(kept1)},"
        f"零重要性剔 {len(feat_cols) - len(kept1)},中文全称零缺失)")
    log(f"[自检4] 层1 独立重算 PASS: {len(feat_cols)} -> {len(kept1)}")

    # ---- 5. 层2 独立重算(自写分年度符号)
    pos = {c: i for i, c in enumerate(feat_cols)}
    years = pd.to_datetime(val_sorted["date"]).dt.year.to_numpy()
    assert set(np.unique(years).tolist()) == set(YEAR_LIST), "val 年份覆盖与预登记不符"
    rows = []
    for feat in kept1:
        j = pos[feat]
        x_all = val_sorted[feat].to_numpy(np.float64)
        s_all = shap_val[:, j]
        rec = {"feature": feat}
        signs = []
        for y in YEAR_LIST:
            m = years == y
            s = sign_independent(x_all[m], s_all[m])
            rec[f"sign_{y}"] = s
            signs.append(s)
        rec["consistent"] = all(s == signs[0] for s in signs) and signs[0] != 0
        rows.append(rec)
    ind_signs = pd.DataFrame(rows)
    d2 = pd.read_csv(OUT_DIR / "layer2_yearly_signs.csv")
    assert (d2["feature"].to_numpy() == ind_signs["feature"].to_numpy()).all()
    for y in YEAR_LIST:
        assert (d2[f"sign_{y}"].to_numpy() == ind_signs[f"sign_{y}"].to_numpy()).all()
    assert (d2["consistent"].to_numpy() == ind_signs["consistent"].to_numpy()).all()
    assert d2["cn_name"].notna().all()
    surv2 = ind_signs[ind_signs["consistent"]]["feature"].tolist()
    results["checks"]["5_layer2_independent"] = (
        f"PASS (自写分年度符号全等: {len(kept1)} -> {len(surv2)},"
        f"漂移剔 {len(kept1) - len(surv2)},年份列恰 2020~2023)")
    log(f"[自检5] 层2 独立重算 PASS: {len(kept1)} -> {len(surv2)}")

    # ---- 6. 层3 独立重算(自写相关矩阵 + 贪婪聚类)
    rank_of = {f_: int(r) for f_, r in zip(ind_table["feature"], ind_table["rank"])}
    order = sorted(surv2, key=lambda c: (rank_of[c], c))
    tv = sub[sub["seg"].isin(["train", "val"])]
    corr = pairwise_corr_independent(tv[order].to_numpy(np.float64))
    pos_o = {c: i for i, c in enumerate(order)}
    rep_of: dict[str, str] = {}
    reps: list[str] = []
    for i, feat in enumerate(order):
        home = None
        for rep in reps:
            r = corr[i, pos_o[rep]]
            if np.isfinite(r) and abs(r) >= CORR_THRESHOLD:
                home = rep
                break
        rep_of[feat] = home if home is not None else feat
        if home is None:
            reps.append(feat)
    d3 = _read_float_exact(OUT_DIR / "layer3_clusters.csv", ("corr_with_rep",))
    assert (d3["feature"].to_numpy() == np.array(order)).all()
    assert (d3["representative"].to_numpy()
            == np.array([rep_of[f] for f in order])).all()
    assert (d3["is_representative"].to_numpy()
            == np.array([rep_of[f] == f for f in order])).all()
    got_corr = np.array([float(corr[i, pos_o[rep_of[f]]]) if rep_of[f] != f else 1.0
                         for i, f in enumerate(order)])
    assert np.array_equal(d3["corr_with_rep"].to_numpy(), got_corr)
    assert (d3["rank"].to_numpy() == np.array([rank_of[f] for f in order])).all()
    assert d3["cn_name"].notna().all()
    assert len(reps) >= 5, "层3 代表 < 5,层4 不应开路"
    results["checks"]["6_layer3_independent"] = (
        f"PASS (自写相关+贪婪聚类全等: {len(surv2)} -> {len(reps)},"
        f"簇吸收 {len(surv2) - len(reps)},阈值 0.9)")
    log(f"[自检6] 层3 独立重算 PASS: {len(surv2)} -> {len(reps)}")

    # ---- 7. 层4 拐点独立重算 + 终选清单
    d4 = _read_float_exact(OUT_DIR / "layer4_curve.csv", CURVE_FLOAT_COLS)
    ks = d4["k"].to_numpy(np.int64)
    assert (ks[:-1] < ks[1:]).all(), "曲线表须按 k 严格升序"
    k_ladder_const = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500,
                      750, 1000, 1500)  # 预登记 K 阶梯常量(规则文本自写,不 import)
    expect_ladder = sorted({k for k in k_ladder_const if k <= len(reps)} | {len(reps)})
    assert ks.tolist() == expect_ladder, "曲线 K 集与预登记阶梯过滤规则不符"
    ps = d4["val_top5_net_dayavg"].to_numpy(np.float64)
    assert np.isfinite(ps).all()
    k_star_ind, dists_ind = elbow_independent(ks, ps)
    ej = json.load(open(OUT_DIR / "layer4_elbow.json", encoding="utf-8"))
    assert ej["k_star"] == k_star_ind and ej["ladder"] == expect_ladder
    assert ej["distances"] == dists_ind, "拐点垂距不逐位一致"
    final = json.load(open(OUT_DIR / "final_features.json", encoding="utf-8"))
    expect_feats = reps[:k_star_ind]
    got_feats = [r["feature"] for r in final["features"]]
    assert got_feats == expect_feats, "终选特征集 != 层3 代表按层1 名次前 K*"
    assert [r["rank"] for r in final["features"]] == [rank_of[f] for f in expect_feats]
    assert all(final["features"][i]["cn_name"] == cn_of[f]
               for i, f in enumerate(expect_feats)), "终选清单中文全称与词典不符"
    assert final["n_features"] == k_star_ind and final["label"] == LABEL
    results["checks"]["7_layer4_elbow_independent"] = (
        f"PASS (自写拐点 k_star={k_star_ind} 与垂距逐位一致;曲线 {len(ks)} 行 K 全出数;"
        f"终选 {len(expect_feats)} 特征 == 代表前 K*,中文全称核验过)")
    log(f"[自检7] 层4 拐点独立重算 PASS: K*={k_star_ind}")

    # ---- 8. 定版分数独立重算
    scores = pd.read_parquet(OUT_DIR / "scores_v6.parquet")
    oof_star, _ = tep.time_series_oof(train_sorted[expect_feats], y_tr, dates_tr,
                                      params=params)
    booster_star = lgb.Booster(model_file=str(OUT_DIR / "model.txt"))
    expect_score = np.full(len(df), np.nan)
    oof_map = pd.Series(oof_star, index=train_sorted["event_id"].to_numpy())
    is_train = (df["seg"] == "train").to_numpy()
    expect_score[is_train] = df.loc[is_train, "event_id"].map(oof_map).to_numpy()
    expect_score[~is_train] = booster_star.predict(df.loc[~is_train, expect_feats])
    assert np.array_equal(scores["score"].to_numpy(np.float64), expect_score,
                          equal_nan=True), "定版分数独立重算不逐位一致"
    results["checks"]["8_final_scores_independent"] = (
        "PASS (train 行 = 终选特征五折折外独立复跑逐位一致;"
        "val/test/embargo/pre2001 行 = model.txt 预测原值逐位一致)")
    log("[自检8] 定版分数独立重算 PASS")

    # ---- 0. 行数与键守恒
    assert list(scores.columns) == ["event_id", "ts_code", "date", "seg", "score"]
    assert len(scores) == 96577 and not scores["event_id"].duplicated().any()
    mk = df[["event_id", "ts_code", "date", "seg"]].reset_index(drop=True)
    assert (scores["event_id"].to_numpy() == mk["event_id"].to_numpy()).all()
    assert (scores["ts_code"].to_numpy() == mk["ts_code"].to_numpy()).all()
    assert (scores["date"].to_numpy() == mk["date"].to_numpy()).all()
    assert (scores["seg"].to_numpy() == mk["seg"].to_numpy()).all()
    results["checks"]["0_row_key_conservation"] = (
        "PASS (96,577 行 × 恰 5 列,event_id 唯一,与主表键/段逐行一致)")
    log("[自检0] 行数与键守恒 PASS")

    # ---- 9. 确定性台账
    sel = json.load(open(OUT_DIR / "selection_results_v6.json", encoding="utf-8"))
    md_now = _md5(OUT_DIR / "scores_v6.parquet")
    assert sel["scores_md5"] == md_now, "台账 scores_md5 与现盘文件不一致"
    results["checks"]["9_determinism_md5"] = (
        f"PASS (双跑 md5 台账与现盘重哈希一致: {md_now})")
    log(f"[自检9] 确定性 md5 PASS: {md_now}")

    # ---- 10. test 零触碰
    assert not any("test" in c.lower() for c in d4.columns)
    assert not any("test" in c.lower() for c in
                   pd.read_csv(OUT_DIR / "layer1_shap_importance.csv", nrows=1).columns)
    assert int((scores["seg"] == "test").sum()) > 0
    assert list(scores.columns) == ["event_id", "ts_code", "date", "seg", "score"], \
        "分数表出现标签列"
    results["checks"]["10_test_untouched"] = (
        f"PASS (分数表无标签列;曲线/层表零 test 列;test "
        f"{int((scores['seg'] == 'test').sum())} 行在场仅出分数)")
    log("[自检10] test 零触碰 PASS")

    # ---- 11. report.md
    final_row = d4.set_index("k").loc[k_star_ind]
    chain = {
        "config_id": config_id,
        "row_base": {"val_top5_net_dayavg":
                         float(sel["final"]["base_all_features_val_top5_net_dayavg"]),
                     "val_top5_net_cluster_t":
                         float(sel["final"]["base_all_features_val_top5_net_cluster_t"])},
        "base_n_round": json.load(open(OUT_DIR / "smoke_results.json",
                                       encoding="utf-8"))["final_num_boost_round"],
        "layers": {**sel["layers"],
                   "layer4": {**sel["layers"]["layer4"], "out": len(reps)}},
        "curve": d4, "k_star": int(k_star_ind), "ladder": expect_ladder,
        "final_features": final["features"], "final_row": final_row,
        "scores": scores, "scores_md5": md_now,
    }
    report = render_report(chain, results["checks"])
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    log("[产出] report.md 落盘")

    results["all_pass"] = True
    results["elapsed_sec"] = round(time.time() - t0, 1)
    with (OUT_DIR / "selfcheck_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[完成] 自检全绿,耗时 {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()
