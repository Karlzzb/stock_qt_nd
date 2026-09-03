#!/usr/bin/env python3
"""特征精选库（issue #27）：当选标签上的 SHAP 五层精选纯函数层。

口径（全部预登记，先于任何精选结果落 issue #27 置顶评论，非调参结果）:
  - 对象: 合并池（#26 正赛）当选标签 hit_N20_k2.0、当选配置（由
    summary_merged.csv + adjudication_merged.json 读出并断言，非手抄）、
    特征池 = 合并池主表 2060 列建模特征（feature_master 权威口径）。
  - 基模: #25/#26 同口径管线（五折 OOF -> 校准层 [p, p²] -> 终模均值轮数）;
    SHAP 用 LightGBM pred_contrib=True（TreeSHAP 精确值），算于 val 段标签
    非 NaN 事件。
  - 层1 重要性排序: importance = mean(|SHAP|); importance 恰为 0 剔除;
    名次 = importance 降序、平局特征名升序。
  - 层2 分年度符号一致性: val 各年份（2019/2020/2021/2022）特征值与其 SHAP
    值的 Pearson 相关定号（成对去 NaN; 年内有效对 < 30 或年内方差为 0 记 0）;
    四年非全同号且非零 -> 漂移剔除。
  - 层3 相关簇去重: train+val 标签非 NaN 行上特征值 Pearson 相关
    （feature_master.pairwise_corr 同口径）; 按层1名次贪婪聚类,
    与既有簇代表 |corr| >= 0.9 入名次最靠前匹配簇, 否则自立为代表; 只留代表。
  - 层4 拐点定容: K 阶梯预登记（见 k_ladder），每 K 取层3幸存按层1名次前 K
    走完整基模管线记 val 头部五名精确率（日加权）; 拐点 = log2(K)-precision
    曲线上距首末连线最远点, 平局取小 K, 末点最远则取末点。
  - 层5 独立复核: 证伪式自写代码重算（不调本模块选择函数），记录落
    t7_review_record.md。

纪律: 选择动作只在 train/val 段; test 段零指标零逐行统计, 分数序列中
test 段 y 一律 NaN（终审前零触碰）。
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

import feature_master as fm
import label_race as lr

# ------------------------------------------------------------- 预登记常量
CORR_CLUSTER_THRESHOLD = 0.9   # 层3 相关簇阈值（T2 同式去重 0.999 之外的近重复簇）
MIN_YEAR_PAIRS = 30            # 层2 年内有效对下限（与 pairwise_corr 同口径）
K_LADDER = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750,
            1000, 1500)        # 层4 候选规模阶梯（外加 N_surviving 自身）
MIN_SURVIVORS = 5              # 层4 开路下限: 幸存特征不足则报错


# ------------------------------------------------------------- 当选配置解析
def resolve_winner(summary: pd.DataFrame, adjudication: dict) -> tuple[str, int, dict]:
    """从标签赛落盘产物解析当选标签与配置（断言守护，不接受手抄）。

    返回 (label, config_id, 完整 LightGBM 参数)。
    """
    winner = adjudication.get("winner")
    assert winner == lr.SNIPER_LABEL, f"裁决 winner={winner} 非预登记狙击标签"
    assert adjudication.get("ap_constraint_passed") is True, "平均精确率中位数约束未过"
    row = summary[summary["candidate"] == winner]
    assert len(row) == 1, f"summary 中 {winner} 行数 {len(row)} != 1"
    row = row.iloc[0]
    config_id = int(row["config_id"])
    expect = lr.GRID[config_id]
    got = {k: row[k] for k in lr.GRID_KEYS}
    for k, v in expect.items():
        assert float(got[k]) == float(v), \
            f"summary 第 {config_id} 行超参 {k}={got[k]} 与预登记网格 {v} 不符"
    return winner, config_id, lr.grid_params(expect)


# ------------------------------------------------------------- SHAP 值
def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP 精确值（pred_contrib=True，末列偏置项剔除）。

    返回 (n_rows, n_features) float64; 特征序与 X.columns 一致。
    """
    raw = booster.predict(X, pred_contrib=True)
    assert raw.shape == (len(X), X.shape[1] + 1), \
        f"pred_contrib 形状 {raw.shape} 与预期 {(len(X), X.shape[1] + 1)} 不符"
    contrib = np.asarray(raw[:, :-1], dtype=np.float64)
    assert np.isfinite(contrib).all(), "SHAP 值含 NaN/inf"
    return contrib


# ------------------------------------------------------------- 层1 重要性排序
def layer1_rank(shap: np.ndarray, feat_cols: list[str]) -> pd.DataFrame:
    """层1: mean(|SHAP|) 重要性排序表（feature/importance/rank/kept 四列）。

    名次 = importance 降序、平局特征名升序（mergesort 确定性）;
    kept = importance 恰大于 0（恰为 0 者从未入树, 剔除并记录）。
    """
    assert shap.ndim == 2 and shap.shape[1] == len(feat_cols)
    importance = np.abs(shap).mean(axis=0)
    table = pd.DataFrame({"feature": feat_cols, "importance": importance})
    table["kept"] = table["importance"] > 0.0
    table = table.sort_values(["importance", "feature"],
                              ascending=[False, True], kind="mergesort")
    table["rank"] = np.arange(len(table))
    return table[["feature", "importance", "rank", "kept"]].reset_index(drop=True)


# ------------------------------------------------------------- 层2 分年度符号一致性
def _sign_of(x: np.ndarray, y: np.ndarray) -> int:
    """成对去 NaN 后的 Pearson 相关符号: +1 / -1 / 0。

    预登记口径: 年内有效对 < MIN_YEAR_PAIRS 或年内特征方差为 0 记 0。
    相关无法定义的另两种形态（SHAP 方差为 0、相关恰为 0.0）同属"无稳定方向",
    一并记 0 —— 只可能触发剔除, 不可能促成幸存。
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < MIN_YEAR_PAIRS:
        return 0
    xs, ys = x[mask], y[mask]
    if xs.std() == 0.0 or ys.std() == 0.0:
        return 0
    r = float(np.corrcoef(xs, ys)[0, 1])
    return 0 if r == 0.0 else (1 if r > 0 else -1)


def layer2_yearly_signs(X: pd.DataFrame, shap: np.ndarray, years: np.ndarray,
                        feat_cols: list[str], year_list: tuple = (2019, 2020, 2021, 2022),
                        ) -> pd.DataFrame:
    """层2: 分年度符号一致性表（feature/sign_<年>.../consistent 列）。

    四年符号全同号且非零 -> consistent=True（幸存）; 否则漂移剔除。
    X 为 val 段特征帧（列序 = feat_cols）, shap 同形, years 为事件日年份数组。
    """
    assert len(X) == shap.shape[0] == len(years)
    assert shap.shape[1] == len(feat_cols)
    years = np.asarray(years)
    observed = set(np.unique(years).tolist())
    assert observed == set(year_list), \
        f"层2 年份覆盖 {sorted(observed)} 与预登记 {sorted(year_list)} 不一致"
    pos = {c: i for i, c in enumerate(X.columns)}
    rows = []
    for j, feat in enumerate(feat_cols):
        x_all = X.iloc[:, pos[feat]].to_numpy(dtype=np.float64)
        s_all = shap[:, j]
        rec = {"feature": feat}
        signs = []
        for y in year_list:
            m = years == y
            s = _sign_of(x_all[m], s_all[m])
            rec[f"sign_{y}"] = s
            signs.append(s)
        rec["consistent"] = all(s == signs[0] for s in signs) and signs[0] != 0
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 层3 相关簇去重
def layer3_clusters(X: pd.DataFrame, feat_cols: list[str],
                    rank_of: dict) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    """层3: 相关簇贪婪去重（|corr| >= 0.9, 簇内留层1名次最前者为代表）。

    X 为 train+val 特征帧; feat_cols 为层2幸存特征; rank_of: feature -> 层1名次。
    按名次升序遍历: 与既有簇代表 |corr| >= threshold 则入名次最靠前的匹配簇,
    否则自立新簇为代表。NaN 相关（成对不足/常数列）不判同簇。
    返回 (representatives 按名次序, clusters 明细表, corr 矩阵)。
    """
    assert feat_cols, "层3 输入特征为空"
    order = sorted(feat_cols, key=lambda c: (rank_of[c], c))
    pos = {c: i for i, c in enumerate(order)}
    x = X[order].to_numpy(np.float64)
    corr = fm.pairwise_corr(x)
    rep_of: dict[str, str] = {}          # member -> representative
    reps: list[str] = []                 # 代表, 按名次序
    for i, feat in enumerate(order):
        home = None
        for rep in reps:                 # reps 按名次升序, 首个匹配即名次最靠前
            r = corr[i, pos[rep]]
            if np.isfinite(r) and abs(r) >= CORR_CLUSTER_THRESHOLD:
                home = rep
                break
        if home is None:
            reps.append(feat)
            rep_of[feat] = feat
        else:
            rep_of[feat] = home
    records = pd.DataFrame({
        "feature": order,
        "representative": [rep_of[f] for f in order],
        "is_representative": [rep_of[f] == f for f in order],
        "corr_with_rep": [float(corr[i, pos[rep_of[f]]])
                          if rep_of[f] != f else 1.0 for i, f in enumerate(order)],
        "rank": [rank_of[f] for f in order],
    })
    return reps, records, corr


# ------------------------------------------------------------- 层4 拐点定容
def k_ladder(n_surviving: int) -> list[int]:
    """预登记 K 阶梯: 常量阶梯过滤 <= n 后并入 n 自身, 升序去重。n < 5 报错。"""
    assert n_surviving >= MIN_SURVIVORS, \
        f"幸存特征 {n_surviving} < {MIN_SURVIVORS}, 层4 不开路"
    ks = sorted({k for k in K_LADDER if k <= n_surviving} | {n_surviving})
    return ks


def find_elbow(curve: pd.DataFrame) -> dict:
    """拐点: x=log2(K)、y=precision, 首末连线垂距最远点; 平局取小 K。

    curve 须含 k/val_precision_at_5_dayavg 两列且按 k 升序、无重复。
    返回 dict(k_star, distances, secant_endpoints)。
    """
    need = {"k", "val_precision_at_5_dayavg"}
    assert need <= set(curve.columns), f"曲线表缺列: {need - set(curve.columns)}"
    ks = curve["k"].to_numpy(dtype=np.int64)
    assert (ks[:-1] < ks[1:]).all(), "曲线表须按 k 严格升序"
    ps = curve["val_precision_at_5_dayavg"].to_numpy(dtype=np.float64)
    assert np.isfinite(ps).all(), "曲线含 NaN"
    if len(ks) == 1:
        return {"k_star": int(ks[0]), "distances": [0.0]}
    xs = np.log2(ks.astype(np.float64))
    p0 = np.array([xs[0], ps[0]])
    p1 = np.array([xs[-1], ps[-1]])
    v = p1 - p0
    norm = float(np.hypot(*v))
    assert norm > 0.0, "k 严格升序保证首末点不重合"
    # 二维叉积显式展开: |v_x*(y-y0) - v_y*(x-x0)| / |v|
    dists = np.abs(v[0] * (ps - p0[1]) - v[1] * (xs - p0[0])) / norm
    # "平局取小 K" 的浮点化: 相对容差 1e-12 内视同并列（共线点浮点尘不判"最远"）
    tol = 1e-12 * max(1.0, float(dists.max()))
    tied = np.flatnonzero(dists >= dists.max() - tol)
    k_star = int(ks[tied[0]])
    return {"k_star": k_star, "distances": [float(d) for d in dists]}


# ------------------------------------------------------------- 分数序列装配
def assemble_scores(keys: pd.DataFrame, prob: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """分数序列: (pool 可选, ts_code, date, event_id, seg, prob, y)。

    keys 须含 ts_code/date/event_id/seg（合并池另有 pool）; prob 允许 NaN
    （train 首块无折外）; test 段调用方必须传 y=NaN（零触碰纪律在此断言守护）。
    """
    need = {"ts_code", "date", "event_id", "seg"}
    assert need <= set(keys.columns), f"键列缺失: {need - set(keys.columns)}"
    assert len(keys) == len(prob) == len(y)
    out = keys.reset_index(drop=True).copy()
    out["prob"] = np.asarray(prob, dtype=np.float64)
    out["y"] = np.asarray(y, dtype=np.float64)
    assert out.loc[out["seg"] == "test", "y"].isna().all(), \
        "test 段 y 必须为 NaN（终审前零触碰）"
    assert not out["event_id"].duplicated().any(), "event_id 不唯一"
    return out
