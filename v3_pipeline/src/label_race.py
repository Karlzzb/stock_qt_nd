#!/usr/bin/env python3
"""标签赛库（issue #26）：十九候选标签、预登记小网格、选配置与裁决纯函数。

口径（全部预登记，先于任何结果落 #26 评论，非调参结果）:
  - 候选十九个：狙击标签 hit_N20_k2.0 一个（各池 labels.parquet div 组，
    定义见 m_scan：T+1 开盘入场、20 交易日内 +2*ATR(14)），
    加两个收益族各自九个视野（3/5/10/15/20/25/30/45/60 个交易日）的
    "收益大于零"二分类版：
      cur       T 收盘入场  -> T+1+h 收盘出场   （复用 label_candidates.cur_return）
      open_exec T+1 开盘入场 -> T+1+h 收盘出场  （复用 label_candidates.open_exec_return）
    收益经 shift 取该票第 h+1 个后续可用 bar，截断/停牌顺延产出 NaN，按段剔除并登记。
  - 样本量兜底（#20 预登记）：主池训练段事件数 2838 < 3000，兜底触发——
    主备合并池（master_main ∪ master_backup 行并集，event_id 加池前缀，不去重）
    升为正赛，主池降为对照；合并池 train 22963 >= 3000。
  - 网格：仅叶子数、叶子最小样本数、学习率、特征采样比例四项，
    3x3x2x2 = 36 组全因子（<= 50 上限），其余参数沿用
    train_eval_pipeline.DEFAULT_LGBM_PARAMS（含复现性四件套与 num_threads=8）。
  - 每配置管线与 #25 完全一致：训练段五折时间序列折外概率 ->
    校准层 [p, p²] -> 全训练段终模（轮数 = 五折 best_iteration 均值）->
    指标表行 = train_oof（折外口径）与 val（终模+校准口径）。
  - 每候选选配置：val 头部五名精确率（日加权）最高者；
    平局取 val 平均精确率较高者；再平局取网格序靠前者。
  - 裁决（正赛 = 合并池，无自由裁量）：当选 = 当选配置 val 头部五名精确率
    （日加权）最高的候选；平局取 val 平均精确率较高者，再平局取候选序靠前者；
    且其 val 平均精确率不得低于十九个候选（各取当选配置）的中位数，
    不满足则本赛季无当选（阴性结论落盘，不降格另选）。
  - 复现性：十九个候选各自的当选配置，OOF 全程复跑一遍逐位一致才生效。
  - test 段零触碰：只在场断言，不出任何数字（#20 终审前每候选只碰一次）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import label_candidates as lc
import train_eval_pipeline as tep

# ------------------------------------------------------------- 预登记常量
RACE_HORIZONS = (3, 5, 10, 15, 20, 25, 30, 45, 60)
RETURN_FAMILIES = ("cur", "open_exec")
SNIPER_LABEL = "hit_N20_k2.0"

FAMILY_FUNCS = {"cur": lc.cur_return, "open_exec": lc.open_exec_return}
FAMILY_CN = {"cur": "T 收盘入场收益大于零", "open_exec": "T+1 开盘入场收益大于零"}

MIN_TRAIN_EVENTS = 3000  # #20 兜底阈值：主池训练段事件数不足则主备合并池升正赛

# 预登记网格：四项超参全因子 3x3x2x2 = 36 组（<= 50）
GRID: list[dict] = [
    {"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": rate,
     "feature_fraction": ff}
    for nl in (15, 31, 63)
    for md in (50, 100, 200)
    for rate in (0.05, 0.10)
    for ff in (0.6, 0.8)
]
GRID_KEYS = ("num_leaves", "min_data_in_leaf", "learning_rate", "feature_fraction")
assert 0 < len(GRID) <= 50, "预登记网格须不超过 50 组"
assert all(set(g) == set(GRID_KEYS) for g in GRID), "网格只允许四项超参"
assert len({tuple(g[k] for k in GRID_KEYS) for g in GRID}) == len(GRID), "网格有重复组"


# ------------------------------------------------------------- 候选标签
def return_label_name(family: str, h: int) -> str:
    assert family in RETURN_FAMILIES, f"未知收益族: {family}"
    assert h in RACE_HORIZONS, f"未知视野: {h}"
    return f"{family}_pos_{h}d"


def candidate_labels() -> list[str]:
    """十九候选，固定顺序：狙击标签在前，其后 cur 族九视野、open_exec 族九视野。"""
    return ([SNIPER_LABEL]
            + [return_label_name(f, h) for f in RETURN_FAMILIES for h in RACE_HORIZONS])


def candidate_cn(name: str) -> str:
    """候选标签中文全名（命名纪律：产物表须带中文全名列）。"""
    if name == SNIPER_LABEL:
        return "背离狙击命中标签（T+1 开盘入场，20 交易日内触及 +2 倍 ATR(14)）"
    for fam in RETURN_FAMILIES:
        prefix = f"{fam}_pos_"
        if name.startswith(prefix) and name.endswith("d"):
            h = int(name[len(prefix):-1])
            assert h in RACE_HORIZONS, f"未知视野: {name}"
            return f"{FAMILY_CN[fam]}二分类标签（{h} 个交易日视野）"
    raise AssertionError(f"未知候选标签: {name}")


def compute_return_labels(d: pd.DataFrame) -> pd.DataFrame:
    """单票日线 -> 十八个"收益大于零"二分类标签列（键列 date 在前）。

    d 须含 trade_date/open/high/low/close；内部按 trade_date 升序重排。
    收益为 NaN（尾部截断/停牌顺延）时标签保持 NaN，不由 (ret > 0) 吞成 0。
    行 t 的标签只用行 > t 的数据（由 label_candidates 的 shift 构造保证，
    头部截断不变性由本模块单测钉死）。
    """
    need = {"trade_date", "open", "high", "low", "close"}
    assert need <= set(d.columns), f"日线缺列: {need - set(d.columns)}"
    d = d.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    out = pd.DataFrame({"date": pd.to_datetime(d["trade_date"])})
    for fam in RETURN_FAMILIES:
        func = FAMILY_FUNCS[fam]
        for h in RACE_HORIZONS:
            ret = func(d, h)
            out[return_label_name(fam, h)] = np.where(
                ret.isna(), np.nan, (ret > 0).to_numpy(dtype=np.float64))
    return out


# ------------------------------------------------------------- 合并池
def build_merged_master(main: pd.DataFrame, backup: pd.DataFrame) -> pd.DataFrame:
    """主备合并池：行并集（不去重），event_id 加池前缀，新增 pool 溯源列。

    两池同 (ts_code,date) 事件属不同背离几何（事件级特征不同），按预登记各留一行；
    pool 为字符串列，feature_master.feature_columns 数值口径自动排除，不进特征。
    dtype 归一：downtrend/hammer_signal 在备池为 object(bool, 含 None 缺失)、
    主池为 int8（T4 构建期漂移），合并前统一为可空 Int8（dtype.kind='i' 仍属
    数值特征口径，缺失保留），使两池特征列口径一致（2060 列）。
    """
    out = []
    for pool, df in (("main", main), ("backup", backup)):
        part = df.copy()
        for col in ("downtrend", "hammer_signal"):
            if col in part.columns and part[col].dtype == object:
                vals = set(part[col].dropna().unique())
                assert vals <= {True, False, 0, 1}, f"{pool}.{col} 含非布尔值: {vals}"
                part[col] = part[col].map({True: 1, False: 0, 1: 1, 0: 0}).astype("Int8")
        part["event_id"] = pool + "_" + part["event_id"].astype(str)
        part["pool"] = pool
        out.append(part)
    merged = pd.concat(out, ignore_index=True)
    assert not merged["event_id"].duplicated().any(), "合并池 event_id 不唯一"
    return merged


# ------------------------------------------------------------- 网格跑训
def grid_params(overrides: dict) -> dict:
    """单组网格参数 -> 完整 LightGBM 参数（其余项沿用 #25 冒烟口径）。"""
    assert set(overrides) <= set(GRID_KEYS), f"网格越权超参: {set(overrides) - set(GRID_KEYS)}"
    params = dict(tep.DEFAULT_LGBM_PARAMS)
    params.update(overrides)
    return params


def run_single_config(train: pd.DataFrame, val: pd.DataFrame, feat_cols: list[str],
                      label_col: str, params: dict) -> tuple[dict, dict]:
    """单候选单配置：五折 OOF -> 校准层 [p, p²] -> 终模 -> train_oof/val 指标行。

    返回 (row, artifacts)；row 可直接作指标表一行，artifacts 含
    oof/oof_mask/val_prob/best_iters 供落盘与复跑断言。test 段不得传入。
    """
    assert train["seg"].eq("train").all() and val["seg"].eq("val").all(), \
        "run_single_config 只接受 train/val 段"
    train = train.sort_values(["date", "ts_code", "event_id"], kind="mergesort")
    X_tr = train[feat_cols]
    y_tr = train[label_col].to_numpy(dtype=np.float64)
    dates_tr = pd.to_datetime(train["date"]).to_numpy()

    oof, best_iters = tep.time_series_oof(X_tr, y_tr, dates_tr, params=params)
    oof_mask = np.isfinite(oof)
    calibrator = tep.SquaredLogitCalibrator().fit(oof[oof_mask], y_tr[oof_mask])
    n_round = tep.final_num_boost_round(best_iters)
    booster = tep.fit_final_model(X_tr, y_tr, num_boost_round=n_round, params=params)

    train_ev = train.loc[oof_mask, ["date", "ts_code", "event_id", "seg"]].copy()
    train_ev["y"] = y_tr[oof_mask]
    train_ev["prob"] = calibrator.predict(oof[oof_mask])
    row_train = tep.evaluate_segment(train_ev)

    val_ev = val[["date", "ts_code", "event_id", "seg"]].copy()
    val_ev["y"] = val[label_col].to_numpy(dtype=np.float64)
    val_ev["prob"] = calibrator.predict(booster.predict(val[feat_cols]))
    row_val = tep.evaluate_segment(val_ev)

    row = {
        "best_iters": ",".join(str(i) for i in best_iters),
        "final_num_boost_round": n_round,
        "calib_coef_p": float(calibrator.coef_[0]),
        "calib_coef_p2": float(calibrator.coef_[1]),
        "calib_intercept": calibrator.intercept_,
    }
    for k_, v_ in row_train.items():
        row[f"train_oof_{k_}"] = v_
    for k_, v_ in row_val.items():
        row[f"val_{k_}"] = v_
    artifacts = {"oof": oof, "oof_mask": oof_mask, "best_iters": best_iters,
                 "val_ev": val_ev}
    return row, artifacts


# ------------------------------------------------------------- 选配置与裁决（纯函数）
def select_best_config(metrics: pd.DataFrame) -> int:
    """每候选选配置：val 头部五名精确率（日加权）最高 -> val 平均精确率较高 -> 网格序靠前。

    metrics 为该候选的指标表（配置为行，须含 config_id/val_precision_at_5_dayavg/
    val_average_precision 列）；返回当选行在表中的位置索引（iloc 口径）。
    """
    need = {"config_id", "val_precision_at_5_dayavg", "val_average_precision"}
    assert need <= set(metrics.columns), f"指标表缺列: {need - set(metrics.columns)}"
    assert len(metrics) > 0, "指标表为空"
    order = metrics.sort_values(
        ["val_precision_at_5_dayavg", "val_average_precision", "config_id"],
        ascending=[False, False, True], kind="mergesort")
    return int(metrics.index.get_loc(order.index[0]))


def adjudicate(summary: pd.DataFrame) -> dict:
    """裁决（预登记，无自由裁量）：十九候选各取当选配置后的总裁决。

    summary 每候选一行，须含 candidate/val_precision_at_5_dayavg/val_average_precision。
    当选 = val 头部五名精确率（日加权）最高（平局 val 平均精确率较高 ->
    candidate_labels() 序靠前），且其 val 平均精确率 >= 十九候选中位数；
    不满足则 winner=None（无当选）。
    """
    need = {"candidate", "val_precision_at_5_dayavg", "val_average_precision"}
    assert need <= set(summary.columns), f"裁决表缺列: {need - set(summary.columns)}"
    canon = candidate_labels()
    assert list(summary["candidate"]) == canon or set(summary["candidate"]) == set(canon), \
        "裁决表候选集与预登记十九候选不一致"
    rank_order = {c: i for i, c in enumerate(canon)}
    s = summary.copy()
    s["_ord"] = s["candidate"].map(rank_order)
    s = s.sort_values(
        ["val_precision_at_5_dayavg", "val_average_precision", "_ord"],
        ascending=[False, False, True], kind="mergesort")
    top = s.iloc[0]
    median_ap = float(s["val_average_precision"].median())
    passed = bool(top["val_average_precision"] >= median_ap)
    return {
        "winner": str(top["candidate"]) if passed else None,
        "winner_val_precision_at_5_dayavg": float(top["val_precision_at_5_dayavg"]),
        "winner_val_average_precision": float(top["val_average_precision"]),
        "median_val_average_precision": median_ap,
        "ap_constraint_passed": passed,
    }
