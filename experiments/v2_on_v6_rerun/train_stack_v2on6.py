#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2-on-v6 复刻线(issue #54) 步骤 5:特征选择 + 训练栈 + 分数表。

预登记军令状 = 同目录 README.md(commit 49b9c5b,冻结);公式唯一来源 = 同目录
anatomy_report.md(解剖报告);v2 原始代码(/tmp/v2_excavation/)只读参考,禁止 import。

README §六 选择规则(v2 程序化复刻):
  1. 基座 LGBM(§七参数)在 train 段(标签非 NaN)以 FULL 604 列训练,
     TimeSeriesSplit 5 折,逐折 feature_importances_ top-250、入选折占比 ≥ 0.6
     → OPTIMIZED_V2ON6(analyze_feature_importance 算法逐字,L630-708);
  2. 定版基座 = OPTIMIZED_V2ON6 上重跑同一 5 折管线(OOF + PR-AUC 折权重);
  3. STABLE_V2ON6:train OOF 栈结构(StandardScaler[pred_lgb, pred_lgb_squared
     + FULL 604])上 collect_lr_coefs(TimeSeriesSplit 12 折,LR 同 §七参数)
     + analyze_coef_stability,selection_rate ≥ 0.6 → STABLE_V2ON6;
  4. 终版元模型输入 = pred_lgb + pred_lgb_squared + STABLE_V2ON6(pred 列若入选
     则去重并披露)。

README §七 训练管线(v2 架构逐项):
  - LGB_PARAMS 逐字 = comm_fun.py L206-229(n_jobs=4 → 按本机核数 28,README 已登记);
  - 折内训练 <10,000 → min_child_samples = max(5, int(len×0.01))(v2 原样);
  - 特征矩阵 float32、inf→NaN→fillna(0);median imputer 在 train fit(v2 冗余保险);
  - 5 折 fit(eval_set=验证折, eval_metric=[auc,binary_logloss], early_stopping 150);
  - 折权重 = 折验证段 PR-AUC 归一化;非 train 段基座概率 = 5 折加权平均;
  - 元模型 LR(l1/saga/C=0.01/balanced/max_iter=1000/fit_intercept/random_state=42),
    输入 StandardScaler(train 栈 fit) 变换后的 [pred_lgb, pred_lgb_squared, STABLE];
  - scores_v2on6.parquet = event_id/ts_code/date/seg/score 恰 5 列 × 96,577 行
    (train 行 = OOF 栈打分;val/test/embargo/pre2001 = 折加权基座 + 元模型;
    标签 NaN 的 train 行 score=NaN;分数表无标签列)。

用法:
  python3 train_stack_v2on6.py                   # run1
  python3 train_stack_v2on6.py --rerun-tag run2  # 第二进程重跑(§九.7 双跑对账)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from build_master_v2on6 import MASTER_FEATURE_COLS  # noqa: E402  列规格单一来源

CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

# v2 comm_fun.py:L206-229 逐字(n_jobs=4 → 28,README §七 已登记「按本机核数落盘」)
LGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'metric': ['auc', 'binary_logloss'],
    'num_leaves': 63,
    'objective': 'binary',
    'n_jobs': 28,
    'random_state': 42,
    'max_depth': 4,
    'min_child_samples': 58,
    'min_split_gain': 0.6,
    'scale_pos_weight': 1.0,
    'verbosity': -1,
    "reg_alpha": 2,
    "reg_lambda": 2,
    "min_child_weight": 0.03,
    'bagging_fraction': 0.875,
    'bagging_freq': 7,
    'feature_fraction': 0.85,
}
N_SPLITS = 5               # comm_fun.py:L183
EARLY_STOP = 150           # stock_model_Tflod_v2.py:L141
TOP_N_SELECT = 250         # L630 top_n
FOLD_RATIO_SELECT = 0.6    # L630 threshold
LR_COEF_SPLITS = 12        # main L462 n_splits=12
RETURN_THRESHOLD = 0.01    # comm_fun.py:L190
LABEL_COL = "future_return_15d"  # SETTLEMENT_DAYS=15(L186-187)
LR_PARAMS = dict(max_iter=1000, class_weight='balanced', random_state=42,
                 C=0.01, solver='saga', penalty='l1', fit_intercept=True)  # L294-304


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [train_stack] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ================================================================ v2 算法逐字 ========
def train_lgb_models(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """v2 train_lgb_models(L93-188)逐字形:5 折 TimeSeriesSplit,OOF 零初始化
    (block-0 行保持 0,v2 原样),early stopping 150,折 PR-AUC,动态 min_child_samples。
    阈值搜索(P16)/样本权重打印(v2 未启用)不复刻。
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    models, fold_scores = [], []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        lgb_params = LGB_PARAMS.copy()
        if len(tr_idx) < 10000:  # v2 L130-132 动态覆盖
            lgb_params['min_child_samples'] = max(5, int(len(tr_idx) * 0.01))
            log(f"  fold {fold + 1}: min_child_samples → {lgb_params['min_child_samples']}")
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                eval_metric=['auc', 'binary_logloss'],
                callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False)])
        oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        models.append(clf)
        fold_score = average_precision_score(y_val, oof[val_idx])
        fold_scores.append(fold_score)
        log(f"  fold {fold + 1}/{N_SPLITS}: 训练 {len(tr_idx)} 验证 {len(val_idx)} "
            f"best_iter {clf.best_iteration_} PR-AUC {fold_score:.4f}")
    oof_score = average_precision_score(y_train, oof)
    log(f"  整体 OOF PR-AUC {oof_score:.4f};各折 {[round(s, 4) for s in fold_scores]}")
    return oof, models, fold_scores


def select_optimized(lgb_models: list, feature_names: list) -> tuple:
    """v2 analyze_feature_importance(L630-708)选择核逐字:逐折 nlargest(250),
    入选折占比 ≥ 0.6;返回(OPTIMIZED 清单,折占比台账)。"""
    importance_df = pd.DataFrame(index=feature_names)
    n_folds = len(lgb_models)
    for i, model in enumerate(lgb_models):
        importance_df[f'fold_{i}'] = model.feature_importances_
    importance_df['importance_mean'] = importance_df.mean(axis=1)
    importance_df['importance_std'] = importance_df.std(axis=1)
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    # ---- v2 逐折 top_n 计数环(追加顺序 = 首见序,逐字)----
    feature_top_counts: list = []
    for i in range(n_folds):
        fold_importance = importance_df[f'fold_{i}']
        top_features_in_fold = fold_importance.nlargest(TOP_N_SELECT).index.tolist()
        for feature in feature_names:
            if feature in top_features_in_fold:
                if i == 0:
                    feature_top_counts.append({'feature': feature, 'count': 1})
                else:
                    for item in feature_top_counts:
                        if item['feature'] == feature:
                            item['count'] += 1
                            break
                    else:
                        feature_top_counts.append({'feature': feature, 'count': 1})
    stability_df = pd.DataFrame(feature_top_counts)
    assert not stability_df.empty, "无任何入选特征,停工待处置"
    stability_df['percentage'] = stability_df['count'] / n_folds
    stable = stability_df[stability_df['percentage'] >= FOLD_RATIO_SELECT].copy()
    stable = stable.sort_values('percentage', ascending=False)  # 稳定排序(v2 原样)
    selected = stable['feature'].tolist()
    summary = pd.DataFrame({
        'feature': stable['feature'],
        'fold_count': stable['count'],
        'fold_percentage': stable['percentage'],
        'importance_mean': [importance_df.loc[f, 'importance_mean'] for f in stable['feature']],
        'importance_std': [importance_df.loc[f, 'importance_std'] for f in stable['feature']],
    })
    return selected, summary, importance_df


def collect_lr_coefs(X: pd.DataFrame, y: pd.Series, feature_names: list,
                     n_splits: int) -> pd.DataFrame:
    """v2 collect_lr_coefs(L711-729)逐字形:12 折 TimeSeriesSplit,逐折新 LR(同参)
    在折内训练段 fit,收集系数(等价于 v2 复用同一 lr 对象逐折 refit)。"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    coef_records = []
    for fold, (train_idx, _) in enumerate(tscv.split(X)):
        lr = LogisticRegression(**LR_PARAMS)
        lr.fit(X.iloc[train_idx], y.iloc[train_idx])
        coef_records.append(pd.Series(lr.coef_[0], index=feature_names,
                                      name=f"fold_{fold}"))
        log(f"  [stable-select] fold {fold + 1}/{n_splits} 完成(训练 {len(train_idx)})")
    return pd.concat(coef_records, axis=1)


def analyze_coef_stability(coef_df: pd.DataFrame) -> pd.DataFrame:
    """v2 analyze_coef_stability(L731-755)逐字。"""
    eps = 1e-9
    stats = pd.DataFrame(index=coef_df.index)
    stats["selection_rate"] = (coef_df != 0).mean(axis=1)
    stats["sign_consistency"] = np.abs(
        np.sign(coef_df).replace(0, np.nan).mean(axis=1))
    abs_coef = coef_df.abs()
    stats["mean_abs_coef"] = abs_coef.mean(axis=1)
    stats["std_abs_coef"] = abs_coef.std(axis=1) + eps
    stats["strength_stability"] = stats["mean_abs_coef"] / stats["std_abs_coef"]
    return stats.sort_values(["selection_rate", "strength_stability"],
                             ascending=False)


# ================================================================ 主流程 ============
def main() -> None:
    ap = argparse.ArgumentParser(description="v2on6 特征选择 + 训练栈 + 分数表")
    ap.add_argument("--rerun-tag", default="run1")
    ap.add_argument("--master-tag", default="run1", help="主表输入标签(默认 run1;P7 修复链用 p7fix)")
    args = ap.parse_args()
    tag = args.rerun_tag
    t_all = time.time()
    log(f"TRAIN STACK START | tag={tag} | n_jobs={LGB_PARAMS['n_jobs']} | "
        f"lightgbm {lgb.__version__}")

    # ---- 输入:主表 + 标签 ----
    master = pd.read_parquet(CACHE_DIR / f"master_v2on6_{args.master_tag}.parquet")
    lab = pd.read_parquet(CACHE_DIR / "labels_v2on6_run1.parquet",
                          columns=["ts_code", "event_date", LABEL_COL])
    df = master.merge(lab, on=["ts_code", "event_date"], how="left", validate="1:1")
    assert len(df) == 96577 and df["event_id"].is_unique
    # 训练行:train 段且标签非 NaN(§五/§九.3;embargo/pre2001/test 不作建模型行)
    df = df.sort_values(["event_date", "event_id"], kind="mergesort") \
           .reset_index(drop=True)  # 时间序(TimeSeriesSplit 前提),同日按 event_id
    tr_mask = (df["seg"] == "train") & df[LABEL_COL].notna()
    train_idx = np.flatnonzero(tr_mask.to_numpy())
    assert (df.loc[tr_mask, "seg"] == "train").all()  # §九.3:训练行 seg 全 train
    y = (df.loc[tr_mask, LABEL_COL] > RETURN_THRESHOLD).astype(int) \
        .reset_index(drop=True)
    log(f"训练行 {len(train_idx)}(train 段 {int((df['seg'] == 'train').sum())} 中 "
        f"标签 NaN {int((df['seg'] == 'train').sum()) - len(train_idx)});"
        f"正样本比例 {y.mean():.4f}")

    # ---- 特征矩阵:float32、inf→NaN→fillna(0)(v2 prepare_features 原样)----
    t0 = time.time()
    feats_all = df[MASTER_FEATURE_COLS].astype(np.float32)
    feats_all = feats_all.replace([np.inf, -np.inf], np.nan)
    n_inf = int(feats_all.isna().sum().sum() - df[MASTER_FEATURE_COLS].isna().sum().sum())
    feats_all = feats_all.fillna(0)
    # median imputer 在 train fit(v2 冗余保险;fillna(0) 后无可填补,断言无操作)
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(feats_all.iloc[train_idx]),
                            columns=MASTER_FEATURE_COLS)
    assert (X_tr_imp.to_numpy(np.float32)
            == feats_all.iloc[train_idx].to_numpy(np.float32)).all(), \
        "imputer 改变了矩阵(应为冗余保险),停工待处置"
    X_all = pd.DataFrame(imputer.transform(feats_all), columns=MASTER_FEATURE_COLS)
    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    log(f"特征矩阵 {X_all.shape}(inf→NaN {n_inf} 单元),imputer 验证无操作"
        f"({time.time() - t0:.0f}s)")

    # ---- §六.1:FULL 基座 → OPTIMIZED_V2ON6 ----
    t0 = time.time()
    log("§六.1 FULL 604 基座 5 折训练开始")
    _, models_full, _ = train_lgb_models(X_train, y)
    optimized, opt_summary, importance_df = select_optimized(
        models_full, MASTER_FEATURE_COLS)
    log(f"OPTIMIZED_V2ON6 = {len(optimized)} 条({time.time() - t0:.0f}s)")

    # ---- §六.2:定版基座(OPTIMIZED 上重跑同一 5 折管线)----
    t0 = time.time()
    log("§六.2 定版基座 5 折训练开始")
    oof, models_opt, fold_scores = train_lgb_models(X_train[optimized], y)
    weights = np.array(fold_scores)
    weights = weights / weights.sum()
    log(f"折权重 {np.round(weights, 4).tolist()}({time.time() - t0:.0f}s)")

    # ---- §六.3:STABLE_V2ON6(train OOF 栈结构,12 折 LR 系数)----
    t0 = time.time()
    stack_cols_all = ["pred_lgb", "pred_lgb_squared"] + MASTER_FEATURE_COLS
    X_stack_train = pd.DataFrame(index=X_train.index)
    X_stack_train["pred_lgb"] = oof
    X_stack_train["pred_lgb_squared"] = oof ** 2
    X_stack_train[MASTER_FEATURE_COLS] = X_train
    scaler = StandardScaler()
    X_stack_scaled = pd.DataFrame(scaler.fit_transform(X_stack_train),
                                  columns=stack_cols_all)
    coef_df = collect_lr_coefs(X_stack_scaled, y, stack_cols_all, LR_COEF_SPLITS)
    stability = analyze_coef_stability(coef_df)
    stable_sel = stability[stability["selection_rate"] >= FOLD_RATIO_SELECT]
    stable_list = stable_sel.index.tolist()
    n_pred_in_stable = sum(c.startswith("pred") for c in stable_list)
    log(f"STABLE_V2ON6 = {len(stable_list)} 条(含 pred 列 {n_pred_in_stable} 条)"
        f"({time.time() - t0:.0f}s)")
    # §六.4:终版元模型输入 = pred_lgb + pred_lgb_squared + STABLE(去重披露)
    meta_cols = ["pred_lgb", "pred_lgb_squared"] \
        + [c for c in stable_list if c not in ("pred_lgb", "pred_lgb_squared")]

    # ---- §七:元模型(train 栈 fit)----
    t0 = time.time()
    scaler_meta = StandardScaler()
    X_meta_train = scaler_meta.fit_transform(X_stack_train[meta_cols])
    lr_meta = LogisticRegression(**LR_PARAMS)
    lr_meta.fit(X_meta_train, y)
    log(f"元模型 fit 完成(输入 {len(meta_cols)} 列)({time.time() - t0:.0f}s)")

    # ---- 全量打分:train = OOF 栈;其他段 = 折加权基座 + 元模型 ----
    t0 = time.time()
    scores = pd.Series(np.nan, index=df.index, dtype=np.float64)
    # train 行(含标签 NaN 行 → score=NaN;block-0 行 pred=0,v2 OOF 原样)
    tr_pos = df.index[df["seg"] == "train"]
    tr_labeled_pos = df.index[tr_mask]
    X_stack_tr = X_stack_train[meta_cols]
    scores.loc[tr_labeled_pos] = lr_meta.predict_proba(
        scaler_meta.transform(X_stack_tr))[:, 1]
    # 非 train 段:折加权基座
    non_tr = df.index[df["seg"] != "train"]
    X_non = X_all.loc[non_tr, optimized]
    pred_non = np.average(
        [m.predict_proba(X_non)[:, 1] for m in models_opt], axis=0, weights=weights)
    X_stack_non = pd.DataFrame(index=non_tr)
    X_stack_non["pred_lgb"] = pred_non
    X_stack_non["pred_lgb_squared"] = pred_non ** 2
    X_stack_non[MASTER_FEATURE_COLS] = X_all.loc[non_tr, MASTER_FEATURE_COLS]
    scores.loc[non_tr] = lr_meta.predict_proba(
        scaler_meta.transform(X_stack_non[meta_cols]))[:, 1]
    log(f"全量打分完成({time.time() - t0:.0f}s)")

    out = pd.DataFrame({
        "event_id": df["event_id"].to_numpy(np.int64),
        "ts_code": df["ts_code"].to_numpy(),
        "date": df["event_date"].to_numpy(),
        "seg": df["seg"].to_numpy(),
        "score": scores.to_numpy(np.float64),
    }).sort_values("event_id", kind="mergesort").reset_index(drop=True)
    assert out.shape == (96577, 5) and out["event_id"].is_unique
    # §九.3:分数表无标签列(列名固定 5 列,机检)
    assert list(out.columns) == ["event_id", "ts_code", "date", "seg", "score"]
    n_nan_train = int(out.loc[out["seg"] == "train", "score"].isna().sum())
    exp_nan = int((df["seg"] == "train").sum()) - len(train_idx)
    assert n_nan_train == exp_nan, f"train NaN 分数 {n_nan_train} != 标签 NaN {exp_nan}"
    assert out.loc[out["seg"] != "train", "score"].notna().all()

    out_path = CACHE_DIR / f"scores_v2on6_{tag}.parquet"
    out.to_parquet(out_path, index=False)
    h = hashlib.md5()
    with open(out_path, "rb") as fh:
        for chunk_b in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk_b)

    # ---- 清单与台账落盘(§九.8)----
    with open(SCRIPT_DIR / "optimized_features_v2on6.json", "w", encoding="utf-8") as f:
        json.dump(optimized, f, ensure_ascii=False, indent=1)
    with open(SCRIPT_DIR / "stable_features_v2on6.json", "w", encoding="utf-8") as f:
        json.dump(stable_list, f, ensure_ascii=False, indent=1)
    sel_results = {
        "tag": tag,
        "optimized_v2on6": {"n": len(optimized), "features": optimized,
                            "summary_csv": "selection_optimized_summary.csv"},
        "stable_v2on6": {"n": len(stable_list), "features": stable_list,
                         "n_pred_cols_selected": int(n_pred_in_stable),
                         "meta_input_cols": meta_cols,
                         "stability_csv": "selection_stable_stability.csv"},
        "fold_scores_opt": fold_scores, "fold_weights": weights.tolist(),
        "lgb_params": LGB_PARAMS, "lr_params": {k: str(v) for k, v in LR_PARAMS.items()},
        "n_train_rows": int(len(train_idx)), "pos_rate": float(y.mean()),
        "scores_md5": h.hexdigest(), "sec": round(time.time() - t_all, 1),
    }
    opt_summary.to_csv(CACHE_DIR / "selection_optimized_summary.csv", index=False)
    stability.to_csv(CACHE_DIR / "selection_stable_stability.csv")
    importance_df.to_csv(CACHE_DIR / "selection_full_importance.csv")
    with open(SCRIPT_DIR / "selection_results_v2on6.json", "w", encoding="utf-8") as f:
        json.dump(sel_results, f, ensure_ascii=False, indent=2, default=str)
    log(f"TRAIN STACK DONE | tag={tag} | OPTIMIZED {len(optimized)} 条 | "
        f"STABLE {len(stable_list)} 条 | scores md5 {h.hexdigest()} | "
        f"{time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
