# -*- coding: utf-8 -*-
"""复诊目标试验 (revival_targets, issue #15): #7 复核遗留 5 候选中判"试"的 2 个。

第一步代价评估结论(详见 revival_targets_report.md):
  试: ②头部聚焦标签(label 侧) 与 ⑤截断排序交互(协议侧);
  不试: ①触障时间(本质是 hit 的命中内细化, 无法修复头部非命中误判, 与二代重叠高)、
        ③胜率x幅度混合(两个已试且判平目标的插值, 验收口径忽略幅度)、
        ④horizon 轴(h10/h20 双侧已夹逼判平, 5/30 无信息存在的先验)。

协议纪律(与三代赛跑同一, 唯一新增 = 清洗宇宙 + 日聚类 bootstrap):
  - 宇宙: 特征矩阵键 ∩ pool_cleaning 组合过滤(excl_combined: 剔 f_st|f_suspend|f_limitup);
    清洗后验证段日加权基线 主池 0.503287 / 备池 0.556050(独立重算断言, 容差 5e-4);
  - 切分: train 2001-2018 探索 / val 2019-01~2022-10 确认(30 交易日段界隔离带) / test 封存;
  - 评估: 验证段每日截面 top3 命中率(对 hit_N20_k2.0), 日加权, 与清洗后零信息基线逐日配对;
  - 统计: 逐日配对 Wilcoxon(与前三代同口径, 剔零差) + 日聚类 bootstrap(B=20000, seed=42,
    按日有放回重抽样非零差配对日, 双侧); 家族 Bonferroni + BH-FDR(主池 8 配置主家族, 16 全家族敏感性);
  - 早停/选参: 验证段日 Rank IC(对训练标签), 网格与 v2 相同 6 组, tiebreak = 验证段 top3(hit);
    已知局限(继承): 验证段既早停又评估, 偏向方向为抬高上报值, 对否定性结论保守。
  - 预登记复活线: 主池某配置 超额 > +2.0pp 且 bootstrap p < 0.05 且 Bonferroni(8) 校正后 < 0.05
    且备池同配置超额同号 → 复活并提请测试段终审; 全部不满足 → 判平关闭。
    +0.3pp 量级明确不算数(噪声带)。

配置矩阵(8 配置 x 双池; 每候选 <=4 配置, 遵守试验预算):
  候选⑤ 截断排序交互(协议侧, lambdarank_truncation_level):
    trunc3_hit__withATRN      label=hit, trunc=3, C4 全 60 特征
    trunc3_hit__noATRN        label=hit, trunc=3, 剔 ATRN 59(ATRN 机械化对照)
    trunc1_hit__withATRN      label=hit, trunc=1(极端头部), C4 60
    trunc3_mfe20atrn__noATRN  label=mfe20/ATRN 逐日秩 31 桶, trunc=3, 剔 ATRN
                              (= 三代主池最优组合 mfe20/ATRN·剔ATRN + 头部截断)
  候选② 头部聚焦标签(label 侧, 截断保持默认 30 ≈ 无截断, 与⑤分离变量):
    head30_mfe20__withATRN    label=1{当日 mfe20 rank_pct>=0.7}(≈每日 top3), C4 60
    head30_mfe20__noATRN      同上, 剔 ATRN 对照
    head10_mfe20__withATRN    label=1{当日 mfe20 rank_pct>=0.9}(≈每日 top1), C4 60
    head30_hitmfe__withATRN   label=1{hit=1 且 当日 mfe20 rank_pct>=0.7}, C4 60
  头部标签逐日截面 rank_pct 在 train/val 段内分别计算(与 bucket_gains 同纪律, 无跨段通道);
  mfe20 复用 target_switch 已归档事件级目标(同 hit 口径 T+1 开盘入场, 复核 200/200 逐位一致)。

输出:
  v3_pipeline/reports/revival_targets/results_revival.json   全部数值(含逐日序列)
  v3_pipeline/reports/revival_targets/progress.log           进度(追加)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(ROOT / "v3_pipeline" / "scripts"))
import run_race_rerun_v2 as rv2  # noqa: E402  复用数据装配/评估/早停代码(与三代同一)
sys.path.insert(0, str(ROOT / "v3_pipeline" / "scripts"))
from run_target_switch import bucket_gains  # noqa: E402  逐日秩 31 桶(三代同函数)

OUT_DIR = ROOT / "v3_pipeline" / "reports" / "revival_targets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"
CLEAN_DIR = ROOT / "v3_pipeline" / "reports" / "pool_cleaning"
TS_DIR = ROOT / "v3_pipeline" / "reports" / "target_switch"

# 清洗后(excl_combined)验证段日加权零信息基线, 摘自 pool_cleaning/baseline_recalc.csv
EXPECTED_CLEAN_BASELINE = {"main": 0.5032867789268494, "backup": 0.5560504198074341}
BOOT_B = 20000
BOOT_SEED = 42

CONFIGS = {
    # ---- 候选⑤: 截断排序交互(协议侧) ----
    "trunc3_hit__withATRN": dict(candidate="c5_trunc", label="hit",
                                 trunc=3, with_atrn=True),
    "trunc3_hit__noATRN": dict(candidate="c5_trunc", label="hit",
                               trunc=3, with_atrn=False),
    "trunc1_hit__withATRN": dict(candidate="c5_trunc", label="hit",
                                 trunc=1, with_atrn=True),
    "trunc3_mfe20atrn__noATRN": dict(candidate="c5_trunc", label="mfe20atrn_gain",
                                     trunc=3, with_atrn=False),
    # ---- 候选②: 头部聚焦标签(label 侧, trunc=30 默认) ----
    "head30_mfe20__withATRN": dict(candidate="c2_headlabel", label="head30_mfe20",
                                   trunc=30, with_atrn=True),
    "head30_mfe20__noATRN": dict(candidate="c2_headlabel", label="head30_mfe20",
                                 trunc=30, with_atrn=False),
    "head10_mfe20__withATRN": dict(candidate="c2_headlabel", label="head10_mfe20",
                                   trunc=30, with_atrn=True),
    "head30_hitmfe__withATRN": dict(candidate="c2_headlabel", label="head30_hitmfe",
                                    trunc=30, with_atrn=True),
}


def log(msg):
    line = f"{msg} {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 清洗宇宙装配
def load_pool_cleaned(name):
    """rv2.load_pool + pool_cleaning 组合过滤(f_any)应用到 train/val 掩码。"""
    df, train_m, val_m = rv2.load_pool(name)
    ex = pd.read_parquet(CLEAN_DIR / f"excluded_events_{name}.parquet",
                         columns=["ts_code", "date", "f_any"])
    ex["date"] = pd.to_datetime(ex["date"])
    n0 = len(df)
    df = df.merge(ex, on=["ts_code", "date"], how="left", validate="1:1")
    assert len(df) == n0 and df["f_any"].notna().all(), f"{name} 清洗键未 1:1 匹配"
    clean = ~df["f_any"].to_numpy(bool)
    train_m = train_m & clean
    val_m = val_m & clean
    log(f"池={name} 清洗应用: 剔除 f_any={int((~clean).sum())} 事件, "
        f"train={int(train_m.sum())} val={int(val_m.sum())}")
    return df, train_m, val_m


def cleaned_baseline(name, df):
    """清洗宇宙上的日加权零信息基线(独立重算: 标签直读, 仅借键集合与清洗标记)。"""
    lab = rv2.load_labels(rv2.POOLS[name]["labdir"])
    keys = df[["ts_code", "date", "f_any"]].copy()
    m = keys.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    seg = rv2.segment_mask(m["date"].to_numpy())
    ok = (seg == "val") & m["hit"].notna().to_numpy() & ~m["f_any"].to_numpy(bool)
    sub = m.loc[ok]
    daily = sub.groupby("date")["hit"].mean()
    dayw = float(daily.mean())
    exp = EXPECTED_CLEAN_BASELINE[name]
    status = "一致" if abs(dayw - exp) < 5e-4 else "!!不符!!"
    log(f"池={name} 清洗基线独立重算: 日加权={dayw:.6f} 事件加权={sub['hit'].mean():.6f} "
        f"信号日={len(daily)} 事件={len(sub)} (预期≈{exp:.6f}, {status})")
    assert status == "一致", f"{name} 清洗基线与 pool_cleaning 归档值不符"
    return {"day_weighted": dayw, "event_weighted": float(sub["hit"].mean()),
            "n_events": int(len(sub)), "n_days": int(len(daily)),
            "yearly_day_weighted": {
                int(y): float(daily[pd.DatetimeIndex(daily.index).year == y].mean())
                for y in sorted(set(pd.DatetimeIndex(daily.index).year))},
            "daily_series": {str(d)[:10]: float(v) for d, v in daily.items()}}


# ================================================================ 目标/标签装配
def attach_mfe(df, pool):
    """复用 target_switch 归档的 mfe20/ret_h10(T+1 开盘口径), 派生 mfe20_atrn。"""
    tgt = pd.read_parquet(TS_DIR / f"targets_{pool}.parquet")
    tgt["date"] = pd.to_datetime(tgt["date"])
    df = df.merge(tgt[["ts_code", "date", "mfe20"]], on=["ts_code", "date"],
                  how="left", validate="1:1")
    atrn = df["ATRN"].to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["mfe20_atrn"] = np.where(atrn > 0, df["mfe20"] / atrn, np.nan)
    return df


def daily_rankpct(df, mask, col):
    """逐日截面 rank_pct(method=average), 仅 mask 内且 col 非 NaN 行; 其余 NaN。"""
    rp = np.full(len(df), np.nan)
    y = df[col].to_numpy(np.float64)
    ok = mask & np.isfinite(y)
    sub = pd.DataFrame({"date": df.loc[ok, "date"].to_numpy(), "y": y[ok]})
    rp[np.nonzero(ok)[0]] = sub.groupby("date")["y"].rank(
        pct=True, method="average").to_numpy()
    return rp


def build_labels(df, train_m, val_m, label_kind):
    """返回 (y_gain 训练标签 float[NaN=不可用], y_cont 早停 IC 用连续值, 标签正例率)。

    label_kind:
      hit            二元 hit (lambdarank 0/1)
      mfe20atrn_gain mfe20/ATRN 逐日秩 31 桶(0..30), 早停用连续 mfe20_atrn
      head30_mfe20   1{当日 mfe20 rank_pct>=0.7}
      head10_mfe20   1{当日 mfe20 rank_pct>=0.9}
      head30_hitmfe  1{hit=1 且 当日 mfe20 rank_pct>=0.7}
    """
    n = len(df)
    if label_kind == "hit":
        y = df["hit"].to_numpy(np.float64)
        return y, y.copy(), float(np.nanmean(y[train_m | val_m]))
    if label_kind == "mfe20atrn_gain":
        gtr = bucket_gains(df, train_m, "mfe20_atrn")
        gva = bucket_gains(df, val_m, "mfe20_atrn")
        y = np.where(np.isfinite(gtr), gtr, gva)  # 行只属于一段, 取非 NaN 者
        y_cont = df["mfe20_atrn"].to_numpy(np.float64)
        pos = float(np.nanmean((y > 0)[train_m | val_m]))
        return y, y_cont, pos
    thr = {"head30_mfe20": 0.7, "head10_mfe20": 0.9,
           "head30_hitmfe": 0.7}[label_kind]
    rp_tr = daily_rankpct(df, train_m, "mfe20")
    rp_va = daily_rankpct(df, val_m, "mfe20")
    rp = np.where(np.isfinite(rp_tr), rp_tr, rp_va)
    y = np.full(n, np.nan)
    fin = np.isfinite(rp)
    y[fin] = (rp[fin] >= thr).astype(np.float64)
    if label_kind == "head30_hitmfe":
        hit = df["hit"].to_numpy(np.float64)
        y[fin] = y[fin] * hit[fin]
    pos = float(np.nanmean(y[train_m | val_m]))
    return y, y.copy(), pos


# ================================================================ 单配置训练(骨架与三代同)
def run_config(df, train_m, val_m, feats, cfg, tag):
    y_gain, y_cont, pos_rate = build_labels(df, train_m, val_m, cfg["label"])
    train_ok = train_m & np.isfinite(y_gain)
    val_ok = val_m & np.isfinite(y_gain)

    X = df[feats].to_numpy(np.float64)
    tr_idx, tr_bounds, gtr = rv2.sort_by_day(df, train_ok)
    va_idx, va_bounds, gva = rv2.sort_by_day(df, val_ok)
    Xtr = X[tr_idx]
    ytr = y_gain[tr_idx].astype(int)
    Xva = X[va_idx]
    yva_gain = y_gain[va_idx].astype(int)
    yva_cont = y_cont[va_idx]

    ic_calc = rv2.DayRankIC(yva_cont, va_bounds)
    feval = rv2.make_feval(ic_calc)

    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr,
                         feature_name=list(feats), free_raw_data=True)
    dval = lgb.Dataset(Xva, label=yva_gain, group=gva, reference=dtrain,
                       free_raw_data=True)

    day_val_full = df["date"].to_numpy()
    hit = df["hit"].to_numpy(np.float64)

    best = None
    for p in rv2.LGBM_GRID:
        params = dict(rv2.LGBM_BASE, **p,
                      lambdarank_truncation_level=cfg["trunc"])
        mdl = lgb.train(params, dtrain, num_boost_round=rv2.N_EST_MAX,
                        valid_sets=[dval], feval=feval,
                        callbacks=[lgb.early_stopping(rv2.EARLY_STOP,
                                                      verbose=False)])
        bi = max(int(mdl.best_iteration), 1)
        sc = mdl.predict(Xva, num_iteration=bi)
        # 选参 key: (验证段日RankIC(对训练标签), tiebreak=验证段top3(hit)) —— 与三代结构一致
        key_ic = float(ic_calc(sc))
        t3, _, _, _ = rv2.topk_metrics(day_val_full[va_idx], hit[va_idx], sc)
        key = (key_ic, float(t3.mean()))
        if best is None or key > best[0]:
            best = (key, p, bi)
    best_params, best_iter = best[1], best[2]

    final = lgb.train(dict(rv2.LGBM_BASE, **best_params,
                           lambdarank_truncation_level=cfg["trunc"]),
                      dtrain, num_boost_round=best_iter)
    prob = final.predict(X, num_iteration=best_iter)
    mt = rv2.eval_segment(df, val_m, prob)  # 评估: 清洗后全验证段 vs hit(与三代一字不变)

    gain = final.feature_importance(importance_type="gain")
    imp = dict(zip(feats, [float(g) for g in gain]))
    out = {
        "candidate": cfg["candidate"], "label": cfg["label"],
        "trunc": cfg["trunc"], "with_atrn": cfg["with_atrn"],
        "n_features": len(feats),
        "n_train_used": int(train_ok.sum()), "n_val_label_ok": int(val_ok.sum()),
        "label_pos_rate": pos_rate,
        "best_params": best_params, "best_iter": best_iter,
        "val": {k: v for k, v in mt.items() if not k.startswith("_")},
        "gain_importance_top15": dict(sorted(imp.items(), key=lambda x: -x[1])[:15]),
        "_daily": mt["_daily"],
    }
    log(f"配置={tag} val_top3={mt['top3']:.4f} val_rankIC_hit={mt['rank_ic']:.4f} "
        f"iter={best_iter} 正例率={pos_rate:.3f} 完成")
    return out


# ================================================================ 日聚类 bootstrap
def bootstrap_daycluster(days, t3, baseline_daily, B=BOOT_B, seed=BOOT_SEED):
    """日聚类 bootstrap: 按日有放回重抽样(日=聚类单位), 重算日加权超额均值, 双侧 p。
    与 Wilcoxon 同一配对集(剔零差日: 当日信号数<=3 时模型无鉴别力)。"""
    b = np.array([baseline_daily[str(d)[:10]] for d in days], np.float64)
    diff = t3 - b
    nz = diff[np.abs(diff) > 1e-12]
    n = len(nz)
    if n < 10:
        return {"boot_p": np.nan, "n_pairs": int(n)}
    rng = np.random.default_rng(seed)
    boot = nz[rng.integers(0, n, size=(B, n))].mean(axis=1)
    p_lo = (1.0 + float((boot <= 0).sum())) / (1.0 + B)
    p_hi = (1.0 + float((boot >= 0).sum())) / (1.0 + B)
    return {"boot_p": float(min(1.0, 2.0 * min(p_lo, p_hi))),
            "boot_mean_diff": float(nz.mean()),
            "boot_ci_2.5%": float(np.quantile(boot, 0.025)),
            "boot_ci_97.5%": float(np.quantile(boot, 0.975)),
            "n_pairs": int(n)}


def bonferroni(p, m):
    return min(p * m, 1.0) if np.isfinite(p) else np.nan


def bh_fdr(pvals):
    items = [(k, p) for k, p in pvals.items() if np.isfinite(p)]
    items.sort(key=lambda x: x[1])
    m = len(items)
    adj, prev = {}, 1.0
    for rank, (k, p) in reversed(list(enumerate(items, 1))):
        prev = min(prev, p * m / rank)
        adj[k] = prev
    return adj


def main():
    t0 = time.time()
    log("阶段=脚本启动 revival_targets(候选②头部聚焦标签 + 候选⑤截断排序交互, "
        "清洗宇宙, 8配置x双池)")
    b28, band = rv2.feature_lists()
    feats_full = ["ATRN"] + b28 + band
    feats_noatrn = b28 + band
    log(f"阶段=名单载入 含ATRN={len(feats_full)} 剔ATRN={len(feats_noatrn)}")

    results = {"seed": rv2.SEED, "grid": rv2.LGBM_GRID, "objective": "lambdarank",
               "boot_B": BOOT_B, "boot_seed": BOOT_SEED,
               "universe": "feature_matrix keys ∩ pool_cleaning excl_combined",
               "baseline_expected": EXPECTED_CLEAN_BASELINE,
               "early_stop_metric": "val daily RankIC (vs train label)",
               "preregistered_gate": "主池超额>+2.0pp 且 boot_p<0.05 且 Bonf(8)<0.05 "
                                     "且备池同号 -> 复活",
               "configs": {k: {kk: vv for kk, vv in v.items()}
                           for k, v in CONFIGS.items()},
               "pools": {}}
    daily_curves = {}
    for pool in ("main", "backup"):
        log(f"阶段=池载入 {pool}")
        df, train_m, val_m = load_pool_cleaned(pool)
        df = attach_mfe(df, pool)
        log(f"池={pool} mfe20 缺失={int(df['mfe20'].isna().sum())} "
            f"mfe20_atrn 缺失={int(df['mfe20_atrn'].isna().sum())}")
        base = cleaned_baseline(pool, df)

        pool_res = {"n_train": int(train_m.sum()), "n_val": int(val_m.sum()),
                    "baseline": {k: v for k, v in base.items()
                                 if k != "daily_series"},
                    "configs": {}}
        for cname, cfg in CONFIGS.items():
            feats = feats_full if cfg["with_atrn"] else feats_noatrn
            tag = f"{pool}/{cname}"
            r = run_config(df, train_m, val_m, feats, cfg, tag)
            daily_curves[tag] = {
                "days": [str(d)[:10] for d in r["_daily"][0]],
                "top3_daily": [float(v) for v in r["_daily"][1]],
            }
            st = rv2.wilcoxon_vs_baseline(r["_daily"][0], r["_daily"][1],
                                          base["daily_series"])
            bt = bootstrap_daycluster(r["_daily"][0], r["_daily"][1],
                                      base["daily_series"])
            r["baseline_day_weighted"] = base["day_weighted"]
            r["excess_pp"] = (r["val"]["top3"] - base["day_weighted"]) * 100.0
            r["stats"] = {**st, **bt}
            r.pop("_daily")
            pool_res["configs"][cname] = r
        results["pools"][pool] = pool_res

    # ================================================================ 多重比较校正(主口径 = bootstrap p)
    main_cfgs = results["pools"]["main"]["configs"]
    fam8 = {k: v["stats"]["boot_p"] for k, v in main_cfgs.items()}
    all_cfgs = {f"{p}/{k}": v["stats"]["boot_p"]
                for p in ("main", "backup")
                for k, v in results["pools"][p]["configs"].items()}
    bh8, bh16 = bh_fdr(fam8), bh_fdr(all_cfgs)
    results["multiple_comparison"] = {
        "family_main8": {k: {"boot_p": p, "bonf": bonferroni(p, 8),
                             "bh": bh8.get(k),
                             "wilcoxon_p": main_cfgs[k]["stats"]["wilcoxon_p"]}
                         for k, p in fam8.items()},
        "family_all16": {k: {"boot_p": p, "bonf": bonferroni(p, 16),
                             "bh": bh16.get(k)}
                         for k, p in all_cfgs.items()},
    }

    # ================================================================ 预登记判定
    verdict = {}
    for k, v in main_cfgs.items():
        exc = v["excess_pp"]
        bp = v["stats"]["boot_p"]
        bonf = results["multiple_comparison"]["family_main8"][k]["bonf"]
        same_sign = results["pools"]["backup"]["configs"][k]["excess_pp"] > 0
        revive = (exc > 2.0) and np.isfinite(bp) and (bp < 0.05) \
            and (bonf < 0.05) and same_sign
        verdict[k] = {"excess_pp": exc, "boot_p": bp, "bonf8": bonf,
                      "backup_excess_pp":
                          results["pools"]["backup"]["configs"][k]["excess_pp"],
                      "backup_same_sign": bool(same_sign),
                      "revive": bool(revive)}
    results["verdict"] = verdict
    n_revive = sum(1 for v in verdict.values() if v["revive"])
    results["final"] = ("任一配置过复活线" if n_revive else
                        "全部判平: 候选②与候选⑤在清洗宇宙下无一过复活线")

    with (OUT_DIR / "results_revival.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    with (OUT_DIR / "daily_curves.json").open("w", encoding="utf-8") as f:
        json.dump(daily_curves, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=全部完成 复活配置数={n_revive} 耗时{time.time()-t0:.0f}s "
        f"-> results_revival.json")


if __name__ == "__main__":
    main()
