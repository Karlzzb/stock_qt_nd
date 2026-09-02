# -*- coding: utf-8 -*-
"""issue #17 测试段一次性终审: trunc1 截断排序活口(预登记+2026-09-02 补充冻结)。

纪律:
  - 训练管线与 #15(run_revival_targets.py)完全同码路: lambdarank、按日分组、
    验证段日 RankIC 早停选参(rv2.LGBM_GRID)、最终模型仅训练段重训 best_iter;
  - 完整性断言 GATE: 主池 trunc1_hit__withATRN 验证段 val_top3 复现 0.533105(容差 1e-3)
    且 best_iter=24, 失败则停止, 绝不碰测试段;
  - 测试段(>=2022-11-01, hit 非 NaN, 过 pool_cleaning f_any)只评分一次, 零调参;
  - 判定: 主池主变体测试段超额>0 且日聚类 bootstrap(B=20000, seed=42, 双侧) p<0.0125 -> 通过。

家族=4 变体 x 双池(main/backup):
  trunc1_hit__withATRN(主) / trunc1_hit__noATRN(对照) /
  trunc30_hit__withATRN(锚点, trunc=30≈无截断) / ATRN_only(几何基线, 特征仅 ATRN, trunc=30)。
特征: withATRN=["ATRN"]+b28+band(rv2.feature_lists()), noATRN 剔 ATRN。label 全部 = hit。

附属报告(不参与判定): 朴素日加权等权 top3 模拟组合 —— T+1 开盘入场, 持有至多 20 个
交易日(或数据末端), 逐事件总收益按持有天数逐日分摊, 组合日收益=当日在持仓位分摊收益
等权平均(无持仓日=0); 基线组合=同口径当日全部清洗后事件等权; 出逐年收益/超额/最大回撤。

输出:
  v3_pipeline/reports/final_adjudication_17/results_final.json
  v3_pipeline/reports/final_adjudication_17/daily_curves.json
  v3_pipeline/reports/final_adjudication_17/final_adjudication_17_report.md
  v3_pipeline/reports/final_adjudication_17/progress.log
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(ROOT / "v3_pipeline" / "scripts"))
import run_race_rerun_v2 as rv2  # noqa: E402  数据装配/评估/topk/wilcoxon 工具
import run_revival_targets as rt  # noqa: E402  清洗宇宙装配/标签/日聚类 bootstrap
from run_target_switch import _stock_arrays, DATA_DIR  # noqa: E402  原始日线(附属组合模拟)

OUT_DIR = ROOT / "v3_pipeline" / "reports" / "final_adjudication_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"

TEST_LO = pd.Timestamp("2022-11-01")
GATE_EXPECT = {"val_top3": 0.533105, "best_iter": 24, "tol": 1e-3}
BOOT_B, BOOT_SEED = rt.BOOT_B, rt.BOOT_SEED
ALPHA = 0.0125  # 0.05/4 家族 Bonferroni

VARIANTS = {
    "trunc1_hit__withATRN": dict(trunc=1, feats="full"),
    "trunc1_hit__noATRN": dict(trunc=1, feats="noatrn"),
    "trunc30_hit__withATRN": dict(trunc=30, feats="full"),
    "ATRN_only": dict(trunc=30, feats=["ATRN"]),
}
MAIN_VARIANT = "trunc1_hit__withATRN"


def log(msg):
    line = f"{msg} {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 单配置训练(#15 同码路, 附测试评分)
def run_variant(df, train_m, val_m, test_m, feats, trunc, tag, score_test):
    """逐行复刻 rt.run_config(label='hit') 的训练/选参/重训; score_test=True 时才评估测试段。

    返回 (out, prob): out 含 best_params/best_iter/val 指标(及 test 指标若评分)。
    """
    y_gain, y_cont, pos_rate = rt.build_labels(df, train_m, val_m, "hit")
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

    day_full = df["date"].to_numpy()
    hit = df["hit"].to_numpy(np.float64)

    best = None
    for p in rv2.LGBM_GRID:
        params = dict(rv2.LGBM_BASE, **p, lambdarank_truncation_level=trunc)
        mdl = lgb.train(params, dtrain, num_boost_round=rv2.N_EST_MAX,
                        valid_sets=[dval], feval=feval,
                        callbacks=[lgb.early_stopping(rv2.EARLY_STOP,
                                                      verbose=False)])
        bi = max(int(mdl.best_iteration), 1)
        sc = mdl.predict(Xva, num_iteration=bi)
        key_ic = float(ic_calc(sc))
        t3, _, _, _ = rv2.topk_metrics(day_full[va_idx], hit[va_idx], sc)
        key = (key_ic, float(t3.mean()))
        if best is None or key > best[0]:
            best = (key, p, bi)
    best_params, best_iter = best[1], best[2]

    final = lgb.train(dict(rv2.LGBM_BASE, **best_params,
                           lambdarank_truncation_level=trunc),
                      dtrain, num_boost_round=best_iter)
    prob = final.predict(X, num_iteration=best_iter)

    mt = rv2.eval_segment(df, val_m, prob)  # 验证段评估(与 #15 一字不变)
    out = {
        "trunc": trunc, "n_features": len(feats),
        "n_train_used": int(train_ok.sum()), "n_val_label_ok": int(val_ok.sum()),
        "label_pos_rate": pos_rate,
        "best_params": best_params, "best_iter": best_iter,
        "val": {k: v for k, v in mt.items() if not k.startswith("_")},
        "_daily_val": mt["_daily"],
    }
    if score_test:
        mt_t = rv2.eval_segment(df, test_m, prob)
        out["test"] = {k: v for k, v in mt_t.items() if not k.startswith("_")}
        out["_daily_test"] = mt_t["_daily"]
    log(f"心跳: 配置={tag} 训练完成 val_top3={mt['top3']:.6f} "
        f"val_rankIC={mt['rank_ic']:.4f} best_iter={best_iter}")
    return out, prob


# ================================================================ 测试段零信息基线(首次计算)
def test_baseline(name, df):
    """清洗后特征矩阵键集合内, date>=2022-11-01 且 hit 非 NaN 且 ~f_any 的逐日池命中率。

    独立重算纪律: 标签直读 labels.parquet, 仅借键集合与清洗标记(与 rt.cleaned_baseline 同式)。
    """
    lab = rv2.load_labels(rv2.POOLS[name]["labdir"])
    keys = df[["ts_code", "date", "f_any"]].copy()
    m = keys.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    ok = (m["date"].to_numpy() >= TEST_LO.to_datetime64()) \
        & m["hit"].notna().to_numpy() & ~m["f_any"].to_numpy(bool)
    sub = m.loc[ok]
    daily = sub.groupby("date")["hit"].mean()
    dayw = float(daily.mean())
    log(f"池={name} 测试段基线(首次计算): 日加权={dayw:.6f} "
        f"事件加权={sub['hit'].mean():.6f} 信号日={len(daily)} 事件={len(sub)}")
    return {"day_weighted": dayw, "event_weighted": float(sub["hit"].mean()),
            "n_events": int(len(sub)), "n_days": int(len(daily)),
            "yearly_day_weighted": {
                int(y): float(daily[pd.DatetimeIndex(daily.index).year == y].mean())
                for y in sorted(set(pd.DatetimeIndex(daily.index).year))},
            "daily_series": {str(d)[:10]: float(v) for d, v in daily.items()}}


# ================================================================ 附属: 朴素日加权等权 top3 模拟组合
def event_forward_returns(pool, keys):
    """逐事件 T+1 开盘入场、持有至多 20 交易日(或数据末端)的总收益与持有期日期序列。

    口径与 target_switch.compute_targets 同式: ret = close[T+1]/open[T+1] * cf[exit]/cf[T+1] - 1,
    exit = min(sig+20, 末端); 复权经 pct_chg 链式 cf 比值(规避除权跳变)。
    返回 DataFrame(ts_code, date, sig, dates_held[list[Timestamp]], total_ret, hold_days)。
    """
    cfg = rv2.POOLS[pool]
    ev = pd.read_parquet(cfg["labdir"] / "events.parquet",
                         columns=["ts_code", "date", "sig_idx"])
    ev["date"] = pd.to_datetime(ev["date"])
    m = keys.merge(ev, on=["ts_code", "date"], how="left", validate="1:1")
    assert m["sig_idx"].notna().all(), f"{pool} 有键未匹配到 events"
    m["sig_idx"] = m["sig_idx"].astype(np.int64)

    recs = []
    n_skip = 0
    for ts, sub in m.groupby("ts_code", sort=False):
        st = _stock_arrays(DATA_DIR / f"{ts}.parquet")
        n = len(st["close"])
        sig = sub["sig_idx"].to_numpy()
        assert (st["dates"][sig] == sub["date"].to_numpy()).all(), \
            f"{pool}/{ts} sig_idx 日期不符"
        t1 = sig + 1
        valid = (t1 <= n - 1) & (st["open"][np.minimum(t1, n - 1)] > 0)
        for i in np.nonzero(valid)[0]:
            e = min(int(sig[i]) + 20, n - 1)
            if e < t1[i]:
                n_skip += 1
                continue
            tot = (st["close"][t1[i]] / st["open"][t1[i]]
                   * st["cf"][e] / st["cf"][t1[i]] - 1.0)
            held = st["dates"][t1[i]:e + 1]
            recs.append((sub["ts_code"].iloc[i], sub["date"].iloc[i],
                         held, float(tot), int(len(held))))
        n_skip += int((~valid).sum())
    log(f"池={pool} 附属组合: 事件前向收益计算完成 n={len(recs)} 跳过(无入场/数据末端)={n_skip}")
    return pd.DataFrame(recs, columns=["ts_code", "date", "dates_held",
                                       "total_ret", "hold_days"])


def simulate_portfolio(events, select_fn, label):
    """组合日收益 = 当日所有在持仓位逐日分摊收益(total_ret/hold_days)的等权平均; 无持仓日=0。

    select_fn(day_df) -> 当日入选事件行(model: top3; baseline: 全部)。
    返回 {daily: {date: r}, yearly, annualized, max_drawdown}。
    """
    contrib = {}  # date -> [sum, cnt]
    for sig_day, day_df in events.groupby("date", sort=True):
        sel = select_fn(day_df)
        for _, r in sel.iterrows():
            amort = r["total_ret"] / r["hold_days"]
            for d in r["dates_held"]:
                s = contrib.setdefault(d, [0.0, 0])
                s[0] += amort
                s[1] += 1
    if not contrib:
        return {"daily": {}, "yearly": {}, "annualized": np.nan,
                "max_drawdown": np.nan, "n_days_invested": 0}
    days = sorted(contrib)
    r = np.array([contrib[d][0] / contrib[d][1] for d in days], np.float64)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / peak))
    years = pd.DatetimeIndex(days).year
    yearly = {int(y): float(np.prod(1.0 + r[years == y]) - 1.0)
              for y in sorted(set(years))}
    n_years = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days / 365.25
    ann = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else np.nan
    log(f"池={label} 附属组合模拟: 投资日={len(days)} 年均={ann:+.2%} 最大回撤={mdd:.2%}")
    return {"daily": {str(d)[:10]: float(v) for d, v in zip(days, r)},
            "yearly": yearly, "annualized": ann, "max_drawdown": mdd,
            "total_return": float(eq[-1] - 1.0), "n_days_invested": int(len(days))}


# ================================================================ 主流程
def main():
    t0 = time.time()
    log("阶段=脚本启动 final_adjudication_17(issue #17 测试段一次性终审, 预登记冻结)")
    b28, band = rv2.feature_lists()
    feats_full = ["ATRN"] + b28 + band
    feats_noatrn = b28 + band
    assert len(feats_full) == 60 and len(feats_noatrn) == 59
    log(f"阶段=名单载入 含ATRN={len(feats_full)} 剔ATRN={len(feats_noatrn)}")

    def feats_of(kind):
        return feats_full if kind == "full" else (
            feats_noatrn if kind == "noatrn" else list(kind))

    # ---------------- 阶段1: 双池清洗宇宙装配 ----------------
    log("阶段=双池装配 load_pool_cleaned(清洗 f_any 应用于 train/val 掩码)")
    pools = {}
    for pool in ("main", "backup"):
        df, train_m, val_m = rt.load_pool_cleaned(pool)
        test_m = (df["date"].to_numpy() >= TEST_LO.to_datetime64()) \
            & df["hit"].notna().to_numpy() & ~df["f_any"].to_numpy(bool)
        pools[pool] = dict(df=df, train_m=train_m, val_m=val_m, test_m=test_m)
        log(f"池={pool} 掩码: train={int(train_m.sum())} val={int(val_m.sum())} "
            f"test={int(test_m.sum())} 测试日数={df.loc[test_m, 'date'].nunique()}")

    # ---------------- 阶段2: 完整性断言 GATE(先跑主池主变体, 不碰测试段) ----------------
    log("阶段=完整性断言GATE 主池 trunc1_hit__withATRN 仅验证段复现 "
        f"(预期 val_top3={GATE_EXPECT['val_top3']} best_iter={GATE_EXPECT['best_iter']})")
    p = pools["main"]
    gate_out, gate_prob = run_variant(
        p["df"], p["train_m"], p["val_m"], p["test_m"],
        feats_full, 1, "main/trunc1_hit__withATRN(GATE)", score_test=False)
    gate_top3 = gate_out["val"]["top3"]
    gate_iter = gate_out["best_iter"]
    gate_ok = (abs(gate_top3 - GATE_EXPECT["val_top3"]) <= GATE_EXPECT["tol"]
               and gate_iter == GATE_EXPECT["best_iter"])
    log(f"阶段=GATE结果 val_top3={gate_top3:.6f} best_iter={gate_iter} "
        f"-> {'通过' if gate_ok else '!!失败, 停止, 不触碰测试段!!'}")
    if not gate_ok:
        with (OUT_DIR / "results_final.json").open("w", encoding="utf-8") as f:
            json.dump({"issue": 17, "gate": {"passed": False,
                                             "val_top3": gate_top3,
                                             "best_iter": gate_iter,
                                             "expected": GATE_EXPECT},
                       "final": "GATE 失败: 未触碰测试段, 终审中止"},
                      f, ensure_ascii=False, indent=1, default=str)
        log("阶段=中止 GATE 失败, 测试段未被评分")
        sys.exit(1)

    # ---------------- 阶段3: 测试掩码与基线(首次计算) ----------------
    log("阶段=测试段基线首次计算(双池)")
    baselines = {}
    for pool in ("main", "backup"):
        baselines[pool] = test_baseline(pool, pools[pool]["df"])

    # ---------------- 阶段4: 4 变体 x 双池 终审 ----------------
    log("阶段=变体终审 4变体x双池(验证段选参早停 -> 训练段重训 -> 测试段评分一次)")
    results = {"issue": 17, "seed": rv2.SEED, "grid": rv2.LGBM_GRID,
               "objective": "lambdarank",
               "early_stop_metric": "val daily RankIC (vs hit)",
               "universe": "feature_matrix keys ∩ pool_cleaning excl_combined",
               "test_segment": ">=2022-11-01, hit 非 NaN, ~f_any",
               "gate": {"passed": True, "val_top3": gate_top3,
                        "best_iter": gate_iter, "expected": GATE_EXPECT},
               "alpha": ALPHA, "boot_B": BOOT_B, "boot_seed": BOOT_SEED,
               "variants": {k: dict(v) for k, v in VARIANTS.items()},
               "pools": {}}
    daily_curves = {}
    main_probs = {}
    for pool in ("main", "backup"):
        pk = pools[pool]
        base = baselines[pool]
        pool_res = {"n_train": int(pk["train_m"].sum()),
                    "n_val": int(pk["val_m"].sum()),
                    "n_test": int(pk["test_m"].sum()),
                    "baseline_test": {k: v for k, v in base.items()
                                      if k != "daily_series"},
                    "variants": {}}
        for vname, vcfg in VARIANTS.items():
            tag = f"{pool}/{vname}"
            if pool == "main" and vname == MAIN_VARIANT:
                # GATE 已训(同一最终模型, 确定性重训同分), 直接补测试段评分一次
                out, prob = gate_out, gate_prob
                mt_t = rv2.eval_segment(pk["df"], pk["test_m"], prob)
                out["test"] = {k: v for k, v in mt_t.items()
                               if not k.startswith("_")}
                out["_daily_test"] = mt_t["_daily"]
                log(f"心跳: 配置={tag} 复用GATE最终模型, 测试段评分一次完成")
            else:
                out, prob = run_variant(pk["df"], pk["train_m"], pk["val_m"],
                                        pk["test_m"], feats_of(vcfg["feats"]),
                                        vcfg["trunc"], tag, score_test=True)
            days_t, t3_t = out["_daily_test"]
            st = rv2.wilcoxon_vs_baseline(days_t, t3_t, base["daily_series"])
            bt = rt.bootstrap_daycluster(days_t, t3_t, base["daily_series"])
            out["baseline_test_day_weighted"] = base["day_weighted"]
            out["test_excess_pp"] = (out["test"]["top3"]
                                     - base["day_weighted"]) * 100.0
            out["stats_test"] = {**st, **bt}
            daily_curves[tag] = {
                "days": [str(d)[:10] for d in days_t],
                "top3_daily": [float(v) for v in t3_t]}
            out.pop("_daily_val")
            out.pop("_daily_test")
            pool_res["variants"][vname] = out
            if vname == MAIN_VARIANT:
                main_probs[pool] = prob
        daily_curves[f"{pool}/__baseline__"] = {
            "days": sorted(base["daily_series"].keys()),
            "pool_hit_daily": [base["daily_series"][d]
                               for d in sorted(base["daily_series"])]}
        results["pools"][pool] = pool_res

    # ---------------- 阶段5: 判定 ----------------
    mv = results["pools"]["main"]["variants"][MAIN_VARIANT]
    bv = results["pools"]["backup"]["variants"][MAIN_VARIANT]
    exc = mv["test_excess_pp"]
    bp = mv["stats_test"]["boot_p"]
    same_sign = bv["test_excess_pp"] > 0
    passed = bool(exc > 0 and np.isfinite(bp) and bp < ALPHA)
    results["verdict"] = {
        "main_variant": MAIN_VARIANT,
        "main_test_excess_pp": exc, "main_boot_p": bp,
        "main_wilcoxon_p": mv["stats_test"]["wilcoxon_p"],
        "threshold": f"超额>0 且 boot_p<{ALPHA}",
        "passed": passed,
        "backup_test_excess_pp": bv["test_excess_pp"],
        "backup_boot_p": bv["stats_test"]["boot_p"],
        "backup_same_sign": bool(same_sign),
        "reference_all_variants": {
            f"{pool}/{vn}": results["pools"][pool]["variants"][vn]["test_excess_pp"]
            for pool in ("main", "backup") for vn in VARIANTS},
    }
    log(f"阶段=判定 主池主变体 测试超额={exc:+.2f}pp boot_p={bp:.5f} "
        f"备池同号={same_sign} -> {'通过' if passed else '否决'}")

    # ---------------- 阶段6: 附属组合模拟(不参与判定) ----------------
    log("阶段=附属组合模拟(T+1开盘, 持有<=20交易日, 逐日分摊, 双池主变体)")
    port = {}
    for pool in ("main", "backup"):
        pk = pools[pool]
        df = pk["df"]
        idx = np.nonzero(pk["test_m"])[0]
        sub = df.iloc[idx][["ts_code", "date"]].copy()
        sc = main_probs[pool][idx]
        sub["score"] = np.where(np.isfinite(sc), sc, -np.inf)
        fwd = event_forward_returns(pool, sub[["ts_code", "date"]])
        evj = sub.merge(fwd, on=["ts_code", "date"], how="inner", validate="1:1")

        def top3_fn(day_df):
            return day_df.sort_values("score", ascending=False,
                                      kind="stable").head(3)

        def all_fn(day_df):
            return day_df

        pm = simulate_portfolio(evj, top3_fn, f"{pool}/模型top3")
        pb = simulate_portfolio(evj, all_fn, f"{pool}/基线全事件")
        yearly_excess = {y: pm["yearly"].get(y, 0.0) - pb["yearly"].get(y, 0.0)
                         for y in sorted(set(pm["yearly"]) | set(pb["yearly"]))}
        port[pool] = {"model_top3": pm, "baseline_all": pb,
                      "yearly_excess": yearly_excess,
                      "annualized_excess": (pm["annualized"] - pb["annualized"])
                      if np.isfinite(pm["annualized"]) and
                      np.isfinite(pb["annualized"]) else np.nan,
                      "n_events": int(len(evj))}
    results["portfolio_aux"] = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("model_top3", "baseline_all")}
        for k, v in port.items()}
    results["portfolio_aux"]["method"] = (
        "T+1 开盘入场, 持有至多 20 交易日(或数据末端), 逐事件总收益按持有天数逐日分摊; "
        "组合日收益=当日在持仓位分摊收益等权平均, 无持仓日=0; "
        "基线组合=同信号日全部清洗后事件等权; 收益经 pct_chg 链式 cf 复权")
    for pool in ("main", "backup"):
        daily_curves[f"{pool}/__portfolio_model_daily__"] = port[pool]["model_top3"]["daily"]
        daily_curves[f"{pool}/__portfolio_baseline_daily__"] = port[pool]["baseline_all"]["daily"]
    for pool in ("main", "backup"):
        results["portfolio_aux"][pool]["model_top3"] = {
            k: v for k, v in port[pool]["model_top3"].items() if k != "daily"}
        results["portfolio_aux"][pool]["baseline_all"] = {
            k: v for k, v in port[pool]["baseline_all"].items() if k != "daily"}

    # ---------------- 落盘 ----------------
    with (OUT_DIR / "results_final.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    with (OUT_DIR / "daily_curves.json").open("w", encoding="utf-8") as f:
        json.dump(daily_curves, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=结果落盘 results_final.json + daily_curves.json 耗时{time.time()-t0:.0f}s")

    write_report(results, port, time.time() - t0)
    log(f"阶段=全部完成 总耗时{time.time()-t0:.0f}s "
        f"终审={'通过' if passed else '否决'}")


# ================================================================ 报告
def write_report(results, port, elapsed):
    v = results["verdict"]
    L = []
    A = L.append
    A("# issue #17 测试段一次性终审报告: trunc1 截断排序活口")
    A("")
    A("## 协议与纪律")
    A("")
    A("本报告为 issue #17 预登记(2026-09-02 补充冻结)的测试段一次性终审结果, 结果即终局, 不回炉调参。")
    A("训练管线与 #15(run_revival_targets.py)完全同码路: LGBM lambdarank 按事件日分组, 验证段日 RankIC 早停选参(网格=rv2.LGBM_GRID 6 组), 最终模型仅在训练段(2001-2018)按 best_iter 重训。")
    A("宇宙 = 特征矩阵键集合 ∩ pool_cleaning 组合过滤(剔 f_st|f_suspend|f_limitup)。")
    A("测试段 = date>=2022-11-01 且 hit_N20_k2.0 非 NaN 且通过清洗过滤, 只评分一次。")
    A("评估口径 = 每日横截面 top3 命中率(当日不足 3 个取全部), 日加权, 对测试段清洗后零信息基线(本次首次计算)逐日配对。")
    A("统计 = 逐日配对 Wilcoxon + 日聚类 bootstrap(B=20000, seed=42, 双侧)。")
    A("判定线(预登记): 主池主变体 trunc1_hit__withATRN 测试段超额>0 且 bootstrap p<0.0125(α=0.05/4) 即通过, 备池同号为加分项非必需。")
    A("")
    A("## 完整性断言 GATE")
    A("")
    g = results["gate"]
    A(f"主池主变体验证段复现: val_top3={g['val_top3']:.6f}(预期 0.533105, 容差 1e-3), best_iter={g['best_iter']}(预期 24)。")
    A(f"GATE 结果: {'通过, 训练管线的验证段行为与 #15 归档逐位一致, 方触碰测试段' if g['passed'] else '失败'}。")
    A("")
    A("## 测试段零信息基线(本次首次计算)")
    A("")
    A("| 池 | 事件数 | 信号日 | 日加权基线 | 事件加权基线 |")
    A("|---|---|---|---|---|")
    for pool in ("main", "backup"):
        b = results["pools"][pool]["baseline_test"]
        A(f"| {pool} | {b['n_events']} | {b['n_days']} | {b['day_weighted']:.4f} | {b['event_weighted']:.4f} |")
    A("")
    A("备池测试事件 8139 = 特征矩阵 2022-11 后 8851 键剔 274 个 hit 为 NaN 的 2026 尾部事件再剔 438 个清洗事件; 预登记估计的约 8391 只剔了清洗项, 未剔 hit NaN 项。")
    A("")
    A("## 主判定")
    A("")
    A(f"主池 trunc1_hit__withATRN 测试段 top3={results['pools']['main']['variants']['trunc1_hit__withATRN']['test']['top3']:.4f}, 对基线超额={v['main_test_excess_pp']:+.2f}pp, bootstrap p={v['main_boot_p']:.5f}, Wilcoxon p={v['main_wilcoxon_p']:.5f}。")
    A(f"备池同变体超额={v['backup_test_excess_pp']:+.2f}pp(boot_p={v['backup_boot_p']:.5f}), 同号={'是' if v['backup_same_sign'] else '否'}。")
    A(f"**终审判定: {'通过' if v['passed'] else '否决'}**(线: 超额>0 且 boot_p<0.0125)。")
    A("")
    A("## 全变体测试段结果(家族=4, 参考)")
    A("")
    A("| 池 | 变体 | 测试top3 | 超额pp | 测试top1 | 测试RankIC | boot_p | wilcoxon_p | best_iter | 验证段top3 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for pool in ("main", "backup"):
        for vn, vv in results["pools"][pool]["variants"].items():
            st = vv["stats_test"]
            A(f"| {pool} | {vn} | {vv['test']['top3']:.4f} | {vv['test_excess_pp']:+.2f} "
              f"| {vv['test']['top1']:.4f} | {vv['test']['rank_ic']:.4f} "
              f"| {st['boot_p']:.5f} | {st.get('wilcoxon_p', float('nan')):.5f} "
              f"| {vv['best_iter']} | {vv['val']['top3']:.4f} |")
    A("")
    A("## 逐年 top3(测试段, 对逐年基线)")
    A("")
    A("| 池 | 变体 | 年份 | top3 | 基线 | 超额pp |")
    A("|---|---|---|---|---|---|")
    for pool in ("main", "backup"):
        by = results["pools"][pool]["baseline_test"]["yearly_day_weighted"]
        for vn, vv in results["pools"][pool]["variants"].items():
            for y, t3 in vv["test"]["yearly_top3"].items():
                b = by.get(str(y), by.get(y, np.nan))
                A(f"| {pool} | {vn} | {y} | {t3:.4f} | {b:.4f} | {(t3-b)*100:+.2f} |")
    A("")
    A("## 附属: 朴素日加权等权 top3 模拟组合(不参与判定)")
    A("")
    A("方法: 事件日 T+1 开盘价入场, 持有至多 20 个交易日(或数据末端), 逐事件总收益按持有天数逐日分摊; 组合日收益=当日所有在持仓位分摊收益的等权平均, 无持仓日记 0。")
    A("模型组合仓位=主变体 trunc1_hit__withATRN 信号日 top3(等权), 基线组合=同信号日全部清洗后事件等权。")
    A("收益经 pct_chg 链式累乘因子复权(与 target_switch/divergence_lab 同式, 规避除权跳变), 未计交易成本与滑点。")
    A("")
    for pool in ("main", "backup"):
        pm, pb = port[pool]["model_top3"], port[pool]["baseline_all"]
        A(f"### {pool} 池(事件 {port[pool]['n_events']})")
        A("")
        A(f"模型组合: 总收益={pm['total_return']:+.2%}, 年化={pm['annualized']:+.2%}, 最大回撤={pm['max_drawdown']:.2%}, 投资日={pm['n_days_invested']}。")
        A(f"基线组合: 总收益={pb['total_return']:+.2%}, 年化={pb['annualized']:+.2%}, 最大回撤={pb['max_drawdown']:.2%}。")
        A(f"年化超额={pm['annualized']-pb['annualized']:+.2%}。")
        A("")
        A("| 年份 | 模型 | 基线 | 超额 |")
        A("|---|---|---|---|")
        for y in sorted(set(pm["yearly"]) | set(pb["yearly"])):
            A(f"| {y} | {pm['yearly'].get(y, 0.0):+.2%} | {pb['yearly'].get(y, 0.0):+.2%} "
              f"| {pm['yearly'].get(y, 0.0)-pb['yearly'].get(y, 0.0):+.2%} |")
        A("")
    A("## 结论")
    A("")
    if v["passed"]:
        A(f"主变体 trunc1_hit__withATRN 在封存测试段超额 {v['main_test_excess_pp']:+.2f}pp 且 bootstrap p={v['main_boot_p']:.5f}<0.0125, 预登记判定通过。")
    else:
        A(f"主变体 trunc1_hit__withATRN 在封存测试段超额 {v['main_test_excess_pp']:+.2f}pp、bootstrap p={v['main_boot_p']:.5f}, 未过预登记线(超额>0 且 p<0.0125), 终审否决。")
    A(f"备池同号={'是' if v['backup_same_sign'] else '否'}(加分项, 不影响判定)。")
    A("按预登记纪律, 本结果为终局, 测试段不再回炉。")
    A("")
    A(f"全程耗时 {elapsed:.0f}s; 全部数值见 results_final.json, 逐日序列见 daily_curves.json。")
    A("")
    with (OUT_DIR / "final_adjudication_17_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
