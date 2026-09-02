#!/usr/bin/env python3
"""事件×特征主表构建驱动（issue #22）。

输入:
  事件表   v3_pipeline/reports/divergence_lab/m_scan/{m_fractal15_full,m_zigzag05_nofilter}/events.parquet
  来源 1   v3_pipeline/reports/feature_matrix/{main,backup}_pool_features.parquet
  来源 2   reports/feature_master/cache/factory_full_{main,backup}.parquet      (regen_factory_full.py)
  来源 3   reports/feature_master/cache/v4daily_snapshot_{main,backup}.parquet  (rebuild_v4_daily_snapshot.py)
输出:
  v3_pipeline/reports/feature_master/master_{main,backup}.parquet
  v3_pipeline/reports/feature_master/master_dictionary.csv
  v3_pipeline/reports/feature_master/master_results.json

验收断言（issue #22 AC）:
  1. 三来源全部合并（行数=事件数, 各来源列在表）
  2. 泄漏排除模式断言（命中列物理缺席）
  3. 去重规则断言（去重后两两 |ρ|<0.999）
  4. 事件日快照与逐日重算一致性（来源3重建脚本前缀抽检零不一致 + 本脚本末端新鲜抽检）

用法: python v3_pipeline/scripts/build_feature_master.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_engine as fe  # noqa: E402
import feature_master as fmx  # noqa: E402
import v4_daily_snapshot as v4s  # noqa: E402

SCAN_DIR = REPO / "v3_pipeline" / "reports" / "divergence_lab" / "m_scan"
FM_DIR = REPO / "v3_pipeline" / "reports" / "feature_matrix"
CACHE_DIR = REPO / "v3_pipeline" / "reports" / "feature_master" / "cache"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "feature_master"
POOLS = {"main": "m_fractal15_full", "backup": "m_zigzag05_nofilter"}

# 末端新鲜抽检规模（确定性种子）
SPOT_STOCKS = 5
SPOT_SEED = 20260902


PROGRESS = OUT_DIR / "master_progress.log"


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# 来源 3（V2 日频族）中文族名映射：按列名前缀匹配，横截面/滞后变体加后缀说明
V2_CN_RULES = [
    ("macd_hist", "指数平滑异同移动平均线柱"), ("macd_signal", "指数平滑异同移动平均线信号线"),
    ("macd", "指数平滑异同移动平均线"), ("rsi_", "相对强弱指标"), ("rsi", "相对强弱指标"),
    ("ma_", "移动均线"), ("bb_", "布林带"), ("volume_ma", "成交量均线"),
    ("volume_ratio", "量比"), ("volume_spike", "放量标记"), ("volume_dryup", "缩量标记"),
    ("volume_trend", "成交量趋势"), ("volume_consistency", "成交量一致性"),
    ("volume_momentum", "成交量动量"), ("volume", "成交量"),
    ("obv", "能量潮"), ("atr", "平均真实波幅"), ("slowk", "随机指标K"), ("slowd", "随机指标D"),
    ("stoch_", "随机指标状态"), ("price_vs_ma", "价格相对均线偏离"), ("ma_arrangement", "均线排列"),
    ("volatility_", "滚动波动率"), ("distance_to_support", "距支撑位距离"),
    ("distance_to_resistance", "距压力位距离"), ("macd_percentile", "MACD滚动百分位"),
    ("hammer", "锤子线形态"), ("doji", "十字线形态"), ("engulfing", "吞没形态"),
    ("downtrend", "下跌趋势标记"), ("pct_change", "日收益率"), ("clv", "收盘位置值"),
    ("upper_shadow_ratio", "上影线占比"), ("body_strength", "K线实体强度"),
    ("signed_vol", "带符号量能强度"), ("pv_corr_10", "量价相关(10日)"),
    ("dist_to_high_60", "距60日高点距离"), ("vol_divergence", "波动率背离"),
    ("vol_gk", "GK波动率"), ("illiq", "非流动性"), ("efficiency_ratio", "效率系数"),
    ("intraday_pos", "日内位置"), ("ret_overnight", "隔夜收益"), ("ret_intraday", "日内收益"),
    ("smart_money_diff", "聪明钱差值"), ("high_mean_20", "20日高点均值"),
    ("low_mean_20", "20日低点均值"), ("support_resistance_ratio", "支撑压力比"),
    ("log_volume", "对数成交量"), ("boxcox_atr", "ATR对数变换(log1p)"),
    ("close_smooth_10", "收盘价EMA10平滑"), ("amihud", "Amihud非流动性(日内)"),
    ("hl_spread", "高低价差"), ("effective_spread", "有效价差估计"),
    ("alpha12", "量价反向(Alpha12)"), ("price_volume_divergence", "价量背离标记"),
    ("price_impact", "价格冲击"), ("price_trend", "价格趋势"),
    ("close_vs_high", "收盘距日高位置"), ("signed_volume_strength", "带符号成交量强度"),
    ("volume_ma_ratio", "成交量均线比"), ("daily_return", "日收益率"), ("amplitude", "振幅"),
    ("rank_return", "当日收益全市场百分位"), ("rank_volume", "当日成交量全市场百分位"),
    ("sh_", "上证指数日特征"), ("sz_", "深证成指日特征"),
    ("sh_sz_sync", "沪深同步性"), ("market_", "市场综合状态"),
    ("open", "开盘价"), ("high", "最高价"), ("low", "最低价"), ("close", "收盘价"),
    ("cs_n", "当日横截面样本数"),
    ("day_of_year", "年内日序"), ("week_of_year", "年内周序"), ("quarter", "季度"),
    ("day_of_month", "月内日序"), ("is_month_end", "月末标记"), ("is_month_start", "月初标记"),
    ("is_quarter_end", "季末标记"), ("is_quarter_start", "季初标记"),
]


def cn_name_s3(col):
    """来源 3 列的中文族名：族名 + 变体后缀（横截面百分位/标准化/滞后）。"""
    base, suffix = col, ""
    if col.endswith("_rankpct"):
        base, suffix = col[: -len("_rankpct")], "｜当日横截面百分位"
    elif col.endswith("_z"):
        base, suffix = col[: -len("_z")], "｜当日横截面稳健标准化"
    import re as _re
    m = _re.match(r"^(.+)_lag_(\d+)$", base)
    if m:
        base, suffix = m.group(1), f"｜滞后{m.group(2)}日" + suffix
    for prefix, cn in sorted(V2_CN_RULES, key=lambda kv: -len(kv[0])):
        if base.startswith(prefix):
            rest = base[len(prefix):]
            return f"{cn}{rest}{suffix}"
    return suffix.lstrip("｜")


def load_events(pool):
    ev = pd.read_parquet(SCAN_DIR / POOLS[pool] / "events.parquet")
    n_raw = len(ev)
    ev = ev[~ev["ts_code"].isin(fe.INDEX_CODES)].reset_index(drop=True)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["seg"] = fmx.segment_of(ev["date"])
    log(f"[{pool}] 事件 {n_raw} -> 剔指数伪股后 {len(ev)}")
    return ev


def spot_check_snapshot(df_master, pool):
    """末端新鲜抽检: 主表中的来源3列值 == 当场前缀重算（逐日重算一致性）。"""
    rng = np.random.default_rng(SPOT_SEED)
    cand = df_master[["ts_code", "date"]].drop_duplicates()
    picks = cand.iloc[rng.choice(len(cand), size=min(SPOT_STOCKS, len(cand)),
                                 replace=False)]
    n_checked, bad = 0, []
    import src.feature_pipeline_v2 as fp2  # 延迟导入, 避免模块级重依赖
    pipe = fp2.FeaturePipeline(None, None)
    for row in picks.itertuples():
        ref = v4s.prefix_recompute_at(
            fe.DATA_DIR / f"{row.ts_code}.parquet", row.ts_code, row.date, pipe=pipe)
        if ref is None:
            continue
        ref = ref.rename(columns={"timestamp": "date", "symbol": "ts_code"})
        mrow = df_master[(df_master["ts_code"] == row.ts_code)
                         & (df_master["date"] == row.date)]
        cmp_cols = [c for c in ref.columns if c in mrow.columns
                    and c not in ("ts_code", "date")]
        cmp_cols = [c for c in cmp_cols
                    if pd.api.types.is_numeric_dtype(mrow[c])]
        a = mrow[cmp_cols].iloc[0].to_numpy(np.float64)
        b = ref[cmp_cols].iloc[0].to_numpy(np.float64)
        n_checked += 1
        if not np.allclose(a, b, rtol=1e-9, atol=0, equal_nan=True):
            diff = [cmp_cols[k] for k in np.where(
                ~np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True))[0]]
            bad.append({"ts_code": row.ts_code, "date": str(row.date.date()),
                        "cols": diff[:10]})
    return {"pool": pool, "n_checked": n_checked, "mismatches": bad}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 加载三来源
    s2 = {p: pd.read_parquet(CACHE_DIR / f"factory_full_{p}.parquet") for p in POOLS}
    s3 = {p: pd.read_parquet(CACHE_DIR / f"v4daily_snapshot_{p}.parquet")
          for p in POOLS}
    merged, src_of, collisions = {}, {}, {}
    for pool in POOLS:
        ev = load_events(pool)
        s1 = pd.read_parquet(FM_DIR / f"{pool}_pool_features.parquet")
        df, so, coll = fmx.merge_sources(ev, s1, s2[pool], s3[pool])
        bool_cols = [c for c in df.columns if df[c].dtype.kind == "b"
                     and c not in fmx.EVENT_META_COLS]
        if bool_cols:  # bool 特征统一为 0/1 数值, 参与去重与训练
            df[bool_cols] = df[bool_cols].astype("int8")
        merged[pool], src_of[pool], collisions[pool] = df, so, coll
        log(f"[{pool}] 合并后 {df.shape} ({time.time()-t0:.0f}s)")

    # ---- AC1: 三来源全部合并
    for pool in POOLS:
        df = merged[pool]
        assert len(df) == len(pd.read_parquet(
            FM_DIR / f"{pool}_pool_features.parquet")), f"{pool} 行数与特征矩阵不一致"
        for tag in ("s1", "s2", "s3"):
            assert any(t == tag for t in src_of[pool].values()), f"{pool} 缺来源 {tag}"

    # ---- 泄漏排除（物理剔除 + 断言）
    leak_records = {}
    for pool in POOLS:
        df = merged[pool]
        bad = fmx.excluded_columns(df.columns)
        leak_records[pool] = bad
        if bad:
            merged[pool] = df.drop(columns=bad)
            for c in bad:
                src_of[pool].pop(c, None)
        fmx.assert_no_leakage(merged[pool].columns)
    log(f"泄漏排除: main={len(leak_records['main'])} backup={len(leak_records['backup'])} 列")

    # ---- 去重（两池 train+val 合并计算, 同一保留集施于两池）
    feat_cols = sorted(set(fmx.feature_columns(merged["main"]))
                       | set(fmx.feature_columns(merged["backup"])))
    pooled = []
    for pool in POOLS:
        df = merged[pool]
        mask = df["seg"].isin(["train", "val"]).to_numpy()
        sub = df.loc[mask].copy()
        for c in feat_cols:
            if c not in sub.columns:
                sub[c] = np.nan
        pooled.append(sub[feat_cols])
    pooled_df = pd.concat(pooled, ignore_index=True)
    src_pooled = {c: (src_of["main"].get(c) or src_of["backup"].get(c))
                  for c in feat_cols}
    keep, drop_records, corr = fmx.dedup_by_correlation(
        pooled_df, feat_cols, np.ones(len(pooled_df), bool), src_of=src_pooled)
    fmx.assert_dedup_clean(corr, keep, feat_cols)
    log(f"去重: {len(feat_cols)} -> {len(keep)} 列 (剔 {len(drop_records)}), "
        f"{time.time()-t0:.0f}s")

    dropped = {r[0] for r in drop_records}
    for pool in POOLS:
        drop_cols = [c for c in merged[pool].columns if c in dropped]
        merged[pool] = merged[pool].drop(columns=drop_cols)

    # ---- AC4: 末端新鲜抽检（来源3 快照 vs 逐日前缀重算）
    spot = [spot_check_snapshot(merged[p], p) for p in POOLS]
    for s in spot:
        assert s["n_checked"] >= SPOT_STOCKS - 2, \
            f"末端抽检覆盖不足: {s['pool']} 仅 {s['n_checked']} 例"
        assert not s["mismatches"], f"末端抽检失败: {s}"
    log(f"末端抽检通过 ({time.time()-t0:.0f}s)")

    # ---- 落盘
    results = {"pools": {}, "leak_excluded": leak_records,
               "dedup": {"threshold": fmx.DEDUP_THRESHOLD, "in": len(feat_cols),
                         "kept": len(keep),
                         "dropped": [{"column": c, "anchor": a, "rho": r}
                                     for c, a, r in drop_records]},
               "collisions": collisions, "spot_check": spot}
    for pool in POOLS:
        df = merged[pool]
        path = OUT_DIR / f"master_{pool}.parquet"
        df.to_parquet(path, index=False)
        results["pools"][pool] = {"rows": int(len(df)), "cols": int(df.shape[1]),
                                  "path": str(path.relative_to(REPO))}
        log(f"[{pool}] 主表 {df.shape} -> {path}")

    # ---- 特征词典（含中文族名/口径列）
    dic1 = pd.read_csv(FM_DIR / "feature_dictionary.csv")
    cn1 = dict(zip(dic1["column"], dic1["family"] + "｜" + dic1["formula"]))
    reg2 = pd.read_csv(REPO / "v3_pipeline" / "reports" / "feature_factory"
                       / "cache" / "factory_registry.csv")
    cn2 = dict(zip(reg2["feature"], reg2["expression"]))
    rows = []
    status = {c: "kept" for c in keep}
    for c, a, r in drop_records:
        status[c] = f"dropped_dedup(anchor={a}, rho={r:.6f})"
    for c in sorted(set(feat_cols)):
        src = src_pooled.get(c)
        cn = cn1.get(c) or cn2.get(c) or (cn_name_s3(c) if src == "s3" else "")
        rows.append({"column": c, "source": src, "cn_name": cn,
                     "status": status.get(c, "dropped_not_in_union")})
    for c in leak_records["main"] + leak_records["backup"]:
        rows.append({"column": c, "source": "", "cn_name": "",
                     "status": "dropped_leakage"})
    dic = pd.DataFrame(rows).drop_duplicates("column")
    dic.to_csv(OUT_DIR / "master_dictionary.csv", index=False)
    log(f"词典 {dic.shape} -> master_dictionary.csv")

    (OUT_DIR / "master_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str))
    log(f"完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
