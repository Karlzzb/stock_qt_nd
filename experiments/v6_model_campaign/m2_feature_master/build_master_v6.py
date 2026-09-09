#!/usr/bin/env python3
"""主表合并驱动:事件×特征主表 v6(战役 #47 M2,issue #49)。

机器来源:v3_pipeline/scripts/build_feature_master.py 改造(#44 §2.4 主表)。
改造点:
  - 单池化: v5 双池 -> v6 单一 96,577 键主表;
  - s0 事件表直透列 13 条并入(README §六因果审查),src_of 标 "s0"(去重优先级最低,
    SOURCE_PRIORITY 缺省 9);
  - 泄漏扫描 = 三套排除模式并集(feature_master.EXCLUDE_PATTERNS ∪
    feature_engine.BLACKLIST_PATTERNS ∪ v4_daily_snapshot.FORBIDDEN_PATTERNS),
    ^rank_ 重叠以白名单 {rank_return, rank_volume} 口径为准(README §三);
  - 段界与隔离带硬断言: assert_segment_integrity + EMBARGO 落码值与
    derive_embargo_bands(上证交易日历) 重算逐日一致 + 段计数对预登记值;
  - 合并/去重/断言计算本体(merge_sources/dedup_by_correlation 等)零改动复用。

输入:
  事件表  experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet(只读)
  s1      cache/s1_event_dictionary.parquet(键 event_id)
  s2      cache/s2_factory_full.parquet(键 ts_code+date)
  s3      cache/s3_v4daily_snapshot.parquet(键 ts_code+date)
  s4      cache/s4_t3_snapshot.parquet(键 ts_code+date)
输出:
  master_v6.parquet(96,577 行) / master_dictionary_v6.csv(含中文全名列)
  master_results_v6.json(全部台账:泄漏剔除/去重/段界/抽检/覆盖率/哈希)

用法: python build_master_v6.py [--out master_v6.parquet] [--results master_results_v6.json]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import feature_engine as fe  # noqa: E402
import feature_master as fmx  # noqa: E402
import geo_features as geo  # noqa: E402
import t3_features as t3  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402
import v4_daily_snapshot as v4s  # noqa: E402

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

# 末端新鲜抽检规模(确定性种子,v5 原值)
SPOT_STOCKS = 5
SPOT_SEED = 20260902
SPOT_S4_CELLS = 4

# s0 事件表直透列 13 条(README §六预登记准入集)
S0_PASSTHROUGH = ["anchor_close", "cross_prev_dif", "cross_dif", "dif_lift",
                  "cross_prev2_dif", "cross_prev2_dea", "cross_prev_dea",
                  "cross_dea", "min_prev_close", "event_close", "event_vol",
                  "event_amount", "anchor_bars"]
S0_CN = {
    "anchor_close": "锚点收盘价(区间(i1,j]最低收盘价)",
    "cross_prev_dif": "前金叉DIF值", "cross_dif": "事件日金叉DIF值",
    "dif_lift": "金叉DIF抬升幅度(DIF[j]-DIF[i1])",
    "cross_prev2_dif": "远金叉DIF值", "cross_prev2_dea": "远金叉DEA值",
    "cross_prev_dea": "前金叉DEA值", "cross_dea": "事件日金叉DEA值",
    "min_prev_close": "前区间最低收盘价",
    "event_close": "事件日收盘价", "event_vol": "事件日成交量",
    "event_amount": "事件日成交额", "anchor_bars": "锚点距事件日交易日数",
}

# 预登记段计数(README §二,跑数后硬核验)
PREREG_SEGMENT_COUNTS = {"train": 50165, "val": 24405, "test": 17902,
                         "pre2001": 1042}
PREREG_BAND_EVENTS = [1313, 1750]

_V4S_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in v4s.FORBIDDEN_PATTERNS]
_FE_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in fe.BLACKLIST_PATTERNS]

# 来源 3(V2 日频族)中文族名映射(v5 build_feature_master 原表逐字)
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
    """来源 3 列的中文族名:族名 + 变体后缀(v5 build_feature_master 原函数逐字)。"""
    base, suffix = col, ""
    if col.endswith("_rankpct"):
        base, suffix = col[: -len("_rankpct")], "｜当日横截面百分位"
    elif col.endswith("_z"):
        base, suffix = col[: -len("_z")], "｜当日横截面稳健标准化"
    m = re.match(r"^(.+)_lag_(\d+)$", base)
    if m:
        base, suffix = m.group(1), f"｜滞后{m.group(2)}日" + suffix
    for prefix, cn in sorted(V2_CN_RULES, key=lambda kv: -len(kv[0])):
        if base.startswith(prefix):
            rest = base[len(prefix):]
            return f"{cn}{rest}{suffix}"
    return suffix.lstrip("｜")


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [master] {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def load_events():
    """v6 事件表 + event_id 铸造(与 s1 驱动同一规则:(ts_code,event_date) 字典序)。"""
    ev = pd.read_parquet(EVENTS_PATH)
    assert not ev["ts_code"].isin(fe.INDEX_CODES).any()
    assert not ev.duplicated(["ts_code", "event_date"]).any()
    ev = ev.sort_values(["ts_code", "event_date"], kind="mergesort").reset_index(drop=True)
    ev["event_id"] = np.arange(len(ev), dtype=np.int64)
    out = ev[["event_id", "ts_code", "event_date", "event_row"] + S0_PASSTHROUGH].copy()
    out = out.rename(columns={"event_date": "date"})
    out["seg"] = fmx.segment_of(out["date"])
    return out


def combined_leakage_scan(columns):
    """三套排除模式并集扫描;^rank_ 重叠以白名单口径为准(README §三)。"""
    bad = []
    for c in columns:
        if c in fmx.RANK_WHITELIST:
            continue
        hit = any(rx.match(c) for rx in fmx._EXCLUDE_RE) \
            or any(rx.match(c) for rx in _FE_BLACKLIST_RE) \
            or any(rx.match(c) for rx in _V4S_FORBIDDEN_RE)
        if hit:
            bad.append(c)
    return bad


def spot_check_snapshot(df_master):
    """末端新鲜抽检: 主表来源3列值 == 当场前缀重算(v5 同口径,单池)。"""
    rng = np.random.default_rng(SPOT_SEED)
    cand = df_master[["ts_code", "date"]].drop_duplicates()
    picks = cand.iloc[rng.choice(len(cand), size=min(SPOT_STOCKS, len(cand)),
                                 replace=False)]
    n_checked, bad = 0, []
    import src.feature_pipeline_v2 as fp2  # 延迟导入,避免模块级重依赖
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
        log(f"  s3 末端抽检 {row.ts_code} {row.date.date()} 完成")
    return {"n_checked": n_checked, "mismatches": bad}


def spot_check_s4(df_master, ctx):
    """来源4 末端抽检: 主表 T3 列值 == 截断前缀重算(v5 同口径,单池)。"""
    rng = np.random.default_rng(SPOT_SEED + 1)
    cand = df_master[["ts_code", "date"]].drop_duplicates()
    picks = cand.iloc[rng.choice(len(cand), size=min(SPOT_S4_CELLS, len(cand)),
                                 replace=False)]
    n_checked, bad = 0, []
    for row in picks.itertuples():
        ref = t3.prefix_recompute_at(t3.STOCK_DATA, ctx, row.ts_code, row.date)
        if ref is None:
            continue
        mrow = df_master[(df_master["ts_code"] == row.ts_code)
                         & (df_master["date"] == row.date)]
        cmp_cols = [c for c in t3.T3_COLUMNS if c in mrow.columns]
        a = mrow[cmp_cols].iloc[0].to_numpy(np.float64)
        b = ref[cmp_cols].iloc[0].to_numpy(np.float64)
        n_checked += 1
        if not np.allclose(a, b, rtol=1e-9, atol=0, equal_nan=True):
            diff = [cmp_cols[k] for k in np.where(
                ~np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True))[0]]
            bad.append({"ts_code": row.ts_code, "date": str(row.date.date()),
                        "cols": diff[:10]})
        log(f"  s4 末端抽检 {row.ts_code} {row.date.date()} 完成")
    return {"n_checked": n_checked, "mismatches": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(SCRIPT_DIR / "master_v6.parquet"))
    ap.add_argument("--results", type=str,
                    default=str(SCRIPT_DIR / "master_results_v6.json"))
    args = ap.parse_args()
    t0 = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 加载事件与各来源
    ev = load_events()
    s1 = pd.read_parquet(CACHE_DIR / "s1_event_dictionary.parquet")
    s2 = pd.read_parquet(CACHE_DIR / "s2_factory_full.parquet")
    s3 = pd.read_parquet(CACHE_DIR / "s3_v4daily_snapshot.parquet")
    s4 = pd.read_parquet(CACHE_DIR / "s4_t3_snapshot.parquet")
    log(f"来源载入: ev {ev.shape} s1 {s1.shape} s2 {s2.shape} "
        f"s3 {s3.shape} s4 {s4.shape} ({time.time() - t0:.0f}s)")

    # 验收 4: event_id 重铸与 s1 键逐位一致(同字典序规则双边独立铸造)
    chk = ev[["event_id", "ts_code", "date"]].merge(
        s1[["event_id", "ts_code", "date"]], on="event_id",
        suffixes=("", "_s1"), validate="1:1")
    assert len(chk) == len(ev) == len(s1), "event_id 双侧行数不守恒"
    assert (chk["ts_code"] == chk["ts_code_s1"]).all(), "event_id 键 ts_code 错位"
    assert (chk["date"] == chk["date_s1"]).all(), "event_id 键 date 错位"
    del chk

    # ---- 合并(s1 键 event_id;s2/s3/s4 键 ts_code+date)
    df, src_of, collisions = fmx.merge_sources(ev, s1, s2, s3, s4)
    for c in S0_PASSTHROUGH:
        src_of[c] = "s0"
    bool_cols = [c for c in df.columns if df[c].dtype.kind == "b"
                 and c not in fmx.EVENT_META_COLS]
    if bool_cols:  # bool 特征统一为 0/1 数值
        df[bool_cols] = df[bool_cols].astype("int8")
    log(f"合并后 {df.shape};碰撞 {len(collisions)} 起(须全为值同去重) "
        f"({time.time() - t0:.0f}s)")

    # ---- 验收 4/5: 行数键守恒 + 来源在场 + 碰撞纪律
    assert len(df) == 96577, f"主表行数 {len(df)} != 96,577"
    assert not df.duplicated(["ts_code", "date"]).any(), "(ts_code,date) 不唯一"
    assert not df.duplicated(["event_id"]).any(), "event_id 不唯一"
    for tag in ("s0", "s1", "s2", "s3", "s4"):
        assert any(t == tag for t in src_of.values()), f"缺来源 {tag}"
    assert all(c["kept_source"] for c in collisions), "碰撞记录形态异常(值异应已报错)"

    # ---- 验收 1: 泄漏物理剔除(三套并集,白名单口径)
    leak_cols = combined_leakage_scan(df.columns)
    if leak_cols:
        df = df.drop(columns=leak_cols)
        for c in leak_cols:
            src_of.pop(c, None)
    fmx.assert_no_leakage(df.columns)
    assert not [c for c in df.columns if any(rx.match(c) for rx in _V4S_FORBIDDEN_RE)]
    log(f"泄漏物理剔除 {len(leak_cols)} 列: {leak_cols[:10]} ({time.time() - t0:.0f}s)")

    # ---- 验收 2: 同式去重(train+val 扣带合并行,pairwise-complete |ρ|>=0.999)
    feat_cols = sorted(fmx.feature_columns(df))
    mask = df["seg"].isin(["train", "val"]).to_numpy()
    keep, drop_records, corr = fmx.dedup_by_correlation(
        df, feat_cols, mask, src_of=src_of)
    fmx.assert_dedup_clean(corr, keep, feat_cols)
    dropped = {r[0] for r in drop_records}
    df = df.drop(columns=[c for c in df.columns if c in dropped])
    log(f"去重: {len(feat_cols)} -> {len(keep)} 列(剔 {len(drop_records)}) "
        f"({time.time() - t0:.0f}s)")

    # ---- 验收 3: 段界与隔离带硬断言
    idx_sh = pd.read_parquet(REPO / "stock_data" / "daily" / "000001.SH.parquet",
                             columns=["trade_date"])
    sh_cal = pd.to_datetime(idx_sh["trade_date"]).sort_values().unique()
    derived = fmx.derive_embargo_bands(sh_cal)
    assert tuple(fmx.EMBARGO) == derived, \
        f"EMBARGO 落码值 {fmx.EMBARGO} 与派生 {derived} 不一致"
    tep.assert_segment_integrity(df, sh_cal)
    seg_counts = df["seg"].value_counts().to_dict()
    assert {k: seg_counts.get(k, 0) for k in PREREG_SEGMENT_COUNTS} \
        == PREREG_SEGMENT_COUNTS, f"段计数不符预登记: {seg_counts}"
    band_ledger = []
    for i, (lo, hi) in enumerate(fmx.EMBARGO):
        in_band = (df["date"] >= lo) & (df["date"] <= hi)
        n_days = int(((sh_cal >= np.datetime64(lo)) & (sh_cal <= np.datetime64(hi))).sum())
        band_ledger.append({"band": i, "lo": str(lo.date()), "hi": str(hi.date()),
                            "trading_days": n_days,
                            "events": int(in_band.sum())})
        assert int(in_band.sum()) == PREREG_BAND_EVENTS[i], \
            f"带 {i} 内事件数 {int(in_band.sum())} != 预登记 {PREREG_BAND_EVENTS[i]}"
    log(f"段界断言通过: {seg_counts};带台账 {band_ledger} ({time.time() - t0:.0f}s)")

    # ---- 验收 6: 末端新鲜抽检(s3 5 格 + s4 4 格)
    spot = spot_check_snapshot(df)
    assert spot["n_checked"] >= SPOT_STOCKS - 2, f"末端抽检覆盖不足: {spot['n_checked']}"
    assert not spot["mismatches"], f"末端抽检失败: {spot}"
    ctx4 = t3.build_ctx()
    spot_s4 = spot_check_s4(df, ctx4)
    assert spot_s4["n_checked"] >= SPOT_S4_CELLS - 2, \
        f"s4 末端抽检覆盖不足: {spot_s4['n_checked']}"
    assert not spot_s4["mismatches"], f"s4 末端抽检失败: {spot_s4}"
    log(f"末端抽检通过(s3 {spot['n_checked']} 格 / s4 {spot_s4['n_checked']} 格) "
        f"({time.time() - t0:.0f}s)")

    # ---- 验收 9: 特征-标签隔离
    fe.assert_no_label_columns(df.columns)
    assert not [c for c in df.columns if c.startswith("label_")]

    # ---- 落盘主表
    df.to_parquet(out_path, index=False)
    log(f"主表 {df.shape} -> {out_path.name} ({time.time() - t0:.0f}s)")

    # ---- 特征词典(中文全名列)
    dic1 = pd.read_csv(CACHE_DIR / "s1_dictionary.csv")
    cn1 = dict(zip(dic1["column"], dic1["cn_name"]))
    reg2 = pd.read_csv(REPO / "v3_pipeline" / "reports" / "feature_factory"
                       / "cache" / "factory_registry.csv")
    cn2 = dict(zip(reg2["feature"], reg2["expression"]))
    status = {c: "kept" for c in keep}
    for c, a, r in drop_records:
        status[c] = f"dropped_dedup(anchor={a}, rho={r:.6f})"
    rows = []
    for c in sorted(set(feat_cols)):
        src = src_of.get(c, "")
        if src == "s0":
            cn = S0_CN.get(c, "")
        elif src == "s1":
            cn = cn1.get(c, "")
        elif src == "s2":
            cn = cn2.get(c, "")
        elif src == "s3":
            cn = cn_name_s3(c)
        elif src == "s4":
            cn = t3.T3_CN.get(c, "")
        else:
            cn = ""
        rows.append({"column": c, "source": src, "cn_name": cn,
                     "status": status.get(c, "dropped_leakage")})
    dic = pd.DataFrame(rows).drop_duplicates("column")
    dic_path = out_path.parent / (
        "master_dictionary_v6.csv" if out_path.name == "master_v6.parquet"
        else f"{out_path.stem}_dictionary.csv")
    dic.to_csv(dic_path, index=False)
    log(f"词典 {dic.shape} -> {dic_path.name}")

    # ---- 覆盖率台账(各来源键覆盖)
    coverage = {
        "s1": {"rows": int(len(s1)), "missing": 0},
        "s2": {"rows": int(len(s2)), "missing": int(len(ev) - len(s2))},
        "s3": {"rows": int(len(s3)), "missing": int(len(ev) - len(s3))},
        "s4": {"rows": int(len(s4)), "missing": int(len(ev) - len(s4))},
    }

    results = {
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "feature_cols_in": len(feat_cols), "feature_cols_kept": len(keep),
        "leak_excluded": leak_cols,
        "dedup": {"threshold": fmx.DEDUP_THRESHOLD,
                  "dropped": [{"column": c, "anchor": a, "rho": r}
                              for c, a, r in drop_records]},
        "collisions": collisions,
        "segment_counts": {k: int(v) for k, v in seg_counts.items()},
        "embargo_bands": band_ledger,
        "embargo_derived_match": True,
        "spot_check_s3": spot, "spot_check_s4": spot_s4,
        "source_coverage": coverage,
        "source_cols": {t: int(sum(1 for v in src_of.values() if v == t))
                        for t in ("s0", "s1", "s2", "s3", "s4")},
        "md5": {},
        "elapsed_sec": time.time() - t0,
    }
    results["md5"][out_path.name] = md5_of(out_path)
    rpath = Path(args.results)
    rpath.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    log(f"结果 -> {rpath}; 总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
