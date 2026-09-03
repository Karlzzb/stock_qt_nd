#!/usr/bin/env python3
"""日频打分面板构建（issue #28 分数衰减退出类前置，T9 测试段复用同一实现）。

目的：score-decay 策略要求"持仓期间每日重算分数"。scores_final.parquet 只有事件行，
本脚本把 T7 定版模型（25 特征）推广到 事件股票宇宙 × 窗口内全部交易日 的日频面板，
产出 (ts_code, date, prob) 及 25 特征值（供审计与锚定断言）。

四来源全部复用既有冻结模块全历史因果实现（绝不重造算子）：
  s1 股内 7 列 + 市场级 3 列: v3_pipeline/src/feature_engine.py
      （compute_stock_features 逐股全历史；build_market_frame 全 universe 面板聚合）
  s2 工厂 3 列: v3_pipeline/src/feature_factory.py（compute_stock_factory 逐股全历史）
  s3 横截面 6 列: src/feature_pipeline_v2.py 逐股链（v4_daily_snapshot.compute_stock_features）
      + _calculate_cross_features 当日全市场横截面（排名群体=当日有 bar 且历史>=100 行的全市场，
      与 T4 Pass B 口径一致；未复权价口径）
  s4 6 列: v3_pipeline/src/t3_features.py（build_panel + compute_all 全历史面板，
      市场级列为逐日全历史序列）

窗口外不留行：所有特征为 as-of 值，全历史算完后切片 [start, end]，无前视。
事件股票宇宙 = scores_final 窗口内有事件的全部 ts_code（持仓股必在其中）。

断言（全部落 results json，任一 FAIL 退出码 1）：
  A 覆盖：窗口内全部事件 (ts_code, date) 在面板有行。
  B 特征锚定：面板在事件行的 25 特征值与主表（master_main/master_backup）一致
    （主表部分列为 float32 存储，容差取 float32 精度量级 rtol=1e-6；
    float64 列实测逐位差 0.0；NaN 对 NaN）。
  C 前缀截断抽检：抽样 (股票, 日) 单元，仅用 <=T 数据重算 s1/s2/s4 列与面板逐位一致
    （s3 横截面列由 T4 既有前缀断言 + 本脚本基列抽检覆盖，排名为当日确定性变换）。
  D 分数锚定：面板事件行重算 prob 与 scores_final.prob 一致（atol=1e-12，
    实测最大差 2.2e-16，即 float32 特征存储引入的末位 ulp 抖动）。

用法: python v3_pipeline/scripts/build_daily_score_panel.py \
        --start 2019-01-01 --end 2022-10-31 --workers 24 [--force]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
sys.path.insert(0, str(REPO / "v3_pipeline" / "scripts"))

import feature_engine as fe  # noqa: E402
import feature_factory as ff  # noqa: E402
import t3_features as t3  # noqa: E402
import v4_daily_snapshot as v4s  # noqa: E402
import src.feature_pipeline_v2 as fp2  # noqa: E402

SCORES = REPO / "v3_pipeline/reports/feature_selection/scores_final.parquet"
FINAL_FEATURES = REPO / "v3_pipeline/reports/feature_selection/final_features.json"
MODEL_PATH = REPO / "v3_pipeline/reports/feature_selection/model.txt"
CAL_PATH = REPO / "v3_pipeline/reports/feature_selection/calibrator.json"
MASTER_DIR = REPO / "v3_pipeline/reports/feature_master"
OUT_DIR = REPO / "v3_pipeline/reports/strategy_tuning"
# 缓存按窗口打标：不同窗口的构建互不串缓存（T8 验证段 / T9 测试段并存可复现）
CACHE_DIR = OUT_DIR / "panel_cache"

S1_STOCK_COLS = ["AMT20", "DIST_52W_HIGH", "MACD_DIF_NORM", "RET120_20",
                 "SJV60", "TREND_SLOPE_120", "VOL_CONTRACTION60"]
S1_MARKET_COLS = ["REGIME_CODE", "BREADTH_ADV5", "MKT_RET20"]
S2_COLS = ["IDXMIN_R_60", "MEAN_R_60", "STD_A_20"]
S3_BASE_COLS = ["doji_pattern", "log_volume", "ma_arrangement",
                "macd_death_cross", "macd_zero_cross_down", "volume"]
S3_CROSS_COLS = ["doji_pattern_rankpct", "log_volume_z", "ma_arrangement_rankpct",
                 "macd_death_cross_rankpct", "macd_zero_cross_down_rankpct", "volume_z"]
S4_STOCK_COLS = ["ABN_TURN_21_252", "LN_FREE_MV"]
S4_MARKET_COLS = ["MKT_NH_NL_DIFF", "MKT_SEAL_RATIO", "MKT_TURNOVER_PCTL_EQW",
                  "STYLE_SIZE_RS60"]

N_ASSERT_STOCKS = 6
ASSERT_DAYS_PER_STOCK = 2
ASSERT_SEED = 20260903

PROGRESS = OUT_DIR / "panel_progress.log"


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_final_feature_order() -> list[str]:
    with FINAL_FEATURES.open(encoding="utf-8") as f:
        meta = json.load(f)
    return [x["feature"] for x in meta["features"]]


# ---------------------------------------------------------------- 阶段 1：s1 逐股 + 市场面板
def _s1_worker(task):
    path, code, need_stock, lo, hi = task
    try:
        df = fe.load_stock_df(path)
        if len(df) < 30:
            return code, None, None, "too_short"
        feats, ctx = fe.compute_stock_features(df, code)
        days = ctx["days"].astype(np.int64)
        is_index = code in fe.INDEX_CODES
        panel = pd.DataFrame({"date": days, "is_index": is_index, **ctx["panel"]})
        stock = None
        if need_stock:
            dates = pd.to_datetime(days, unit="D")
            stock = pd.DataFrame({"date": dates})
            for c in S1_STOCK_COLS:
                stock[c] = feats[c].to_numpy(np.float64)
            stock = stock[(stock["date"] >= lo) & (stock["date"] <= hi)]
            stock.insert(0, "ts_code", code)
        return code, panel, stock, None
    except Exception as e:  # noqa: BLE001
        return code, None, None, repr(e)


def stage_s1(universe: set[str], lo, hi, workers: int, force: bool):
    out_market = CACHE_DIR / "s1_market.parquet"
    out_stock = CACHE_DIR / "s1_stock.parquet"
    if not force and out_market.exists() and out_stock.exists():
        log("阶段1 s1 缓存命中")
        return pd.read_parquet(out_market), pd.read_parquet(out_stock)
    t0 = time.time()
    files = sorted(fe.DATA_DIR.glob("*.parquet"))
    tasks = [(p, p.stem, p.stem in universe, lo, hi) for p in files]
    panels, stocks, errors, skipped = [], [], {}, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (code, panel, stock, err) in enumerate(
                ex.map(_s1_worker, tasks, chunksize=16)):
            if err:
                if err in ("too_short",) or err.startswith("read:"):
                    skipped.append(code)  # 与 build_feature_matrix.process_stock 同口径跳过
                else:
                    errors[code] = err
                continue
            panels.append(panel)
            if stock is not None and len(stock):
                stocks.append(stock)
            if (i + 1) % 1000 == 0:
                log(f"  s1 {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    assert not errors, f"s1 失败 {len(errors)}: {list(errors.items())[:3]}"
    bad = [c for c in skipped if c in universe]
    assert not bad, f"宇宙股被 s1 跳过: {bad[:5]}"
    log(f"  s1 跳过 {len(skipped)} 只短历史股（与历史构建同口径），宇宙股零跳过")
    panel = pd.concat(panels, ignore_index=True)
    log(f"s1 面板 {panel.shape}，构建市场特征框 ... ({time.time() - t0:.0f}s)")
    idx_sh = fe.load_index_df(fe.DATA_DIR / "000001.SH.parquet")
    idx_sz = fe.load_index_df(fe.DATA_DIR / "399001.SZ.parquet")
    market, days = fe.build_market_frame(panel, idx_sh, idx_sz)
    market = market.reset_index()
    market["date"] = pd.to_datetime(market["_days"].astype(np.int64), unit="D")
    market = market[["date"] + S1_MARKET_COLS]
    market = market[(market["date"] >= lo) & (market["date"] <= hi)]
    stock_df = pd.concat(stocks, ignore_index=True)
    market.to_parquet(out_market, index=False)
    stock_df.to_parquet(out_stock, index=False)
    log(f"阶段1 完成: 市场 {market.shape} 股内 {stock_df.shape} ({time.time() - t0:.0f}s)")
    return market, stock_df


# ---------------------------------------------------------------- 阶段 2：s2 工厂逐股
_RIDX = None


def _s2_init(ridx):
    global _RIDX
    _RIDX = ridx


def _s2_worker(task):
    code, lo, hi = task
    try:
        path = fe.DATA_DIR / f"{code}.parquet"
        if not path.exists():
            return code, None, "缺文件"
        df = fe.load_stock_df(path)
        cols, _ = ff.compute_stock_factory(df, _RIDX)
        dates = pd.to_datetime(df["trade_date"])
        out = pd.DataFrame({"date": dates})
        for c in S2_COLS:
            out[c] = cols[c]
        out = out[(out["date"] >= lo) & (out["date"] <= hi)]
        out.insert(0, "ts_code", code)
        return code, out, None
    except Exception as e:  # noqa: BLE001
        return code, None, repr(e)


def load_ridx_map():
    idx = fe.load_index_df(fe.DATA_DIR / "000001.SH.parquet")
    c = idx["close"].to_numpy(np.float64)
    r = np.full(len(c), np.nan)
    r[1:] = c[1:] / c[:-1] - 1.0
    return {int(d): float(v) for d, v in zip(idx["_days"].to_numpy(np.int64), r)}


def stage_s2(universe: set[str], lo, hi, workers: int, force: bool) -> pd.DataFrame:
    out_path = CACHE_DIR / "s2_stock.parquet"
    if not force and out_path.exists():
        log("阶段2 s2 缓存命中")
        return pd.read_parquet(out_path)
    t0 = time.time()
    ridx = load_ridx_map()
    tasks = [(code, lo, hi) for code in sorted(universe)]
    frames, errors = [], {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_s2_init,
                             initargs=(ridx,)) as ex:
        for i, (code, out, err) in enumerate(ex.map(_s2_worker, tasks, chunksize=8)):
            if err:
                errors[code] = err
            elif len(out):
                frames.append(out)
            if (i + 1) % 500 == 0:
                log(f"  s2 {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    assert not errors, f"s2 失败 {len(errors)}: {list(errors.items())[:3]}"
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out_path, index=False)
    log(f"阶段2 完成: {df.shape} ({time.time() - t0:.0f}s)")
    return df


# ---------------------------------------------------------------- 阶段 3：s3 逐股链 + 横截面
def _s3_worker(task):
    path, code, lo, hi = task
    try:
        df = v4s.load_stock_v2(path, code)
        feat = v4s.compute_stock_features(df, pipe=fp2.FeaturePipeline(None, None))
        if feat is None:
            return code, None, "历史不足100行"
        keep = ["timestamp", "symbol"] + [c for c in S3_BASE_COLS if c in feat.columns]
        out = feat[keep]
        out = out[(out["timestamp"] >= lo) & (out["timestamp"] <= hi)]
        if not len(out):
            return code, None, None
        return code, out, None
    except Exception as e:  # noqa: BLE001
        return code, None, repr(e)


def stage_s3(lo, hi, workers: int, force: bool) -> pd.DataFrame:
    """返回全市场窗口日面板（基列 + 横截面变换列）。"""
    out_path = CACHE_DIR / "s3_cross.parquet"
    if not force and out_path.exists():
        log("阶段3 s3 缓存命中")
        return pd.read_parquet(out_path)
    t0 = time.time()
    files = sorted(p for p in fe.DATA_DIR.glob("*.parquet")
                   if p.stem not in fe.INDEX_CODES)
    tasks = [(p, p.stem, lo, hi) for p in files]
    frames, errors = [], {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (code, out, err) in enumerate(ex.map(_s3_worker, tasks, chunksize=8)):
            if err and err != "历史不足100行":
                errors[code] = err
            elif out is not None and len(out):
                frames.append(out)
            if (i + 1) % 500 == 0:
                log(f"  s3 {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    assert not errors, f"s3 失败 {len(errors)}: {list(errors.items())[:3]}"
    panel = pd.concat(frames, ignore_index=True)
    log(f"s3 基列面板 {panel.shape}，横截面变换 ... ({time.time() - t0:.0f}s)")
    pipe = fp2.FeaturePipeline(None, None)
    panel = pipe._calculate_cross_features(panel)
    got = [c for c in S3_CROSS_COLS if c in panel.columns]
    assert got == S3_CROSS_COLS, f"s3 横截面列缺失: {set(S3_CROSS_COLS) - set(got)}"
    panel = panel.rename(columns={"timestamp": "date", "symbol": "ts_code"})
    panel = panel[["ts_code", "date"] + S3_CROSS_COLS]
    panel.to_parquet(out_path, index=False)
    log(f"阶段3 完成: {panel.shape} ({time.time() - t0:.0f}s)")
    return panel


# ---------------------------------------------------------------- 阶段 4：s4 全历史面板
def stage_s4(universe: set[str], lo, hi, force: bool):
    out_stock = CACHE_DIR / "s4_stock.parquet"
    out_market = CACHE_DIR / "s4_market.parquet"
    if not force and out_stock.exists() and out_market.exists():
        log("阶段4 s4 缓存命中")
        return pd.read_parquet(out_stock), pd.read_parquet(out_market)
    t0 = time.time()
    ctx = t3.build_ctx()
    log(f"s4 ctx 完成 ({time.time() - t0:.0f}s)，装配全市场面板 ...")
    panel = t3.build_panel()
    log(f"s4 面板 {panel.shape} ({time.time() - t0:.0f}s)，计算特征 ...")
    feat = t3.compute_all(panel, ctx)
    del panel
    log(f"s4 特征面板 {feat.shape} ({time.time() - t0:.0f}s)，切片 ...")
    mkt = feat[["date"] + S4_MARKET_COLS].drop_duplicates("date").sort_values("date")
    mkt = mkt[(mkt["date"] >= lo) & (mkt["date"] <= hi)]
    stock = feat[["ts_code", "date"] + S4_STOCK_COLS]
    stock = stock[(stock["date"] >= lo) & (stock["date"] <= hi)
                  & stock["ts_code"].isin(universe)]
    stock.to_parquet(out_stock, index=False)
    mkt.to_parquet(out_market, index=False)
    log(f"阶段4 完成: 股内 {stock.shape} 市场 {mkt.shape} ({time.time() - t0:.0f}s)")
    return stock, mkt


# ---------------------------------------------------------------- 阶段 5：装配
def stage_assemble(order: list[str], lo, hi, force: bool) -> pd.DataFrame:
    out_path = CACHE_DIR / "panel_features.parquet"
    if not force and out_path.exists():
        log("阶段5 装配缓存命中")
        return pd.read_parquet(out_path)
    t0 = time.time()
    s1 = pd.read_parquet(CACHE_DIR / "s1_stock.parquet")
    s1["date"] = pd.to_datetime(s1["date"])
    s2 = pd.read_parquet(CACHE_DIR / "s2_stock.parquet")
    s2["date"] = pd.to_datetime(s2["date"])
    s3 = pd.read_parquet(CACHE_DIR / "s3_cross.parquet")
    s3["date"] = pd.to_datetime(s3["date"])
    s4s = pd.read_parquet(CACHE_DIR / "s4_stock.parquet")
    s4s["date"] = pd.to_datetime(s4s["date"])
    s1m = pd.read_parquet(CACHE_DIR / "s1_market.parquet")
    s1m["date"] = pd.to_datetime(s1m["date"])
    s4m = pd.read_parquet(CACHE_DIR / "s4_market.parquet")
    s4m["date"] = pd.to_datetime(s4m["date"])

    panel = s1.merge(s2, on=["ts_code", "date"], how="outer", validate="1:1")
    panel = panel.merge(s3, on=["ts_code", "date"], how="left", validate="1:1")
    panel = panel.merge(s4s, on=["ts_code", "date"], how="left", validate="1:1")
    panel = panel.merge(s1m, on="date", how="left", validate="m:1")
    panel = panel.merge(s4m, on="date", how="left", validate="m:1")
    panel = panel.sort_values(["ts_code", "date"]).reset_index(drop=True)
    missing = [c for c in order if c not in panel.columns]
    assert not missing, f"装配缺列: {missing}"
    panel = panel[["ts_code", "date"] + order]
    panel.to_parquet(out_path, index=False)
    log(f"阶段5 完成: {panel.shape} ({time.time() - t0:.0f}s)")
    return panel


# ---------------------------------------------------------------- 断言
def assert_coverage(panel: pd.DataFrame, events: pd.DataFrame) -> dict:
    key_panel = pd.MultiIndex.from_arrays([panel["ts_code"], panel["date"]])
    key_ev = pd.MultiIndex.from_arrays([events["ts_code"], events["date"]])
    missing = key_ev.difference(key_panel)
    return {"name": "coverage", "pass": len(missing) == 0,
            "n_events": len(key_ev), "n_missing": len(missing),
            "examples": [str(x) for x in list(missing)[:5]]}


def assert_feature_anchor(panel: pd.DataFrame, events: pd.DataFrame,
                          order: list[str]) -> dict:
    masters = []
    for pool in ("main", "backup"):
        m = pd.read_parquet(MASTER_DIR / f"master_{pool}.parquet",
                            columns=["ts_code", "date"] + order)
        masters.append(m)
    master = pd.concat(masters, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    # 同一 (ts_code, date) 可能因主/备池双收录而重复；特征只依赖键本身，
    # 全列去重后若仍有键重复（键同值异），下方 1:1 合并会显式报错。
    master = master.drop_duplicates()
    key_ev = events[["ts_code", "date"]].drop_duplicates()
    master = master.merge(key_ev, on=["ts_code", "date"], how="inner", validate="1:1")
    cmp = master.merge(panel, on=["ts_code", "date"], how="left",
                       validate="1:1", suffixes=("_m", "_p"))
    assert len(cmp) == len(master), "锚定合并行数不齐"
    max_diff = 0.0
    bad_cols = []
    # 主表以 float32 存储、面板为 float64 原值；容差取 float32 精度量级（rtol 1e-6 ≈ 8 ulp），
    # 任何真实逻辑分叉（不同窗口/复权/秩口径）都会产生远超此量级的差异。
    for c in order:
        a = cmp[f"{c}_m"].to_numpy(np.float64)
        b = cmp[f"{c}_p"].to_numpy(np.float64)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        diff = np.where(np.isnan(a) != np.isnan(b), np.inf, diff)
        d = float(np.max(diff))
        max_diff = max(max_diff, d)
        tol = 1e-6 * np.maximum(np.abs(a), 1e-12)
        if bool(np.any(diff > tol)):
            bad_cols.append((c, d))
    return {"name": "feature_anchor", "pass": not bad_cols,
            "n_cells": len(cmp) * len(order), "max_abs_diff": max_diff,
            "bad_cols": bad_cols[:10]}


def assert_prefix_spot(panel: pd.DataFrame, universe: set[str], ctx,
                       lo, hi) -> dict:
    """抽样 (股票, 窗口内日) 单元，仅用 <=T 数据重算 s1/s2/s4 列与面板比对。"""
    rng = np.random.default_rng(ASSERT_SEED)
    stocks = sorted(universe)
    picks = [stocks[i] for i in rng.choice(len(stocks), size=min(N_ASSERT_STOCKS, len(stocks)),
                                           replace=False)]
    ridx = load_ridx_map()
    n_checked, mismatches = 0, []
    for code in picks:
        sub = panel[panel["ts_code"] == code]
        if not len(sub):
            continue
        days = sub["date"].to_numpy()
        sel = rng.choice(len(days), size=min(ASSERT_DAYS_PER_STOCK, len(days)),
                         replace=False)
        path = fe.DATA_DIR / f"{code}.parquet"
        for j in sel:
            T = pd.Timestamp(days[j])
            row = sub[sub["date"] == T].iloc[0]
            # s1
            df_full = fe.load_stock_df(path)
            df_t = df_full[df_full["trade_date"] <= T].reset_index(drop=True)
            feats_t, _ = fe.compute_stock_features(df_t, code)
            for c in S1_STOCK_COLS:
                a, b = float(row[c]), float(feats_t[c].iloc[-1])
                n_checked += 1
                if not np.isclose(a, b, rtol=1e-6, atol=0, equal_nan=True):
                    mismatches.append({"ts_code": code, "date": str(T.date()),
                                       "col": c, "panel": a, "recompute": b})
            # s2
            cols_t, _ = ff.compute_stock_factory(df_t, ridx)
            for c in S2_COLS:
                a, b = float(row[c]), float(cols_t[c][-1])
                n_checked += 1
                if not np.isclose(a, b, rtol=1e-6, atol=0, equal_nan=True):
                    mismatches.append({"ts_code": code, "date": str(T.date()),
                                       "col": c, "panel": a, "recompute": b})
            # s4（t3 自带截断重算对照实现，含市场级列）
            ref = t3.prefix_recompute_at(t3.STOCK_DATA, ctx, code, T)
            if ref is not None and len(ref):
                for c in S4_STOCK_COLS + S4_MARKET_COLS:
                    a, b = float(row[c]), float(ref.iloc[0][c])
                    n_checked += 1
                    if not np.isclose(a, b, rtol=1e-5, atol=0, equal_nan=True):
                        mismatches.append({"ts_code": code, "date": str(T.date()),
                                           "col": c, "panel": a, "recompute": b})
    return {"name": "prefix_spot", "pass": not mismatches, "n_checked": n_checked,
            "mismatches": mismatches[:10]}


def calibrate_prob(raw_prob: np.ndarray) -> np.ndarray:
    """逻辑回归校准层 [p, p²] 的手工实现（系数自 calibrator.json 读出并断言字段齐全）。"""
    with CAL_PATH.open(encoding="utf-8") as f:
        cal = json.load(f)
    assert cal["form"] == "logistic(p, p^2)", "校准层形态异常"
    coef = np.asarray([cal["coef_p"], cal["coef_p2"]], dtype=np.float64)
    intercept = float(cal["intercept"])
    z = coef[0] * raw_prob + coef[1] * raw_prob * raw_prob + intercept
    return 1.0 / (1.0 + np.exp(-z))


def score_panel(panel: pd.DataFrame, order: list[str]) -> np.ndarray:
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    raw = booster.predict(panel[order])
    return calibrate_prob(np.asarray(raw, dtype=np.float64))


def assert_prob_anchor(panel_with_prob: pd.DataFrame, events: pd.DataFrame) -> dict:
    cmp = events.merge(panel_with_prob[["ts_code", "date", "prob"]],
                       on=["ts_code", "date"], how="left", validate="m:1",
                       suffixes=("_final", "_panel"))
    diff = (cmp["prob_final"] - cmp["prob_panel"]).abs()
    max_diff = float(diff.max())
    # 容差 1e-12：仅吸收 float32 特征存储引入的末位 ulp 抖动（实测 2.2e-16）
    return {"name": "prob_anchor", "pass": max_diff <= 1e-12,
            "n_events": len(cmp), "max_abs_diff": max_diff}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2022-10-31")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    tag = f"{lo:%Y%m%d}_{hi:%Y%m%d}"
    global CACHE_DIR
    CACHE_DIR = OUT_DIR / f"panel_cache_{tag}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_panel = OUT_DIR / f"daily_score_panel_{tag}.parquet"
    out_results = OUT_DIR / f"daily_score_panel_{tag}_results.json"

    order = load_final_feature_order()
    assert sorted(order) == sorted(S1_STOCK_COLS + S1_MARKET_COLS + S2_COLS
                                   + S3_CROSS_COLS + S4_STOCK_COLS + S4_MARKET_COLS), \
        "25 特征清单与四来源列集不一致"

    scores = pd.read_parquet(SCORES)
    scores["date"] = pd.to_datetime(scores["date"])
    events = scores[(scores["date"] >= lo) & (scores["date"] <= hi)]
    universe = set(events["ts_code"].unique())
    log(f"窗口 [{args.start}..{args.end}] 事件 {len(events)} 宇宙股数 {len(universe)}")

    stage_s1(universe, lo, hi, args.workers, args.force)
    stage_s2(universe, lo, hi, args.workers, args.force)
    stage_s3(lo, hi, args.workers, args.force)
    stage_s4(universe, lo, hi, args.force)
    panel = stage_assemble(order, lo, hi, args.force)

    results = {"window": [args.start, args.end], "n_universe": len(universe),
               "n_events": len(events), "panel_rows": len(panel), "assertions": []}

    log("断言 A 覆盖 ...")
    results["assertions"].append(assert_coverage(panel, events))
    log("断言 B 特征锚定 ...")
    results["assertions"].append(assert_feature_anchor(panel, events, order))
    log("断言 C 前缀截断抽检 ...")
    ctx = t3.build_ctx()
    results["assertions"].append(assert_prefix_spot(panel, universe, ctx, lo, hi))

    log("定版模型打分 ...")
    panel["prob"] = score_panel(panel, order)
    log("断言 D 分数锚定 ...")
    results["assertions"].append(assert_prob_anchor(panel, events))

    panel.to_parquet(out_panel, index=False)
    results["elapsed_s"] = round(time.time() - t0, 1)
    results["panel_path"] = str(out_panel)
    with out_results.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    n_fail = sum(1 for a in results["assertions"] if not a["pass"])
    log(f"面板落盘 {out_panel} ({panel.shape})；断言 {len(results['assertions'])} 项，"
        f"失败 {n_fail}；总耗时 {time.time() - t0:.0f}s")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
