#!/usr/bin/env python3
"""来源 1:事件级特征词典 v6 重建驱动(战役 #47 M2,issue #49)。

机器来源:`build_feature_matrix.py` 自 git 历史 4c7717f^ 恢复(#44 §2.4 来源 1),
几何族按 #42 重定义(v5 的 17 条 V1 事件结构特征 DIV_* 不重建、不带入)。

输入:
  事件表   experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet(只读)
  原始日线 stock_data/daily/*.parquet(只读)
输出:
  cache/s1_event_dictionary.parquet   (96,577 行,键 event_id)
  cache/s1_dictionary.csv             (列名/规范名/家族/层/公式/中文全称)
  cache/s1_results.json               (台账:对账/种子布尔/NaN 披露/耗时)

口径要点(README 军令状 §一/§四/§七):
  - 个股加载 = scanner 口径(sort_values(trade_date) + reset_index),与 M1 事件表行号对齐;
    实测全市场 0 文件存在 NaN close / 重复 trade_date,与 v5 load_stock_df 口径等价。
  - P0/P1 股内特征复用 feature_engine.compute_stock_features(复权链口径,v5 原样)。
  - v6 几何族 40 条在原始价 + 原始价 MACD 上计算(#42 钉死不复权),逐股与事件表
    落盘列逐位对账(不等即停线,断言 6 的构建内执行)。
  - 种子布尔 F6~F10 在 float64 上判定;全表计数对 frozen 值硬断言。
  - 市场/宽度/日历特征 = feature_engine.build_market_frame(全 universe 面板,池无关);
    RET20_CSR = 当日全 universe 横截面百分位(v5 原口径)。

用法: python build_event_dictionary.py [--workers 24] [--sample 0] [--out cache/s1_event_dictionary.parquet]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import talib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import feature_engine as fe  # noqa: E402
import geo_features as geo  # noqa: E402

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

# v5 的 17 条 V1 事件结构特征(v6 不重建,注册表差集断言用)
V1_EVENT_COLS = set(fe.EVENT_FEATURE_COLS)

# 事件表行号/日期列 -> 事件表落盘值列 的逐位对账清单(构建内断言 6)
# (行号列, 值列, 取值数组名);数组名: dif/dea/close/vol/amount
RECON_CHECKS = [
    ("event_row", "cross_dif", "dif"), ("event_row", "cross_dea", "dea"),
    ("event_row", "event_close", "close"), ("event_row", "event_vol", "vol"),
    ("event_row", "event_amount", "amount"),
    ("cross_prev_row", "cross_prev_dif", "dif"), ("cross_prev_row", "cross_prev_dea", "dea"),
    ("cross_prev2_row", "cross_prev2_dif", "dif"), ("cross_prev2_row", "cross_prev2_dea", "dea"),
    ("anchor_row", "anchor_close", "close"),
    ("min_prev_row", "min_prev_close", "close"),
]


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [s1] {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_events():
    """v6 事件表 + event_id 铸造((ts_code, event_date) 字典序序号,#44 §2.4)。"""
    ev = pd.read_parquet(EVENTS_PATH)
    assert not ev["ts_code"].isin(fe.INDEX_CODES).any(), "事件表含指数伪股"
    assert not ev.duplicated(["ts_code", "event_date"]).any()
    ev = ev.sort_values(["ts_code", "event_date"], kind="mergesort").reset_index(drop=True)
    ev["event_id"] = np.arange(len(ev), dtype=np.int64)
    return ev


def scan_load(path):
    """scanner 口径加载(M1 逐字):sort + reset_index,不 dropna 不 dedup。"""
    df = pd.read_parquet(path)
    return df.sort_values("trade_date").reset_index(drop=True)


def _worker(task):
    """单股:P0/P1 全历史特征 + 面板列 + v6 几何族(事件行) + 逐位对账。"""
    path, code, sid, evrecs, idx_dates, idx_r = task
    try:
        df = scan_load(path)
    except Exception as e:  # noqa: BLE001
        return dict(panel=None, rows=None, error=f"read: {e}", n_events=0)
    if len(df) < 30:
        return dict(panel=None, rows=None, error="too_short", n_events=0)
    try:
        feats, ctx = fe.compute_stock_features(df, code, idx_dates, idx_r)
    except Exception as e:  # noqa: BLE001
        return dict(panel=None, rows=None, error=f"compute: {e}", n_events=0)

    days = ctx["days"]
    panel = pd.DataFrame({"sid": np.full(len(days), sid, np.int32), "date": days,
                          "is_index": code in fe.INDEX_CODES, **ctx["panel"]})
    if not evrecs:
        return dict(panel=panel, rows=None, error=None, n_events=0)

    # ---- v6 几何族(原始价 + 原始价 MACD)
    C = df["close"].to_numpy(np.float64)
    V = pd.to_numeric(df["vol"], errors="coerce").to_numpy(np.float64)
    A = pd.to_numeric(df["amount"], errors="coerce").to_numpy(np.float64) \
        if "amount" in df.columns else np.full(len(df), np.nan)
    dif, dea, _ = talib.MACD(C)
    j = np.array([r[1] for r in evrecs], np.int64)
    i1 = np.array([r[2] for r in evrecs], np.int64)
    i2 = np.array([r[3] for r in evrecs], np.int64)
    a = np.array([r[4] for r in evrecs], np.int64)
    p_low = np.array([r[5] for r in evrecs], np.int64)
    eid = np.array([r[0] for r in evrecs], np.int64)

    # 构建内逐位对账(README §七.6:不等即停线)
    arrays = {"dif": dif, "dea": dea, "close": C, "vol": V, "amount": A}
    for row_col, val_col, arr_name in RECON_CHECKS:
        rows = np.array([r[{"event_row": 1, "cross_prev_row": 2, "cross_prev2_row": 3,
                            "anchor_row": 4, "min_prev_row": 5}[row_col]] for r in evrecs],
                        np.int64)
        got = arrays[arr_name][rows]
        want = np.array([r[6][val_col] for r in evrecs], np.float64)
        if not np.array_equal(got, want, equal_nan=True):
            bad = int((~((got == want) | (np.isnan(got) & np.isnan(want)))).sum())
            raise RuntimeError(f"{code}: 对账失败 {val_col}({bad} 行不等)")

    geo_cols = geo.compute_geo_events(C, V, dif, dea, i2, i1, j, a, p_low)

    blk = feats.iloc[j].reset_index(drop=True)
    for c in geo.GEO_COLUMNS:
        blk[c] = geo_cols[c]
    blk.insert(0, "event_id", eid)
    blk = blk.astype({c: np.float32 for c in blk.columns if c != "event_id"})
    return dict(panel=panel, rows=blk, error=None, n_events=len(eid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sample", type=int, default=0, help="只取前 N 个文件(冒烟)")
    ap.add_argument("--out", type=str, default=str(CACHE_DIR / "s1_event_dictionary.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 指数(IVOL60 与市场特征用,v5 原口径)
    idx_sh = fe.load_index_df(fe.DATA_DIR / "000001.SH.parquet")
    idx_sz = fe.load_index_df(fe.DATA_DIR / "399001.SZ.parquet")
    idx_dates = idx_sh["_days"].to_numpy(np.int32)
    c_idx = pd.Series(idx_sh["close"].to_numpy(np.float64))
    idx_r = (c_idx / c_idx.shift(1) - 1.0).to_numpy(np.float64)

    # ---- 事件表 + event_id 铸造
    ev = load_events()
    log(f"事件 {len(ev)} 行,涉股 {ev['ts_code'].nunique()},"
        f"event_id 铸造 = (ts_code, event_date) 字典序")
    per_stock = defaultdict(list)
    for r in ev.itertuples():
        per_stock[r.ts_code].append(
            (int(r.event_id), int(r.event_row), int(r.cross_prev_row),
             int(r.cross_prev2_row), int(r.anchor_row), int(r.min_prev_row),
             {"cross_dif": r.cross_dif, "cross_dea": r.cross_dea,
              "event_close": r.event_close, "event_vol": r.event_vol,
              "event_amount": r.event_amount, "cross_prev_dif": r.cross_prev_dif,
              "cross_prev_dea": r.cross_prev_dea, "cross_prev2_dif": r.cross_prev2_dif,
              "cross_prev2_dea": r.cross_prev2_dea, "anchor_close": r.anchor_close,
              "min_prev_close": r.min_prev_close}))

    # ---- 全 universe 并行计算
    files = sorted(fe.DATA_DIR.glob("*.parquet"))
    if args.sample:
        files = files[: args.sample]
    tasks = [(str(p), p.stem, sid, per_stock.get(p.stem, []), idx_dates, idx_r)
             for sid, p in enumerate(files)]
    log(f"加载并计算 {len(tasks)} 只股票 (workers={args.workers}) ...")
    panels, blocks = [], []
    n_err, errs = 0, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_worker, tasks, chunksize=16)):
            if r["error"]:
                n_err += 1
                errs[tasks[i][1]] = r["error"]
                continue
            if r["panel"] is not None:
                panels.append(r["panel"])
            if r["rows"] is not None:
                blocks.append(r["rows"])
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    panel = pd.concat(panels, ignore_index=True)
    log(f"面板 {len(panel)} 行,跳过/失败 {n_err} 只 ({time.time() - t0:.0f}s)")

    # ---- 市场特征框(date 级,v5 原口径)
    market, _ = fe.build_market_frame(panel, idx_sh, idx_sz)
    log(f"市场特征框 {market.shape} ({time.time() - t0:.0f}s)")

    # ---- 装配来源 1 矩阵
    rows = pd.concat(blocks, ignore_index=True)
    rows = rows.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    df = ev[["event_id", "ts_code", "event_date"]].rename(columns={"event_date": "date"})
    df = df.merge(rows, on="event_id", how="left", validate="1:1")
    days = fe._to_days(df["date"])
    mkt = market.reindex(days)
    mkt.index = df.index
    df = pd.concat([df, mkt.reset_index(drop=True).astype(np.float32)], axis=1)
    df["RET20_CSR"] = fe.attach_ret20_csr(
        panel, days, df["RET20"].to_numpy(np.float64)).astype(np.float32)

    # 列集断言:注册表差集(V1 事件 17 条不产出) + 几何族 40 + 元数据
    produced = set(df.columns) - {"event_id", "ts_code", "date"}
    expect = (set(fe.FEATURE_REGISTRY) - V1_EVENT_COLS) | set(geo.GEO_COLUMNS)
    missing = expect - produced
    extra = produced - expect
    assert not missing, f"应产出而未产出: {sorted(missing)[:10]}"
    assert not extra, f"未注册注入列: {sorted(extra)[:10]}"
    feat_cols = [c for c in fe.FEATURE_REGISTRY if c in df.columns] + geo.GEO_COLUMNS
    df = df[["event_id", "ts_code", "date"] + feat_cols]
    fe.assert_no_blacklisted(df.columns)
    fe.assert_no_label_columns(df.columns)
    fe.assert_no_inf(df[feat_cols])
    assert len(df) == len(ev), "来源 1 行数 != 事件数"

    # ---- 种子布尔全表硬断言(README §四)
    seed_counts = {c: int(df[c].to_numpy(np.float64).sum()) for c in geo.SEED_FROZEN_COUNTS}
    assert seed_counts == geo.SEED_FROZEN_COUNTS, f"种子计数不符: {seed_counts}"
    s = {c: df[c].astype(bool).to_numpy() for c in geo.SEED_FROZEN_COUNTS}
    assert not (s["SEED_V6_3"] & ~s["SEED_V6_2"]).any()
    assert not (s["SEED_V6_2"] & ~s["SEED_V6_1"]).any()
    assert not (s["SEED_V6_1"] & ~s["SEED_V6_4"]).any()
    assert not (s["SEED_V6_5"] & ~s["SEED_V6_4"]).any()
    log(f"种子布尔全表对账通过: {seed_counts} ({time.time() - t0:.0f}s)")

    # ---- 几何族不变量(README §七.7 构建内部分)
    assert (df["DD20"].to_numpy(np.float64) <= 1e-12 + 1e-6).all(), "dd20 符号违反"
    assert (df["ANCHOR_BREAK_DEPTH"].to_numpy(np.float64) < 0).all(), "D3 构造 < 0 违反"
    d2 = df["ANCHOR_REL_POS"].to_numpy(np.float64)
    assert ((d2 > 0) & (d2 <= 1 + 1e-6)).all(), "D2 值域 (0,1] 违反"

    # ---- 落盘
    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)
    log(f"来源 1 矩阵 {df.shape} -> {out_path.name} ({time.time() - t0:.0f}s)")

    # 词典(中文全名列)
    dic_rows = []
    for c, spec in fe.FEATURE_REGISTRY.items():
        if c in V1_EVENT_COLS:
            continue
        dic_rows.append(dict(column=c, feature=spec.feature, layer=spec.layer,
                             family=spec.family, formula=spec.formula,
                             cn_name=f"{spec.family}｜{spec.formula}",
                             event_only=spec.event_only))
    for c in geo.GEO_COLUMNS:
        dic_rows.append(dict(column=c, feature=c, layer="v6_geo",
                             family=f"v6几何族{geo.GEO_ID[c]}", formula=geo.GEO_FORMULA[c],
                             cn_name=geo.GEO_CN[c], event_only=True))
    dic = pd.DataFrame(dic_rows)
    dic_path = CACHE_DIR / "s1_dictionary.csv"
    dic.to_csv(dic_path, index=False)

    nan_disclosure = {c: float(df[c].isna().mean()) for c in geo.GUARDED_NAN_COLS}
    results = {
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "feature_cols": len(feat_cols),
        "geo_cols": len(geo.GEO_COLUMNS),
        "registry_cols": len(feat_cols) - len(geo.GEO_COLUMNS),
        "v1_event_cols_dropped": sorted(V1_EVENT_COLS),
        "seed_counts": seed_counts,
        "guarded_nan_rate": nan_disclosure,
        "a_eq_j_share": float((ev["anchor_bars"] == 0).mean()),
        "errors": errs,
        "elapsed_sec": time.time() - t0,
    }
    (CACHE_DIR / "s1_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str))
    log(f"完成 {time.time() - t0:.0f}s;守卫 NaN 披露: {nan_disclosure}")


if __name__ == "__main__":
    main()
