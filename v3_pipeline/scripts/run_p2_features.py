# -*- coding: utf-8 -*-
"""P2 补数据特征: 实现 + 三关筛查 (issue #14).

依据:
  - 特征主表 v3_pipeline/reports/feature_harvest/feature_master_spec.md §4 (P2 15 条) + §7 (泄漏防线)
  - 三关口径 v3_pipeline/reports/feature_screening/review_report.md (含几何剥离实验 D)
  - 数据契约 stock_data/SUPPLEMENTARY_DATA.md

产出 (v3_pipeline/reports/p2_features/):
  p2_feature_values_{main,backup}.parquet  事件行 P2 特征值 (+ AUX_THR_PCT 几何辅助列)
  per_feature_ic.csv                      逐特征 IC/三关/几何剥离结果
  p2_features_report.md                   15 条逐个判定报告
  progress.log                            阶段日志 + 心跳
  run_meta.json                           计数与截断对拍结果

纪律:
  - 全历史逐股计算 -> 末端按事件行切片; 滚动算子 min_periods=window.
  - 统计只用 train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31 (含隔离带剔除);
    2022-11 后只落特征值, 不入任何统计.
  - 池清洗: excluded_events_*.parquet 的 f_any (f_st|f_suspend|f_limitup).
    ST_FLAG 例外: f_st 清洗会抽干其截面变差, 另出 f_suspend|f_limitup 口径作其主读数.
  - ths_index/ths_member 仅当前快照 = 前瞻性, 一律不用 (LINK_MOM 记不可算).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import talib

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
import feature_engine as fe  # noqa: E402

REPO = SCRIPT_DIR.parents[1]
OUT_DIR = REPO / "v3_pipeline" / "reports" / "p2_features"
CACHE_DIR = OUT_DIR / "_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "progress.log"

FM_DIR = REPO / "v3_pipeline" / "reports" / "feature_matrix"
POOLS = {
    "main": {
        "features": FM_DIR / "main_pool_features.parquet",
        "labdir": REPO / "v3_pipeline" / "reports" / "divergence_lab" / "w_fractal_o15_s20",
        "excluded": REPO / "v3_pipeline" / "reports" / "pool_cleaning" / "excluded_events_main.parquet",
    },
    "backup": {
        "features": FM_DIR / "backup_pool_features.parquet",
        "labdir": REPO / "v3_pipeline" / "reports" / "divergence_lab" / "w_zigzag_p05_s5",
        "excluded": REPO / "v3_pipeline" / "reports" / "pool_cleaning" / "excluded_events_backup.parquet",
    },
}

TRAIN_LO, TRAIN_HI = pd.Timestamp("2001-01-01"), pd.Timestamp("2018-12-31")
VAL_LO, VAL_HI = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-10-31")
EMBARGO = [(pd.Timestamp("2018-11-19"), pd.Timestamp("2018-12-28")),
           (pd.Timestamp("2022-09-13"), pd.Timestamp("2022-10-31"))]
MIN_DAY_N = 5
MIN_YEAR_DAYS = 10
YEARS = list(range(2001, 2023))

DATA_DAILY = REPO / "stock_data" / "daily"
DATA_BASIC = REPO / "stock_data" / "daily_basic"
DATA_LIMIT = REPO / "stock_data" / "stk_limit"
DATA_INDEX = REPO / "stock_data" / "index"
DATA_META = REPO / "stock_data" / "meta"

# 参考特征 (共线性对照, 从主特征矩阵取): 几何控制 + 流动性 + 反转 + 制度近似版
REF_FEATS = ["ATRN", "VOL20", "AMP20", "ILLIQ20", "AMT20", "RET20",
             "LIMITCNT20", "LIMITDOWN_CNT20", "DIST_LIMIT", "MKT_RET20", "MKT_MA60"]
GEO_CONTROLS = ["ATRN", "VOL20", "AMP20"]

# P2 输出列 (黑名单安全: 无 ^rank_/^next_/^future_ 等前缀)
TURN_COLS = ["TURN20", "ABTURN", "STDTURN20", "FLOAT_MCAP"]
LIMIT_COLS = ["LIMITCNT20_X", "LIMITDOWN_CNT20_X", "DIST_LIMIT_X"]
IND_COLS = ["IND_REL_STR", "IND_MOM20", "IND_RS_RANK", "IND_BREADTH_MA20", "IND_RET5_RANK"]
IDX_SPECS = [("000300.SH", "MKT_CSI300"), ("000905.SH", "MKT_CSI500"),
             ("000852.SH", "MKT_CSI1000"), ("399006.SZ", "MKT_CHINEXT")]
MKT_COLS = ["RETAIL_SENT"] + [f"{p}_{s}" for _, p in IDX_SPECS
                              for s in ("RET20", "MA60", "VOL20_PCT")]
ST_COLS = ["ST_FLAG"]
CROSS_SECTIONAL = TURN_COLS + LIMIT_COLS + IND_COLS + ST_COLS  # 参与逐日 Rank IC 三关
ALL_P2 = TURN_COLS + LIMIT_COLS + IND_COLS + ST_COLS + MKT_COLS


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


# ================================================================ 数据加载 (带缓存)
def _days_int(s) -> np.ndarray:
    return fe._to_days(s)


def _read_one_stock(path: str):
    """轻量面板: date/r1/ret20/above_ma20/open_adj/atr14/close/pre_close (全历史, 因果)."""
    code = Path(path).stem
    try:
        df = fe.load_stock_df(path)
    except Exception:  # noqa: BLE001
        return None
    if len(df) < 30 or code in fe.INDEX_CODES:
        return None
    C = df["close"].to_numpy(np.float64)
    O = pd.to_numeric(df["open"], errors="coerce").to_numpy(np.float64)
    H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64)
    PC = pd.to_numeric(df["pre_close"], errors="coerce").to_numpy(np.float64) \
        if "pre_close" in df.columns else np.full(len(df), np.nan)
    pct = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(np.float64)
    R = pct / 100.0
    cf = np.cumprod(1.0 + np.where(np.isfinite(R), R, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        f_adj = np.where(C > 0, cf / C, np.nan)
    ca, oa, ha, la = cf, O * f_adj, H * f_adj, L * f_adj
    scf = pd.Series(cf)
    ret20 = (scf / scf.shift(20) - 1.0).to_numpy()
    mean_ca20 = pd.Series(ca).rolling(20, min_periods=20).mean().to_numpy()
    above = np.where(np.isfinite(mean_ca20), (ca > mean_ca20).astype(np.float64), np.nan)
    atr14 = talib.ATR(ha, la, ca, timeperiod=14)
    out = pd.DataFrame({
        "ts_code": code,
        "date": _days_int(df["trade_date"]).astype(np.int32),
        "r1": np.where(np.isfinite(R), R, np.nan),
        "ret20": ret20,
        "above_ma20": above,
        "open_adj": oa,
        "atr14": atr14,
        "close": C,
        "pre_close": PC,
    })
    out.iloc[0, out.columns.get_loc("r1")] = np.nan  # 首日收益不存在 (engine 口径)
    return out


def _read_one_basic(path: str):
    try:
        df = pd.read_parquet(path, columns=["ts_code", "trade_date", "turnover_rate",
                                            "float_share", "close"])
    except Exception:  # noqa: BLE001
        return None
    df["date"] = _days_int(df["trade_date"]).astype(np.int32)
    return df[["ts_code", "date", "turnover_rate", "float_share", "close"]]


def _read_one_limit(path: str):
    try:
        df = pd.read_parquet(path, columns=["ts_code", "trade_date", "up_limit", "down_limit"])
    except Exception:  # noqa: BLE001
        return None
    df["date"] = _days_int(df["trade_date"]).astype(np.int32)
    return df[["ts_code", "date", "up_limit", "down_limit"]]


def load_panel_cached(name, reader, glob_dir, workers, heartbeat=1000):
    cache = CACHE_DIR / f"{name}.parquet"
    if cache.exists():
        log(f"[缓存] {name} <- {cache}")
        return pd.read_parquet(cache)
    files = sorted(str(p) for p in glob_dir.glob("*.parquet"))
    log(f"[加载] {name}: {len(files)} 文件 (workers={workers})")
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(reader, files, chunksize=32)):
            if r is not None:
                parts.append(r)
            if (i + 1) % heartbeat == 0:
                log(f"  [心跳] {name} {i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(cache, index=False)
    log(f"[加载] {name} 完成 {df.shape} ({time.time()-t0:.0f}s), 缓存 -> {cache}")
    return df


# ================================================================ 特征计算 (纯函数, 截断对拍复用)
def db_stock_features(db: pd.DataFrame) -> pd.DataFrame:
    """换手/市值族 (P2 #1-4): 全历史逐股 rolling, min_periods=window."""
    db = db.sort_values(["ts_code", "date"], kind="mergesort").reset_index(drop=True)
    turn = pd.to_numeric(db["turnover_rate"], errors="coerce")
    g = turn.groupby(db["ts_code"], sort=False)
    db["TURN20"] = g.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    m5 = g.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    m60 = g.rolling(60, min_periods=60).mean().reset_index(level=0, drop=True)
    db["ABTURN"] = fe._safe_div(m5.to_numpy(), m60.to_numpy())
    db["STDTURN20"] = g.rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    fshare = pd.to_numeric(db["float_share"], errors="coerce").to_numpy(np.float64)
    cdb = pd.to_numeric(db["close"], errors="coerce").to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        db["FLOAT_MCAP"] = np.where((fshare > 0) & (cdb > 0), np.log(cdb * fshare), np.nan)
    return db


def retail_sentiment(db: pd.DataFrame) -> pd.Series:
    """RETAIL_SENT (P2 #5): 全市场换手率加总 20 日均值的 250 日时序分位. 市场级 (日内截面恒定)."""
    turn = pd.to_numeric(db["turnover_rate"], errors="coerce")
    valid = turn.notna()
    cnt = valid.groupby(db["date"]).sum()
    s = turn[valid].groupby(db["date"][valid]).sum()
    s = s.where(cnt >= 100)  # 覆盖过薄的早年不定义
    s = s.sort_index()
    m20 = s.rolling(20, min_periods=20).mean()
    pct = fe._ts_pct_rank(m20.to_numpy(np.float64), 250)
    return pd.Series(pct, index=m20.index, name="RETAIL_SENT")


def index_market_features() -> pd.DataFrame:
    """MKT_CSI300 族 (P2 #13): 4 宽基指数 × (RET20, MA60, VOL20_PCT). 市场级."""
    frames = []
    for code, prefix in IDX_SPECS:
        df = pd.read_parquet(DATA_INDEX / f"{code}.parquet")
        df = df.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
        C = df["close"].to_numpy(np.float64)
        if "pct_chg" in df.columns and df["pct_chg"].notna().any():
            r = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(np.float64) / 100.0
        else:
            r = np.concatenate([[np.nan], C[1:] / C[:-1] - 1.0])
        sc = pd.Series(C)
        vol20 = pd.Series(r).rolling(20, min_periods=20).std().to_numpy()
        frames.append(pd.DataFrame({
            "date": _days_int(df["trade_date"]).astype(np.int32),
            f"{prefix}_RET20": (sc / sc.shift(20) - 1.0).to_numpy(),
            f"{prefix}_MA60": (sc / sc.rolling(60, min_periods=60).mean() - 1.0).to_numpy(),
            f"{prefix}_VOL20_PCT": fe._ts_pct_rank(vol20, 250),
        }))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def limit_stock_features(panel: pd.DataFrame, limit: pd.DataFrame) -> pd.DataFrame:
    """STK_LIMIT 精确版 (P2 #14): 2007-01-04 起; 此前/北交所无覆盖 -> NaN."""
    df = panel[["ts_code", "date", "close", "pre_close"]].merge(
        limit, on=["ts_code", "date"], how="left")
    df = df.sort_values(["ts_code", "date"], kind="mergesort").reset_index(drop=True)
    C = df["close"].to_numpy(np.float64)
    PC = df["pre_close"].to_numpy(np.float64)
    up = df["up_limit"].to_numpy(np.float64)
    dn = df["down_limit"].to_numpy(np.float64)
    has = np.isfinite(up) & np.isfinite(dn)
    flag_up = np.where(has, (C >= up - 0.005).astype(np.float64), np.nan)
    flag_dn = np.where(has, (C <= dn + 0.005).astype(np.float64), np.nan)
    su = pd.Series(flag_up).groupby(df["ts_code"], sort=False)
    sd = pd.Series(flag_dn).groupby(df["ts_code"], sort=False)
    df["LIMITCNT20_X"] = su.rolling(20, min_periods=20).sum().reset_index(level=0, drop=True)
    df["LIMITDOWN_CNT20_X"] = sd.rolling(20, min_periods=20).sum().reset_index(level=0, drop=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        limit_mag = np.where(has & (PC > 0), (up - PC) / PC, np.nan)
        dl = np.where(np.isfinite(limit_mag) & (limit_mag > 0.001),
                      (C / PC - 1.0) / limit_mag, np.nan)
    df["DIST_LIMIT_X"] = np.clip(dl, -1.5, 1.5)
    return df[["ts_code", "date"] + LIMIT_COLS]


# ---------------------------------------------------------------- 行业族 (P2 #7-11)
def build_l1_intervals():
    """申万 L1 历史成分 -> {ts_code: (starts, ends, ind_idx)}; 重叠区间取 in_date 最新者."""
    cls = pd.read_parquet(DATA_META / "sw_index_classify.parquet")
    l1 = cls[cls["level"] == "L1"][["index_code", "industry_name"]].reset_index(drop=True)
    ind_codes = l1["index_code"].tolist()
    ind_pos = {c: i for i, c in enumerate(ind_codes)}
    mem = pd.read_parquet(DATA_META / "sw_index_member.parquet")
    mem = mem[mem["index_code"].isin(ind_codes)]
    starts = pd.to_datetime(mem["in_date"], format="%Y%m%d").to_numpy("datetime64[D]").astype(np.int32)
    ends = pd.to_datetime(mem["out_date"], format="%Y%m%d").to_numpy("datetime64[D]")
    ends = np.where(pd.isna(ends), np.int32(30000), ends.astype(np.int32))
    mem = mem.assign(_s=starts, _e=ends, _i=mem["index_code"].map(ind_pos).astype(np.int16))
    intervals = {}
    for code, g in mem.sort_values("_s").groupby("con_code"):
        intervals[code] = (g["_s"].to_numpy(), g["_e"].to_numpy(), g["_i"].to_numpy())
    return intervals, ind_codes, l1.set_index("index_code")["industry_name"].to_dict()


def attach_industry(codes: np.ndarray, dates: np.ndarray, intervals) -> np.ndarray:
    """逐行 as-of 行业 (int16, -1=未覆盖). 重叠取 in_date 最新区间, 过期不回退."""
    out = np.full(len(dates), -1, np.int16)
    s = pd.Series(codes)
    for code, sub in s.groupby(s, sort=False):
        if code not in intervals:
            continue
        starts, ends, inds = intervals[code]
        pos = sub.index.to_numpy()
        i = np.searchsorted(starts, dates[pos], side="right") - 1
        ok = i >= 0
        if not ok.any():
            continue
        pos2, ii = pos[ok], i[ok]
        covered = dates[pos2] < ends[ii]
        out[pos2[covered]] = inds[ii[covered]]
    return out


def industry_day_frames(panel: pd.DataFrame, intervals, n_ind: int, min_members=5):
    """行业日聚合 -> (ind_day DataFrame, panel 行级行业数组).

    ind_day: (ind, date) 索引, 列 ind_ret1(等权日收益均值)/breadth/n_members/n_breadth.
    行业动量: 有效日 (成员>=min_members) 等权日收益累乘链的 20/5 日收益; 不足 20/5 个有效日 -> NaN.
    """
    codes = panel["ts_code"].to_numpy()
    dates = panel["date"].to_numpy(np.int32)
    ind_arr = attach_industry(codes, dates, intervals)
    df = pd.DataFrame({
        "ind": ind_arr, "date": dates,
        "r1": panel["r1"].to_numpy(np.float64),
        "above": panel["above_ma20"].to_numpy(np.float64),
    })
    df = df[df["ind"] >= 0]
    agg = df.groupby(["ind", "date"]).agg(
        ind_ret1=("r1", "mean"), n_members=("r1", "count"),
        breadth=("above", "mean"), n_breadth=("above", "count")).reset_index()
    agg["ind_ret1"] = agg["ind_ret1"].where(agg["n_members"] >= min_members)
    agg["breadth"] = agg["breadth"].where(agg["n_breadth"] >= min_members)
    mom20 = np.full(len(agg), np.nan)
    mom5 = np.full(len(agg), np.nan)
    for k, g in agg.groupby("ind", sort=False):
        g = g.sort_values("date")
        r = g["ind_ret1"].to_numpy(np.float64)
        lvl = np.cumprod(1.0 + np.where(np.isfinite(r), r, 0.0))
        sl = pd.Series(lvl, index=g.index)
        valid_cnt20 = pd.Series(np.isfinite(r).astype(np.float64), index=g.index) \
            .rolling(20, min_periods=20).sum()
        valid_cnt5 = pd.Series(np.isfinite(r).astype(np.float64), index=g.index) \
            .rolling(5, min_periods=5).sum()
        m20 = (sl / sl.shift(20) - 1.0).where(valid_cnt20 == 20)
        m5 = (sl / sl.shift(5) - 1.0).where(valid_cnt5 == 5)
        mom20[g.index.to_numpy()] = m20.to_numpy()
        mom5[g.index.to_numpy()] = m5.to_numpy()
    agg["IND_MOM20"] = mom20
    agg["IND_MOM5"] = mom5
    # 行业横截面分位 (当日 >=10 个有效行业)
    def cs_rank(col):
        r = agg.groupby("date")[col].rank(pct=True)
        n = agg.groupby("date")[col].count()
        return r.where(agg["date"].map(n) >= 10)
    agg["IND_RS_RANK"] = cs_rank("IND_MOM20")
    agg["IND_RET5_RANK"] = cs_rank("IND_MOM5")
    return agg, ind_arr


def industry_features_at(agg: pd.DataFrame, ind_arr_query: np.ndarray,
                         dates_query: np.ndarray) -> pd.DataFrame:
    """查询行 (行业, 日期) -> IND_MOM20/IND_RS_RANK/IND_BREADTH_MA20/IND_RET5_RANK."""
    key = agg.set_index(["ind", "date"])
    cols = ["IND_MOM20", "IND_RS_RANK", "IND_RET5_RANK", "breadth"]
    out = {c: np.full(len(dates_query), np.nan) for c in cols}
    for ind, sub in key.groupby(level=0, sort=False):
        pos = np.where(ind_arr_query == ind)[0]
        if len(pos) == 0:
            continue
        idx_dates = sub.index.get_level_values(1).to_numpy(np.int32)
        i = np.searchsorted(idx_dates, dates_query[pos])
        ok = (i < len(idx_dates)) & (i >= 0)
        hit = np.zeros(len(pos), bool)
        hit[ok] = idx_dates[i[ok]] == dates_query[pos][ok]
        for c in cols:
            vals = sub[c].to_numpy(np.float64)
            out[c][pos[hit]] = vals[i[hit]]
    return pd.DataFrame({
        "IND_MOM20": out["IND_MOM20"], "IND_RS_RANK": out["IND_RS_RANK"],
        "IND_RET5_RANK": out["IND_RET5_RANK"], "IND_BREADTH_MA20": out["breadth"],
    })


# ---------------------------------------------------------------- ST_FLAG (P2 #12)
def build_name_index():
    nc = pd.read_parquet(DATA_META / "namechange.parquet")
    nc["start"] = pd.to_datetime(nc["start_date"], format="%Y%m%d")
    idx = {}
    for code, g in nc.sort_values("start").groupby("ts_code"):
        starts = g["start"].to_numpy("datetime64[D]").astype(np.int32)
        is_st = np.array(["ST" in nm for nm in g["name"].to_numpy()])
        idx[code] = (starts, is_st)
    return idx


def st_flags(codes: np.ndarray, dates: np.ndarray, name_index) -> np.ndarray:
    """as-of: T 时点名称 = 最新 start_date<=T 的行; 含 'ST' 即 1 (run_pool_cleaning 同口径)."""
    out = np.full(len(dates), np.nan)
    s = pd.Series(codes)
    for code, sub in s.groupby(s, sort=False):
        pos = sub.index.to_numpy()
        if code not in name_index:
            out[pos] = 0.0
            continue
        starts, is_st = name_index[code]
        i = np.searchsorted(starts, dates[pos], side="right") - 1
        ok = i >= 0
        out[pos[~ok]] = 0.0
        out[pos[ok]] = is_st[i[ok]].astype(np.float64)
    return out


# ================================================================ 事件/标签/清洗
def load_pool_frame(pool: str) -> pd.DataFrame:
    cfg = POOLS[pool]
    fm = pd.read_parquet(cfg["features"], columns=["event_id", "ts_code", "date"] + REF_FEATS)
    fm["date"] = pd.to_datetime(fm["date"])
    ev = pd.read_parquet(cfg["labdir"] / "events.parquet", columns=["ts_code", "date"])
    lb = pd.read_parquet(cfg["labdir"] / "labels.parquet")
    n = len(ev)
    div = lb.iloc[:n].reset_index(drop=True)
    assert (div["group"] == "div").all()
    lab = pd.DataFrame({
        "ts_code": ev["ts_code"].values,
        "date": pd.to_datetime(ev["date"].values),
        "hit": div["hit_N20_k2.0"].values,
        "ret_h10": div["ret_h10"].values,
    })
    assert not lab.duplicated(["ts_code", "date"]).any()
    df = fm.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    ex = pd.read_parquet(cfg["excluded"])
    ex["date"] = pd.to_datetime(ex["date"])
    df = df.merge(ex[["ts_code", "date", "f_st", "f_suspend", "f_limitup", "f_any"]],
                  on=["ts_code", "date"], how="left", validate="1:1")
    assert df["f_any"].notna().all(), f"{pool}: 清洗标记未全覆盖"
    in_emb = np.zeros(len(df), bool)
    for lo, hi in EMBARGO:
        in_emb |= (df["date"] >= lo) & (df["date"] <= hi)
    df["seg"] = "drop"
    df.loc[(df["date"] >= TRAIN_LO) & (df["date"] <= TRAIN_HI), "seg"] = "train"
    df.loc[(df["date"] >= VAL_LO) & (df["date"] <= VAL_HI), "seg"] = "val"
    df.loc[in_emb, "seg"] = "embargo"
    df["_d"] = _days_int(df["date"]).astype(np.int32)
    return df


# ================================================================ 三关统计 (复刻 run_feature_screening)
def daily_rank_ic(df, feat_cols, ycol, min_n=MIN_DAY_N):
    cnt = df.groupby("date").size()
    ok_dates = cnt[cnt >= min_n].index
    sub = df.loc[df["date"].isin(ok_dates), ["date", ycol] + feat_cols]
    g = sub.groupby("date")
    ranks = g[feat_cols + [ycol]].rank()
    keyed = ranks.groupby(sub["date"].values)
    centered = ranks - keyed.transform("mean")
    cy = centered[ycol]
    C = centered[feat_cols]
    num = C.mul(cy, axis=0).groupby(sub["date"].values).sum()
    denx = C.pow(2).groupby(sub["date"].values).sum()
    deny = cy.pow(2).groupby(sub["date"].values).sum()
    ic = num.div(np.sqrt(denx.mul(deny, axis=0)))
    return ic.sort_index()


def agg_ic(ic_daily, lo, hi):
    seg = ic_daily.loc[(ic_daily.index >= lo) & (ic_daily.index <= hi)]
    out = pd.DataFrame({"ic_mean": seg.mean(), "ic_std": seg.std(), "n_days": seg.count()})
    out["icir"] = out["ic_mean"] / out["ic_std"]
    return out


def yearly_ic(ic_daily):
    df = ic_daily.copy()
    df["year"] = df.index.year
    yrs, days = {}, {}
    for y, sub in df.groupby("year"):
        sub = sub.drop(columns="year")
        yrs[y] = sub.mean()
        days[y] = sub.count()
    return pd.DataFrame(yrs), pd.DataFrame(days)


def year_consistency(yh_row, yd_row, direction):
    valid = [y for y in YEARS if pd.notna(yh_row.get(y)) and yd_row.get(y, 0) >= MIN_YEAR_DAYS]
    if not valid:
        return np.nan, 0
    agree = sum(1 for y in valid if np.sign(yh_row[y]) == direction)
    return agree / len(valid), len(valid)


def partial_rank_ic_daily(df, feat, ycol, xcol, min_n=10):
    """实验 D 口径: 逐日 rank(feat)/rank(y) 各自对 rank(x) 残差化后的相关."""
    out = {}
    for d, g in df.groupby("date"):
        v = g[[feat, ycol, xcol]].dropna()
        if len(v) < min_n:
            continue
        rf = v[feat].rank().to_numpy(np.float64)
        ry = v[ycol].rank().to_numpy(np.float64)
        rx = v[xcol].rank().to_numpy(np.float64)
        if rx.std() == 0 or rf.std() == 0:
            continue
        X = np.column_stack([np.ones(len(v)), rx])
        res_f = rf - X @ np.linalg.lstsq(X, rf, rcond=None)[0]
        res_y = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
        if res_f.std() == 0 or res_y.std() == 0:
            continue
        out[d] = float(np.corrcoef(res_f, res_y)[0, 1])
    return pd.Series(out).sort_index()


def seg_stats(series: pd.Series, lo, hi):
    seg = series.loc[(series.index >= lo) & (series.index <= hi)]
    if len(seg) < 20:
        return np.nan, np.nan, 0
    return float(seg.mean()), float(seg.mean() / seg.std()) if seg.std() > 0 else np.nan, len(seg)


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t_start = time.time()
    log("=" * 70)
    log("P2 补数据特征: 实现 + 三关筛查 启动")

    # ---------------- 阶段 1: 事件/标签/清洗 ----------------
    log("[阶段1] 事件/标签/池清洗加载")
    pools = {p: load_pool_frame(p) for p in POOLS}
    for p, df in pools.items():
        log(f"  [{p}] 事件 {len(df)}; train {(df['seg']=='train').sum()} "
            f"val {(df['seg']=='val').sum()} embargo {(df['seg']=='embargo').sum()} "
            f"test(封存,仅落值) {(df['date']>VAL_HI).sum()}; "
            f"清洗剔除(f_any) train+val: "
            f"{int(df.loc[df['seg'].isin(['train','val']),'f_any'].sum())}")

    # ---------------- 阶段 2: 面板加载 ----------------
    log("[阶段2] 轻量日线面板 (全 universe, 复权链/ret20/above_ma20/ATR14)")
    panel = load_panel_cached("light_panel", _read_one_stock, DATA_DAILY, args.workers)
    panel = panel.sort_values(["ts_code", "date"], kind="mergesort").reset_index(drop=True)
    log(f"  面板 {panel.shape}, 股票 {panel['ts_code'].nunique()}")

    log("[阶段3] daily_basic 面板 (换手率/流通股本)")
    db = load_panel_cached("daily_basic_panel", _read_one_basic, DATA_BASIC, args.workers)
    log(f"  daily_basic {db.shape}, 股票 {db['ts_code'].nunique()}")

    log("[阶段4] stk_limit 面板 (涨跌停精确价, 2007+)")
    limit = load_panel_cached("stk_limit_panel", _read_one_limit, DATA_LIMIT, args.workers)
    log(f"  stk_limit {limit.shape}")

    # ---------------- 阶段 3: 特征计算 (全历史, 因果) ----------------
    log("[阶段5] 换手/市值族 (TURN20/ABTURN/STDTURN20/FLOAT_MCAP)")
    t0 = time.time()
    db = db_stock_features(db)
    log(f"  完成 ({time.time()-t0:.0f}s)")

    log("[阶段6] 市场级特征 (RETAIL_SENT + 4 宽基指数族)")
    retail = retail_sentiment(db)
    idx_mkt = index_market_features()
    mkt_frame = idx_mkt.copy()
    mkt_frame["RETAIL_SENT"] = mkt_frame["date"].map(retail)
    log(f"  市场框 {mkt_frame.shape}; RETAIL_SENT 非空 {mkt_frame['RETAIL_SENT'].notna().sum()} 日")

    log("[阶段7] STK_LIMIT 精确版 (LIMITCNT20_X/LIMITDOWN_CNT20_X/DIST_LIMIT_X)")
    t0 = time.time()
    limit_feats = limit_stock_features(panel, limit)
    log(f"  完成 {limit_feats.shape} ({time.time()-t0:.0f}s); "
        f"LIMITCNT20_X 非空率 {limit_feats['LIMITCNT20_X'].notna().mean():.3f}")

    log("[阶段8] 申万 L1 行业族 (IND_REL_STR/IND_MOM20/IND_RS_RANK/IND_BREADTH_MA20/IND_RET5_RANK)")
    t0 = time.time()
    intervals, ind_codes, ind_names = build_l1_intervals()
    ind_agg, panel_ind = industry_day_frames(panel, intervals, len(ind_codes))
    ind_coverage = float((panel_ind >= 0).mean())
    log(f"  行业日框 {ind_agg.shape} ({time.time()-t0:.0f}s); 面板行业覆盖率 {ind_coverage:.3f}")

    log("[阶段9] ST_FLAG (namechange as-of)")
    name_index = build_name_index()

    # ---------------- 阶段 4: 事件行切片装配 ----------------
    log("[阶段10] 事件行特征装配 (双池)")
    # 个股级查询结构: (ts_code, date) -> 行位置
    panel_key = panel["ts_code"].to_numpy() + "|" + panel["date"].astype(str)
    panel_pos = pd.Series(np.arange(len(panel)), index=panel_key)
    # thr_pct = 2*ATR14[T] / open_adj[T+1] (个股自身下一根 bar, 与标签机器同口径)
    thr_map = {}
    pos_arr = panel_pos
    codes_p = panel["ts_code"].to_numpy()
    for pool, df in pools.items():
        t0 = time.time()
        key = df["ts_code"].to_numpy() + "|" + df["_d"].astype(str)
        pos = panel_pos.reindex(key).to_numpy()
        assert np.isfinite(pos).all(), f"{pool}: 有事件日不在面板内"
        pos = pos.astype(np.int64)
        nxt = pos + 1
        same = (nxt < len(panel)) & (codes_p[np.minimum(nxt, len(panel) - 1)] == df["ts_code"].to_numpy())
        atr_t = panel["atr14"].to_numpy(np.float64)[pos]
        open_n = panel["open_adj"].to_numpy(np.float64)[np.where(same, nxt, pos)]
        with np.errstate(divide="ignore", invalid="ignore"):
            thr = np.where(same & (open_n > 0), 2.0 * atr_t / open_n, np.nan)
        df["AUX_THR_PCT"] = thr

        # 换手/市值族 + 涨跌停精确版: (ts_code, date) merge
        dbs = db[["ts_code", "date"] + TURN_COLS]
        df = df.merge(dbs, left_on=["ts_code", "_d"], right_on=["ts_code", "date"],
                      how="left", suffixes=("", "_db")).drop(columns=["date_db"])
        df = df.merge(limit_feats, left_on=["ts_code", "_d"], right_on=["ts_code", "date"],
                      how="left", suffixes=("", "_lf")).drop(columns=["date_lf"])

        # 市场级: 按日期 join
        mkt = mkt_frame.set_index("date")
        for c in MKT_COLS:
            df[c] = df["_d"].map(mkt[c])

        # 行业族: as-of 行业 + 行业日框查询
        ind_q = attach_industry(df["ts_code"].to_numpy(), df["_d"].to_numpy(), intervals)
        indf = industry_features_at(ind_agg, ind_q, df["_d"].to_numpy())
        df["IND_MOM20"] = indf["IND_MOM20"].to_numpy()
        df["IND_RS_RANK"] = indf["IND_RS_RANK"].to_numpy()
        df["IND_RET5_RANK"] = indf["IND_RET5_RANK"].to_numpy()
        df["IND_BREADTH_MA20"] = indf["IND_BREADTH_MA20"].to_numpy()
        df["IND_REL_STR"] = df["RET20"].to_numpy(np.float64) - df["IND_MOM20"].to_numpy(np.float64)
        df.loc[ind_q < 0, IND_COLS] = np.nan

        # ST_FLAG
        df["ST_FLAG"] = st_flags(df["ts_code"].to_numpy(), df["_d"].to_numpy(), name_index)
        pools[pool] = df
        n_nan = df[ALL_P2].isna().mean()
        log(f"  [{pool}] 装配完成 ({time.time()-t0:.0f}s); NaN 率: "
            + ", ".join(f"{c}={n_nan[c]:.3f}" for c in ALL_P2))

    # ---------------- 阶段 5: 泄漏防线 ----------------
    log("[阶段11] 泄漏防线: 列名黑名单断言")
    fe.assert_no_blacklisted(ALL_P2 + ["AUX_THR_PCT"])
    for pool, df in pools.items():
        bad = df[ALL_P2].to_numpy(np.float64)
        assert not np.isinf(bad).any(), f"{pool}: 特征含 inf"
    log("  黑名单/inf 断言通过")

    log("[阶段12] 泄漏防线: 截断历史重算对拍")
    trunc_results = run_truncation_tests(panel, db, limit, intervals, name_index,
                                         ind_codes, mkt_frame, pools, args.workers)
    for k, v in trunc_results.items():
        log(f"  {k}: {v['summary']}")
    n_fail = sum(1 for v in trunc_results.values() if not v["pass"])
    assert n_fail == 0, f"截断对拍失败 {n_fail} 项"

    # ---------------- 阶段 6: 三关筛查 ----------------
    log("[阶段13] 三关筛查: 单变量逐日 Rank IC + 双池复现")
    results = {}
    for pool, df in pools.items():
        use = df[df["seg"].isin(["train", "val"]) & ~df["f_any"]].copy()
        # 市场级特征日内截面恒定 -> Rank IC 结构性 NaN, 只跑截面特征
        ic_hit = daily_rank_ic(use, CROSS_SECTIONAL, "hit")
        y_hit, y_days = yearly_ic(ic_hit)
        results[pool] = {
            "train": agg_ic(ic_hit, TRAIN_LO, TRAIN_HI),
            "val": agg_ic(ic_hit, VAL_LO, VAL_HI),
            "yearly": y_hit, "yearly_days": y_days,
            "n_use": len(use),
            # ST_FLAG 变体: 只去 f_suspend|f_limitup (保留 ST 事件)
            "use_nost": df[df["seg"].isin(["train", "val"])
                           & ~(df["f_suspend"] | df["f_limitup"])].copy(),
        }
        ic_nost = daily_rank_ic(results[pool]["use_nost"], ["ST_FLAG"], "hit")
        results[pool]["train_nost"] = agg_ic(ic_nost, TRAIN_LO, TRAIN_HI)
        results[pool]["val_nost"] = agg_ic(ic_nost, VAL_LO, VAL_HI)
        y_nost, yd_nost = yearly_ic(ic_nost)
        results[pool]["yearly_nost"] = y_nost
        results[pool]["yearly_days_nost"] = yd_nost
        log(f"  [{pool}] IC 完成: 清洗后样本 {len(use)} (ST_FLAG 变体 "
            f"{len(results[pool]['use_nost'])})")

    log("[阶段14] 几何剥离 (实验 D: 对 AUX_THR_PCT/ATRN 偏秩相关, 流动性族必过)")
    geo = {}
    for pool, df in pools.items():
        use = df[df["seg"].isin(["train", "val"]) & ~df["f_any"]]
        for f in TURN_COLS + LIMIT_COLS + IND_COLS:
            sub = use[["date", "hit", f, "AUX_THR_PCT", "ATRN", "seg"]].dropna(subset=[f])
            tr = sub[sub["seg"] == "train"]
            rho_thr = float(tr[f].corr(tr["AUX_THR_PCT"], method="spearman")) if len(tr) > 100 else np.nan
            pic_thr = partial_rank_ic_daily(sub, f, "hit", "AUX_THR_PCT")
            pic_atrn = partial_rank_ic_daily(sub, f, "hit", "ATRN")
            geo[(pool, f)] = {
                "rho_thr_train": rho_thr,
                "pthr_train": seg_stats(pic_thr, TRAIN_LO, TRAIN_HI),
                "pthr_val": seg_stats(pic_thr, VAL_LO, VAL_HI),
                "patrn_train": seg_stats(pic_atrn, TRAIN_LO, TRAIN_HI),
                "patrn_val": seg_stats(pic_atrn, VAL_LO, VAL_HI),
            }
        log(f"  [{pool}] 几何剥离完成")
    log("  (几何剥离逐日偏相关完成)")

    # ---------------- 阶段 7: 汇总判定 ----------------
    log("[阶段15] 汇总: 三关 + 共线性 + 定档")
    perf_rows = []
    for f in CROSS_SECTIONAL:
        r = {"feature": f}
        for pool in ["main", "backup"]:
            R = results[pool]
            for seg in ["train", "val"]:
                A = R[seg].loc[f] if f in R[seg].index else None
                r[f"{pool}_{seg}_ic"] = A["ic_mean"] if A is not None else np.nan
                r[f"{pool}_{seg}_icir"] = A["icir"] if A is not None else np.nan
                r[f"{pool}_{seg}_days"] = int(A["n_days"]) if A is not None else 0
            yh = R["yearly"].loc[f] if f in R["yearly"].index else pd.Series(dtype=float)
            yd = R["yearly_days"].loc[f] if f in R["yearly_days"].index else pd.Series(dtype=float)
            tr_ic = r[f"{pool}_train_ic"]
            d = np.sign(tr_ic) if pd.notna(tr_ic) and tr_ic != 0 else 1.0
            share, n_years = year_consistency(yh, yd, d)
            r[f"{pool}_year_share"] = share
            r[f"{pool}_n_years"] = n_years
            g = geo[(pool, f)] if (pool, f) in geo else None
            if g:
                r[f"{pool}_rho_thr"] = g["rho_thr_train"]
                r[f"{pool}_pthr_ic_val"], r[f"{pool}_pthr_icir_val"], r[f"{pool}_pthr_days_val"] = g["pthr_val"]
                r[f"{pool}_pthr_ic_train"], r[f"{pool}_pthr_icir_train"], r[f"{pool}_pthr_days_train"] = g["pthr_train"]
                r[f"{pool}_patrn_ic_val"], r[f"{pool}_patrn_icir_val"], _ = g["patrn_val"]
        # 逐年 IC (主池, 审计用; ST_FLAG 用 f_suspend|f_limitup 口径)
        y_src = results["main"]["yearly"]
        if f == "ST_FLAG":
            y_src = results["main"]["yearly_nost"]
        if f in y_src.index:
            for y in YEARS:
                r[f"y{y}"] = y_src.loc[f].get(y, np.nan)
        perf_rows.append(r)
    # ST_FLAG 变体 (f_suspend|f_limitup 口径) 作为主读数列
    for pool in ["main", "backup"]:
        R = results[pool]
        row = next(r for r in perf_rows if r["feature"] == "ST_FLAG")
        for seg in ["train", "val"]:
            A = R[f"{seg}_nost"].loc["ST_FLAG"] if "ST_FLAG" in R[f"{seg}_nost"].index else None
            row[f"{pool}_{seg}_ic_nost"] = A["ic_mean"] if A is not None else np.nan
            row[f"{pool}_{seg}_icir_nost"] = A["icir"] if A is not None else np.nan
            row[f"{pool}_{seg}_days_nost"] = int(A["n_days"]) if A is not None else 0
        yh = R["yearly_nost"].loc["ST_FLAG"] if "ST_FLAG" in R["yearly_nost"].index else pd.Series(dtype=float)
        yd = R["yearly_days_nost"].loc["ST_FLAG"] if "ST_FLAG" in R["yearly_days_nost"].index else pd.Series(dtype=float)
        tr_ic = row[f"{pool}_train_ic_nost"]
        d = np.sign(tr_ic) if pd.notna(tr_ic) and tr_ic != 0 else 1.0
        share, n_years = year_consistency(yh, yd, d)
        row[f"{pool}_year_share_nost"] = share
        row[f"{pool}_n_years_nost"] = n_years
    perf = pd.DataFrame(perf_rows)

    def gate_pass(r, pool, suffix=""):
        tr, va = r[f"{pool}_train_ic{suffix}"], r[f"{pool}_val_ic{suffix}"]
        share = r[f"{pool}_year_share{suffix}"]
        icir = r[f"{pool}_val_icir{suffix}"]
        if pd.isna(tr) or pd.isna(va):
            return False, "IC 全 NaN(截面无变差或样本不足)"
        g1 = np.sign(tr) == np.sign(va) and tr != 0
        g2 = pd.notna(share) and share >= 0.55
        g3 = pd.notna(icir) and abs(icir) >= 0.1
        fails = []
        if not g1:
            fails.append("符号相反")
        if not g2:
            fails.append(f"年度一致率{share:.2f}<0.55" if pd.notna(share) else "无有效年度")
        if not g3:
            fails.append(f"|val ICIR|={icir:.3f}<0.1" if pd.notna(icir) else "val ICIR NaN")
        return g1 and g2 and g3, ";".join(fails)

    main_pass, main_why, backup_pass, backup_why = [], [], [], []
    for _, r in perf.iterrows():
        suffix = "_nost" if r["feature"] == "ST_FLAG" else ""
        p, w = gate_pass(r, "main", suffix)
        main_pass.append(p)
        main_why.append(w)
        p2, w2 = gate_pass(r, "backup", suffix)
        backup_pass.append(p2)
        backup_why.append(w2)
    perf["main_pass"], perf["main_fail"] = main_pass, main_why
    perf["backup_pass"], perf["backup_fail"] = backup_pass, backup_why

    # 共线性 (train 主池, P2 幸存者 + 参考特征)
    survivors = perf.loc[perf["main_pass"], "feature"].tolist()
    coll_df = pd.DataFrame()
    drop_by_coll = {}
    if survivors:
        dfm = pools["main"]
        use_tr = dfm[dfm["seg"].isin(["train"]) & ~dfm["f_any"]]
        cols = survivors + REF_FEATS
        corr = use_tr[cols].corr(method="spearman")
        adj = (corr.abs() > 0.85).copy()
        adj = pd.DataFrame(np.where(np.eye(len(adj), dtype=bool), False, adj.values),
                           index=adj.index, columns=adj.columns)
        from scipy.sparse.csgraph import connected_components
        n_comp, labels = connected_components(adj.values, directed=False)
        rows = []
        for c in range(n_comp):
            members = [cols[i] for i in np.where(labels == c)[0]]
            if len(members) <= 1:
                continue
            mem_p2 = [m for m in members if m in survivors]
            non_geo = [m for m in members if m not in GEO_CONTROLS]
            cands = non_geo if non_geo else members
            vic = perf.set_index("feature")["main_val_icir"]
            keep = max(cands, key=lambda m: abs(vic.get(m, 0) or 0))
            for m in members:
                rho_max = corr.loc[m, members].drop(index=m).abs().max()
                rows.append({"feature": m, "cluster": c, "kept": m == keep or m in GEO_CONTROLS,
                             "max_abs_rho": float(rho_max),
                             "is_p2": m in survivors, "is_geo_control": m in GEO_CONTROLS})
                if m in mem_p2 and m != keep:
                    drop_by_coll[m] = f"共线性簇#{c}败给{keep}"
        coll_df = pd.DataFrame(rows)
        log(f"  共线性簇: {coll_df['cluster'].nunique() if len(coll_df) else 0} 个; "
            f"P2 被汰: {list(drop_by_coll)}")

    # 定档 (含几何剥离结论)
    decisions = []
    for _, r in perf.iterrows():
        f = r["feature"]
        rho = r.get("main_rho_thr", np.nan)
        raw_icir = abs(r["main_val_icir"]) if pd.notna(r["main_val_icir"]) else 0
        geo_note = ""
        if f == "ST_FLAG":
            geo_note = "不适用(0/1 制度哑变量, 非波动/流动性族)"
        elif pd.notna(rho):
            retained = []
            for seg in ["train", "val"]:
                raw = abs(r[f"main_{seg}_ic"])
                pt = abs(r.get(f"main_pthr_ic_{seg}", np.nan))
                if pd.notna(raw) and raw > 1e-9 and pd.notna(pt):
                    retained.append(pt / raw)
            retain_ratio = float(np.mean(retained)) if retained else np.nan
            pthr_days = r.get("main_pthr_days_train", 0) + r.get("main_pthr_days_val", 0)
            if pd.isna(retain_ratio):
                geo_note = (f"偏相关样本不足(ρ_thr={rho:.2f}, 有效日 {int(pthr_days)}, "
                            f"日内近恒定), 不作几何判定")
            elif abs(rho) >= 0.85 and retain_ratio < 0.3:
                geo_note = f"几何实锤(ρ_thr={rho:.2f},偏相关保留{retain_ratio:.0%})"
            elif retain_ratio >= 0.5:
                geo_note = f"非几何(ρ_thr={rho:.2f},偏相关保留{retain_ratio:.0%})"
            else:
                geo_note = f"部分几何/不确定(ρ_thr={rho:.2f},偏相关保留{retain_ratio:.0%})"
        is_geo_hammer = geo_note.startswith("几何实锤")
        if not r["main_pass"]:
            decisions.append(("淘汰", "主池三关未过: " + r["main_fail"], geo_note))
        elif f in drop_by_coll:
            decisions.append(("淘汰", drop_by_coll[f], geo_note))
        elif is_geo_hammer:
            decisions.append(("几何控制", "三关过但几何实锤, 降为控制变量", geo_note))
        elif r["backup_pass"]:
            decisions.append(("A", "双池双段三关皆过" + (", " + geo_note if geo_note else ""), geo_note))
        else:
            decisions.append(("B", "主池过、备池未过: " + r["backup_fail"], geo_note))
    perf["decision"] = [d[0] for d in decisions]
    perf["decision_reason"] = [d[1] for d in decisions]
    perf["geo_verdict"] = [d[2] for d in decisions]

    # ---------------- 阶段 8: 产物落盘 ----------------
    log("[阶段16] 产物落盘")
    perf.to_csv(OUT_DIR / "per_feature_ic.csv", index=False, float_format="%.5f")
    if len(coll_df):
        coll_df.to_csv(OUT_DIR / "collinearity_clusters.csv", index=False, float_format="%.5f")
    for pool, df in pools.items():
        out_cols = ["event_id", "ts_code", "date"] + ALL_P2 + ["AUX_THR_PCT"]
        out = df[out_cols].copy()
        for c in ALL_P2 + ["AUX_THR_PCT"]:
            out[c] = out[c].astype(np.float32)
        out.to_parquet(OUT_DIR / f"p2_feature_values_{pool}.parquet", index=False)
        log(f"  p2_feature_values_{pool}.parquet {out.shape}")

    with open(OUT_DIR / "run_meta.json", "w") as fh:
        json.dump({
            "pools": {p: {"n_events": int(len(d)),
                          "n_train": int((d["seg"] == "train").sum()),
                          "n_val": int((d["seg"] == "val").sum()),
                          "n_test_untouched": int((d["date"] > VAL_HI).sum()),
                          "nan_rate": {c: float(d[c].isna().mean()) for c in ALL_P2}}
                      for p, d in pools.items()},
            "industry_coverage_panel": ind_coverage,
            "truncation": trunc_results,
            "decisions": perf.set_index("feature")["decision"].to_dict(),
            "industry_names": ind_names,
        }, fh, ensure_ascii=False, indent=1, default=str)

    write_report(perf, coll_df, pools, results, trunc_results, ind_names)
    log(f"完成, 总耗时 {(time.time()-t_start)/60:.1f} 分钟")
    print(perf[["feature", "decision", "decision_reason"]].to_string())


# ================================================================ 截断对拍
def _cmp_val(a, b):
    """allclose 口径单值比对: err <= atol + rtol*|a| 时返回 0, 否则返回超出倍数."""
    err = abs(a - b)
    bound = 1e-9 + 1e-6 * abs(a)
    return 0.0 if err <= bound else err / bound


def run_truncation_tests(panel, db_full, limit, intervals, name_index, ind_codes,
                         mkt_frame, pools, workers):
    """主表 7.2-1: 截断至 T 重算 = 全历史 T 处值.

    样本 A (个股级): 3 只长历史股票 × 50 个日期, 逐对重算换手/市值/涨跌停族.
    样本 B (面板级): 12 个日期, 面板截断重算行业/市场族 (全市场聚合) + 当日事件股的个股族,
    与全历史运行在 (stock, T) 处全量比对. 容差 allclose(rtol=1e-6, atol=1e-9).
    """
    rng = np.random.default_rng(42)
    res = {}
    db_sorted = db_full  # 已按 (ts_code,date) 排序且含特征列

    # ---- 样本 A ----
    stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
    stocks = [s for s in stocks if s in set(panel["ts_code"])]
    cols_a = TURN_COLS + LIMIT_COLS
    worst_a = {c: 0.0 for c in cols_a}
    n_pairs = 0
    limit_by_stock = {c: g for c, g in limit.groupby("ts_code", sort=False)}
    db_by_stock = {c: g for c, g in db_sorted.groupby("ts_code", sort=False)}
    empty_lim = pd.DataFrame(columns=["ts_code", "date", "up_limit", "down_limit"])
    for code in stocks:
        pdb = db_by_stock.get(code)
        plim = limit_by_stock.get(code)
        ppanel = panel[panel["ts_code"] == code]
        if pdb is None or len(ppanel) < 300:
            continue
        dates_pool = ppanel["date"].to_numpy()
        lo, hi = _days_int(pd.Series([TRAIN_LO])), _days_int(pd.Series([VAL_HI]))
        cand = dates_pool[(dates_pool >= lo[0]) & (dates_pool <= hi[0])]
        pick = rng.choice(cand, size=min(50, len(cand)), replace=False)
        lf_full = limit_stock_features(ppanel, plim if plim is not None else empty_lim)
        for T in pick:
            sub_db = pdb[pdb["date"] <= T]
            rec = db_stock_features(sub_db.drop(columns=TURN_COLS, errors="ignore"))
            full_row = pdb[pdb["date"] == T]
            rec_row = rec[rec["date"] == T]
            if len(full_row) and len(rec_row):
                for c in TURN_COLS:
                    a, b = full_row[c].iloc[0], rec_row[c].iloc[0]
                    if pd.isna(a) and pd.isna(b):
                        continue
                    assert not (pd.isna(a) ^ pd.isna(b)), f"{code}@{T} {c} NaN 不一致"
                    worst_a[c] = max(worst_a[c], _cmp_val(float(a), float(b)))
                    n_pairs += 1
            sub_panel = ppanel[ppanel["date"] <= T]
            sub_lim = plim[plim["date"] <= T] if plim is not None else empty_lim
            rec_l = limit_stock_features(sub_panel, sub_lim)
            fr = lf_full[lf_full["date"] == T]
            rr = rec_l[rec_l["date"] == T]
            if len(fr) and len(rr):
                for c in LIMIT_COLS:
                    a, b = fr[c].iloc[0], rr[c].iloc[0]
                    if pd.isna(a) and pd.isna(b):
                        continue
                    assert not (pd.isna(a) ^ pd.isna(b)), f"{code}@{T} {c} NaN 不一致"
                    worst_a[c] = max(worst_a[c], _cmp_val(float(a), float(b)))
                    n_pairs += 1
    ok = all(v == 0.0 for v in worst_a.values())
    res["样本A_个股级截断"] = {
        "pass": ok, "n_pairs": n_pairs,
        "worst": {k: float(v) for k, v in worst_a.items()},
        "summary": f"{len(stocks)} 股 × 50 日, 比对 {n_pairs} 值, "
                   f"容差超出倍数峰值 {max(worst_a.values()):.2e} -> {'通过' if ok else '失败'}",
    }

    # ---- 样本 B ----
    ev_dates = np.sort(pd.concat([pools["main"]["_d"], pools["backup"]["_d"]]).unique())
    lo2002 = _days_int(pd.Series([pd.Timestamp("2002-01-01")]))[0]
    hi_val = _days_int(pd.Series([VAL_HI]))[0]
    ev_dates = ev_dates[(ev_dates >= lo2002) & (ev_dates <= hi_val)]
    picks = rng.choice(ev_dates, size=12, replace=False)
    cols_b = TURN_COLS + LIMIT_COLS + IND_COLS + MKT_COLS + ST_COLS
    worst_b = {c: 0.0 for c in cols_b}
    n_cmp = 0

    def upd(c, a, b):
        nonlocal n_cmp
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        assert not (np.isnan(a) ^ np.isnan(b)).any(), f"{c} NaN 模式不一致"
        both = ~np.isnan(a)
        if both.any():
            w = max(_cmp_val(float(x), float(y)) for x, y in zip(a[both], b[both]))
            worst_b[c] = max(worst_b[c], w)
            n_cmp += int(both.sum())

    for T in picks:
        ev_T = {p: df[df["_d"] == T] for p, df in pools.items()}
        ev_codes = set()
        for e in ev_T.values():
            ev_codes.update(e["ts_code"])
        if not ev_codes:
            continue
        sub_panel = panel[panel["date"] <= T]
        sub_db_ev = db_full[(db_full["date"] <= T) & (db_full["ts_code"].isin(ev_codes))]
        sub_lim_ev = limit[(limit["date"] <= T) & (limit["ts_code"].isin(ev_codes))]
        # 个股族 (逐股独立, 子集重算 == 全量重算)
        rec_db = db_stock_features(sub_db_ev.drop(columns=TURN_COLS, errors="ignore"))
        sub_panel_ev = sub_panel[sub_panel["ts_code"].isin(ev_codes)]
        rec_lf = limit_stock_features(sub_panel_ev, sub_lim_ev)
        # 市场族 (全市场聚合, 必须全量截断)
        rec_retail = retail_sentiment(db_full[db_full["date"] <= T])
        rec_mkt = index_market_features()
        rec_mkt = rec_mkt[rec_mkt["date"] <= T]
        # 行业族 (全市场聚合)
        rec_agg, _ = industry_day_frames(sub_panel, intervals, len(ind_codes))
        for pool, e in ev_T.items():
            if len(e) == 0:
                continue
            m = e.merge(rec_db[["ts_code", "date"] + TURN_COLS],
                        left_on=["ts_code", "_d"], right_on=["ts_code", "date"], how="left",
                        suffixes=("", "_rec"))
            for c in TURN_COLS:
                upd(c, e[c].to_numpy(), m[c + "_rec"].to_numpy())
            m2 = e.merge(rec_lf, left_on=["ts_code", "_d"], right_on=["ts_code", "date"],
                         how="left", suffixes=("", "_rec"))
            for c in LIMIT_COLS:
                upd(c, e[c].to_numpy(), m2[c + "_rec"].to_numpy())
            ind_q = attach_industry(e["ts_code"].to_numpy(), e["_d"].to_numpy(), intervals)
            indf = industry_features_at(rec_agg, ind_q, e["_d"].to_numpy())
            rec_rel = e["RET20"].to_numpy(np.float64) - indf["IND_MOM20"].to_numpy(np.float64)
            upd("IND_MOM20", e["IND_MOM20"].to_numpy(), indf["IND_MOM20"].to_numpy())
            upd("IND_RS_RANK", e["IND_RS_RANK"].to_numpy(), indf["IND_RS_RANK"].to_numpy())
            upd("IND_RET5_RANK", e["IND_RET5_RANK"].to_numpy(), indf["IND_RET5_RANK"].to_numpy())
            upd("IND_BREADTH_MA20", e["IND_BREADTH_MA20"].to_numpy(),
                indf["IND_BREADTH_MA20"].to_numpy())
            upd("IND_REL_STR", e["IND_REL_STR"].to_numpy(),
                np.where(ind_q < 0, np.nan, rec_rel))
            for c in MKT_COLS:
                if c == "RETAIL_SENT":
                    b = np.full(len(e), rec_retail.get(T, np.nan))
                else:
                    row = rec_mkt[rec_mkt["date"] == T]
                    b = np.full(len(e), row[c].iloc[0] if len(row) else np.nan)
                upd(c, e[c].to_numpy(), b)
            b = st_flags(e["ts_code"].to_numpy(), e["_d"].to_numpy(), name_index)
            upd("ST_FLAG", e["ST_FLAG"].to_numpy(), b)
    ok = all(v == 0.0 for v in worst_b.values())
    res["样本B_面板级截断"] = {
        "pass": ok, "n_pairs": n_cmp, "n_dates": 12,
        "worst": {k: float(v) for k, v in worst_b.items()},
        "summary": f"12 个截断日 × 当日全部事件, 比对 {n_cmp} 值, "
                   f"容差超出倍数峰值 {max(worst_b.values()):.2e} -> {'通过' if ok else '失败'}",
    }
    return res


# ================================================================ 报告
def _cell(x) -> str:
    """markdown 表格单元转义."""
    s = str(x)
    return s.replace("|", "\\|").replace("\n", "<br>")


def write_report(perf, coll_df, pools, results, trunc_results, ind_names):
    SPEC15 = [
        ("1", "TURN20", "换手率20日均值", "Mean(turnover_rate,20)", "流动性"),
        ("2", "ABTURN", "换手异动(5日/60日)", "Mean(turn,5)/Mean(turn,60)", "流动性"),
        ("3", "STDTURN20", "换手率20日标准差", "Std(turnover_rate,20)", "流动性"),
        ("4", "FLOAT_MCAP", "流通市值对数", "ln(close×float_share)", "流动性/规模"),
        ("5", "RETAIL_SENT", "全市场换手情绪", "全市场换手率加总20日均值的250日分位", "市场情绪(市场级)"),
        ("6", "IDEAL_REV", "理想反转", "20日按单笔金额(成交额/成交笔数)分组切割的反转差", "反转"),
        ("7", "IND_REL_STR", "行业相对强度", "RET20_个股 − RET20_申万一级行业(等权)", "行业"),
        ("8", "IND_MOM20", "行业20日动量", "申万一级行业等权指数20日收益", "行业"),
        ("9", "IND_RS_RANK", "行业强度横截面分位", "行业20日收益在31个一级行业中的当日分位", "行业"),
        ("10", "IND_BREADTH_MA20", "行业站上20日线占比", "行业内复权收盘>20日均线成分占比", "行业"),
        ("11", "IND_RET5_RANK", "行业5日动量横截面分位", "行业5日收益当日分位", "行业"),
        ("12", "ST_FLAG", "ST状态标记", "namechange as-of 重建T时点名称含ST", "制度"),
        ("13", "MKT_CSI300族", "宽基指数市场族(12列)", "沪深300/中证500/中证1000/创业板指 × (RET20/MA60乖离/20日波动250日分位)", "市场状态(市场级)"),
        ("14", "STK_LIMIT精确版", "精确涨跌停族(3列)", "stk_limit精确价: 涨停/跌停20日次数 + 距涨停幅度", "制度"),
        ("15", "LINK_MOM", "概念股联动动量", "同概念/共提及股票组合20日平均收益", "联动"),
    ]
    COL2SPEC = {"TURN20": "1", "ABTURN": "2", "STDTURN20": "3", "FLOAT_MCAP": "4",
                "RETAIL_SENT": "5", "IND_REL_STR": "7", "IND_MOM20": "8", "IND_RS_RANK": "9",
                "IND_BREADTH_MA20": "10", "IND_RET5_RANK": "11", "ST_FLAG": "12",
                "LIMITCNT20_X": "14", "LIMITDOWN_CNT20_X": "14", "DIST_LIMIT_X": "14"}
    perf_idx = perf.set_index("feature")

    lines = []
    lines.append("# P2 补数据特征：实现与三关筛查报告")
    lines.append("")
    lines.append("> 脚本：`v3_pipeline/scripts/run_p2_features.py`（可复跑，面板缓存于 `_cache/`）。")
    lines.append("> 口径：特征主表 §4（P2 15 条）+ §7 泄漏防线；三关与几何剥离按 feature_screening/review_report.md 实验 D。")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}。")
    lines.append("")
    lines.append("## 0. 数据与纪律")
    lines.append("")
    lines.append("- 数据源：`daily_basic/`（换手率/流通股本，8715 日全量）、`stk_limit/`（2007-01-04 起，北交所无覆盖）、"
                 "`index/`（4 宽基）、`meta/namechange.parquet`（ST 历史）、`meta/sw_index_classify+member`（申万 2021 版一级行业，含 in/out 历史）。")
    lines.append("- **禁用**：`ths_index/ths_member`（仅当前快照=前瞻性）与 `stock_basic_industry`（当前快照，无历史）——未用于任何历史特征。")
    lines.append("- 切分：train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31（含两段隔离带剔除）；2022-11 后测试段封存，只落特征值不入统计。")
    lines.append("- 池清洗：统计样本 = 事件 ∩ `f_any==False`（f_st|f_suspend|f_limitup，来自 pool_cleaning）。"
                 "ST_FLAG 例外：f_st 会抽干其截面变差，其主读数用 `f_suspend|f_limitup` 口径（保留 ST 事件）。")
    lines.append("- 主池 m_fractal15_full（标签 w_fractal_o15_s20）、备池 m_zigzag05_nofilter（标签 w_zigzag_p05_s5），标签 hit_N20_k2.0。")
    lines.append("- 全部特征只用于信号日 T 收盘后可得数据；行业归属按 T 时点 in_date≤T<out_date as-of 重建；"
                 "申万只覆盖其评级股票（面板行业覆盖率见 run_meta），未覆盖股特征为 NaN。")
    lines.append("")
    lines.append("### 泄漏防线：截断历史重算对拍（主表 7.2-1）")
    lines.append("")
    for k, v in trunc_results.items():
        lines.append(f"- **{k}**：{v['summary']}")
    lines.append("- 列名黑名单 14 条正则断言通过；全部输出列无 inf。")
    lines.append("")
    lines.append("## 1. 15 条逐个判定总表")
    lines.append("")
    lines.append("| # | 规范名 | 中文全名 | 判定 | 关键读数 |")
    lines.append("|---|---|---|---|---|")
    for num, spec, cn, formula, fam in SPEC15:
        cols = [c for c in perf_idx.index if COL2SPEC.get(c) == num]
        if num == "6":
            verdict, detail = "不可算", "个股级成交笔数不可得（daily/daily_basic/stk_factor 均无；daily_info 仅交易所级汇总）——主表标注，实测确认"
        elif num == "15":
            verdict, detail = "不可算", "概念板块历史成分不可得：TS 概念无成分接口；同花顺 ths_member 仅当前快照（前瞻性禁用）"
        elif num == "5":
            v = perf_idx  # market-level
            detail = "市场级特征（日内截面恒定），逐日 Rank IC 结构性 NaN，三关不适用（同 MKT_SH_VOLUME_SIGNAL 先例，应走时序口径评估）；已落值供建模层使用"
            verdict = "市场级-结构性N/A"
        elif num == "13":
            detail = "12 列（4 指数×3）全为市场级，日内截面恒定，逐日 Rank IC 结构性 NaN；已落值"
            verdict = "市场级-结构性N/A"
        else:
            vs = [perf_idx.loc[c, "decision"] for c in cols]
            order = {"A": 0, "B": 1, "几何控制": 2, "淘汰": 3}
            verdict = sorted(vs, key=lambda x: order[x])[0] if len(set(vs)) == 1 else "/".join(
                f"{c}:{v}" for c, v in zip(cols, vs))
            bits = []
            for c in cols:
                r = perf_idx.loc[c]
                suffix = "_nost" if c == "ST_FLAG" else ""
                sfx_note = "（f_suspend|f_limitup 口径）" if c == "ST_FLAG" else ""
                bits.append(f"{c}{sfx_note}: 主train/val IC {r[f'main_train_ic{suffix}']:+.3f}/{r[f'main_val_ic{suffix}']:+.3f}"
                            f"(ICIR {r[f'main_val_icir{suffix}']:+.2f}), 备val {r[f'backup_val_ic{suffix}']:+.3f}"
                            f"(ICIR {r[f'backup_val_icir{suffix}']:+.2f})"
                            + (f", {r['geo_verdict']}" if r['geo_verdict'] else ""))
            detail = "<br>".join(bits)
        lines.append(f"| {num} | `{spec}` | {cn} | **{verdict}** | {_cell(detail)} |")
    lines.append("")
    lines.append("## 2. 三关明细（截面特征，配置为行）")
    lines.append("")
    lines.append("主口径：逐日 Rank IC（label=hit_N20_k2.0，清洗后样本）。"
                 "g1=train/val 符号一致；g2=年度方向一致率≥0.55（有效年≥10 个 IC 日）；g3=|val ICIR|≥0.1。")
    lines.append("")
    hdr = ["特征", "主train IC", "主val IC", "主val ICIR", "主年一致率", "备train IC", "备val IC", "备val ICIR", "备年一致率", "主关", "备关", "判定"]
    lines.append("|" + "|".join(hdr) + "|")
    lines.append("|" + "---|" * len(hdr))
    for _, r in perf.iterrows():
        suffix = "_nost" if r["feature"] == "ST_FLAG" else ""
        note = "（f_suspend\\|f_limitup 口径）" if r["feature"] == "ST_FLAG" else ""
        lines.append(
            f"| `{r['feature']}`{note} | {r[f'main_train_ic{suffix}']:+.4f} | {r[f'main_val_ic{suffix}']:+.4f} "
            f"| {r[f'main_val_icir{suffix}']:+.3f} | {r[f'main_year_share{suffix}']:.2f} "
            f"| {r[f'backup_train_ic{suffix}']:+.4f} | {r[f'backup_val_ic{suffix}']:+.4f} "
            f"| {r[f'backup_val_icir{suffix}']:+.3f} | {r[f'backup_year_share{suffix}']:.2f} "
            f"| {'过' if r['main_pass'] else _cell(r['main_fail'])} | {'过' if r['backup_pass'] else _cell(r['backup_fail'])} "
            f"| **{r['decision']}** |")
    lines.append("")
    backup_only = perf.loc[~perf["main_pass"] & perf["backup_pass"], "feature"].tolist()
    if backup_only:
        lines.append(f"**备池单向幸存（主池未过、备池三关全过，按主池优先原则淘汰，存档备查）**："
                     f"{', '.join(f'`{f}`' for f in backup_only)}。")
        lines.append("")
    lines.append("## 3. 几何剥离（实验 D 口径，波动/流动性族必过）")
    lines.append("")
    lines.append("逐日偏秩相关：rank(特征) 与 rank(hit) 各自对 rank(thr_pct) 残差化后的相关；"
                 "thr_pct = 2×ATR14[T]/复权open[T+1]（个股自身下一根 bar，与标签机器同口径）。")
    lines.append("ρ_thr = train 段 pooled spearman(特征, thr_pct)。判定：\\|ρ\\|≥0.85 且偏相关保留<30% → 几何实锤；保留≥50% → 非几何。")
    lines.append("保留率可超过 100%：控制几何门槛后偏相关绝对值大于原始 IC（几何成分与原信号方向相反）。")
    lines.append("")
    hdr = ["特征", "ρ_thr(主train)", "主train偏IC", "主val偏IC", "主val偏ICIR", "备val偏IC", "备val偏ICIR", "几何判定"]
    lines.append("|" + "|".join(hdr) + "|")
    lines.append("|" + "---|" * len(hdr))
    def _f3(x):
        return f"{x:+.4f}" if pd.notna(x) else "—"
    def _f2(x):
        return f"{x:+.3f}" if pd.notna(x) else "—"
    for _, r in perf.iterrows():
        if pd.isna(r.get("main_rho_thr")):
            continue
        lines.append(
            f"| `{r['feature']}` | {_f2(r['main_rho_thr'])} | {_f3(r.get('main_pthr_ic_train'))} "
            f"| {_f3(r.get('main_pthr_ic_val'))} | {_f2(r.get('main_pthr_icir_val'))} "
            f"| {_f3(r.get('backup_pthr_ic_val'))} | {_f2(r.get('backup_pthr_icir_val'))} "
            f"| {_cell(r['geo_verdict'])} |")
    lines.append("")
    if len(coll_df):
        lines.append("## 4. 共线性簇（train 主池 spearman \\|ρ\\|>0.85，含 P0 参考特征）")
        lines.append("")
        lines.append("| 簇 | 特征 | max\\|ρ\\| | 保留 | 备注 |")
        lines.append("|---|---|---|---|---|")
        for _, r in coll_df.iterrows():
            note = ("P0 几何控制" if r["is_geo_control"] else ("P2" if r["is_p2"] else "P0/P1 参考"))
            lines.append(f"| {r['cluster']} | `{r['feature']}` | {r['max_abs_rho']:.3f} "
                         f"| {'是' if r['kept'] else '否'} | {note} |")
        lines.append("")
        p2_rows = coll_df[coll_df["is_p2"]]
        for _, r in p2_rows.iterrows():
            mates = coll_df[(coll_df["cluster"] == r["cluster"]) & ~coll_df["is_p2"]
                            & ~coll_df["is_geo_control"]]["feature"].tolist()
            if mates:
                lines.append(f"- 注意：`{r['feature']}` 与 P0/P1 已有特征 "
                             f"{', '.join(f'`{m}`' for m in mates)} 同簇"
                             f"（max\\|ρ\\|={r['max_abs_rho']:.2f}），精确版相对近似版的增量信息有限。")
        lines.append("")
    lines.append("## 5. 产物")
    lines.append("")
    lines.append("- `p2_feature_values_{main,backup}.parquet`：事件行 P2 特征值（含 AUX_THR_PCT 几何辅助列，非特征）。")
    lines.append("- `per_feature_ic.csv`：逐特征双池双段 IC/ICIR/年度一致率/几何剥离数值/判定。")
    lines.append("- `collinearity_clusters.csv`：共线性簇明细。")
    lines.append("- `run_meta.json`：计数与截断对拍结果；`progress.log`：阶段日志+心跳。")
    lines.append("")
    lines.append("## 6. 数据缺陷与口径备注")
    lines.append("")
    lines.append("- `stk_limit` 2007 年以前无数据：精确涨跌停族在 2007-01-04 前为 NaN（有效年度 2007 起 16 年），北交所（.BJ）全期无覆盖为 NaN。")
    lines.append("- 申万一级行业 as-of 重建：重叠区间（168 只股票存在重叠）取 in_date 最新者，过期不回退；未评级股票为 NaN。")
    lines.append("- `daily_basic` 早年（1991-1995）少量缺行按 NaN 处理；2016-06-30 等个别日期小批量缺行（接口缺报）同。")
    lines.append("- ST_FLAG 的 as-of 规则与 pool_cleaning f_st 完全同口径（已通过 ±5% 涨跌幅交叉验证）。")
    lines.append("- IND_MOM20 与 IND_RS_RANK 的逐日 Rank IC 完全相同（日内截面单调变换），建模时二选一；IND_REL_STR = 个股 RET20 − 行业 RET20，与 RET20 族存在构造性相关。")
    lines.append("- LIMITCNT20_X 年度一致率 1.00 仅覆盖 4 个有效年（2016/2020/2021/2022；2015 年 8 个 IC 日未达 10 日下限且为正），主池早期日截面过薄时分辨力有限（复核报告 §3-4 同源风险）。")
    lines.append("- DIST_LIMIT_X 对长期停牌复牌首日（无涨跌幅限制但接口挂公式值）会产出大比值，已截断至 ±1.5。")
    (OUT_DIR / "p2_features_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"  报告 -> {OUT_DIR / 'p2_features_report.md'}")


if __name__ == "__main__":
    main()
