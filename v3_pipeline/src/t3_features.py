#!/usr/bin/env python3
"""v5-T4 新特征计算库（issue #24，候选清单 = t3_candidate_preregistration.md 30 条）。

命名口径：T3 指候选清单所属战役步骤（issue #23 调研），本库实现该清单，
文件与列族以 t3_/T3_ 标记；T4 指本实现入池票据（issue #24），报告以 T4 标记。

输入面板（长表，行 = 个股交易日）：
  ts_code, date, open, high, low, close, pre_close, pct_chg, amount   (stock_data/daily)
  turnover_rate_f, volume_ratio, pb, pe_ttm, dv_ttm, free_share, circ_mv  (daily_basic)
  up_limit, down_limit                                                   (stk_limit)
静态上下文 ctx（截断不变量，全部自带历史生效区间，T 时点可重建）：
  calendar       交易日历（daily_basic 文件名并集，非面板派生）
  list_date      universe 上市日期
  st_intervals   namechange 名称区间 -> 风险警示状态
  ind_intervals  sw_index_member(L1, SW2021) 成分区间 -> 申万一级行业
  idx_close      000300/000852/399006 收盘序列
  mkt_ret        市场日收益（2002-01-04 起 000300.pct_chg，此前 daily/000001.SH
                 旧 schema 升序重排后 close 自差分；拼接点按清单 #25 约定）

纪律（与清单 §0 一致）：
  - 全部算子为 rolling/shift/cumprod 回看形式或 T 时点横截面聚合；
    截断 [T-1600 自然日, T]（≈1060 交易日，覆盖最大 750 交易日滚动窗）重算与
    全历史值逐位一致，由 prefix_recompute_at 对拍保证。
  - 交易日历、ST/行业区间、指数序列经 ctx 传入，与面板截断解耦。
  - 1996-12-16 前涨跌停类特征 NaN；2007-01-04 起用 stk_limit 精确价
    （容差 0.005 元），之前用 pct_chg 阈值近似（主板 10%、ST/*ST 5%，
    容差 limit_pct-0.5 个百分点；近似段无 300/688/北交所标的，阈值仅此两档）。
  - 上市未满 5 个交易日的个股：涨跌停标记 NaN（IPO 初期非标准制度），
    且不进入任何横截面分母（§0 时点纪律）。
  - 复权口径：跨日价格比较一律用 pct_chg 链 CF = cumprod(1+pct_chg/100)；
    单日截面比率用原始价。CF 只在同股同窗内做比较，整体归一化无关。
  - 已实锤口径事实（T3 评审记录）：pb/pe_ttm 从不存负值（负值股=NaN）；
    dv_ttm 无分红股为 NaN 而非 0，显式 NaN->0 映射且"无分红/缺失"不可分。
  - min_periods 口径（清单未写死者，在此登记）：滚动 20/21 窗 >=15，
    252 窗 >=126，500 窗 >=250，750 窗 >=250，250 滚动矩 >=126，
    60 日残差窗 >=45 有效日，5 日平滑 >=3，20 日动量窗 NaN 天数 >5 记 NaN。
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STOCK_DATA = REPO / "stock_data"
INDEX_CODES = ("000001.SH", "399001.SZ")  # daily/ 下的指数伪股文件

EXACT_LIMIT_START = pd.Timestamp("2007-01-04")   # stk_limit 覆盖起点
LIMIT_RULE_START = pd.Timestamp("1996-12-16")    # ±10% 涨跌停恢复日
HS300_RET_START = pd.Timestamp("2002-01-04")     # 000300 日收益起点
PRICE_TOL = 0.005                                # 精确段价格容差（元）
APPROX_TOL_PP = 0.5                              # 近似段阈值容差（百分点）
MIN_LIST_DAYS = 5                                # 新股横截面/涨跌停剔除窗口
LOOKBACK_DAYS = 1600                             # 截断对拍窗口（自然日）

# 特征列（入快照、入主表）；口径标记列同样数值化、同为 T 时点可得因果量
T3_COLUMNS = [
    # 换手/流动性/规模/估值
    "TURN_F20", "STR20", "ABN_TURN_21_252", "VOLUME_RATIO_T", "VOLUME_RATIO_MA5",
    "AMIHUD_TURN20", "LN_FREE_MV", "BP_IND_Z", "EP_TSRANK_500", "DV_TTM",
    # 涨跌停制度（个股级）
    "LIMITUP_SEALED_EXACT", "LIMITUP_SEALED_SRC", "ONEWORD_LIMITUP",
    "TOUCH_LIMITUP_FAIL", "TOUCH_FAIL_DEPTH", "CONSEC_LIMITUP", "CONSEC_SUSP_GAP",
    "DOWNLIMIT_UNSEALED", "DOWNTOUCH_RECOVER",
    # 市场情绪/宽度（市场级，事件日快照取值）
    "MKT_SEAL_RATIO", "MKT_PROMOTE_RATE", "MKT_DOWNLIMIT_Z",
    "MKT_LIMITUP_PREM", "MKT_LIMITUP_PREM_MA5",
    "MKT_NH_NL_DIFF", "MKT_NH_NL_RATIO",
    "MKT_TURNOVER_PCTL", "MKT_TURNOVER_PCTL_EQW",
    # 行业/风格/制度
    "IND_MOM20_EQW", "IND_EXCESS_RET20", "IND_BREADTH_MA60", "IND_CAPIT_PCT120",
    "RESID_MOM60", "RESID_MOM60_MKSRC", "RESID_MOM60_INDNA",
    "STYLE_SIZE_RS60", "STYLE_GV_RS60",
    "ST_STATUS", "PAR_VALUE_GAP", "DAYS_BELOW_PAR", "LIST_AGE",
]

# 中文全名词典（命名用全称纪律）
T3_CN = {
    "TURN_F20": "自由流通换手率二十日均值",
    "STR20": "量稳换手率（换手二十日波动）",
    "ABN_TURN_21_252": "异常换手率（短期对长期中枢比值）",
    "VOLUME_RATIO_T": "交易所口径量比当日值",
    "VOLUME_RATIO_MA5": "交易所口径量比五日均值",
    "AMIHUD_TURN20": "换手口径 Amihud 非流动性二十日均值",
    "LN_FREE_MV": "自由流通市值对数",
    "BP_IND_Z": "账面市值比行业内标准化值",
    "EP_TSRANK_500": "盈利收益率五百日历史分位",
    "DV_TTM": "股息率滚动十二个月值",
    "LIMITUP_SEALED_EXACT": "精确收盘封涨停标记",
    "LIMITUP_SEALED_SRC": "封涨停判定口径来源（2=精确价 1=阈值近似 0=缺失）",
    "ONEWORD_LIMITUP": "一字涨停标记",
    "TOUCH_LIMITUP_FAIL": "触涨停未封标记（炸板）",
    "TOUCH_FAIL_DEPTH": "炸板回落深度（涨停价与收盘价相对差）",
    "CONSEC_LIMITUP": "个股连续涨停天数",
    "CONSEC_SUSP_GAP": "连板跨越停牌缺口标记",
    "DOWNLIMIT_UNSEALED": "触跌停未封标记（撬板）",
    "DOWNTOUCH_RECOVER": "撬板回收幅度（收盘价与跌停价相对差）",
    "MKT_SEAL_RATIO": "市场封板率",
    "MKT_PROMOTE_RATE": "市场连板晋级率",
    "MKT_DOWNLIMIT_Z": "市场跌停家数滚动标准化恐慌度",
    "MKT_LIMITUP_PREM": "昨日涨停股当日平均溢价",
    "MKT_LIMITUP_PREM_MA5": "昨日涨停股当日平均溢价五日均值",
    "MKT_NH_NL_DIFF": "市场二百五十日新高新低家数差占比",
    "MKT_NH_NL_RATIO": "市场二百五十日新高占新高新低家数比",
    "MKT_TURNOVER_PCTL": "全市场换手率七百五十日滚动分位（成交额加权）",
    "MKT_TURNOVER_PCTL_EQW": "全市场换手率七百五十日滚动分位（等权对照）",
    "IND_MOM20_EQW": "申万一级行业二十日等权动量",
    "IND_EXCESS_RET20": "个股二十日收益对行业超额",
    "IND_BREADTH_MA60": "行业内六十日均线上方个股占比",
    "IND_CAPIT_PCT120": "行业内一百二十日新低附近个股占比",
    "RESID_MOM60": "剔除市场与行业后的六十日残差动量（跳过近五日）",
    "RESID_MOM60_MKSRC": "残差动量市场收益口径拼接标记（2=纯沪深300 1=混合 0=纯上证综指）",
    "RESID_MOM60_INDNA": "残差动量窗口内行业收益缺失退化标记",
    "STYLE_SIZE_RS60": "大小盘风格六十日相对强弱",
    "STYLE_GV_RS60": "成长价值风格六十日相对强弱",
    "ST_STATUS": "风险警示状态（3=退市整理期 2=*ST 1=ST 0=正常）",
    "PAR_VALUE_GAP": "面值退市距离（原始收盘价对数）",
    "DAYS_BELOW_PAR": "连续低于面值天数",
    "LIST_AGE": "上市时长对数",
}

DAILY_COLS = ["trade_date", "open", "high", "low", "close", "pre_close",
              "pct_chg", "amount"]
BASIC_COLS = ["ts_code", "trade_date", "turnover_rate_f", "volume_ratio", "pb",
              "pe_ttm", "dv_ttm", "free_share", "circ_mv"]
LIMIT_COLS = ["ts_code", "trade_date", "up_limit", "down_limit"]


# ---------------------------------------------------------------- 数据加载

def _read_daily(path):
    df = pd.read_parquet(path, columns=DAILY_COLS)
    df["ts_code"] = path.stem
    return df


def _read_basic(path):
    return pd.read_parquet(path, columns=BASIC_COLS)


def _read_limit(path):
    return pd.read_parquet(path, columns=LIMIT_COLS)


def _read_many(paths, reader, workers):
    if workers and workers > 1 and len(paths) > 64:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(reader, paths, chunksize=16))
    else:
        parts = [reader(p) for p in paths]
    return pd.concat(parts, ignore_index=True)


def _to_date(s):
    """trade_date 兼容：datetime 直转，字符串按 yyyymmdd 解析。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    return pd.to_datetime(s, format="%Y%m%d")


def build_panel(data_root=STOCK_DATA, date_lo=None, date_hi=None, workers=0):
    """读日线+daily_basic+stk_limit 装配长表面板（float32 降精度省内存）。

    date_lo/date_hi: 仅保留该区间（截断重算用）；None 为全历史。
    面板只含有 bar 的交易日（停牌日天然缺席）。
    """
    data_root = Path(data_root)
    daily_paths = [p for p in sorted((data_root / "daily").glob("*.parquet"))
                   if p.stem not in INDEX_CODES]

    def _in_range(stem):
        d = pd.Timestamp(stem)
        return (date_lo is None or d >= pd.Timestamp(date_lo)) and \
               (date_hi is None or d <= pd.Timestamp(date_hi))

    basic_paths = [p for p in sorted((data_root / "daily_basic").glob("*.parquet"))
                   if _in_range(p.stem)]
    limit_paths = [p for p in sorted((data_root / "stk_limit").glob("*.parquet"))
                   if _in_range(p.stem)]

    panel = _read_many(daily_paths, _read_daily, workers)
    panel["date"] = _to_date(panel["trade_date"])
    panel = panel.drop(columns=["trade_date"])
    if date_lo is not None:
        panel = panel[panel["date"] >= pd.Timestamp(date_lo)]
    if date_hi is not None:
        panel = panel[panel["date"] <= pd.Timestamp(date_hi)]

    basic = _read_many(basic_paths, _read_basic, workers)
    basic["date"] = _to_date(basic["trade_date"])
    basic = basic.drop(columns=["trade_date"])
    limit = _read_many(limit_paths, _read_limit, workers)
    limit["date"] = _to_date(limit["trade_date"])
    limit = limit.drop(columns=["trade_date"])

    panel = panel.merge(basic, on=["ts_code", "date"], how="left", validate="m:1")
    panel = panel.merge(limit, on=["ts_code", "date"], how="left", validate="m:1")
    num_cols = [c for c in panel.columns if c not in ("ts_code", "date")]
    panel[num_cols] = panel[num_cols].astype(np.float32)
    panel = panel.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return panel


def _parse_ymd(s):
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def _st_of_name(name):
    """名称 -> 风险警示状态：3=退市整理期（含"退"）2=*ST 1=ST 0=正常。"""
    if not isinstance(name, str):
        return 0
    n = name.upper().replace("＊", "*").replace(" ", "")
    if "退" in n:
        return 3
    if "*ST" in n:
        return 2
    if "ST" in n:
        return 1
    return 0


def build_ctx(data_root=STOCK_DATA):
    """静态上下文：交易日历、上市日期、ST 区间、行业区间、指数序列、市场收益。

    日历取 daily_basic 文件名并集（= 全交易日），与面板截断解耦；
    所有区间表均为历史生效区间记录，T 时点归属可无前视重建。
    """
    data_root = Path(data_root)
    calendar = np.array(sorted(
        pd.Timestamp(p.stem) for p in (data_root / "daily_basic").glob("*.parquet")),
        dtype="datetime64[ns]")

    uni = pd.read_parquet(data_root / "universe" / "universe_latest.parquet")
    list_date = dict(zip(uni["ts_code"], pd.to_datetime(uni["list_date"])))

    nm = pd.read_parquet(data_root / "meta" / "namechange.parquet").copy()
    nm["start"] = _parse_ymd(nm["start_date"])
    nm["end"] = _parse_ymd(nm["end_date"])
    nm["status"] = [_st_of_name(x) for x in nm["name"]]
    st_intervals = {c: g[["start", "end", "status"]]
                    .sort_values("start").reset_index(drop=True)
                    for c, g in nm.groupby("ts_code")}

    cls = pd.read_parquet(data_root / "meta" / "sw_index_classify.parquet")
    l1 = set(cls.loc[(cls["level"] == "L1") & (cls["src"] == "SW2021"),
                     "index_code"])
    mem = pd.read_parquet(data_root / "meta" / "sw_index_member.parquet")
    mem = mem[mem["index_code"].isin(l1)].copy()
    mem["start"] = _parse_ymd(mem["in_date"])
    mem["end"] = _parse_ymd(mem["out_date"])
    ind_intervals = {c: g[["start", "end", "index_code"]]
                     .sort_values("start").reset_index(drop=True)
                     for c, g in mem.groupby("con_code")}

    idx_close = {}
    for code in ("000300.SH", "000852.SH", "399006.SZ"):
        df = pd.read_parquet(data_root / "index" / f"{code}.parquet",
                             columns=["trade_date", "close"])
        idx_close[code] = pd.Series(
            df["close"].to_numpy(),
            index=pd.to_datetime(df["trade_date"], format="%Y%m%d")).sort_index()

    hs300 = pd.read_parquet(data_root / "index" / "000300.SH.parquet",
                            columns=["trade_date", "pct_chg"])
    hs300_ret = pd.Series(hs300["pct_chg"].to_numpy(),
                          index=pd.to_datetime(hs300["trade_date"],
                                               format="%Y%m%d")).sort_index()
    # 上证综指（daily/ 旧 schema：字符串日期、无 pct_chg，升序重排后 close 自差分）
    sh = pd.read_parquet(data_root / "daily" / "000001.SH.parquet",
                         columns=["trade_date", "close"])
    sh["trade_date"] = pd.to_datetime(sh["trade_date"], format="%Y%m%d")
    sh = sh.sort_values("trade_date")
    sh_ret = pd.Series(sh["close"].pct_change().to_numpy() * 100.0,
                       index=sh["trade_date"])
    mkt_ret = pd.concat([sh_ret[sh_ret.index < HS300_RET_START],
                         hs300_ret[hs300_ret.index >= HS300_RET_START]])
    mkt_ret = mkt_ret[~mkt_ret.index.duplicated(keep="last")].sort_index()

    return {"calendar": calendar, "list_date": list_date,
            "st_intervals": st_intervals, "ind_intervals": ind_intervals,
            "idx_close": idx_close, "mkt_ret": mkt_ret}


# ---------------------------------------------------------------- 通用算子

def ts_rank_last(arr, window, min_count):
    """时序分位：当前值在尾随 window 个观测（含自身）中的分位，NaN 感知。

    rank = (小于当前的个数 + 0.5*等于当前的个数) / 窗口内有效个数；
    有效个数 < min_count 或当前值 NaN 时记 NaN。arr 为一维数值序列。
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    if n < window:
        for i in range(min_count - 1, n):
            out[i] = _rank_in_window(arr[: i + 1], min_count)
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(arr, window)
    x = sw[:, -1]
    valid = np.isfinite(sw)
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        less = ((sw < x[:, None]) & valid).sum(axis=1)
        eq = ((sw == x[:, None]) & valid).sum(axis=1)
    r = (less + 0.5 * eq) / np.where(cnt > 0, cnt, np.nan)
    r[cnt < min_count] = np.nan
    r[~np.isfinite(x)] = np.nan
    out[window - 1:] = r
    if min_count < window:  # 前缀段（窗口不足全长但可满足 min_count）
        for i in range(min_count - 1, min(window - 1, n)):
            out[i] = _rank_in_window(arr[: i + 1], min_count)
    return out


def _rank_in_window(w, min_count):
    valid = np.isfinite(w)
    cnt = int(valid.sum())
    x = w[-1]
    if cnt < min_count or not np.isfinite(x):
        return np.nan
    return float((np.sum(w[valid] < x) + 0.5 * np.sum(w[valid] == x)) / cnt)


def streak_of_ones(flag):
    """连续 1 的 streak 计数（遇 0 归零）；flag 的 NaN 断链且对应位置记 NaN。

    flag: pd.Series（单股时序，0/1/NaN）。返回同索引 float64 Series。
    """
    f = flag.fillna(0).astype(np.float64)
    run = (f == 0).cumsum()
    s = f.groupby(run).cumsum()
    return s.where(flag.notna())


def _groll(panel, col, window, min_periods, how):
    """逐股 rolling 聚合，按面板行位对齐返回 float64 ndarray。

    panel 须已按 (ts_code, date) 排序并 reset_index；groupby(sort=False)
    rolling 结果索引为 (ts_code, 原行位)，与面板行位一一对应。
    """
    g = panel.groupby("ts_code", sort=False)[col]
    r = getattr(g.rolling(window, min_periods=min_periods), how)()
    out = r.droplevel(0)
    assert out.index.is_monotonic_increasing and len(out) == len(panel)
    return out.to_numpy(np.float64)


def _gshift(panel, col, periods):
    """逐股 shift，按面板行位对齐返回 ndarray。"""
    return panel.groupby("ts_code", sort=False)[col].shift(periods) \
        .to_numpy(np.float64)


def _assign_interval(panel, intervals, val_col, default=np.nan,
                     dtype=np.float64):
    """区间归属：对每行 (ts_code, date) 找 start<=date<=end 的区间值。

    无匹配记 default；重叠区间按 start 排序后者覆盖（取最后生效）。
    dtype=np.float64 用于数值归属（ST 状态等），dtype=object 用于字符串
    归属（行业代码）。
    """
    out = np.full(len(panel), default, dtype=dtype)
    dates = panel["date"].to_numpy()
    far = np.datetime64("2262-01-01")
    for code, pos in panel.groupby("ts_code", sort=False).indices.items():
        iv = intervals.get(code)
        if iv is None or len(iv) == 0:
            continue
        starts = iv["start"].to_numpy()
        ends = iv["end"].to_numpy()
        ends = np.where(pd.isna(ends), far, ends)
        vals = iv[val_col].to_numpy()
        d = dates[pos]
        k = np.searchsorted(starts, d, side="right") - 1
        hit = k >= 0
        k_safe = np.clip(k, 0, None)
        hit &= d <= ends[k_safe]
        out[pos[hit]] = vals[k_safe[hit]]
    return out


# ---------------------------------------------------------------- 特征计算

def compute_all(panel, ctx):
    """全量 T3 特征计算。panel 为 build_panel 产出；ctx 为 build_ctx 产出。

    纯因果：截断 [T-LOOKBACK_DAYS, T] 的面板重算，T 日行与全历史计算逐位一致。
    返回 DataFrame(ts_code, date, *T3_COLUMNS)。
    """
    panel = panel.sort_values(["ts_code", "date"]).reset_index(drop=True)
    cal = ctx["calendar"]

    # ---- 基础量：日历位置 / 上市交易日数 / 可交易标记 / CF 复权链
    pos_t = np.searchsorted(cal, panel["date"].to_numpy()).astype(np.float64)
    panel["_cal_pos"] = pos_t
    ld = pd.to_datetime(panel["ts_code"].map(ctx["list_date"]))
    pos_list = np.searchsorted(cal, ld.to_numpy()).astype(np.float64)
    age = pos_t - pos_list
    age[ld.isna().to_numpy()] = np.nan
    age[age < 0] = np.nan
    panel["_age"] = age
    panel["_tradable"] = age >= MIN_LIST_DAYS

    gr = 1.0 + panel["pct_chg"].astype(np.float64) / 100.0
    gr = gr.fillna(1.0)  # bar 存在而 pct_chg 缺失极罕见，视作零变动并登记
    panel["_cf"] = gr.groupby(panel["ts_code"], sort=False).cumprod().to_numpy()

    # ---- 制度归属：#28 ST 状态与申万一级行业（T 时点区间重建，禁快照回填）
    panel["ST_STATUS"] = _assign_interval(
        panel, ctx["st_intervals"], "status", default=0.0)
    panel["_ind"] = _assign_interval(panel, ctx["ind_intervals"],
                                     "index_code", dtype=object)

    # ---- 涨跌停制度族（个股级 #10-#14）
    _add_limit_features(panel)

    # ---- 换手/流动性/规模/估值（#1-#9）
    _add_turnover_valuation(panel)

    # ---- 市场收益与行业日表（#21 行业动量、#23 宽度、#24 新低占比）
    panel["_mkt_ret"] = panel["date"].map(ctx["mkt_ret"]).astype(np.float64)
    ind_daily = _industry_daily(panel)
    panel = panel.merge(
        ind_daily, left_on=["date", "_ind"], right_on=["date", "ind"],
        how="left", validate="m:1").drop(columns=["ind"])
    panel = panel.rename(columns={"ind_ret": "_ind_ret"})
    panel = panel.sort_values(["ts_code", "date"]).reset_index(drop=True)

    # ---- #22 个股二十日收益对行业超额
    pct = panel["pct_chg"].astype(np.float64)
    panel["_log1p"] = np.where(np.isfinite(pct), np.log1p(pct / 100.0), np.nan)
    cnt20 = _groll(panel, "pct_chg", 20, 1, "count")
    slog20 = _groll(panel, "_log1p", 20, 1, "sum")
    ret20 = np.where(cnt20 >= 15, (np.exp(slog20) - 1.0) * 100.0, np.nan)
    panel["IND_EXCESS_RET20"] = ret20 - panel["IND_MOM20_EQW"].to_numpy()

    # ---- #25 残差动量
    _add_resid_mom(panel)

    # ---- 市场级日序列（#15-#20）与风格（#26/#27），按 date 映射回面板
    mkt_daily = _market_daily(panel, ctx)
    panel = panel.merge(mkt_daily, on="date", how="left", validate="m:1")

    # ---- #29 面值距离 / 低于面值天数；#30 上市时长
    panel["PAR_VALUE_GAP"] = np.log(panel["close"].astype(np.float64))
    below = (panel["close"] < 1.0).astype(np.float64).where(
        panel["close"].notna())
    panel["DAYS_BELOW_PAR"] = below.groupby(
        panel["ts_code"], sort=False).transform(streak_of_ones).to_numpy()
    panel["LIST_AGE"] = np.log1p(panel["_age"])

    return panel[["ts_code", "date", *T3_COLUMNS]].copy()


def _exact_approx_flag(ok, exact_ok, exact_hit, approx_hit):
    """精确段取精确判定、近似段取阈值判定的 0/1 标记；ok 之外记 NaN。

    ok/exact_ok/exact_hit/approx_hit 均为等长布尔 ndarray（近似段掩码
    由 ok & ~exact_ok 蕴含，调用方无需再传）。
    """
    out = np.full(len(ok), np.nan)
    m = ok & exact_ok
    out[m] = exact_hit[m].astype(np.float64)
    m = ok & ~exact_ok
    out[m] = approx_hit[m].astype(np.float64)
    return out


def _add_limit_features(panel):
    """#10-#14：精确段（2007+）stk_limit 价格判定，近似段 pct_chg 阈值判定。

    上市未满 5 个交易日一律 NaN；1996-12-16 前 NaN；
    精确段缺 stk_limit 行记 NaN（不回退近似）。
    """
    dates = panel["date"].to_numpy()
    exact_ok = (dates >= np.datetime64(EXACT_LIMIT_START)) & \
        panel["up_limit"].notna().to_numpy()
    approx_ok = (dates < np.datetime64(EXACT_LIMIT_START)) & \
        (dates >= np.datetime64(LIMIT_RULE_START)) & \
        panel["pre_close"].notna().to_numpy()
    age_ok = panel["_age"].to_numpy() >= MIN_LIST_DAYS

    # 近似段阈值（百分数）：ST/*ST/退 5%，否则 10%（近似段无其他板块）
    st = panel["ST_STATUS"].to_numpy()
    limit_pct = np.where(st >= 1, 5.0, 10.0)
    thr = limit_pct - APPROX_TOL_PP

    close = panel["close"].astype(np.float64).to_numpy()
    open_ = panel["open"].astype(np.float64).to_numpy()
    high = panel["high"].astype(np.float64).to_numpy()
    low = panel["low"].astype(np.float64).to_numpy()
    pre = panel["pre_close"].astype(np.float64).to_numpy()
    pct_chg = panel["pct_chg"].astype(np.float64).to_numpy()
    up = panel["up_limit"].astype(np.float64).to_numpy()
    down = panel["down_limit"].astype(np.float64).to_numpy()

    up_ok = age_ok & (exact_ok | approx_ok) & np.isfinite(close)
    # --- 封涨停 (#10)：精确 close>=up-0.005；近似 pct_chg >= limit_pct-0.5
    sealed = _exact_approx_flag(up_ok, exact_ok,
                                close >= up - PRICE_TOL, pct_chg >= thr)
    # --- 触板 / 炸板 (#12)
    with np.errstate(invalid="ignore", divide="ignore"):
        touch = _exact_approx_flag(up_ok, exact_ok,
                                   high >= up - PRICE_TOL,
                                   (high / pre - 1.0) * 100.0 >= thr)
    touch_fail = np.where(up_ok, ((touch == 1) & (sealed == 0))
                          .astype(np.float64), np.nan)
    up_apx = pre * (1.0 + limit_pct / 100.0)
    up_eff = np.where(exact_ok, up, up_apx)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where((touch == 1) & (up_eff > 0),
                         (up_eff - close) / up_eff, np.nan)
    # --- 一字涨停 (#11)：仅精确段可判
    oneword = np.where(
        up_ok & exact_ok,
        ((np.abs(open_ - up) <= PRICE_TOL) & (np.abs(high - up) <= PRICE_TOL)
         & (np.abs(low - up) <= PRICE_TOL) & (np.abs(close - up) <= PRICE_TOL))
        .astype(np.float64), np.nan)
    # --- 触跌停 / 撬板 (#14)
    down_apx = pre * (1.0 - limit_pct / 100.0)
    down_eff = np.where(exact_ok, down, down_apx)
    down_ok = up_ok & np.isfinite(down_eff)
    with np.errstate(invalid="ignore", divide="ignore"):
        down_touch = _exact_approx_flag(down_ok, exact_ok,
                                        low <= down + PRICE_TOL,
                                        (low / pre - 1.0) * 100.0 <= -thr)
    down_sealed = _exact_approx_flag(down_ok, exact_ok,
                                     close <= down + PRICE_TOL,
                                     pct_chg <= -thr)
    down_unsealed = np.where(down_ok, ((down_touch == 1) & (down_sealed == 0))
                             .astype(np.float64), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        recover = np.where((down_touch == 1) & (down_eff > 0),
                           (close - down_eff) / down_eff, np.nan)

    panel["_sealed"] = sealed
    panel["_down_sealed"] = down_sealed
    panel["_touched"] = np.where(up_ok, ((sealed == 1) | (touch == 1))
                                 .astype(np.float64), np.nan)
    panel["LIMITUP_SEALED_EXACT"] = sealed
    panel["LIMITUP_SEALED_SRC"] = np.where(
        up_ok & exact_ok, 2.0, np.where(up_ok & ~exact_ok, 1.0, 0.0))
    panel["ONEWORD_LIMITUP"] = oneword
    panel["TOUCH_LIMITUP_FAIL"] = touch_fail
    panel["TOUCH_FAIL_DEPTH"] = depth
    panel["DOWNLIMIT_UNSEALED"] = down_unsealed
    panel["DOWNTOUCH_RECOVER"] = recover

    # --- #13 连板递推：streak，NaN 断链；停牌缺口不打断（无 bar）但标注
    panel["CONSEC_LIMITUP"] = panel.groupby("ts_code", sort=False)["_sealed"] \
        .transform(streak_of_ones).to_numpy()
    gap = panel.groupby("ts_code", sort=False)["_cal_pos"] \
        .transform(lambda s: (s.diff() > 1).astype(np.float64)
                   .fillna(0.0)).to_numpy()
    consec = panel["CONSEC_LIMITUP"].to_numpy()
    panel["CONSEC_SUSP_GAP"] = np.where(np.isfinite(consec),
                                        np.where(consec > 0, gap, 0.0),
                                        np.nan)


def _add_turnover_valuation(panel):
    """#1-#9：换手/流动性/规模/估值个股侧滚动量。"""
    panel["_tof"] = panel["turnover_rate_f"].astype(np.float64)
    panel["TURN_F20"] = _groll(panel, "_tof", 20, 15, "mean")
    panel["STR20"] = _groll(panel, "_tof", 20, 15, "std")
    m21 = _groll(panel, "_tof", 21, 15, "mean")
    m252 = _groll(panel, "_tof", 252, 126, "mean")
    panel["ABN_TURN_21_252"] = np.where(
        np.isfinite(m252) & (m252 > 0) & np.isfinite(m21),
        np.log(m21 / m252), np.nan)
    panel["VOLUME_RATIO_T"] = panel["volume_ratio"].astype(np.float64)
    panel["VOLUME_RATIO_MA5"] = _groll(panel, "VOLUME_RATIO_T", 5, 3, "mean")
    tof = panel["_tof"].to_numpy()
    pct = panel["pct_chg"].astype(np.float64).to_numpy()
    panel["_amihud_t"] = np.where(
        np.isfinite(tof) & np.isfinite(pct),
        np.abs(pct) / np.maximum(tof, 0.01), np.nan)
    panel["AMIHUD_TURN20"] = _groll(panel, "_amihud_t", 20, 15, "mean")

    close = panel["close"].astype(np.float64)
    free_mv = close * panel["free_share"].astype(np.float64)  # 元×万股=万元
    ln_free = np.where(np.isfinite(free_mv) & (free_mv > 0),
                       np.log(free_mv), np.nan)
    circ = panel["circ_mv"].astype(np.float64)
    ln_circ = np.where(np.isfinite(circ) & (circ > 0), np.log(circ), np.nan)
    panel["LN_FREE_MV"] = np.where(np.isfinite(ln_free), ln_free, ln_circ)

    # #7 行业内 z-score（行业有效成员 <10 记 NaN；pb NaN 股不参与）
    pb = panel["pb"].astype(np.float64)
    panel["_bp"] = np.where(np.isfinite(pb) & (pb > 0), 1.0 / pb, np.nan)
    g = panel.groupby(["date", "_ind"], sort=False)["_bp"]
    mu = g.transform("mean").to_numpy()
    sd = g.transform("std").to_numpy()
    cnt = g.transform("count").to_numpy()
    z = (panel["_bp"].to_numpy() - mu) / np.where(sd > 0, sd, np.nan)
    panel["BP_IND_Z"] = np.where(cnt >= 10, z, np.nan)

    # #8 盈利收益率时序分位（亏损股 pe_ttm NaN -> EP NaN，交模型处理）
    pe = panel["pe_ttm"].astype(np.float64)
    panel["_ep"] = np.where(np.isfinite(pe) & (pe > 0), 1.0 / pe, np.nan)
    panel["EP_TSRANK_500"] = panel.groupby("ts_code", sort=False)["_ep"] \
        .transform(lambda s: ts_rank_last(s.to_numpy(), 500, 250)).to_numpy()

    # #9 股息率：显式 NaN->0（无分红/数据缺失不可分），不做 ffill
    panel["DV_TTM"] = panel["dv_ttm"].astype(np.float64).fillna(0.0)


def _industry_daily(panel):
    """行业日表：等权日收益、二十日动量、六十日宽度、一百二十日新低占比。

    行 = (date, ind)。行业有效个股 <5 记 NaN；动量窗内行业收益 NaN >5 天记 NaN。
    成分按 T 时点区间归属（面板含当时退市股历史 bar），禁当前快照回填。
    """
    df = panel.loc[panel["_ind"].notna(),
                   ["date", "_ind", "ts_code", "pct_chg", "_cf"]].copy()
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    df["_ma60"] = _groll(df, "_cf", 60, 60, "mean")
    df["_min120"] = _groll(df, "_cf", 120, 120, "min")
    df["_above60"] = (df["_cf"] > df["_ma60"]).astype(np.float64) \
        .where(df["_ma60"].notna())
    df["_near_low120"] = (df["_cf"] <= df["_min120"] * 1.02) \
        .astype(np.float64).where(df["_min120"].notna())

    g = df.groupby(["date", "_ind"], sort=True)
    agg = g.agg(n=("pct_chg", "count"), ind_ret=("pct_chg", "mean"),
                n_b=("_above60", "count"), breadth=("_above60", "mean"),
                n_c=("_near_low120", "count"), capit=("_near_low120", "mean"))
    agg = agg.reset_index().rename(columns={"_ind": "ind"})
    agg["ind_ret"] = agg["ind_ret"].where(agg["n"] >= 5)
    agg["IND_BREADTH_MA60"] = agg["breadth"].where(agg["n_b"] >= 5)
    agg["IND_CAPIT_PCT120"] = agg["capit"].where(agg["n_c"] >= 5)

    agg = agg.sort_values(["ind", "date"]).reset_index(drop=True)
    log1p_ind = np.where(np.isfinite(agg["ind_ret"]),
                         np.log1p(agg["ind_ret"] / 100.0), np.nan)
    agg["_log1p_ind"] = log1p_ind
    cnt = agg.groupby("ind", sort=False)["ind_ret"] \
        .rolling(20, min_periods=1).count().droplevel(0).to_numpy()
    slog = agg.groupby("ind", sort=False)["_log1p_ind"] \
        .rolling(20, min_periods=1).sum().droplevel(0).to_numpy()
    agg["IND_MOM20_EQW"] = np.where(cnt >= 15,
                                    (np.exp(slog) - 1.0) * 100.0, np.nan)
    return agg[["date", "ind", "ind_ret", "IND_BREADTH_MA60",
                "IND_CAPIT_PCT120", "IND_MOM20_EQW"]]


def _add_resid_mom(panel):
    """#25 残差动量：[T-64, T-5] 窗口残差日收益求和（>=45 有效日）。

    残差 = pct_chg - 市场收益 - 行业收益；行业 NaN 日退化为仅减市场，
    窗口内含任一退化日即置 RESID_MOM60_INDNA=1。市场收益缺失日记 NaN 跳过。
    口径拼接标记：窗口内市场收益来源全为沪深300 记 2、混合记 1、纯上证记 0。
    """
    resid = panel["pct_chg"].astype(np.float64) - panel["_mkt_ret"] \
        - panel["_ind_ret"].fillna(0.0)
    panel["_resid"] = resid.where(panel["_mkt_ret"].notna())
    panel["_resid_s60"] = _groll(panel, "_resid", 60, 45, "sum")
    panel["RESID_MOM60"] = _gshift(panel, "_resid_s60", 5)
    # 退化日 = 市场收益在场而行业收益缺失的交易日；窗口内取 max（任一即标）
    panel["_indna"] = np.where(panel["_mkt_ret"].notna(),
                               panel["_ind_ret"].isna().astype(np.float64),
                               np.nan)
    panel["_indna_m60"] = _groll(panel, "_indna", 60, 45, "max")
    src = np.where(panel["_mkt_ret"].isna(), np.nan,
                   (panel["date"] >= HS300_RET_START).astype(np.float64))
    panel["_mksrc"] = src
    panel["_mksrc_sum"] = _groll(panel, "_mksrc", 60, 45, "sum")
    panel["_mksrc_cnt"] = _groll(panel, "_mksrc", 60, 45, "count")
    wsum = _gshift(panel, "_mksrc_sum", 5)
    wcnt = _gshift(panel, "_mksrc_cnt", 5)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = wsum / np.where(wcnt > 0, wcnt, np.nan)
    mksrc = np.where(frac >= 1.0, 2.0, np.where(frac <= 0.0, 0.0, 1.0))
    finite = np.isfinite(panel["RESID_MOM60"])
    panel["RESID_MOM60_MKSRC"] = np.where(finite, mksrc, np.nan)
    panel["RESID_MOM60_INDNA"] = np.where(
        finite, _gshift(panel, "_indna_m60", 5), np.nan)


def _market_daily(panel, ctx):
    """市场级日序列 #15-#20 与风格 #26/#27。行 = ctx 交易日历全日。

    横截面分母 = 可交易集合（有 bar 且上市满 5 个交易日，§0 时点纪律）。
    面板缺失的交易日以 NaN 行补齐（滚动窗口按日历连续，天然跳过）。
    """
    cal = ctx["calendar"]
    d = panel.loc[panel["_tradable"],
                  ["ts_code", "date", "pct_chg", "amount", "circ_mv", "_tof",
                   "_sealed", "_touched", "_down_sealed", "CONSEC_LIMITUP"]
                  ].copy()

    # 昨日（上一交易日）封板/连板状态：按 ctx 日历上一交易日对齐
    pos = np.searchsorted(cal, d["date"].to_numpy())
    d["_prev_date"] = np.where(pos > 0, cal[np.clip(pos - 1, 0, None)],
                               np.datetime64("NaT"))
    prev = d[["ts_code", "date", "CONSEC_LIMITUP", "_sealed"]].rename(
        columns={"date": "_prev_date", "CONSEC_LIMITUP": "_prev_consec",
                 "_sealed": "_prev_sealed"})
    d = d.merge(prev, on=["ts_code", "_prev_date"], how="left", validate="m:1")

    g = d.groupby("date", sort=True)
    mkt = g.agg(n_sealed=("_sealed", _sum_eq1),
                n_touch=("_touched", _sum_eq1),
                n_trad=("ts_code", "count"),
                n_down=("_down_sealed", _sum_eq1),
                sum_amount=("amount", "sum"),
                sum_circ_mv=("circ_mv", "sum"),
                mean_tof=("_tof", "mean"),
                n_promote_den=("_prev_consec", _count_ge1),
                ).reset_index()
    promote = d.loc[d["_prev_consec"] >= 1]
    mkt["n_promote_num"] = promote.groupby("date", sort=True)["_sealed"] \
        .apply(_sum_eq1).reindex(mkt["date"]).fillna(0.0).to_numpy()
    prem_base = d.loc[d["_prev_sealed"] == 1]
    mkt["MKT_LIMITUP_PREM"] = prem_base.groupby("date", sort=True)["pct_chg"] \
        .mean().reindex(mkt["date"]).to_numpy()

    # 面板缺失交易日补 NaN 行，滚动窗口按日历连续
    mkt = mkt.set_index("date").reindex(cal).rename_axis("date").reset_index()
    has_limit_era = mkt["date"] >= LIMIT_RULE_START

    mkt["MKT_SEAL_RATIO"] = np.where(
        has_limit_era & (mkt["n_touch"] >= 10),
        mkt["n_sealed"] / mkt["n_touch"], np.nan)
    mkt["MKT_PROMOTE_RATE"] = np.where(
        has_limit_era & (mkt["n_promote_den"] > 0),
        mkt["n_promote_num"] / mkt["n_promote_den"], np.nan)
    nd = mkt["n_down"].where(has_limit_era)
    mu = nd.rolling(250, min_periods=126).mean()
    sd = nd.rolling(250, min_periods=126).std()
    mkt["MKT_DOWNLIMIT_Z"] = ((nd - mu) / sd.where(sd > 0)).to_numpy()
    mkt["MKT_LIMITUP_PREM_MA5"] = mkt["MKT_LIMITUP_PREM"] \
        .rolling(5, min_periods=3).mean().to_numpy()

    # #19 新高新低（CF 口径，250 日窗须满 250 个有效观测，次新股天然排除）
    nh = _nh_nl(panel)
    mkt = mkt.merge(nh, on="date", how="left", validate="1:1")
    mkt["MKT_NH_NL_DIFF"] = np.where(
        mkt["n_trad19"] > 0,
        (mkt["n_nh"] - mkt["n_nl"]) / mkt["n_trad19"], np.nan)
    nh_nl = mkt["n_nh"] + mkt["n_nl"]
    mkt["MKT_NH_NL_RATIO"] = np.where(nh_nl > 0, mkt["n_nh"] / nh_nl, np.nan)

    # #20 全市场换手率（成交额加权 = sum(amount)/sum(circ_mv)，等权为对照）；
    # amount 千元 / circ_mv 万元，比值与文献口径差常数倍，分位不受影响
    mkt["_to_w"] = mkt["sum_amount"] / mkt["sum_circ_mv"].where(
        mkt["sum_circ_mv"] > 0)
    for src_col, out_col in (("_to_w", "MKT_TURNOVER_PCTL"),
                             ("mean_tof", "MKT_TURNOVER_PCTL_EQW")):
        ma5 = mkt[src_col].rolling(5, min_periods=3).mean().to_numpy()
        mkt[out_col] = ts_rank_last(ma5, 750, 250)

    # #26/#27 风格相对强弱（指数 60 日对数收益差）
    c_den = ctx["idx_close"]["000300.SH"]
    for out_col, num in (("STYLE_SIZE_RS60", "000852.SH"),
                         ("STYLE_GV_RS60", "399006.SZ")):
        c_num = ctx["idx_close"][num]
        rs = np.log(c_num / c_num.shift(59)) - np.log(c_den / c_den.shift(59))
        mkt[out_col] = mkt["date"].map(rs).astype(np.float64).to_numpy()

    keep = ["date", "MKT_SEAL_RATIO", "MKT_PROMOTE_RATE", "MKT_DOWNLIMIT_Z",
            "MKT_LIMITUP_PREM", "MKT_LIMITUP_PREM_MA5", "MKT_NH_NL_DIFF",
            "MKT_NH_NL_RATIO", "MKT_TURNOVER_PCTL", "MKT_TURNOVER_PCTL_EQW",
            "STYLE_SIZE_RS60", "STYLE_GV_RS60"]
    return mkt[keep]


def _sum_eq1(s):
    return float((s == 1).sum())


def _count_ge1(s):
    return float((s >= 1).sum())


def _nh_nl(panel):
    """#19 计数表：逐日 NH250/NL250 家数与可交易分母（行 = 面板出现的交易日）。"""
    df = panel.loc[panel["_tradable"], ["date", "ts_code", "_cf"]].copy()
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    df["_max250"] = _groll(df, "_cf", 250, 250, "max")
    df["_min250"] = _groll(df, "_cf", 250, 250, "min")
    df["_nh"] = (df["_cf"] >= df["_max250"]).astype(np.float64) \
        .where(df["_max250"].notna())
    df["_nl"] = (df["_cf"] <= df["_min250"]).astype(np.float64) \
        .where(df["_min250"].notna())
    g = df.groupby("date", sort=True)
    return g.agg(n_nh=("_nh", "sum"), n_nl=("_nl", "sum"),
                 n_trad19=("ts_code", "count")).reset_index()


# ---------------------------------------------------------------- 快照与对拍

def snapshot(feat_panel, keys):
    """从特征面板抽取事件键行。keys: DataFrame(ts_code, date)，池内唯一。"""
    k = keys.copy()
    k["date"] = pd.to_datetime(k["date"])
    return k.merge(feat_panel, on=["ts_code", "date"], how="left",
                   validate="1:1")


def prefix_recompute_at(data_root, ctx, ts_code, T, lookback_days=LOOKBACK_DAYS,
                        workers=0):
    """时点一致性对照实现：只用 [T-lookback, T] 数据重算，取 (ts_code, T) 行。

    日历/区间/指数序列来自 ctx（截断不变量），与全历史计算共用同一 ctx。
    """
    T = pd.Timestamp(T)
    lo = T - pd.Timedelta(days=lookback_days)
    panel = build_panel(data_root, date_lo=lo, date_hi=T, workers=workers)
    if len(panel) == 0:
        return None
    feat = compute_all(panel, ctx)
    row = feat[(feat["ts_code"] == ts_code) & (feat["date"] == T)]
    return row if len(row) else None
