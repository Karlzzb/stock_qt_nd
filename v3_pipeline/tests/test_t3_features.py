#!/usr/bin/env python3
"""t3_features 单元测试（issue #24）。

合成小面板已知值测试：通用算子、涨跌停判定（精确/近似双口径）、ST/行业
区间归属、行业聚合、市场级聚合、残差动量拼接标记、前缀稳定性（截断重算
== 全历史）、泄漏列名断言。测试不触磁盘：ctx 与 panel 均手工构造。
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_master as fmx  # noqa: E402
import t3_features as t3  # noqa: E402

CAL = pd.date_range("2020-01-02", periods=40, freq="B").to_numpy()


def make_ctx(calendar=CAL, list_date=None, st=None, ind=None, mkt_ret=None,
             idx_close=None):
    """手工静态上下文。st/ind: {code: DataFrame(start,end,status|index_code)}。"""
    codes = list((list_date or {"A": "2019-01-02"}).keys())
    if list_date is None:
        list_date = {c: pd.Timestamp("2019-01-02") for c in codes}
    if mkt_ret is None:
        mkt_ret = pd.Series(0.5, index=pd.DatetimeIndex(calendar))
    if idx_close is None:
        idx = np.linspace(100, 200, len(calendar))
        idx_close = {c: pd.Series(idx, index=pd.DatetimeIndex(calendar))
                     for c in ("000300.SH", "000852.SH", "399006.SZ")}
    return {"calendar": np.asarray(calendar, dtype="datetime64[ns]"),
            "list_date": list_date,
            "st_intervals": st or {},
            "ind_intervals": ind or {},
            "idx_close": idx_close, "mkt_ret": mkt_ret}


def make_panel(rows):
    """rows: list of dict；缺省列自动补常规值。返回按 (ts_code,date) 排序面板。"""
    defaults = {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
                "pre_close": 10.0, "pct_chg": 0.0, "amount": 1e4,
                "turnover_rate_f": 2.0, "volume_ratio": 1.0, "pb": 2.0,
                "pe_ttm": 20.0, "dv_ttm": 1.5, "free_share": 1e5,
                "circ_mv": 1e6, "up_limit": np.nan, "down_limit": np.nan}
    recs = []
    for r in rows:
        d = {**defaults, **r}
        d["date"] = pd.Timestamp(d["date"])
        recs.append(d)
    df = pd.DataFrame(recs)
    return df.sort_values(["ts_code", "date"]).reset_index(drop=True)


def stock_rows(code, dates, **kw):
    return [{"ts_code": code, "date": d, **kw} for d in dates]


# ---------------------------------------------------------------- 通用算子

def test_ts_rank_last_known_values():
    arr = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    # window=3, min_count=2：i=2 时窗口 [1,2,3]，3 的分位 = (2+0.5)/3
    r = t3.ts_rank_last(arr, window=3, min_count=2)
    assert np.isnan(r[0])
    assert r[1] == pytest.approx((0 + 0.5 * 1 + 1) / 2)  # [1,2]: less=1,eq=1 ->(1+0.5)/2
    assert r[2] == pytest.approx(2.5 / 3)
    assert r[4] == pytest.approx((0 + 0.5) / 3)          # [3,2,1]: 1 最小


def test_ts_rank_last_nan_aware():
    arr = np.array([1.0, np.nan, 3.0, 4.0])
    r = t3.ts_rank_last(arr, window=3, min_count=2)
    assert np.isnan(r[1])                      # 当前值 NaN -> NaN
    # 窗口 [nan, 3, 4]：有效 [3,4] 共 2 个 -> (1+0.5)/2
    assert r[3] == pytest.approx(1.5 / 2)


def test_ts_rank_min_count_gate():
    arr = np.arange(10.0)
    r = t3.ts_rank_last(arr, window=5, min_count=4)
    assert np.all(np.isnan(r[:3]))   # 前缀有效数 <4
    assert r[3] == pytest.approx(3.5 / 4)  # [0,1,2,3] 中 3 的分位


def test_streak_of_ones():
    s = pd.Series([1.0, 1.0, 0.0, 1.0, np.nan, 1.0, 1.0])
    out = t3.streak_of_ones(s)
    assert out.tolist()[:4] == [1.0, 2.0, 0.0, 1.0]
    assert np.isnan(out.iloc[4])
    assert out.tolist()[5:] == [1.0, 2.0]      # NaN 断链后重新计数


# ---------------------------------------------------------------- 区间归属

def test_st_status_intervals():
    st = {"A": pd.DataFrame({
        "start": pd.to_datetime(["2019-01-01", "2020-01-10", "2020-02-01"]),
        "end": pd.to_datetime(["2020-01-09", "2020-01-31", None]),
        "status": [0.0, 1.0, 2.0]})}
    panel = make_panel(stock_rows("A", CAL[:30]))
    panel["_age"] = 100.0
    out = t3._assign_interval(panel, st, "status", default=0.0)
    s = pd.Series(out, index=panel["date"])
    assert s.loc["2020-01-09"] == 0.0
    assert s.loc["2020-01-10"] == 1.0
    assert s.loc["2020-01-31"] == 1.0
    assert s.loc["2020-02-03"] == 2.0   # 开放区间延续


def test_industry_interval_str():
    ind = {"A": pd.DataFrame({
        "start": pd.to_datetime(["2019-01-01"]),
        "end": pd.to_datetime([None]),
        "index_code": ["801010.SI"]})}
    panel = make_panel(stock_rows("A", CAL[:5]) + stock_rows("B", CAL[:5]))
    out = t3._assign_interval(panel, ind, "index_code", dtype=object)
    assert (out[panel["ts_code"].to_numpy() == "A"] == "801010.SI").all()
    assert pd.isna(out[panel["ts_code"].to_numpy() == "B"]).all()


# ---------------------------------------------------------------- 涨跌停判定

def test_limit_flags_exact_era():
    # 精确段：up_limit 精确价判定；CAL[10..13] 为四个场景日，前置 10 天满足新股窗口
    rows = stock_rows("A", CAL[:10])
    rows += [
        # 封板：close 容差内
        {"ts_code": "A", "date": CAL[10], "close": 11.00, "high": 11.00,
         "open": 10.5, "low": 10.4, "pre_close": 10.0, "pct_chg": 10.0,
         "up_limit": 11.00, "down_limit": 9.00},
        # 炸板：high 触板 close 未封
        {"ts_code": "A", "date": CAL[11], "close": 10.50, "high": 12.10,
         "open": 11.0, "low": 10.4, "pre_close": 11.0, "pct_chg": -4.545,
         "up_limit": 12.10, "down_limit": 9.90},
        # 一字板
        {"ts_code": "A", "date": CAL[12], "close": 11.55, "high": 11.55,
         "open": 11.55, "low": 11.55, "pre_close": 10.5, "pct_chg": 10.0,
         "up_limit": 11.55, "down_limit": 9.45},
        # 撬板：low 触跌停 close 回收
        {"ts_code": "A", "date": CAL[13], "close": 10.0, "high": 10.2,
         "open": 9.9, "low": 9.45, "pre_close": 10.5, "pct_chg": -4.76,
         "up_limit": 11.55, "down_limit": 9.45},
    ]
    panel = make_panel(rows)
    ctx = make_ctx(list_date={"A": "2019-01-02"})
    feat = t3.compute_all(panel, ctx)
    feat = feat.set_index("date")
    d1 = feat.loc[CAL[10]]
    assert d1["LIMITUP_SEALED_EXACT"] == 1.0
    assert d1["LIMITUP_SEALED_SRC"] == 2.0
    assert d1["TOUCH_LIMITUP_FAIL"] == 0.0
    d2 = feat.loc[CAL[11]]
    assert d2["LIMITUP_SEALED_EXACT"] == 0.0
    assert d2["TOUCH_LIMITUP_FAIL"] == 1.0
    assert d2["TOUCH_FAIL_DEPTH"] == pytest.approx((12.10 - 10.50) / 12.10)
    d3 = feat.loc[CAL[12]]
    assert d3["ONEWORD_LIMITUP"] == 1.0
    assert d3["LIMITUP_SEALED_EXACT"] == 1.0
    d4 = feat.loc[CAL[13]]
    assert d4["DOWNLIMIT_UNSEALED"] == 1.0
    assert d4["DOWNTOUCH_RECOVER"] == pytest.approx((10.0 - 9.45) / 9.45)


def test_limit_flags_approx_era_pre2007():
    # 1996-12-16 后、2007 前的近似段：主板 10% 阈值容差 0.5pp
    cal = pd.date_range("2000-01-03", periods=30, freq="B").to_numpy()
    rows = stock_rows("A", cal[:10], pct_chg=1.0)
    rows.append({"ts_code": "A", "date": cal[10], "pct_chg": 9.8,
                 "pre_close": 10.0, "close": 10.98, "high": 10.99})
    rows.append({"ts_code": "A", "date": cal[11], "pct_chg": 10.2,
                 "pre_close": 10.98, "close": 12.10, "high": 12.10})
    panel = make_panel(rows)
    ctx = make_ctx(calendar=cal, list_date={"A": "1999-01-04"})
    feat = t3.compute_all(panel, ctx).set_index("date")
    # 9.8 >= 10-0.5=9.5 -> 近似封板
    assert feat.loc[cal[10], "LIMITUP_SEALED_EXACT"] == 1.0
    assert feat.loc[cal[10], "LIMITUP_SEALED_SRC"] == 1.0
    assert feat.loc[cal[11], "LIMITUP_SEALED_EXACT"] == 1.0
    assert feat.loc[cal[11], "CONSEC_LIMITUP"] == 2.0     # 连板递推
    assert np.isnan(feat.loc[cal[10], "ONEWORD_LIMITUP"])  # 近似段不可判


def test_limit_flags_approx_st_threshold():
    # ST 股近似段 5% 阈值
    cal = pd.date_range("2000-01-03", periods=30, freq="B").to_numpy()
    st = {"A": pd.DataFrame({
        "start": pd.to_datetime(["1999-01-01"]), "end": pd.to_datetime([None]),
        "status": [1.0]})}
    rows = stock_rows("A", cal[:10])
    rows.append({"ts_code": "A", "date": cal[10], "pct_chg": 4.8,
                 "pre_close": 10.0, "close": 10.48, "high": 10.48})
    panel = make_panel(rows)
    ctx = make_ctx(calendar=cal, list_date={"A": "1999-01-04"}, st=st)
    feat = t3.compute_all(panel, ctx).set_index("date")
    assert feat.loc[cal[10], "LIMITUP_SEALED_EXACT"] == 1.0  # 4.8 >= 5-0.5
    assert feat.loc[cal[10], "ST_STATUS"] == 1.0


def test_limit_flags_new_stock_nan():
    # 上市未满 5 个交易日：涨跌停判定 NaN
    rows = stock_rows("A", CAL[:3], close=11.0, high=11.0, pct_chg=10.0,
                      up_limit=11.0, down_limit=9.0)
    panel = make_panel(rows)
    ctx = make_ctx(list_date={"A": CAL[0]})
    feat = t3.compute_all(panel, ctx)
    assert feat["LIMITUP_SEALED_EXACT"].isna().all()
    assert (feat["LIMITUP_SEALED_SRC"] == 0.0).all()


def test_limit_flags_pre_1996_nan():
    cal = pd.date_range("1995-01-03", periods=10, freq="B").to_numpy()
    rows = stock_rows("A", cal, pct_chg=15.0)
    panel = make_panel(rows)
    ctx = make_ctx(calendar=cal, list_date={"A": "1994-01-03"})
    feat = t3.compute_all(panel, ctx)
    assert feat["LIMITUP_SEALED_EXACT"].isna().all()
    assert feat["CONSEC_LIMITUP"].isna().all()


# ---------------------------------------------------------------- 个股滚动量

def test_turnover_and_valuation():
    rows = stock_rows("A", CAL[:30], turnover_rate_f=2.0, volume_ratio=1.5,
                      pb=2.0, pe_ttm=25.0, dv_ttm=np.nan, free_share=1e5,
                      close=10.0)
    panel = make_panel(rows)
    ctx = make_ctx(list_date={"A": "2019-01-02"})
    feat = t3.compute_all(panel, ctx)
    last = feat.iloc[-1]
    assert last["TURN_F20"] == pytest.approx(2.0)
    assert last["STR20"] == pytest.approx(0.0, abs=1e-12)
    assert last["VOLUME_RATIO_MA5"] == pytest.approx(1.5)
    # dv_ttm NaN -> 0
    assert last["DV_TTM"] == 0.0
    # LN_FREE_MV = ln(10 * 1e5)
    assert last["LN_FREE_MV"] == pytest.approx(np.log(1e6))
    # 次新股长窗 NaN（30 天 < 126）
    assert np.isnan(last["ABN_TURN_21_252"])
    assert np.isnan(last["EP_TSRANK_500"])
    # LIST_AGE = ln(1+29)
    assert last["LIST_AGE"] == pytest.approx(np.log1p(29))


def test_ln_free_mv_fallback_circ_mv():
    rows = stock_rows("A", CAL[:6], free_share=np.nan, circ_mv=5e5,
                      close=8.0)
    panel = make_panel(rows)
    ctx = make_ctx(list_date={"A": "2019-01-02"})
    feat = t3.compute_all(panel, ctx)
    assert feat["LN_FREE_MV"].iloc[-1] == pytest.approx(np.log(5e5))


def test_bp_ind_z_known_value():
    # 同行业两股 A(pb=2) B(pb=4)，异行业 C(pb=8) 单独成组（成员<10 -> NaN）
    ind_iv = pd.DataFrame({
        "start": pd.to_datetime(["2019-01-01"]), "end": pd.to_datetime([None])})
    ind = {"A": ind_iv.assign(index_code="I1")[["start", "end", "index_code"]],
           "B": ind_iv.assign(index_code="I1")[["start", "end", "index_code"]],
           "C": ind_iv.assign(index_code="I2")[["start", "end", "index_code"]]}
    rows = (stock_rows("A", CAL[:6], pb=2.0) + stock_rows("B", CAL[:6], pb=4.0)
            + stock_rows("C", CAL[:6], pb=8.0))
    panel = make_panel(rows)
    ctx = make_ctx(list_date={c: "2019-01-02" for c in "ABC"}, ind=ind)
    feat = t3.compute_all(panel, ctx)
    # 成员数 2 < 10 -> 全部 NaN（口径：行业有效成员 <10 记 NaN）
    assert feat["BP_IND_Z"].isna().all()


def test_bp_ind_z_eleven_members():
    codes = [f"S{i:02d}" for i in range(11)]
    ind = {c: pd.DataFrame({
        "start": pd.to_datetime(["2019-01-01"]), "end": pd.to_datetime([None]),
        "index_code": ["I1"]}) for c in codes}
    rows = []
    for i, c in enumerate(codes):
        rows += stock_rows(c, CAL[:6], pb=float(i + 1))
    panel = make_panel(rows)
    ctx = make_ctx(list_date={c: "2019-01-02" for c in codes}, ind=ind)
    feat = t3.compute_all(panel, ctx)
    bp = 1.0 / np.arange(1, 12.0)
    mu, sd = bp.mean(), bp.std(ddof=1)
    got = feat.loc[feat["ts_code"] == "S00", "BP_IND_Z"].iloc[-1]
    assert got == pytest.approx((bp[0] - mu) / sd)


def test_par_value_gap_and_days_below():
    rows = (stock_rows("A", CAL[:3], close=0.9)
            + stock_rows("A", CAL[3:6], close=1.5))
    panel = make_panel(rows)
    ctx = make_ctx(list_date={"A": "2019-01-02"})
    feat = t3.compute_all(panel, ctx)
    assert feat["DAYS_BELOW_PAR"].tolist() == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    assert feat["PAR_VALUE_GAP"].iloc[0] == pytest.approx(np.log(0.9))


# ---------------------------------------------------------------- 行业/市场聚合

def test_industry_mom_and_excess():
    # 行业 I1 两股每日 +1%，I2 一股（成员<5 -> 行业收益 NaN）
    codes1 = [f"A{i}" for i in range(5)]
    ind = {}
    for c in codes1:
        ind[c] = pd.DataFrame({
            "start": pd.to_datetime(["2019-01-01"]),
            "end": pd.to_datetime([None]), "index_code": ["I1"]})
    ind["B0"] = pd.DataFrame({
        "start": pd.to_datetime(["2019-01-01"]),
        "end": pd.to_datetime([None]), "index_code": ["I2"]})
    rows = []
    for c in codes1:
        rows += stock_rows(c, CAL[:25], pct_chg=1.0)
    rows += stock_rows("B0", CAL[:25], pct_chg=2.0)
    panel = make_panel(rows)
    ctx = make_ctx(list_date={c: "2019-01-02" for c in codes1 + ["B0"]},
                   ind=ind)
    feat = t3.compute_all(panel, ctx)
    a0 = feat[feat["ts_code"] == "A0"].iloc[-1]
    # 行业 20 日动量 = (1.01^20 - 1) * 100
    assert a0["IND_MOM20_EQW"] == pytest.approx((1.01 ** 20 - 1) * 100,
                                                rel=1e-9)
    # 个股 20 日收益与行业相同 -> 超额 0
    assert a0["IND_EXCESS_RET20"] == pytest.approx(0.0, abs=1e-9)
    b0 = feat[feat["ts_code"] == "B0"].iloc[-1]
    assert np.isnan(b0["IND_MOM20_EQW"])     # I2 成员 <5
    assert np.isnan(b0["IND_EXCESS_RET20"])


def test_market_seal_ratio_and_promote():
    # 精确段两日：d0 三股封板（A,B,C），d1 触板 4 家封 2 家
    d0, d1 = CAL[10], CAL[11]
    lim = {"up_limit": 11.0, "down_limit": 9.0}
    rows = []
    for c in ("A", "B", "C", "D"):
        rows += stock_rows(c, CAL[:10], **lim)
    for c in ("A", "B", "C"):
        rows.append({"ts_code": c, "date": d0, "close": 11.0, "high": 11.0,
                     "pct_chg": 10.0, **lim})
    rows.append({"ts_code": "D", "date": d0, "close": 10.0, "high": 10.2,
                 "pct_chg": 0.0, **lim})
    # d1: A 再封、B 炸板、C 平、D 触板未封
    rows.append({"ts_code": "A", "date": d1, "close": 12.1, "high": 12.1,
                 "pct_chg": 10.0, "up_limit": 12.1, "down_limit": 9.9})
    rows.append({"ts_code": "B", "date": d1, "close": 11.5, "high": 12.1,
                 "pct_chg": 4.5, "up_limit": 12.1, "down_limit": 9.9})
    rows.append({"ts_code": "C", "date": d1, "close": 11.0, "high": 11.2,
                 "pct_chg": 0.0, "up_limit": 12.1, "down_limit": 9.9})
    rows.append({"ts_code": "D", "date": d1, "close": 10.5, "high": 11.0,
                 "pct_chg": 5.0, "up_limit": 11.0, "down_limit": 9.0})
    panel = make_panel(rows)
    ctx = make_ctx(list_date={c: "2019-01-02" for c in "ABCD"})
    feat = t3.compute_all(panel, ctx)
    f1 = feat[feat["date"] == d1].set_index("ts_code")
    # 触板 3 家（A 封、B 炸、D 触未封）<10 -> 封板率 NaN（分母下限口径）
    assert np.isnan(f1["MKT_SEAL_RATIO"].iloc[0])
    # 晋级率：昨日连板>=1 的有 A,B,C 三家（d0 封板）；今日再封仅 A -> 1/3
    assert f1["MKT_PROMOTE_RATE"].iloc[0] == pytest.approx(1 / 3)
    # 昨日涨停股（A,B,C）今日平均溢价 = (10.0 + 4.5 + 0.0)/3
    assert f1["MKT_LIMITUP_PREM"].iloc[0] == pytest.approx(4.833333, rel=1e-6)
    # A 连板 2 天
    assert f1.loc["A", "CONSEC_LIMITUP"] == 2.0


def test_mkt_nh_nl_diff():
    # 两股：A 单调上涨（250 日新高），B 单调下跌（新低）；窗口缩短以可测
    cal = pd.date_range("2018-01-02", periods=300, freq="B").to_numpy()
    rows = []
    for i, d in enumerate(cal):
        rows.append({"ts_code": "A", "date": d, "pct_chg": 0.5, "close": 10.0})
        rows.append({"ts_code": "B", "date": d, "pct_chg": -0.5,
                     "close": 10.0})
    panel = make_panel(rows)
    ctx = make_ctx(calendar=cal,
                   list_date={"A": "2017-01-02", "B": "2017-01-02"})
    feat = t3.compute_all(panel, ctx)
    last = feat[feat["date"] == cal[-1]]
    # A 创 250 日新高、B 创新低 -> diff = (1-1)/2 = 0, ratio = 1/2
    assert last["MKT_NH_NL_DIFF"].iloc[0] == pytest.approx(0.0)
    assert last["MKT_NH_NL_RATIO"].iloc[0] == pytest.approx(0.5)


def test_resid_mom_and_mksrc():
    # 市场收益 0.5%/日，无行业归属（退化），个股 1.5%/日 -> 残差 1.0%/日
    cal = pd.date_range("2019-01-02", periods=120, freq="B").to_numpy()
    rows = stock_rows("A", cal, pct_chg=1.5)
    panel = make_panel(rows)
    mkt_ret = pd.Series(0.5, index=pd.DatetimeIndex(cal))
    ctx = make_ctx(calendar=cal, list_date={"A": "2018-01-02"},
                   mkt_ret=mkt_ret)
    feat = t3.compute_all(panel, ctx)
    last = feat.iloc[-1]
    # 窗口 [T-64, T-5] 共 60 日 x 1.0% = 60.0
    assert last["RESID_MOM60"] == pytest.approx(60.0, rel=1e-9)
    assert last["RESID_MOM60_INDNA"] == 1.0
    # 全部日期 >= 2002 -> 纯沪深300 口径记 2
    assert last["RESID_MOM60_MKSRC"] == 2.0


def test_resid_mom_indna_window_level():
    # T 日行业在场、窗口 [T-64,T-5] 内行业全缺失 -> 窗口级退化标记 = 1
    #（T 日截面口径会误判为 0；预登记 #25 要求标记窗口内退化日）
    cal = pd.date_range("2019-01-02", periods=120, freq="B").to_numpy()
    rows = stock_rows("A", cal, pct_chg=1.5)
    panel = make_panel(rows)
    mkt_ret = pd.Series(0.5, index=pd.DatetimeIndex(cal))
    ind = {"A": pd.DataFrame({
        "start": [pd.Timestamp(cal[-1])],
        "end": [pd.NaT],
        "index_code": ["801010.SI"]})}
    ctx = make_ctx(calendar=cal, list_date={"A": "2018-01-02"},
                   mkt_ret=mkt_ret, ind=ind)
    feat = t3.compute_all(panel, ctx)
    last = feat.iloc[-1]
    assert np.isfinite(last["RESID_MOM60"])
    assert last["RESID_MOM60_INDNA"] == 1.0


def test_style_rs60():
    cal = pd.date_range("2019-01-02", periods=100, freq="B").to_numpy()
    rows = stock_rows("A", cal)
    panel = make_panel(rows)
    c300 = pd.Series(np.exp(np.linspace(0, 0.6, 100)),
                     index=pd.DatetimeIndex(cal))
    c852 = pd.Series(np.exp(np.linspace(0, 1.2, 100)),
                     index=pd.DatetimeIndex(cal))
    ctx = make_ctx(calendar=cal, list_date={"A": "2018-01-02"},
                   idx_close={"000300.SH": c300, "000852.SH": c852,
                              "399006.SZ": c852})
    feat = t3.compute_all(panel, ctx)
    # ln 差：59 日间隔的 log 收益差 = (1.2-0.6) * 59/99
    expect = (1.2 - 0.6) * 59 / 99
    assert feat["STYLE_SIZE_RS60"].iloc[-1] == pytest.approx(expect, rel=1e-9)


# ---------------------------------------------------------------- 前缀稳定性

def test_prefix_stability_synthetic():
    # 全历史计算 vs 截断 [.., T] 重算：T 行逐位一致
    cal = pd.date_range("2019-01-02", periods=120, freq="B").to_numpy()
    codes = ["A", "B", "C"]
    rows = []
    rng = np.random.default_rng(7)
    for c in codes:
        for d in cal:
            rows.append({"ts_code": c, "date": d,
                         "pct_chg": float(rng.normal(0, 1.5)),
                         "close": 10.0, "turnover_rate_f": float(rng.uniform(1, 3)),
                         "up_limit": 11.0, "down_limit": 9.0})
    panel_full = make_panel(rows)
    ctx = make_ctx(calendar=cal, list_date={c: "2018-01-02" for c in codes})
    feat_full = t3.compute_all(panel_full, ctx)
    T = cal[-1]
    panel_cut = panel_full[panel_full["date"] <= T].copy()
    feat_cut = t3.compute_all(panel_cut, ctx)
    for c in codes:
        a = feat_full[(feat_full["ts_code"] == c)
                      & (feat_full["date"] == T)][t3.T3_COLUMNS]
        b = feat_cut[(feat_cut["ts_code"] == c)
                     & (feat_cut["date"] == T)][t3.T3_COLUMNS]
        assert np.allclose(a.to_numpy(np.float64), b.to_numpy(np.float64),
                           rtol=1e-9, atol=0, equal_nan=True), c


def test_no_leakage_column_names():
    assert not fmx.excluded_columns(t3.T3_COLUMNS)
    assert set(t3.T3_CN) == set(t3.T3_COLUMNS)
