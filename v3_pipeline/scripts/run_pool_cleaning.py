# -*- coding: utf-8 -*-
"""双信号池清洗与可交易性过滤 (pool_cleaning)。

过滤器(逐个独立标记, 再出组合):
  1. f_st       : 信号日 T 处于 ST/*ST 期(namechange as-of 重建: 最新 start_date<=T 的名称,
                  end_date 口径经抽样交叉验证为"旧名称最后有效日(含)", 被下一 start 取代);
  2. f_limitup  : T+1(交易所历下一交易日, 记 d1)开盘价 >= up_limit - 0.01 元(容差 1 分),
                  一字涨停开盘不可买入; d1 缺 stk_limit(2007 年前)不做此过滤、单独计数;
  3. f_suspend  : d1 无本地行情行, 或 vol<=0/NaN, 或 open NaN/<=0(停牌/无成交)。
                  标签机器(divergence_lab)用个股自身下一根 bar 作 T+1 入场, 停牌事件标签
                  入口被顺延而未剔除——本过滤器按交易所历口径补上, 脚本量化其规模。

段划分与标签与 run_race_rerun_v2.py 完全一致:
  train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31,
  隔离带 2018-11-19~2018-12-28 与 2022-09-13~2022-10-31 不入统计,
  2022-11 以后为测试段: 只做过滤标记与计数, 不出任何命中率数字。
  2001 年前事件(pre2001)不入任何统计段, 仅计数。

基线口径(与 race_rerun_v2 独立基线一致):
  事件加权命中率 = hit 均值; 日加权零信息基线(top3 口径) = 当日池命中率的信号日均值。
  hit = hit_N20_k2.0 (T+1 开盘入场, 20 交易日内 +2*ATR(14))。

输出:
  v3_pipeline/reports/pool_cleaning/excluded_events_{main,backup}.parquet
  v3_pipeline/reports/pool_cleaning/baseline_recalc.csv
  v3_pipeline/reports/pool_cleaning/st_crossval_evidence.csv
  v3_pipeline/reports/pool_cleaning/test_segment_filter_counts.csv
  v3_pipeline/reports/pool_cleaning/pool_cleaning_report.md
  v3_pipeline/reports/pool_cleaning/progress.log (追加)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/karl/repos/personal/stock_qt_nd")
FM_DIR = ROOT / "v3_pipeline" / "reports" / "feature_matrix"
DAILY_DIR = ROOT / "stock_data" / "daily"
BASIC_DIR = ROOT / "stock_data" / "daily_basic"
LIMIT_DIR = ROOT / "stock_data" / "stk_limit"
NC_PATH = ROOT / "stock_data" / "meta" / "namechange.parquet"
OUT_DIR = ROOT / "v3_pipeline" / "reports" / "pool_cleaning"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"

POOLS = {
    "main": {
        "features": FM_DIR / "main_pool_features.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_fractal_o15_s20",
    },
    "backup": {
        "features": FM_DIR / "backup_pool_features.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_zigzag_p05_s5",
    },
}

TRAIN_LO, TRAIN_HI = pd.Timestamp("2001-01-01"), pd.Timestamp("2018-12-31")
VAL_LO, VAL_HI = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-10-31")
EMBARGO = [(pd.Timestamp("2018-11-19"), pd.Timestamp("2018-12-28")),
           (pd.Timestamp("2022-09-13"), pd.Timestamp("2022-10-31"))]
LIMIT_DATA_START = pd.Timestamp("2007-01-04")  # stk_limit 接口数据起点(SUPPLEMENTARY_DATA.md §1.2)
LIMIT_TOL = 0.01  # 一字涨停判定容差 1 分

EXPECTED_VAL_BASELINE = {  # race_rerun_v2 review 复算值, 用于口径自检
    "main": dict(dayw=0.499727, evw=0.580019, days=316, events=3143),
    "backup": dict(dayw=0.557332, evw=0.605003, days=735, events=7076),
}


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 标签/段划分(与 v2 同口径)
def load_labels(labdir):
    ev = pd.read_parquet(labdir / "events.parquet", columns=["ts_code", "date"])
    lb = pd.read_parquet(labdir / "labels.parquet")
    n = len(ev)
    div = lb.iloc[:n].reset_index(drop=True)
    assert (div["group"] == "div").all()
    lab = pd.DataFrame({
        "ts_code": ev["ts_code"].values,
        "date": pd.to_datetime(ev["date"].values),
        "hit": div["hit_N20_k2.0"].values,
    })
    assert not lab.duplicated(["ts_code", "date"]).any()
    return lab


def segment_of(dates):
    """v2 口径 + drop 细分: pre2001 / test(2022-11 以后)。"""
    seg = np.full(len(dates), "test", dtype=object)
    seg[dates < TRAIN_LO] = "pre2001"
    seg[(dates >= TRAIN_LO) & (dates <= TRAIN_HI)] = "train"
    seg[(dates >= VAL_LO) & (dates <= VAL_HI)] = "val"
    for lo, hi in EMBARGO:
        seg[(dates >= lo) & (dates <= hi)] = "embargo"
    return seg


# ================================================================ 名称/ST as-of 索引
def build_name_index():
    nc = pd.read_parquet(NC_PATH)
    nc["start"] = pd.to_datetime(nc["start_date"], format="%Y%m%d")
    idx = {}
    for code, g in nc.sort_values("start").groupby("ts_code"):
        starts = g["start"].to_numpy()
        names = g["name"].to_numpy()
        idx[code] = (starts, np.array(["ST" in nm for nm in names]), names)
    return idx


def st_flags(codes, dates, name_index):
    """as-of: T 时点名称 = 最新 start<=T 的行; 无前知行则记未知(False)。"""
    out = np.zeros(len(dates), bool)
    for code, pos in _group_positions(codes):
        if code not in name_index:
            continue
        starts, is_st, _ = name_index[code]
        i = np.searchsorted(starts, dates[pos], side="right") - 1
        ok = i >= 0
        out[pos[ok]] = is_st[i[ok]]
    return out


def _group_positions(codes):
    s = pd.Series(codes)
    for code, sub in s.groupby(s):
        yield code, sub.index.to_numpy()


# ================================================================ 交易日历 / d1
def load_calendar():
    days = sorted(p.stem for p in BASIC_DIR.glob("*.parquet"))
    return pd.to_datetime(pd.Series(days, dtype="str"), format="%Y%m%d").to_numpy()


def next_trade_days(cal, dates):
    pos = np.searchsorted(cal, dates, side="right")
    d1 = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos < len(cal)
    d1[ok] = cal[pos[ok]]
    return d1, ok


# ================================================================ d1 行情(open/vol) + 个股次bar核查
def attach_d1_bars(df):
    """对每个事件取 d1 当日 open/vol(本地日线), 并核查个股自身次 bar 是否=交易所历 d1。"""
    n = len(df)
    open_d1 = np.full(n, np.nan)
    vol_d1 = np.full(n, np.nan)
    bar_on_d1 = np.zeros(n, bool)
    own_next_eq_d1 = np.zeros(n, bool)
    own_next_after_d1 = np.zeros(n, bool)
    d1_na = df["d1"].isna().to_numpy()
    df["_pos"] = np.arange(n)
    for code, sub in df.groupby("ts_code"):
        path = DAILY_DIR / f"{code}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path, columns=["trade_date", "open", "vol"])
        d = d.drop_duplicates("trade_date").sort_values("trade_date")
        dd = d["trade_date"].to_numpy().astype("datetime64[ns]")
        # d1 行情
        m = sub.loc[~sub["d1"].isna()]
        j = np.searchsorted(dd, m["d1"].to_numpy())
        hit = (j < len(dd)) & (dd[np.minimum(j, len(dd) - 1)] == m["d1"].to_numpy())
        pos = m["_pos"].to_numpy()
        bar_on_d1[pos[hit]] = True
        open_d1[pos[hit]] = d["open"].to_numpy()[j[hit]]
        vol_d1[pos[hit]] = d["vol"].to_numpy()[j[hit]]
        # 个股自身次 bar(T 在本地日线中的位置之后一行)与交易所历 d1 的关系
        jt = np.searchsorted(dd, sub["date"].to_numpy())
        has_t = (jt < len(dd)) & (dd[np.minimum(jt, len(dd) - 1)] == sub["date"].to_numpy())
        has_next = has_t & (jt + 1 < len(dd))
        nb = np.full(len(sub), np.datetime64("NaT"), dtype="datetime64[ns]")
        nb[has_next] = dd[jt[has_next] + 1]
        p2 = sub["_pos"].to_numpy()
        cmp_ok = ~d1_na[p2] & has_next
        own_next_eq_d1[p2[cmp_ok]] = nb[cmp_ok] == sub["d1"].to_numpy()[cmp_ok]
        own_next_after_d1[p2[cmp_ok]] = nb[cmp_ok] > sub["d1"].to_numpy()[cmp_ok]
    df.drop(columns=["_pos"], inplace=True)
    df["bar_on_d1"] = bar_on_d1
    df["open_d1"] = open_d1
    df["vol_d1"] = vol_d1
    df["own_next_eq_d1"] = own_next_eq_d1
    df["own_next_after_d1"] = own_next_after_d1
    return df


# ================================================================ stk_limit
def attach_up_limit(df):
    """对 d1>=2007-01-04 且 d1 有行情的事件, 读 stk_limit 取 up_limit。"""
    df["up_limit_d1"] = np.nan
    m = df.loc[df["bar_on_d1"] & (df["d1"] >= LIMIT_DATA_START)]
    by_day = {d: g for d, g in m.groupby(m["d1"].dt.strftime("%Y%m%d"))}
    for dstr, g in by_day.items():
        path = LIMIT_DIR / f"{dstr}.parquet"
        if not path.exists():
            continue
        lim = pd.read_parquet(path)
        lim = lim[lim["ts_code"].isin(set(g["ts_code"]))]
        lu = dict(zip(lim["ts_code"], lim["up_limit"]))
        vals = [lu.get(c, np.nan) for c in g["ts_code"]]
        df.loc[g.index, "up_limit_d1"] = vals
    return df


# ================================================================ 过滤器标记
def build_flags(pool):
    cfg = POOLS[pool]
    fm = pd.read_parquet(cfg["features"], columns=["ts_code", "date"])
    fm["date"] = pd.to_datetime(fm["date"])
    lab = load_labels(cfg["labdir"])
    df = fm.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    dates = df["date"].to_numpy()
    df["seg"] = segment_of(dates)
    return df


def baseline_table(df, filters):
    """filters: {name: bool array(True=剔除)}。返回长表行。"""
    rows = []
    lab_ok = df["hit"].notna().to_numpy()
    for seg in ("train", "val"):
        base = (df["seg"].to_numpy() == seg) & lab_ok
        for fname, fmask in filters.items():
            keep = base & ~fmask
            sub = df.loc[keep, ["date", "hit"]]
            n_ev = len(sub)
            daily = sub.groupby("date")["hit"].mean()
            rows.append({
                "segment": seg, "filter_set": fname,
                "n_events": int(n_ev), "n_days": int(len(daily)),
                "event_weighted_hit": float(sub["hit"].mean()) if n_ev else np.nan,
                "day_weighted_hit": float(daily.mean()) if len(daily) else np.nan,
            })
    return rows


# ================================================================ ST 区间交叉验证
def st_cross_validation(name_index, cal, n_sample=20, seed=42):
    """抽样主板 ST 股, 用 ±5% 涨跌幅限制验证 as-of 区间重建。

    检查(每股票×每 ST 段):
      a. 段内 |pct_chg|>5.5% 违例天数(价格 rounding 上限约 5.3%, 阈值 5.5% 保守);
      b. 段内触及限制天数(|pct_chg|>=4.8%)——限制确实以 5% 生效的正向证据;
      c. 摘帽后首个交易日 |pct_chg|>5.5%(限制恰在新名称 start 日解除的证据);
      d. 非 ST 日 |pct_chg|>5.5% 天数(对照: 5% 界限是 ST 段特异的);
      e. 2007 年后 ST 日 stk_limit up_limit/pre_close∈[1.045,1.055] 占比(精确口径复核)。
    """
    cands = []
    for code, (starts, is_st, names) in name_index.items():
        if not is_st.any() or code[:2] not in ("60", "00"):
            continue
        if code[:3] == "688" or not (DAILY_DIR / f"{code}.parquet").exists():
            continue
        first_st = starts[is_st][0]
        cands.append((first_st, code))
    cands.sort()
    pick = np.linspace(0, len(cands) - 1, n_sample).round().astype(int)
    sample = [cands[i][1] for i in sorted(set(pick))]

    rows = []
    lim_ratio_days, lim_ratio_ok = 0, 0
    for code in sample:
        starts, is_st, names = name_index[code]
        d = pd.read_parquet(DAILY_DIR / f"{code}.parquet",
                            columns=["trade_date", "pct_chg", "pre_close"])
        d = d.drop_duplicates("trade_date").sort_values("trade_date")
        td = d["trade_date"].to_numpy().astype("datetime64[ns]")
        pct = np.abs(d["pct_chg"].to_numpy(np.float64))
        i = np.searchsorted(starts, td, side="right") - 1
        st_day = np.zeros(len(td), bool)
        ok = i >= 0
        st_day[ok] = is_st[i[ok]]
        # 段边界: ST 段首日的行情日索引
        prev = np.roll(st_day, 1); prev[0] = False
        seg_start = np.nonzero(st_day & ~prev)[0]
        nxt = np.roll(st_day, -1); nxt[-1] = False
        seg_end = np.nonzero(st_day & ~nxt)[0]
        n_st = int(st_day.sum())
        viol = int((pct[st_day] > 5.5).sum())
        bind = int((pct[st_day] >= 4.8).sum())
        nonst_gt = int((pct[~st_day] > 5.5).sum())
        # 摘帽后首个交易日(== 下一名称 start 日当天或之后的首个行情日)
        unseal_moves = []
        for e in seg_end:
            if e + 1 < len(td):
                unseal_moves.append(float(pct[e + 1]))
        unseal_gt = int(sum(x > 5.5 for x in unseal_moves))
        # e: 2007 后 ST 日的 up_limit/pre_close 精确比
        st_dates_post07 = td[st_day & (td >= np.datetime64(LIMIT_DATA_START))]
        if len(st_dates_post07):
            pc = dict(zip(td, d["pre_close"].to_numpy()))
            for dstr_arr in np.array_split(
                    pd.Series(st_dates_post07).dt.strftime("%Y%m%d").unique(),
                    max(1, (len(pd.Series(st_dates_post07).dt.strftime("%Y%m%d").unique()) + 499) // 500)):
                for dstr in dstr_arr:
                    path = LIMIT_DIR / f"{dstr}.parquet"
                    if not path.exists():
                        continue
                    lim = pd.read_parquet(path)
                    r = lim[lim["ts_code"] == code]
                    if len(r):
                        ratio = float(r["up_limit"].iloc[0]) / float(pc[np.datetime64(pd.Timestamp(dstr))])
                        lim_ratio_days += 1
                        lim_ratio_ok += int(1.045 <= ratio <= 1.055)
        rows.append({
            "ts_code": code, "n_st_intervals": int(len(seg_start)),
            "st_days": n_st, "viol_gt_5p5": viol, "bind_ge_4p8": bind,
            "nonst_days_gt_5p5": nonst_gt, "unseal_first_days": len(unseal_moves),
            "unseal_first_day_gt_5p5": unseal_gt,
            "st_start_dates": ",".join(str(pd.Timestamp(td[s]).date()) for s in seg_start),
        })
    ev = pd.DataFrame(rows)
    return ev, sample, lim_ratio_days, lim_ratio_ok


# ================================================================ 主流程
def main():
    t0 = time.time()
    log("[阶段2] 脚本启动 run_pool_cleaning(三过滤器+基线重算)")
    cal = load_calendar()
    name_index = build_name_index()
    log(f"[阶段2] 交易日历 {len(cal)} 天 ({pd.Timestamp(cal[0]).date()}~{pd.Timestamp(cal[-1]).date()}), "
        f"namechange 索引 {len(name_index)} 只")

    all_rows, test_rows = [], []
    for pool in POOLS:
        log(f"[阶段3] 池={pool} 装配事件/标签/段")
        df = build_flags(pool)
        d1, d1_ok = next_trade_days(cal, df["date"].to_numpy())
        df["d1"] = d1
        # ST 过滤
        df["f_st"] = st_flags(df["ts_code"].to_numpy(), df["date"].to_numpy(), name_index)
        # d1 行情
        log(f"[阶段3] 池={pool} 读取 d1 行情")
        df = attach_d1_bars(df)
        # 停牌/无成交过滤
        df["f_suspend"] = (~df["bar_on_d1"] & df["d1"].notna()) | \
                          (df["bar_on_d1"] & ((df["vol_d1"] <= 0) | df["vol_d1"].isna()
                                              | df["open_d1"].isna() | (df["open_d1"] <= 0)))
        # 一字涨停过滤
        log(f"[阶段3] 池={pool} 读取 stk_limit 并判定一字涨停开盘")
        df = attach_up_limit(df)
        has_limit_data = df["d1"] >= LIMIT_DATA_START
        df["limitup_evaluable"] = has_limit_data & df["bar_on_d1"] & df["up_limit_d1"].notna()
        df["f_limitup"] = df["limitup_evaluable"] & \
                          (df["open_d1"] >= df["up_limit_d1"] - LIMIT_TOL - 1e-9)
        df["f_any"] = df["f_st"] | df["f_suspend"] | df["f_limitup"]

        # 核查: 标签机器对停牌的处理(个股次 bar 被顺延的规模)
        gap = df["own_next_after_d1"] & df["d1"].notna()
        log(f"[阶段3] 池={pool} 标签机器停牌顺延事件数={int(gap.sum())} "
            f"(全部被 f_suspend 覆盖={bool((df.loc[gap, 'f_suspend']).all()) if gap.any() else 'NA'})")

        # 过滤标记产物
        out_cols = ["ts_code", "date", "seg", "d1", "f_st", "f_suspend", "f_limitup",
                    "limitup_evaluable", "f_any"]
        df[out_cols].to_parquet(OUT_DIR / f"excluded_events_{pool}.parquet", index=False)
        log(f"[阶段4] 池={pool} 过滤标记落盘 excluded_events_{pool}.parquet")

        # 基线重算(仅 train/val)
        filters = {
            "raw": np.zeros(len(df), bool),
            "excl_st": df["f_st"].to_numpy(),
            "excl_limitup": df["f_limitup"].to_numpy(),
            "excl_suspend": df["f_suspend"].to_numpy(),
            "excl_combined": df["f_any"].to_numpy(),
        }
        for r in baseline_table(df, filters):
            r["pool"] = pool
            all_rows.append(r)
        # 口径自检: raw val 复现 race_rerun_v2 复核值
        exp = EXPECTED_VAL_BASELINE[pool]
        got = [r for r in all_rows if r["pool"] == pool and r["segment"] == "val"
               and r["filter_set"] == "raw"][0]
        assert abs(got["day_weighted_hit"] - exp["dayw"]) < 5e-4, (pool, got)
        assert got["n_events"] == exp["events"] and got["n_days"] == exp["days"], (pool, got)
        log(f"[阶段5] 池={pool} raw val 口径自检通过: 日加权={got['day_weighted_hit']:.6f} "
            f"事件加权={got['event_weighted_hit']:.6f} 日={got['n_days']} 事件={got['n_events']}")

        # 测试段/其他段: 只报计数
        for seg in ("test", "embargo", "pre2001"):
            sm = (df["seg"] == seg).to_numpy()
            sub = df.loc[sm]
            pre07 = sm & df["d1"].notna().to_numpy() & (df["d1"] < LIMIT_DATA_START).to_numpy()
            test_rows.append({
                "pool": pool, "segment": seg, "n_events": int(len(sub)),
                "f_st": int(sub["f_st"].sum()), "f_suspend": int(sub["f_suspend"].sum()),
                "f_limitup": int(sub["f_limitup"].sum()),
                "limitup_not_evaluable_pre2007": int(pre07.sum()),
                "d1_beyond_data_end": int((sm & df["d1"].isna().to_numpy()).sum()),
                "f_any": int(sub["f_any"].sum()),
            })

    bt = pd.DataFrame(all_rows)[["pool", "segment", "filter_set", "n_events", "n_days",
                                 "event_weighted_hit", "day_weighted_hit"]]
    bt.to_csv(OUT_DIR / "baseline_recalc.csv", index=False)
    tc = pd.DataFrame(test_rows)
    tc.to_csv(OUT_DIR / "test_segment_filter_counts.csv", index=False)
    log("[阶段5] 基线重算表落盘 baseline_recalc.csv / test_segment_filter_counts.csv")

    # ST 交叉验证
    log("[阶段6] ST 区间交叉验证(抽样 20 只主板 ST 股)")
    ev, sample, lr_days, lr_ok = st_cross_validation(name_index, cal)
    ev.to_csv(OUT_DIR / "st_crossval_evidence.csv", index=False)
    tot_viol = int(ev["viol_gt_5p5"].sum())
    log(f"[阶段6] 交叉验证完成: {len(ev)} 只, 段内>5.5%违例天数合计={tot_viol}, "
        f"摘帽首日>5.5%次数={int(ev['unseal_first_day_gt_5p5'].sum())}/{int(ev['unseal_first_days'].sum())}, "
        f"2007后 ST 日 up/pre∈[1.045,1.055] {lr_ok}/{lr_days}")

    log(f"[阶段7] 全部完成 耗时{time.time()-t0:.0f}s (报告另由本目录 md 汇总)")
    return bt, tc, ev, (lr_days, lr_ok)


if __name__ == "__main__":
    main()
