#!/usr/bin/env python3
"""背离信号稳定性分析(复用 divergence_event_study 的产物,不重跑检测)。

前提: earliest_detector.py 与 src/divergence_detector.py (V1) 算法等价(见报告第 5 节),
因此直接复用上一轮全量事件研究结果(events.parquet / returns_wide.parquet)。

产出:
  a. 分年度: 每年事件数、各 horizon×口径 背离胜率 - C1 胜率 的超额,及正超额年份占比。
  b. 分市场阶段: 全样本等权日收益构造市场代理,120 日滚动收益 >+10% 为上涨、<-10% 为下跌、
     其余为震荡;各阶段内背离超额(vs C1 / vs C2)。
  c. 信号频率: 每年每股平均触发次数。
结果追加到 reports/divergence_event_study.md 第 5 节,并落盘 stability_*.csv。
"""
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import divergence_event_study as des  # noqa: E402

RAW = des.RAW_DIR
SEED = des.SEED
HORIZONS = des.HORIZONS
MIN_LEN = des.MIN_LEN
REGIME_WIN = 120      # 市场阶段滚动窗口(交易日)
REGIME_TH = 0.10      # 上涨/下跌阈值
MIN_YEAR_N = 100      # 年度统计最小样本


def main():
    t0 = time.time()
    files = sorted(des.DATA_DIR.glob("*.parquet"))
    print("重载全量股票(确定性重建 C1 日期)...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(des.load_stock, [str(f) for f in files], chunksize=16))
    stocks = [r for r in results if "error" not in r]
    print(f"有效股票 {len(stocks)} ({time.time()-t0:.0f}s)", flush=True)

    events = pd.read_parquet(RAW / "events.parquet")
    wide = pd.read_parquet(RAW / "returns_wide.parquet")
    ev_by_sym = {s: g["t_idx"].to_numpy() for s, g in events.groupby("ts_code")}

    # ---- 确定性重建 C1 采样(与 divergence_event_study 完全相同的种子与顺序)
    c1_date, c1_stock = [], []
    c1_check_ra, c1_check_rb = [], []
    for st in stocks:
        t_idx = ev_by_sym.get(st["symbol"])
        if t_idx is None or len(t_idx) == 0:
            continue
        n = len(st["cf"])
        mask = np.ones(n, dtype=bool)
        mask[: MIN_LEN - 1] = False
        mask[t_idx] = False
        days = np.nonzero(mask)[0]
        if len(days) == 0:
            continue
        rng = np.random.default_rng(SEED + zlib.crc32(st["symbol"].encode()))
        pick = rng.choice(days, size=len(t_idx), replace=len(days) < len(t_idx))
        ra, rb = des.fwd_ret_matrix(st, pick)
        c1_check_ra.append(ra)
        c1_check_rb.append(rb)
        c1_date.append(st["dates"][pick])
        c1_stock.extend([st["symbol"]] * len(pick))
    c1_date = np.concatenate(c1_date)
    ra_cat = np.concatenate(c1_check_ra)
    # 校验:与 returns_wide 的 c1 块逐值一致 => 行对齐正确
    c1_wide = wide[wide["group"] == "c1"].reset_index(drop=True)
    assert len(c1_wide) == len(c1_date), (len(c1_wide), len(c1_date))
    for j, h in enumerate(HORIZONS):
        a = c1_wide[f"ret_a_h{h}"].to_numpy(np.float64)
        b = ra_cat[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[ok], b[ok], atol=1e-5), f"c1 ret_a_h{h} 不一致"
        assert (np.isnan(a) == np.isnan(b)).all(), f"c1 ret_a_h{h} NaN 掩码不一致"
    print(f"C1 确定性重建校验通过 (n={len(c1_date)}) ({time.time()-t0:.0f}s)", flush=True)

    div = wide[wide["group"] == "div"].reset_index(drop=True)
    c2 = wide[wide["group"] == "c2"].reset_index(drop=True)
    assert len(div) == len(events) == len(c2)
    ev_dates = events["date"].to_numpy("datetime64[ns]")
    ev_year = pd.DatetimeIndex(ev_dates).year.to_numpy()
    c1_year = pd.DatetimeIndex(c1_date).year.to_numpy()

    # ---- 市场代理: 全样本等权日收益
    all_dates = np.unique(np.concatenate([st["dates"] for st in stocks]))
    cnt = np.zeros(len(all_dates))
    ssum = np.zeros(len(all_dates))
    for st in stocks:
        g = np.searchsorted(all_dates, st["dates"])
        ret = st["cf"][1:] / st["cf"][:-1] - 1.0
        np.add.at(cnt, g[1:], 1)
        np.add.at(ssum, g[1:], ret)
    with np.errstate(invalid="ignore", divide="ignore"):
        mkt_ret = np.where(cnt > 0, ssum / cnt, 0.0)
    mkt_cf = np.cumprod(1.0 + mkt_ret)
    roll = np.full(len(all_dates), np.nan)
    roll[REGIME_WIN:] = mkt_cf[REGIME_WIN:] / mkt_cf[:-REGIME_WIN] - 1.0
    regime = np.where(roll > REGIME_TH, "up", np.where(roll < -REGIME_TH, "down", "side"))
    regime[:(REGIME_WIN + 1)] = "side"  # 窗口未走完,归入震荡(不参与亦可,占比小)
    ev_regime = regime[np.searchsorted(all_dates, ev_dates.astype("datetime64[ns]").astype("int64"))]
    c1_regime = regime[np.searchsorted(all_dates, c1_date.astype("datetime64[ns]").astype("int64"))]

    # ---- 每年活跃股票数与信号频率
    year_of_dates = pd.DatetimeIndex(all_dates).year.to_numpy()
    years = np.arange(year_of_dates.min(), year_of_dates.max() + 1)
    stock_years = set()
    for si, st in enumerate(stocks):
        ys = np.unique(pd.DatetimeIndex(st["dates"]).year.to_numpy())
        for y in ys:
            stock_years.add((st["symbol"], int(y)))
    active = pd.Series([sy for _, sy in stock_years]).value_counts().sort_index()

    # ---- 统计函数
    def win(x):
        x = x[np.isfinite(x)]
        return (np.mean(x > 0) * 100, len(x)) if len(x) else (np.nan, 0)

    # a. 分年度
    yearly_rows = []
    for y in years:
        dm = ev_year == y
        cm = c1_year == y
        row = dict(year=int(y), n_events=int(dm.sum()),
                   n_active=int(active.get(y, 0)))
        row["freq_per_stock"] = row["n_events"] / row["n_active"] if row["n_active"] else np.nan
        for entry, col in (("a", "ret_a"), ("b", "ret_b")):
            for j, h in enumerate(HORIZONS):
                dcol = div[f"{col}_h{h}"].to_numpy(np.float64)[dm]
                ccol = c1_wide[f"{col}_h{h}"].to_numpy(np.float64)[cm]
                dw, dn = win(dcol)
                cw, cn = win(ccol)
                row[f"{entry}{h}_div"] = dw
                row[f"{entry}{h}_c1"] = cw
                row[f"{entry}{h}_ex"] = dw - cw if dn >= MIN_YEAR_N and cn >= MIN_YEAR_N else np.nan
        yearly_rows.append(row)
    ydf = pd.DataFrame(yearly_rows)
    ydf.to_csv(RAW / "stability_yearly.csv", index=False)

    # 正超额年份占比(仅统计样本足够的年份)
    pos_rows = []
    for entry in ("a", "b"):
        for h in HORIZONS:
            ex = ydf[f"{entry}{h}_ex"].dropna()
            pos_rows.append(dict(entry=entry, h=h, n_years=len(ex),
                                 pos_share=float((ex > 0).mean()) if len(ex) else np.nan,
                                 mean_ex=float(ex.mean()) if len(ex) else np.nan,
                                 min_ex=float(ex.min()) if len(ex) else np.nan,
                                 max_ex=float(ex.max()) if len(ex) else np.nan))
    posdf = pd.DataFrame(pos_rows)

    # b. 分市场阶段
    reg_rows = []
    for rg in ("up", "down", "side"):
        dm = ev_regime == rg
        cm = c1_regime == rg
        for entry, col in (("a", "ret_a"), ("b", "ret_b")):
            for j, h in enumerate(HORIZONS):
                dv = div[f"{col}_h{h}"].to_numpy(np.float64)[dm]
                c1v = c1_wide[f"{col}_h{h}"].to_numpy(np.float64)[cm]
                c2v = c2[f"{col}_h{h}"].to_numpy(np.float64)[dm]  # c2 与事件同日,同阶段
                dw, dn = win(dv)
                c1w, _ = win(c1v)
                c2w, _ = win(c2v)
                reg_rows.append(dict(regime=rg, entry=entry, h=h, n=dn,
                                     div_win=dw, c1_win=c1w, ex_c1=dw - c1w,
                                     c2_win=c2w, ex_c2=dw - c2w))
    rdf = pd.DataFrame(reg_rows)
    rdf.to_csv(RAW / "stability_regime.csv", index=False)

    # ---- 追加报告
    L = []
    L.append("\n## 5. 最早版本同一性、正确性与稳定性(追加)\n")
    L.append("### 5.1 算法同一性(earliest_detector.py vs V1)\n")
    L.append("- 逐行 diff 两版本类体:唯一差异是 V1 的输出字典多一个 `is_quick_divergence` 字段"
             "(V1 L183;且因 L146 间隔<5 即跳过,该字段恒为 0,属死字段);导入不同"
             "(earliest 依赖未随附的 `comm_fun_v2`,V1 依赖 `comm_fun.model_config`),"
             "但二者在类方法内均未被使用。")
    L.append("- 参数/条件/边界/调用语义完全一致:window=6/step=3、lookback_lows=2、min_macd_change=0.001、"
             "间隔<5 跳过、NaN 跳过、>=100 行门槛、timestamp==当日过滤、逐日截断调用语义。")
    L.append("- 实证: stub `comm_fun_v2` 后,用 earliest 类在 3 只股票 2108 个截断日上对拍本研究的因果模拟,"
             "0 mismatches。**结论: 两版本算法等价,上一轮全量有效性结论直接适用于最早版本,不重跑。**\n")
    L.append("### 5.2 最早版本正确性(生产逐日截断调用语义下)\n")
    L.append("- `_find_close_lows`(earliest L199-219): 与 V1 相同,截断到 T 时含 T 的窗口只用 ≤T 数据,"
             "无 T 后信息;对全历史一次性运行则有 +5 根 K 线泄漏(与 V1 审计结论相同)。")
    L.append("- `detect_daily_divergence` L37 的 >=100 行门槛: 逐日截断下,每只股票上市最初 99 个交易日"
             "不会产生任何信号(覆盖性限制,非泄漏);MACD 12/26/9 的 ~33 根 NaN 预热期被该门槛完全覆盖。")
    L.append("- L48-49 当日过滤、L83-93 量能特征(仅用两低点当日 volume)、L95-123 基础特征(仅用背离点自身)"
             "均无 T 后信息。**结论: 生产语义下正确、无泄漏;注意输入需含 `volume` 列(原始 parquet 为 `vol`,"
             "调用方需改名)。**\n")
    L.append("### 5.3 分年度稳定性(超额 = 背离胜率 - 同股随机日 C1 胜率, 百分点)\n")
    L.append(f"- 市场代理: 全样本等权日收益; 年度统计要求该年事件与 C1 有效样本均 >= {MIN_YEAR_N}。")
    L.append("- 信号频率(每年每股平均触发次数)逐年: 见下表 freq 列; 明细 `stability_yearly.csv`。\n")
    for entry, title in (("a", "口径 (a) T收盘→T+h收盘"), ("b", "口径 (b) T+1开盘→T+1+h收盘")):
        L.append(f"**{title}: 各 horizon 正超额年份占比 / 平均超额(pp)**\n")
        L.append("| h | 有效年数 | 正超额年份占比 | 平均超额pp | 最差pp | 最好pp |")
        L.append("|---|---|---|---|---|---|")
        for h in HORIZONS:
            r = posdf[(posdf.entry == entry) & (posdf.h == h)].iloc[0]
            L.append(f"| {h} | {r['n_years']} | {r['pos_share']*100:.0f}% | "
                     f"{r['mean_ex']:+.2f} | {r['min_ex']:+.2f} | {r['max_ex']:+.2f} |")
        L.append("")
    L.append("逐年明细(事件数/频率/各 horizon 超额):\n")
    L.append("| 年份 | 事件数 | 活跃股 | freq | a5 | a10 | a15 | a20 | a30 | a60 | b5 | b10 | b15 | b20 | b30 | b60 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ydf.iterrows():
        def f(c):
            v = r.get(c)
            return f"{v:+.1f}" if pd.notna(v) else "-"
        L.append(f"| {int(r['year'])} | {int(r['n_events'])} | {int(r['n_active'])} | "
                 f"{r['freq_per_stock']:.1f} | " + " | ".join(f(f"{e}{h}_ex") for e in "ab" for h in (5, 10, 15, 20, 30, 60)) + " |")
    L.append("")
    L.append("### 5.4 分市场阶段(等权代理 120 日滚动收益 ±10% 划分)\n")
    L.append("| 阶段 | 口径 | h | n | 背离胜率% | C1胜率% | Δwin(C1)pp | C2胜率% | Δwin(C2)pp |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rname = {"up": "上涨", "down": "下跌", "side": "震荡"}
    for rg in ("up", "down", "side"):
        for entry in ("a", "b"):
            for h in HORIZONS:
                r = rdf[(rdf.regime == rg) & (rdf.entry == entry) & (rdf.h == h)].iloc[0]
                L.append(f"| {rname[rg]} | {entry} | {h} | {r['n']} | {r['div_win']:.2f} | "
                         f"{r['c1_win']:.2f} | {r['ex_c1']:+.2f} | {r['c2_win']:.2f} | {r['ex_c2']:+.2f} |")
    L.append("")
    md_path = des.OUT_DIR / "divergence_event_study.md"
    with open(md_path, "a") as f:
        f.write("\n".join(L) + "\n")
    print(f"完成 ({time.time()-t0:.0f}s),已追加到 {md_path}", flush=True)


if __name__ == "__main__":
    main()
