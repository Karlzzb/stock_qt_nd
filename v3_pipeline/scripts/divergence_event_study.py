#!/usr/bin/env python3
"""MACD 底背离事件研究(从零实现,严格无泄漏)。

背离定义: src/divergence_detector.py 的 DivergenceDetector (V1)。
审计结论(详见报告):
  - V1 的 _find_close_lows 用 6-bar 滑动窗口(步长 3,锚定索引 0)取窗口最小值作为低点。
    若对全历史一次性运行,低点 g 的确认会用到 g 之后最多 5 根 K 线 => 未来函数。
  - 生产用法(feature_pipeline.py:670-672)把输入截断到目标日 T,且只保留 timestamp==T 的
    背离点,因此生产信号本身不含 T 之后的信息;但信号日 T 必须本身就是"当前低点"。
  - 本脚本因此实现【逐日截断因果模拟】:对每一天 t,只用 <=t 的数据,精确复现 V1 在
    生产截断语义下的检测结果(含其网格相位怪癖),并用真实 DivergenceDetector 类抽样对拍。

前向收益口径(用 pct_chg 链式累乘,规避除权除息的价格跳变):
  (a) 信号日 T 收盘买入 -> T+h 收盘卖出:  ret = prod(1+pct[T+1..T+h]/100) - 1
  (b) T+1 开盘买入 -> T+1+h 收盘卖出:     ret = (C[T+1]/O[T+1]) * prod(1+pct[T+2..T+1+h]/100) - 1
"""
import argparse
import glob
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "stock_data" / "daily"
OUT_DIR = REPO / "v3_pipeline" / "reports"
RAW_DIR = OUT_DIR / "divergence_event_study"

HORIZONS = [3, 5, 10, 15, 20, 25, 30, 45, 60]
WINDOW = 6          # V1 _find_close_lows window_size
STEP = 3            # V1 step = window_size // 2
MIN_MACD_CHANGE = 0.001
MIN_LEN = 100       # V1 要求输入长度 >= 100 => 信号日 t >= 99
SEED = 42
N_COMBO = len(HORIZONS) * 2  # Bonferroni 校正因子: 18 个 horizon×口径组合


# ---------------------------------------------------------------- 检测核心
def simulate_events_idx(close: np.ndarray, macd: np.ndarray):
    """逐日截断因果模拟 V1 检测。返回 [(t, compare_rank, prev_idx), ...]。

    推导(与 V1 在生产截断语义下严格等价):
    - 窗口为 6-bar、步长 3、锚定索引 0。截断到第 t 日时,含 t 的窗口有两个:
      末尾偏窗 [s, t] (s = 3*floor(t/3),长度 1/2/3) 和倒数第二窗 [s-3, t] (长度 4/5/6,
      仅 t≡2 (mod 3) 时为完整 6-bar 窗口)。
    - t 为"当前低点"的充要条件(t 是任一含 t 窗口的最小值,首次出现口径):
        t % 3 == 0: 恒成立(末尾偏窗仅 [t] 一根,V1 怪癖,原样保留);
        t % 3 == 1: close[t] < close[t-1];
        t % 3 == 2: close[t] < min(close[t-2..t-1])。
    - 当 t 为当前低点时,末尾偏窗最小值必为 t;但倒数第二窗的最小值 q(t) 可能 < t,
      此时 q(t) 也是低点且排在 t 之前,必须计入"前低"序列。
    - 前低序列 = (已走完窗口的去重最小值索引) ∪ {q(t) if q(t) < t},取最后两个比较;
      间隔 < 5 跳过;MACD NaN 跳过;close[t] < close[prev] 且
      macd[t] > macd[prev] + 0.001 记为底背离;需至少 2 个前低(V1 L127)。
    """
    n = len(close)
    if n < MIN_LEN + 1:
        return []
    k_full = (n - WINDOW) // STEP + 1
    sw = np.lib.stride_tricks.sliding_window_view(close, WINDOW)[::STEP][:k_full]
    min_idx = np.argmin(sw, axis=1) + STEP * np.arange(k_full)
    lows, first_k = np.unique(min_idx, return_index=True)
    avail = STEP * first_k + (WINDOW - 1)  # 每个低点首次可知的日期(窗口走完日)

    t_arr = np.arange(n)
    phase = t_arr % STEP
    is_low = np.zeros(n, dtype=bool)
    is_low[phase == 0] = True
    p1 = phase == 1
    is_low[p1] = close[p1] < close[t_arr[p1] - 1]
    p2 = (phase == 2) & (t_arr >= 2)
    is_low[p2] = close[p2] < np.minimum(close[t_arr[p2] - 1], close[t_arr[p2] - 2])

    # q(t): 倒数第二窗 [t-(phase+3), t] 的最小值索引(首次出现)
    q = np.full(n, -1)
    for ph, w in ((0, 3), (1, 4), (2, 5)):
        tt = t_arr[(phase == ph) & (t_arr >= w)]
        if len(tt) == 0:
            continue
        swq = np.lib.stride_tricks.sliding_window_view(close, w + 1)  # swq[i]=close[i..i+w]
        q[tt] = tt - w + np.argmin(swq[tt - w], axis=1)

    macd_nan = np.isnan(macd)
    events = []
    for t in t_arr[is_low & (t_arr >= MIN_LEN - 1)]:
        m = int(np.searchsorted(avail, t, side="right"))
        if m >= 1 and lows[m - 1] == t:
            m -= 1  # t 自身(相位2时来自完整窗口)不计入前低
        qt = int(q[t])
        cand = lows[max(0, m - 2):m].tolist()
        if 0 <= qt < t:
            pos = int(np.searchsorted(lows, qt))
            if not (pos < m and lows[pos] == qt):
                cand.append(qt)
                cand.sort()
        if len(cand) < 2:  # V1: len(close_lows) <= 2 直接返回
            continue
        for rank, prev in ((1, int(cand[-1])), (2, int(cand[-2]))):
            if t - prev < 5:
                continue
            if macd_nan[t] or macd_nan[prev]:
                continue
            if close[t] < close[prev] and macd[t] > macd[prev] + MIN_MACD_CHANGE:
                events.append((int(t), rank, prev))
                break  # 同一 (股票,日期) 只记一个事件
    return events


# ---------------------------------------------------------------- 数据加载
def load_stock(path):
    """读取单股 parquet,计算 MACD,跑因果模拟。返回紧凑字典。"""
    try:
        df = pd.read_parquet(path, columns=["trade_date", "open", "close", "pct_chg"])
    except Exception as e:
        return {"symbol": Path(path).stem, "error": str(e)}
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    if len(df) < 30:
        return {"symbol": Path(path).stem, "error": "too_short"}
    close = df["close"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    pct = df["pct_chg"].to_numpy(np.float64)
    dates = df["trade_date"].to_numpy("datetime64[ns]").astype("int64")
    macd = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0]
    events = simulate_events_idx(close, macd)
    # pct_chg 链式累乘因子(全历史,因果:仅用当日及以前 pct)
    pct_safe = np.where(np.isfinite(pct), pct, 0.0)
    cf = np.cumprod(1.0 + pct_safe / 100.0)
    return {
        "symbol": Path(path).stem,
        "dates": dates,
        "open": open_,
        "close": close,
        "cf": cf,
        "events": events,
    }


def fwd_ret_matrix(st, t_arr):
    """对信号日数组 t_arr,返回 (len, 9) 的 ret_a / ret_b;数据不足为 NaN。"""
    cf, close, open_, n = st["cf"], st["close"], st["open"], len(st["cf"])
    t_arr = np.asarray(t_arr, dtype=np.int64)
    ra = np.full((len(t_arr), len(HORIZONS)), np.nan, dtype=np.float64)
    rb = np.full_like(ra, np.nan)
    for j, h in enumerate(HORIZONS):
        oka = t_arr + h <= n - 1
        ia = t_arr[oka]
        ra[oka, j] = cf[ia + h] / cf[ia] - 1.0
        t1 = np.minimum(t_arr + 1, n - 1)
        okb = (t_arr + 1 + h <= n - 1) & (open_[t1] > 0)
        ib = t_arr[okb]
        rb[okb, j] = (close[ib + 1] / open_[ib + 1]) * (cf[ib + 1 + h] / cf[ib + 1]) - 1.0
    return ra, rb


# ---------------------------------------------------------------- 对拍验证
def validate_against_v1(stocks, n_check_days=400, seed=SEED):
    """用真实 DivergenceDetector 在逐日截断数据上对拍因果模拟结果。"""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    from divergence_detector import DivergenceDetector  # noqa: E402

    det = DivergenceDetector()
    rng = np.random.default_rng(seed)
    total_mismatch = 0
    for st in stocks:
        close, n = st["close"], len(st["close"])
        macd = talib.MACD(close, 12, 26, 9)[0]
        idx = pd.to_datetime(st["dates"])
        df = pd.DataFrame(
            {"close": close, "macd": macd, "volume": np.ones(n)}, index=idx
        )
        sim_days = {e[0] for e in st["events"]}
        cand = np.arange(MIN_LEN - 1, n)
        sim_arr = np.fromiter(sim_days, int, len(sim_days)) if sim_days else np.array([], int)
        non_evt = np.setdiff1d(cand, sim_arr)
        check = sorted(sim_days) + rng.choice(non_evt, size=min(n_check_days, len(non_evt)), replace=False).tolist()
        mism = 0
        for t in check:
            res = det.detect_daily_divergence(df.iloc[: t + 1], st["symbol"], idx[t].date())
            v1_hit = len(res) > 0
            sim_hit = t in sim_days
            if v1_hit != sim_hit:
                mism += 1
                print(f"  MISMATCH {st['symbol']} t={t} date={idx[t].date()} v1={v1_hit} sim={sim_hit}")
        total_mismatch += mism
        print(f"  {st['symbol']}: checked {len(check)} days ({len(sim_days)} sim events), mismatches={mism}")
    return total_mismatch


# ---------------------------------------------------------------- 统计
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - half, c + half)


def prop_ztest(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan
    z = ((k1 / n1) - (k2 / n2)) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def group_stats(x):
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, win=np.nan, mean=np.nan, median=np.nan)
    return dict(n=n, win=float(np.mean(x > 0)), mean=float(np.mean(x)), median=float(np.median(x)))


def compare(d, c):
    """d/c 为收益数组。返回检验 p 值。"""
    d = d[np.isfinite(d)]
    c = c[np.isfinite(c)]
    out = {}
    if len(d) >= 10 and len(c) >= 10:
        out["p_t"] = float(stats.ttest_ind(d, c, equal_var=False).pvalue)
        out["p_mw"] = float(stats.mannwhitneyu(d, c, alternative="two-sided").pvalue)
        out["p_prop"] = float(prop_ztest(int((d > 0).sum()), len(d), int((c > 0).sum()), len(c)))
    else:
        out["p_t"] = out["p_mw"] = out["p_prop"] = np.nan
    return out


# ---------------------------------------------------------------- 报告
AUDIT_MD = """## 1. 实现审计(src/divergence_detector.py V1)

**① 低点识别是否用到未来数据?**
- `_find_close_lows`(L194-214):以 window_size=6、step=3、锚定索引 0 的滑动窗口取每个窗口的
  收盘价最小值(L206 `idxmin`)作为"局部低点"。若对**全历史一次性运行**,低点 g 是否成立由包含 g 的
  窗口 `[3k, 3k+6)` 决定,会用上 g 之后最多 5 根 K 线 —— **存在未来函数**。
- 但生产调用(`src/feature_pipeline.py:670-672`)传入的是截断到目标日 T 的数据,且
  `detect_daily_divergence` 只保留 `timestamp == T` 的背离点(L42-43)。此时含 T 的窗口只含 ≤T 的数据,
  **生产信号不含 T 之后的信息**;代价是信号日 T 必须自身就是"当前低点"。
- V1 网格怪癖(原样保留):T≡0 (mod 3) 时末尾偏窗只有 1 根 K 线,T 恒被判为低点;
  T≡1 (mod 3) 时只需 `close[T] < close[T-1]`;T≡2 (mod 3) 时需 `close[T]` 低于前 2 根。
  因此事件非常密集(约 1/3 的交易日是候选低点)。

**② 背离判定与文档口径是否一致?** 一致。
- 价格创新低:L157 `current_close < prev_close`;MACD 未新低:L158 `current_macd > prev_macd + 0.001`
  (阈值 L133 `min_macd_change=0.001`,为 DIF 绝对值阈值)。
- 注意口径细节:仅与最近 2 个低点比较(L132 `lookback_lows=2`,L142),非"历史新低";
  两低点间隔 < 5 根 K 线跳过(L146);MACD 为 NaN 跳过(L153);需至少 2 个前低(L127)。

**③ 是否存在 T 日之后的信息进入 T 日信号?**
- 生产截断语义下:**无**。`_check_volume_signal`(L77-87)只用两低点当日成交量;
  `_calculate_basic_features_historical`(L89-117)只用背离点自身字段。
- 风险点:若绕过生产截断、直接对全历史跑一次检测再把低点日期当信号日,则引入最多 +5 根 K 线的泄漏。
  本研究不这么做,而是实现逐日截断的因果模拟(等价于生产语义),并用真实 V1 类抽样对拍验证一致性。
"""


def fmt_pct(x, nd=2):
    return "NA" if x is None or not np.isfinite(x) else f"{x*100:.{nd}f}"


def build_report(meta, rows, verdict_rows):
    L = []
    L.append("# MACD 底背离事件研究(从零,无泄漏)\n")
    L.append(f"- 生成时间: {meta['timestamp']}")
    L.append(f"- 数据: `{DATA_DIR}` 共 {meta['n_stocks']} 只股票, "
             f"区间 {meta['date_min']} ~ {meta['date_max']}, 剔除无数据文件 {meta['n_skipped']} 个")
    L.append(f"- 背离定义: V1 (talib MACD 12/26/9 的 DIF 线; 6-bar 窗口低点; 与最近 2 个前低比较, "
             f"间隔>=5; close 新低且 DIF 高出前低 >0.001)")
    L.append(f"- 检测方式: 逐日截断因果模拟(等价生产语义), 与真实 V1 类对拍 mismatches={meta['val_mismatch']}")
    L.append(f"- 背离事件数: {meta['n_events']} (涉及 {meta['n_event_stocks']} 只股票)")
    L.append(f"- 收益口径: pct_chg 链式累乘(含分红再投资口径,避免除权跳变); "
             f"(a) T 收盘->T+h 收盘; (b) T+1 开盘->T+1+h 收盘")
    L.append(f"- 对照1: 同股随机非背离交易日(每股等量); 对照2: 同日随机非背离股票(每事件 1 只)")
    L.append(f"- 随机种子: {SEED}; Bonferroni 校正因子: {N_COMBO} (9 horizons × 2 口径), "
             f"按 (对照×检验类型) 族内校正, 校正后 p<0.05 记显著")
    L.append(f"- 数据不足 h 日剔除计数: events {meta['drop_events']}, "
             f"对照1 {meta['drop_c1']}, 对照2 {meta['drop_c2']}\n")
    L.append(AUDIT_MD)
    L.append("\n## 2. 主结果(每个 horizon × 口径)\n")
    L.append("收益单位 %; 胜率为收益>0 占比; Δwin=背离胜率-对照胜率(百分点); "
             "p_t=Welch t 检验, p_mw=Mann-Whitney, p_prop=两比例 z 检验(均为与对照比较); "
             "B 列 = Bonferroni 校正后仍显著的对照/检验组合。\n")
    for entry, title in (("a", "口径 (a): T 收盘买入 → T+h 收盘卖出"),
                         ("b", "口径 (b): T+1 开盘买入 → T+1+h 收盘卖出")):
        L.append(f"### {title}\n")
        L.append("| h | n | 背离胜率% | 背离均% | 背离中位% | C1胜率% | C1均% | C2胜率% | C2均% | "
                 "Δwin(C1) | Δwin(C2) | p_t(C1) | p_mw(C1) | p_prop(C1) | p_t(C2) | p_mw(C2) | p_prop(C2) | B |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["entry"] != entry:
                continue
            sigs = []
            for ctrl in ("c1", "c2"):
                for test in ("p_t", "p_mw", "p_prop"):
                    p = r[ctrl][test]
                    if np.isfinite(p) and p * N_COMBO < 0.05:
                        sigs.append(f"{ctrl}:{test[2:]}")
            dw1 = r['d']['win'] - r['c1']['win'] if np.isfinite(r['d']['win']) and np.isfinite(r['c1']['win']) else np.nan
            dw2 = r['d']['win'] - r['c2']['win'] if np.isfinite(r['d']['win']) and np.isfinite(r['c2']['win']) else np.nan
            L.append(
                f"| {r['h']} | {r['d']['n']} | {fmt_pct(r['d']['win'])} | {fmt_pct(r['d']['mean'])} | "
                f"{fmt_pct(r['d']['median'])} | {fmt_pct(r['c1']['win'])} | {fmt_pct(r['c1']['mean'])} | "
                f"{fmt_pct(r['c2']['win'])} | {fmt_pct(r['c2']['mean'])} | "
                f"{fmt_pct(dw1)} | {fmt_pct(dw2)} | "
                + " | ".join(f"{r[c][t]:.4g}" if np.isfinite(r[c][t]) else "NA"
                             for c in ("c1", "c2") for t in ("p_t", "p_mw", "p_prop"))
                + f" | {','.join(sigs) if sigs else '-'} |"
            )
        L.append("")
    L.append("## 3. 对用户先验的裁决专用表(10d / 15d 胜率)\n")
    L.append("| 口径 | h | n | 背离胜率% | Wilson 95% CI | C1胜率% | C2胜率% | 先验(>55%)是否成立 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in verdict_rows:
        L.append(f"| {v['entry']} | {v['h']} | {v['n']} | {fmt_pct(v['win'])} | "
                 f"[{fmt_pct(v['lo'])}, {fmt_pct(v['hi'])}] | {fmt_pct(v['c1win'])} | {fmt_pct(v['c2win'])} | "
                 f"{v['verdict']} |")
    L.append("")
    L.append("## 4. 局限与说明\n")
    L.append("- 未考虑涨跌停无法成交、停牌、滑点与手续费;T+1 开盘买入口径假设 T+1 可成交。")
    L.append("- 同一股票相邻事件的前向收益区间相互重叠,事件间不完全独立,t 检验的独立性假设近似成立。")
    L.append("- 对照2 的可选池为当日有数据的所有非背离股票(含已退市股,退市前数据正常参与)。")
    L.append("- 原始结果: `v3_pipeline/reports/divergence_event_study/` 下 events.parquet / "
             "returns_wide.parquet / stats.json。")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="只跑前 N 只股票(调试用)")
    ap.add_argument("--validate", type=int, default=3, help="对拍验证股票数(0 跳过)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    if args.sample:
        files = files[: args.sample]
    print(f"加载并检测 {len(files)} 只股票 ...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(load_stock, files, chunksize=16)):
            results.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    stocks = [r for r in results if "error" not in r]
    print(f"有效股票 {len(stocks)}, 剔除 {len(results)-len(stocks)}; 耗时 {time.time()-t0:.0f}s", flush=True)

    # ---- 对拍验证
    val_mismatch = -1
    if args.validate > 0:
        print(f"对拍验证({args.validate} 只股票,真实 V1 类 vs 因果模拟)...", flush=True)
        with_events = [s for s in stocks if s["events"]]
        pool = with_events[: args.validate]
        val_mismatch = validate_against_v1(pool)
        print(f"对拍 mismatches = {val_mismatch}", flush=True)

    # ---- 事件表与前向收益(向量化)
    ev_id, ev_stock, ev_t, ev_rank, ev_prev = [], [], [], [], []
    ev_ra, ev_rb = [], []
    c1_ra, c1_rb = [], []
    for si, st in enumerate(stocks):
        evs = st["events"]
        if not evs:
            continue
        t_idx = np.array([e[0] for e in evs], dtype=np.int64)
        ra, rb = fwd_ret_matrix(st, t_idx)
        base = len(ev_id)
        ev_id.extend(range(base, base + len(t_idx)))
        ev_stock.extend([si] * len(t_idx))
        ev_t.extend(t_idx.tolist())
        ev_rank.extend([e[1] for e in evs])
        ev_prev.extend([e[2] for e in evs])
        ev_ra.append(ra)
        ev_rb.append(rb)
        # 对照1: 同股随机日(等量)
        n = len(st["cf"])
        mask = np.ones(n, dtype=bool)
        mask[: MIN_LEN - 1] = False
        mask[t_idx] = False
        days = np.nonzero(mask)[0]
        if len(days) == 0:
            continue
        rng = np.random.default_rng(SEED + zlib.crc32(st["symbol"].encode()))
        pick = rng.choice(days, size=len(t_idx), replace=len(days) < len(t_idx))
        ra1, rb1 = fwd_ret_matrix(st, pick)
        c1_ra.append(ra1)
        c1_rb.append(rb1)
    n_events = len(ev_id)
    ev_ra = np.concatenate(ev_ra) if ev_ra else np.empty((0, len(HORIZONS)))
    ev_rb = np.concatenate(ev_rb) if ev_rb else np.empty((0, len(HORIZONS)))
    c1_ra = np.concatenate(c1_ra) if c1_ra else np.empty((0, len(HORIZONS)))
    c1_rb = np.concatenate(c1_rb) if c1_rb else np.empty((0, len(HORIZONS)))
    print(f"背离事件数: {n_events} ({time.time()-t0:.0f}s)", flush=True)

    events_df = pd.DataFrame(dict(
        event_id=np.array(ev_id, dtype=np.int64),
        stock_idx=np.array(ev_stock, dtype=np.int32),
        ts_code=[stocks[s]["symbol"] for s in ev_stock],
        t_idx=np.array(ev_t, dtype=np.int32),
        date=pd.to_datetime([stocks[s]["dates"][t] for s, t in zip(ev_stock, ev_t)]),
        compare_rank=np.array(ev_rank, dtype=np.int8),
        prev_idx=np.array(ev_prev, dtype=np.int32),
    ))

    # ---- 对照2: 同日随机股
    print("构造对照2(同日随机股)...", flush=True)
    all_dates = np.unique(np.concatenate([st["dates"] for st in stocks]))
    g_parts, s_parts, r_parts = [], [], []
    for si, st in enumerate(stocks):
        g = np.searchsorted(all_dates, st["dates"])
        g_parts.append(g.astype(np.int32))
        s_parts.append(np.full(len(g), si, np.int32))
        r_parts.append(np.arange(len(g), dtype=np.int32))
    G = np.concatenate(g_parts)
    S = np.concatenate(s_parts)
    R = np.concatenate(r_parts)
    order = np.argsort(G, kind="stable")
    G, S, R = G[order], S[order], R[order]
    bounds = np.searchsorted(G, np.arange(len(all_dates) + 1))
    # 每个事件日的背离股票集合(含事件股票自身 => 自动排除自身)
    evt_stocks_by_g = {}
    ev_g = np.empty(n_events, dtype=np.int32)
    for i, (si, t) in enumerate(zip(ev_stock, ev_t)):
        gv = int(np.searchsorted(all_dates, stocks[si]["dates"][t]))
        ev_g[i] = gv
        evt_stocks_by_g.setdefault(gv, set()).add(si)
    pools = {}
    for gv, banned in evt_stocks_by_g.items():
        lo, hi = bounds[gv], bounds[gv + 1]
        keep = np.isin(S[lo:hi], list(banned), invert=True)
        pools[gv] = (S[lo:hi][keep], R[lo:hi][keep])
    c2_ra = np.full((n_events, len(HORIZONS)), np.nan)
    c2_rb = np.full_like(c2_ra, np.nan)
    rng2 = np.random.default_rng(SEED)
    no_pool = 0
    for i in range(n_events):
        ps, pr = pools[ev_g[i]]
        if len(ps) == 0:
            no_pool += 1
            continue
        j = int(rng2.integers(len(ps)))
        ra, rb = fwd_ret_matrix(stocks[int(ps[j])], [int(pr[j])])
        c2_ra[i] = ra[0]
        c2_rb[i] = rb[0]
        if (i + 1) % 200000 == 0:
            print(f"  c2 {i+1}/{n_events} ({time.time()-t0:.0f}s)", flush=True)
    print(f"对照2完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 落盘原始结果
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(RAW_DIR / "events.parquet", index=False)
    hcols_a = [f"ret_a_h{h}" for h in HORIZONS]
    hcols_b = [f"ret_b_h{h}" for h in HORIZONS]

    def wide(group, ids, ra, rb):
        df = pd.DataFrame(ra.astype(np.float32), columns=hcols_a)
        df[hcols_b] = rb.astype(np.float32)
        df.insert(0, "event_id", ids)
        df.insert(1, "group", group)
        return df

    wide_df = pd.concat([
        wide("div", np.arange(n_events, dtype=np.int64), ev_ra, ev_rb),
        wide("c1", np.arange(len(c1_ra), dtype=np.int64), c1_ra, c1_rb),
        wide("c2", np.arange(n_events, dtype=np.int64), c2_ra, c2_rb),
    ], ignore_index=True)
    wide_df.to_parquet(RAW_DIR / "returns_wide.parquet", index=False)

    # ---- 统计
    print("统计检验 ...", flush=True)
    rows, verdict_rows = [], []
    stats_json = {}
    for entry, (d_all, c1_all, c2_all) in {
        "a": (ev_ra, c1_ra, c2_ra), "b": (ev_rb, c1_rb, c2_rb)
    }.items():
        for j, h in enumerate(HORIZONS):
            d, c1, c2 = d_all[:, j], c1_all[:, j], c2_all[:, j]
            r = dict(entry=entry, h=h, d=group_stats(d), c1=group_stats(c1), c2=group_stats(c2))
            r["c1"].update(compare(d, c1))
            r["c2"].update(compare(d, c2))
            rows.append(r)
            stats_json[f"{entry}{h}"] = r
            if h in (10, 15):
                dd = d[np.isfinite(d)]
                k, n = int((dd > 0).sum()), len(dd)
                lo, hi = wilson_ci(k, n)
                win = k / n if n else np.nan
                verdict_rows.append(dict(
                    entry=entry, h=h, n=n, win=win, lo=lo, hi=hi,
                    c1win=r["c1"]["win"], c2win=r["c2"]["win"],
                    verdict=("成立" if (np.isfinite(win) and win > 0.55 and lo > 0.55)
                             else ("点估计过线但CI未过线" if np.isfinite(win) and win > 0.55 else "不成立")),
                ))

    drop = lambda a: {f"h{h}": int(np.isnan(a[:, j]).sum()) for j, h in enumerate(HORIZONS)}
    date_min = pd.Timestamp(min(st["dates"][0] for st in stocks)).date().isoformat()
    date_max = pd.Timestamp(max(st["dates"][-1] for st in stocks)).date().isoformat()
    meta = dict(
        timestamp=pd.Timestamp.now().isoformat(timespec="seconds"),
        n_stocks=len(stocks), n_skipped=len(results) - len(stocks),
        date_min=date_min, date_max=date_max,
        n_events=n_events,
        n_event_stocks=int(events_df["ts_code"].nunique()) if n_events else 0,
        val_mismatch=val_mismatch,
        drop_events={"a": drop(ev_ra), "b": drop(ev_rb)},
        drop_c1={"a": drop(c1_ra), "b": drop(c1_rb)},
        drop_c2={"a": drop(c2_ra), "b": drop(c2_rb), "no_pool_events": no_pool},
        sample=args.sample, seed=SEED,
    )
    with open(RAW_DIR / "stats.json", "w") as f:
        json.dump(dict(meta=meta, results=stats_json, verdict=verdict_rows), f, ensure_ascii=False, indent=2, default=float)
    md = build_report(meta, rows, verdict_rows)
    out_md = OUT_DIR / "divergence_event_study.md"
    out_md.write_text(md)
    print(f"完成,耗时 {time.time()-t0:.0f}s\n报告: {out_md}\n原始结果: {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
