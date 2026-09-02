# -*- coding: utf-8 -*-
"""特征工厂层 (G 层) 生成器: 字段 × 算子 × 窗口 笛卡尔积批量生成.

口径唯一依据: v3_pipeline/reports/feature_harvest/feature_master_spec.md 第 6 章.
数据契约: 主表 1.1/1.2 —— 本地日线 (不复权), 用 pct_chg 链重构复权序列 CF,
跨日价格比较一律在复权链上; 全部特征信号日 T 收盘后可算 (严格因果).

设计决定 (在 feature_factory_report.md 逐条备案):
  D1. 归一化 baked-in: 复权价格类 (OPEN/HIGH/LOW/CLOSE/CF) 的水平型算子输出除以
      当日 C̃; VWAP 类除以当日原始 C (A/V 为原始价尺度, 与 C 同日同尺度, 无复权失真);
      量/额类 (V/A) 水平型输出除以 (当日值+1), STD/SLOPE/RESI 除以 (窗口均值+1);
      LOGV 与收益类 (R/RETON/RETID/HLR) 尺度自由不归一.
      理由: 赛跑主变体是逐日截面秩变换, 跨股可比的特征才有意义 (qlib Alpha158 同思路).
  D2. 算子集 19+3: MEAN/STD/MAX/MIN/MED/Q20/Q80/SKEW/KURT/TSRANK/IDXMAX/IDXMIN/
      SLOPE/RSQ/RESI/EMA/WMA/REF/DELTA 全字段适用; SUM 与 COUNT(>0) 仅收益类
      (R/RETON/RETID; 价格类 SUM 与 MEAN 精确线性冗余, HLR 求和无意义).
      VAR 不生成 (与 STD 逐行单调等价, 截面秩下完全冗余); -1x 反向不生成
      (树模型对符号翻转不变). 截断对拍/黑名单断言不受影响.
  D3. cs_rank / zscore_ts 第二层不批量生成 (主表 6.2 标注为可选层, 开启后列数翻倍
      超 2500 上限; 赛跑协议自带逐日截面秩变体 __DR, 功能等价).
  D4. NaN 纪律: 滚动算子严格全窗口出值 (窗口内任一 NaN -> NaN); EMA 用
      min_periods=w 递推 (adjust=False), 截断对拍只看 T 点值, 因果性不受影响.
  D5. 全部 float64; 表达式字符串逐列写入注册表, 防口径漂移.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------- 常量
WINDOWS = [3, 5, 10, 20, 30, 60]

# 字段类
F_PADJ = ["OPEN", "HIGH", "LOW", "CLOSE", "CF"]   # 复权价格链, 归一除数 C̃
F_VWAP = ["VWAP"]                                  # 原始价尺度, 归一除数 C
F_VOL = ["V", "A"]                                 # 量额, 归一除数 (当日值+1)
F_LOGV = ["LOGV"]                                  # log 量, 尺度自由
F_RET = ["R", "RETON", "RETID", "HLR"]             # 收益/比率, 尺度自由
FIELDS = F_PADJ + F_VWAP + F_VOL + F_LOGV + F_RET  # 13 个基础字段

# 单序列算子 (19 个, 全字段适用)
OPS19 = ["MEAN", "STD", "MAX", "MIN", "MED", "Q20", "Q80", "SKEW", "KURT",
         "TSRANK", "IDXMAX", "IDXMIN", "SLOPE", "RSQ", "RESI", "EMA", "WMA",
         "REF", "DELTA"]
# 水平型算子 (需按字段类归一); 其余算子 (SKEW/KURT/TSRANK/IDXMAX/IDXMIN/RSQ) 尺度自由
_LEVEL_OPS = {"MEAN", "STD", "MAX", "MIN", "MED", "Q20", "Q80", "EMA", "WMA",
              "REF", "DELTA", "SLOPE", "RESI"}
SUM_FIELDS = ["R", "RETON", "RETID"]     # SUM 仅收益类 (D2)
COUNT_FIELDS = ["R", "RETON", "RETID"]   # COUNT(x>0) 仅收益类

# 双序列算子对
CORR_PAIRS = [("CLOSE", "V"), ("R", "V"), ("VWAP", "V"), ("HLR", "V"),
              ("R", "A"), ("CLOSE", "A"), ("R", "RIDX"), ("CLOSE", "RIDX")]
COV_PAIRS = [("R", "RIDX"), ("R", "LOGV"), ("CLOSE", "LOGV"), ("HLR", "LOGV")]

EXPR_FIELD = {"OPEN": "Õ", "HIGH": "H̃", "LOW": "L̃", "CLOSE": "C̃", "CF": "CF",
              "VWAP": "A/V", "V": "V", "A": "A", "LOGV": "log(V+1)",
              "R": "pct_chg/100", "RETON": "O/PC-1", "RETID": "C/O-1",
              "HLR": "H/L", "RIDX": "R_idx(000001.SH)"}


# ---------------------------------------------------------------- 基础算子 (严格全窗口出值)
def _place(n, w, vals):
    out = np.full(n, np.nan)
    out[w - 1:] = vals
    return out


def _roll_reduce(x, w):
    """对 (x,w) 一次滑窗, 返回 dict of 各归约结果 (长度 n, 头部 NaN)."""
    n = len(x)
    out = {}
    if n < w:
        for k in ("MEAN", "STD", "MAX", "MIN", "MED", "Q20", "Q80", "SKEW", "KURT"):
            out[k] = np.full(n, np.nan)
        return out
    sw = sliding_window_view(x, w)
    with np.errstate(invalid="ignore"):
        mu = sw.mean(axis=1)
        sd = sw.std(axis=1)
        med = np.median(sw, axis=1)
        q20, q80 = np.quantile(sw, [0.2, 0.8], axis=1)
        z = sw - mu[:, None]
        m2 = (z ** 2).mean(axis=1)
        m3 = (z ** 3).mean(axis=1)
        m4 = (z ** 4).mean(axis=1)
        skew = np.where(m2 > 0, m3 / np.power(m2, 1.5), np.nan)
        kurt = np.where(m2 > 0, m4 / (m2 ** 2) - 3.0, np.nan)
    # NaN 传播即严格 (mean/std/max/min/median/quantile 遇 NaN 得 NaN)
    res = {"MEAN": mu, "STD": sd, "MAX": sw.max(axis=1), "MIN": sw.min(axis=1),
           "MED": med, "Q20": q20, "Q80": q80, "SKEW": skew, "KURT": kurt}
    for k, v in res.items():
        out[k] = _place(n, w, v)
    return out


def _slope(x, w):
    """对时间 0..w-1 的 OLS 斜率; 全窗口才出值 (feature_engine._slope 同口径)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    t = np.arange(w, dtype=np.float64)
    k = t - t.mean()
    k /= (k * k).sum()
    valid = np.isfinite(x)
    y0 = np.where(valid, x, 0.0)
    num = np.correlate(y0, k, mode="valid")
    cnt = np.correlate(valid.astype(np.float64), np.ones(w), mode="valid")
    out[w - 1:] = np.where(cnt == w, num, np.nan)
    return out


def _rsq_resi(x, w, slope, mu):
    """滚动 OLS 的 R^2 与末端残差 (用已算出的 slope/MEAN, 严格全窗口)."""
    n = len(x)
    d2 = w * (w * w - 1) / 12.0
    rsq = np.full(n, np.nan)
    resi = np.full(n, np.nan)
    if n < w:
        return rsq, resi
    sw = sliding_window_view(x, w)
    m2 = ((sw - sw.mean(axis=1)[:, None]) ** 2).mean(axis=1)
    sst = _place(n, w, m2 * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        rsq = np.where(sst > 0, (slope ** 2) * d2 / sst, np.nan)
        resi = x - (mu + slope * (w - 1) / 2.0)
    return rsq, resi


def _tsrank(x, w):
    """当前值在过去 w 日 (含当日) 时序分位 = (less + 0.5*ties)/w; 全窗口才出值."""
    x = np.asarray(x, np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    sw = sliding_window_view(x, w)
    last = sw[:, -1:]
    bad = np.isnan(sw).any(axis=1)
    less = (sw < last).sum(axis=1)
    ties = (sw == last).sum(axis=1)
    val = (less + 0.5 * ties) / w
    out[w - 1:] = np.where(bad, np.nan, val)
    return out


def _idx_extreme(x, w, mode):
    """最值距今天数 / w (0=今天); 全窗口才出值; ties 取最旧."""
    x = np.asarray(x, np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    sw = sliding_window_view(x, w)
    bad = np.isnan(sw).any(axis=1)
    fill = -np.inf if mode == "max" else np.inf
    sw2 = np.where(np.isnan(sw), fill, sw)
    idx = np.argmax(sw2, axis=1) if mode == "max" else np.argmin(sw2, axis=1)
    dist = ((w - 1) - idx).astype(np.float64) / w
    out[w - 1:] = np.where(bad, np.nan, dist)
    return out


def _ema(x, w):
    """EMA alpha=2/(w+1), adjust=False, min_periods=w (D4)."""
    return (pd.Series(x, copy=False)
            .ewm(alpha=2.0 / (w + 1), adjust=False, min_periods=w)
            .mean().to_numpy(np.float64))


def _wma(x, w):
    """DecayLinear: 线性权重 1..w 的滚动加权均值; 全窗口才出值."""
    x = np.asarray(x, np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    wt = np.arange(1, w + 1, dtype=np.float64)
    valid = np.isfinite(x)
    y0 = np.where(valid, x, 0.0)
    num = np.correlate(y0, wt, mode="valid")
    cnt = np.correlate(valid.astype(np.float64), np.ones(w), mode="valid")
    out[w - 1:] = np.where(cnt == w, num / wt.sum(), np.nan)
    return out


def _ref(x, w):
    out = np.full(len(x), np.nan)
    out[w:] = x[:-w]
    return out


def _delta(x, w):
    return x - _ref(x, w)


def _roll_sum_count(x, w, count=False):
    x = np.asarray(x, np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    if count:
        v = np.where(np.isnan(x), np.nan, (x > 0).astype(np.float64))
    else:
        v = x
    valid = np.isfinite(v)
    y0 = np.where(valid, v, 0.0)
    s = np.correlate(y0, np.ones(w), mode="valid")
    cnt = np.correlate(valid.astype(np.float64), np.ones(w), mode="valid")
    vals = np.where(cnt == w, s, np.nan)
    if count:
        vals = vals / w
    out[w - 1:] = vals
    return out


def _roll_cov_corr(x, y, w, kind):
    """滚动 cov/corr (总体矩, ddof=0); 任一序列窗口内含 NaN -> NaN."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    ok = np.isfinite(x) & np.isfinite(y)
    x0 = np.where(ok, x, 0.0)
    y0 = np.where(ok, y, 0.0)
    ones = np.ones(w)
    sx = np.correlate(x0, ones, mode="valid")
    sy = np.correlate(y0, ones, mode="valid")
    sxx = np.correlate(x0 * x0, ones, mode="valid")
    syy = np.correlate(y0 * y0, ones, mode="valid")
    sxy = np.correlate(x0 * y0, ones, mode="valid")
    cnt = np.correlate(ok.astype(np.float64), ones, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / w - (sx / w) * (sy / w)
        vx = sxx / w - (sx / w) ** 2
        vy = syy / w - (sy / w) ** 2
        if kind == "corr":
            den = np.sqrt(np.where(vx > 0, vx, np.nan) * np.where(vy > 0, vy, np.nan))
            vals = cov / den
        else:
            vals = cov
    out[w - 1:] = np.where(cnt == w, vals, np.nan)
    return out


# ---------------------------------------------------------------- 单股生成
def build_base_fields(df, ridx_by_date):
    """从单股日线 (load_stock_df 口径) 构造 13 基础字段 + 归一除数. 全 float64."""
    O = df["open"].to_numpy(np.float64)
    H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64)
    C = df["close"].to_numpy(np.float64)
    PC = df["pre_close"].to_numpy(np.float64)
    V = df["vol"].to_numpy(np.float64)
    A = df["amount"].to_numpy(np.float64)
    R = df["pct_chg"].to_numpy(np.float64) / 100.0
    CF = np.cumprod(1.0 + np.where(np.isfinite(R), R, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        f_adj = np.where(C > 0, CF / C, np.nan)
    fields = {
        "OPEN": O * f_adj, "HIGH": H * f_adj, "LOW": L * f_adj, "CLOSE": CF,
        "CF": CF,
        "VWAP": np.where(V > 0, A / V, np.nan),
        "V": V, "A": A, "LOGV": np.log(np.where(V > -1, V, np.nan) + 1.0),
        "R": R,
        "RETON": np.where(PC > 0, O / PC - 1.0, np.nan),
        "RETID": np.where(O > 0, C / O - 1.0, np.nan),
        "HLR": np.where(L > 0, H / L, np.nan),
    }
    # 指数日收益对齐到个股交易日 (缺日/无指数 -> NaN, 窗口遇 NaN 即 NaN)
    days_i = df["trade_date"].to_numpy("datetime64[D]").astype(np.int64)
    if ridx_by_date is not None:
        fields["RIDX"] = np.array([ridx_by_date.get(int(d), np.nan) for d in days_i],
                                  np.float64)
    else:
        fields["RIDX"] = np.full(len(df), np.nan)
    divisors = {"PADJ": CF, "VWAP": C}
    return fields, divisors


def _norm_divisor(field, op, cur, mu_w, divisors):
    """归一除数 (D1); 返回 None 表示不归一."""
    if op not in _LEVEL_OPS:
        return None
    if field in F_PADJ:
        return divisors["PADJ"]
    if field in F_VWAP:
        return divisors["VWAP"]
    if field in F_VOL:
        if op in ("STD", "SLOPE", "RESI"):
            return mu_w + 1.0
        return cur + 1.0
    return None


def _expr(op, field, w, norm):
    e = EXPR_FIELD[field]
    opmap = {"MEAN": f"Mean({e},{w})", "STD": f"Std({e},{w})",
             "MAX": f"Max({e},{w})", "MIN": f"Min({e},{w})",
             "MED": f"Med({e},{w})", "Q20": f"Quant({e},{w},0.2)",
             "Q80": f"Quant({e},{w},0.8)", "SKEW": f"Skew({e},{w})",
             "KURT": f"Kurt({e},{w})", "TSRANK": f"TsRank({e},{w})",
             "IDXMAX": f"IdxMax({e},{w})/{w}", "IDXMIN": f"IdxMin({e},{w})/{w}",
             "SLOPE": f"Slope({e},{w})", "RSQ": f"Rsquare({e},{w})",
             "RESI": f"Resi({e},{w})", "EMA": f"EMA({e},{w})",
             "WMA": f"DecayLinear({e},{w})", "REF": f"Ref({e},{w})",
             "DELTA": f"Delta({e},{w})", "SUM": f"Sum({e},{w})",
             "COUNT": f"Count({e}>0,{w})/{w}"}
    s = opmap[op]
    if norm == "PADJ":
        s += "/C̃"
    elif norm == "VWAP":
        s += "/C"
    elif norm == "VOLCUR":
        s += "/(x+1)"
    elif norm == "VOLMU":
        s += "/(Mean+1)"
    return s


def compute_stock_factory(df, ridx_by_date=None):
    """单股全历史生成全部工厂特征.

    返回 (cols, registry): cols = {特征名: float64 数组 (与 df 行对齐)},
    registry = [(name, op, field, window, expression)].
    """
    fields, divisors = build_base_fields(df, ridx_by_date)
    n = len(df)
    cols, registry = {}, []

    def _add(name, arr, op, field, w, norm_tag):
        cols[name] = np.asarray(arr, np.float64)
        registry.append((name, op, field, w, _expr(op, field, w, norm_tag)))

    for field in FIELDS:
        x = fields[field]
        cur = x
        for w in WINDOWS:
            red = _roll_reduce(x, w)
            slope = _slope(x, w)
            rsq, resi = _rsq_resi(x, w, slope, red["MEAN"])
            ema = _ema(x, w)
            wma = _wma(x, w)
            tsr = _tsrank(x, w)
            idxmax = _idx_extreme(x, w, "max")
            idxmin = _idx_extreme(x, w, "min")
            ref = _ref(x, w)
            delta = cur - ref
            raw = {"MEAN": red["MEAN"], "STD": red["STD"], "MAX": red["MAX"],
                   "MIN": red["MIN"], "MED": red["MED"], "Q20": red["Q20"],
                   "Q80": red["Q80"], "SKEW": red["SKEW"], "KURT": red["KURT"],
                   "TSRANK": tsr, "IDXMAX": idxmax, "IDXMIN": idxmin,
                   "SLOPE": slope, "RSQ": rsq, "RESI": resi, "EMA": ema,
                   "WMA": wma, "REF": ref, "DELTA": delta}
            for op in OPS19:
                arr = raw[op]
                div = _norm_divisor(field, op, cur, red["MEAN"], divisors)
                norm_tag = None
                if div is not None:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        arr = np.where(div > 0, arr / div, np.nan)
                    if field in F_PADJ:
                        norm_tag = "PADJ"
                    elif field in F_VWAP:
                        norm_tag = "VWAP"
                    elif op in ("STD", "SLOPE", "RESI"):
                        norm_tag = "VOLMU"
                    else:
                        norm_tag = "VOLCUR"
                _add(f"{op}_{field}_{w}", arr, op, field, w, norm_tag)
        # SUM / COUNT (收益类)
        if field in SUM_FIELDS:
            for w in WINDOWS:
                _add(f"SUM_{field}_{w}", _roll_sum_count(x, w), "SUM", field, w, None)
        if field in COUNT_FIELDS:
            for w in WINDOWS:
                _add(f"COUNT_{field}_{w}", _roll_sum_count(x, w, count=True),
                     "COUNT", field, w, None)

    # 双序列
    for f1, f2 in CORR_PAIRS:
        for w in WINDOWS:
            arr = _roll_cov_corr(fields[f1], fields[f2], w, "corr")
            name = f"CORR_{f1}_{f2}_{w}"
            cols[name] = arr
            registry.append((name, "CORR", f"{f1}|{f2}", w,
                             f"Corr({EXPR_FIELD[f1]},{EXPR_FIELD[f2]},{w})"))
    for f1, f2 in COV_PAIRS:
        for w in WINDOWS:
            arr = _roll_cov_corr(fields[f1], fields[f2], w, "cov")
            name = f"COV_{f1}_{f2}_{w}"
            cols[name] = arr
            registry.append((name, "COV", f"{f1}|{f2}", w,
                             f"Cov({EXPR_FIELD[f1]},{EXPR_FIELD[f2]},{w})"))

    # K 线 9 式 + 价格快照 4 式 (原始价, 单日截面比率, 复权安全)
    O = df["open"].to_numpy(np.float64)
    H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64)
    C = df["close"].to_numpy(np.float64)
    hl = H - L
    with np.errstate(invalid="ignore", divide="ignore"):
        body_top = np.maximum(O, C)
        body_bot = np.minimum(O, C)
        kbar = {
            "KBAR_KMID": (C - O) / O,
            "KBAR_KLEN": hl / O,
            "KBAR_KMID2": (C - O) / (hl + 1e-12),
            "KBAR_KUP": (H - body_top) / O,
            "KBAR_KUP2": (H - body_top) / (hl + 1e-12),
            "KBAR_KLOW": (body_bot - L) / O,
            "KBAR_KLOW2": (body_bot - L) / (hl + 1e-12),
            "KBAR_KSFT": (2 * C - H - L) / O,
            "KBAR_KSFT2": (2 * C - H - L) / (hl + 1e-12),
        }
        snap = {
            "SNAP_O": O / C, "SNAP_H": H / C, "SNAP_L": L / C,
            "SNAP_VWAP": fields["VWAP"] / C,
        }
    for name, arr in {**kbar, **snap}.items():
        arr = np.where(np.isfinite(arr), arr, np.nan)
        cols[name] = arr
        registry.append((name, "KBAR" if name.startswith("KBAR") else "SNAP",
                         "OHLC", 0, name))
    return cols, registry


# ---------------------------------------------------------------- 泄漏黑名单 (主表 7.1, 14 条)
BLACKLIST_PATTERNS = [
    r"^stop_loss_", r"^future_", r"^next_", r"^label", r"^mfr_",
    r"^cur_return$|^max_forward_return$|^open_exec_return",
    r"^rank_future_|^rank_open_exec_", r"^ret_h\d+$", r"^hit_N",
    r"^mfe_|^mae_|^tmfe|^tmae", r"^dyn_", r"^entry_date$", r"^rank_",
]
