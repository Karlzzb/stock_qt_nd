#!/usr/bin/env python3
"""v6 专属几何特征族(六族 40 条,#42 清单逐字落码)。

口径源头:reports/wayfinder/42-v6-geometric-features.md §一记号与 §四清单。
  - 价格序列 = 个股不复权日线,按 scanner 口径(trade_date 升序 reset_index)0 基行号。
  - DIF/DEA = talib.MACD(close) 12/26/9,全历史一次性计算(引理 2:递推无前视)。
  - 金叉三元组行号 i2/i1/j、锚点行 a、前区间最低行 p_low 均由 M1 事件表落盘列供给。
  - 43 条中 E2/E3/E4 由日频快照覆盖不进本层(族E 立规),入层 40 条。
  - 种子布尔 F6~F10 在 float64 的 dd20/bounce 上判定(精度守卫,README §四)。

每条特征的全部输入均为行号 <= j 的行及其派生 DIF/DEA(引理 1/2/3),
最晚可知时点 = 事件日(第 j 行)收盘。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 列名录(40 条,构建顺序固定)
# (列名, 编号, 中文全称, 精确公式, 量纲)
GEO_SPECS = [
    ("GC_GAP_NEAR_BARS", "A1", "金叉间隔近端交易日数", "j - i1", "交易日"),
    ("GC_GAP_FAR_BARS", "A2", "金叉间隔远端交易日数", "i1 - i2", "交易日"),
    ("GC_GAP_RATIO", "A3", "金叉间隔比值", "(j - i1) / (i1 - i2)", "无量纲"),
    ("DIF_EVENT_CLOSE_NORM", "A4", "事件日DIF收盘价归一水平", "DIF[j] / close[j]", "无量纲"),
    ("DIF_PREV_CLOSE_NORM", "A5", "前金叉DIF收盘价归一水平", "DIF[i1] / close[i1]", "无量纲"),
    ("DIF_LIFT_CLOSE_NORM", "A6", "金叉DIF抬升幅度收盘价归一", "(DIF[j] - DIF[i1]) / close[j]", "无量纲"),
    ("DIF_LIFT_RATE_CLOSE_NORM", "A7", "金叉DIF抬升速率收盘价归一",
     "(DIF[j] - DIF[i1]) / ((j - i1) * close[j])", "无量纲/交易日"),
    ("DEA_EVENT_CLOSE_NORM", "A8", "事件日DEA收盘价归一水平", "DEA[j] / close[j]", "无量纲"),
    ("DEA_LIFT_CLOSE_NORM", "A9", "金叉DEA抬升幅度收盘价归一", "(DEA[j] - DEA[i1]) / close[j]", "无量纲"),
    ("DIF_DEA_GAP_CLOSE_NORM", "A10", "事件日DIF对DEA开口收盘价归一", "(DIF[j] - DEA[j]) / close[j]", "无量纲"),
    ("DIF_PREV2_CLOSE_NORM", "A11", "远金叉DIF收盘价归一水平", "DIF[i2] / close[i2]", "无量纲"),
    ("DIF_LIFT_PREV_CLOSE_NORM", "A12", "前一对金叉DIF抬升幅度收盘价归一",
     "(DIF[i1] - DIF[i2]) / close[i1]", "无量纲"),
    ("DD20", "B0", "前二十交易日最大收盘回撤", "close[j] / max(close[max(0,j-20)..j]) - 1", "无量纲"),
    ("DD20_START_BARS", "B1", "回撤起点距事件日交易日数",
     "j - argmax_{t in W} close[t](并列取最早)", "交易日"),
    ("DD20_AVG_SPEED", "B2", "急跌段平均跌速",
     "dd20 / max(1, j - argmax_{t in W} close[t])(事件日即窗口最高时定义值为 0)", "无量纲/交易日"),
    ("DD20_DOWN_DAY_SHARE", "B3", "急跌窗口下跌日占比",
     "(1 / (|W| - 1)) * sum_{t in W, t > t0} 1[close[t] < close[t-1]](|W|=1 时 NaN)", "无量纲"),
    ("DD20_MAX_DAY_DROP", "B4", "急跌窗口最大单日跌幅",
     "min_{t in W, t > t0} (close[t] / close[t-1] - 1)(|W|=1 时 NaN)", "无量纲"),
    ("DD20_LOCAL_LOW_COUNT", "B5", "急跌窗口内局部低点次数",
     "#{t in (t0, j): close[t] < close[t-1] 且 close[t] < close[t+1]}", "计数"),
    ("DD20_PATH_EFFICIENCY", "B6", "急跌路径效率系数",
     "|dd20| / sum_{t in W, t > t0} |close[t] / close[t-1] - 1|(分母为 0 时 NaN)", "无量纲"),
    ("BOUNCE", "C0", "信号日相对锚点已弹幅度", "close[j] / anchor_close - 1", "无量纲"),
    ("BOUNCE_AVG_SPEED", "C1", "反弹段平均弹速", "bounce / max(1, j - a)", "无量纲/交易日"),
    ("BOUNCE_MAX_GIVEBACK", "C2", "反弹途中最大回吐幅度",
     "min_{t in [a, j]} (close[t] / max(close[a..t]) - 1)(<= 0)", "无量纲"),
    ("BOUNCE_UP_DAY_SHARE", "C3", "反弹段上涨日占比",
     "(1 / max(1, j - a)) * sum_{t = a+1..j} 1[close[t] > close[t-1]]", "无量纲"),
    ("BOUNCE_VOL_RATIO", "C4", "反弹段均量对锚点前基线量比",
     "mean(vol[a+1..j]) / mean(vol[max(0,a-20)..a])(分母为 0 时 NaN;a=j 时分子取 vol[j])", "无量纲"),
    ("ANCHOR_DIST_BARS", "D1", "锚点距事件日交易日数", "j - a", "交易日"),
    ("ANCHOR_REL_POS", "D2", "锚点在后区间内相对位置", "(a - i1) / (j - i1)(值域 (0, 1])", "无量纲"),
    ("ANCHOR_BREAK_DEPTH", "D3", "区间新低下破深度",
     "anchor_close / min_prev - 1,min_prev = min(close[(i2, i1]])(由构造 < 0)", "无量纲"),
    ("MIN_PREV_DIST_BARS", "D4", "前区间最低收盘距事件日交易日数",
     "j - p_low,p_low = argmin close[(i2, i1]](并列取最早)", "交易日"),
    ("MIN_PREV_DRAWDOWN", "D5", "前区间自身回撤深度", "min_prev / max(close[(i2, i1]]) - 1", "无量纲"),
    ("DD20_VOL_BASELINE_RATIO", "E1", "急跌窗口均量对前六十日均量比值",
     "mean(vol[W]) / mean(vol[max(0,j-80)..t0-1])(基线窗口不足 60 日按实际长度;分母为 0 时 NaN)", "无量纲"),
    ("DD20_DIST_TH15", "F1", "dd20距负零点一五阈值距离", "dd20 + 0.15", "无量纲"),
    ("DD20_DIST_TH20", "F2", "dd20距负零点二零阈值距离", "dd20 + 0.20", "无量纲"),
    ("DD20_DIST_TH25", "F3", "dd20距负零点二五阈值距离", "dd20 + 0.25", "无量纲"),
    ("BOUNCE_DIST_LO", "F4", "bounce距下界零点零二距离", "bounce - 0.02", "无量纲"),
    ("BOUNCE_DIST_HI", "F5", "bounce距上界零点零八距离", "0.08 - bounce", "无量纲"),
    ("SEED_V6_1", "F6", "种子v6-1成员资格标记", "1[dd20 <= -0.15 且 0.02 < bounce <= 0.08]", "布尔"),
    ("SEED_V6_2", "F7", "种子v6-2成员资格标记", "1[dd20 <= -0.20 且 0.02 < bounce <= 0.08]", "布尔"),
    ("SEED_V6_3", "F8", "种子v6-3成员资格标记", "1[dd20 <= -0.25 且 0.02 < bounce <= 0.08]", "布尔"),
    ("SEED_V6_4", "F9", "种子v6-4成员资格标记", "1[dd20 <= -0.15]", "布尔"),
    ("SEED_V6_5", "F10", "种子v6-5成员资格标记", "1[dd20 <= -0.25]", "布尔"),
]

GEO_COLUMNS = [s[0] for s in GEO_SPECS]
GEO_CN = {s[0]: s[2] for s in GEO_SPECS}
GEO_FORMULA = {s[0]: s[3] for s in GEO_SPECS}
GEO_ID = {s[0]: s[1] for s in GEO_SPECS}

# 含守卫 NaN 分支的特征(#42 断言 11 披露义务)
GUARDED_NAN_COLS = ("DD20_DOWN_DAY_SHARE", "DD20_MAX_DAY_DROP",
                    "DD20_PATH_EFFICIENCY", "BOUNCE_VOL_RATIO",
                    "DD20_VOL_BASELINE_RATIO")

# 种子阈值(#31 §2.2 冻结)
SEED_THRESHOLDS = {  # 列名: (dd20 阈值, 是否带 bounce 带)
    "SEED_V6_1": (-0.15, True),
    "SEED_V6_2": (-0.20, True),
    "SEED_V6_3": (-0.25, True),
    "SEED_V6_4": (-0.15, False),
    "SEED_V6_5": (-0.25, False),
}
BOUNCE_LO, BOUNCE_HI = 0.02, 0.08
# frozen 计数(#31/#48 对账值,全表硬断言用)
SEED_FROZEN_COUNTS = {"SEED_V6_1": 5580, "SEED_V6_2": 2419, "SEED_V6_3": 1159,
                      "SEED_V6_4": 11888, "SEED_V6_5": 3271}


def _safe_div(num, den):
    num = np.asarray(num, np.float64)
    den = np.asarray(den, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def compute_geo_events(close, vol, dif, dea, i2, i1, j, a, p_low):
    """对单股全部事件计算 40 条几何特征。

    close/vol: 个股不复权日线全历史数组(行号对齐事件表行号);
    dif/dea: talib.MACD(close) 全历史;
    i2/i1/j/a/p_low: 各事件的行号数组(int64,来自事件表落盘列)。
    返回 dict[str, float64 数组],长度 = len(j),键序 == GEO_COLUMNS。
    """
    C = np.asarray(close, np.float64)
    V = np.asarray(vol, np.float64)
    DIF = np.asarray(dif, np.float64)
    DEA = np.asarray(dea, np.float64)
    i2 = np.asarray(i2, np.int64)
    i1 = np.asarray(i1, np.int64)
    j = np.asarray(j, np.int64)
    a = np.asarray(a, np.int64)
    p_low = np.asarray(p_low, np.int64)
    m = len(j)

    out: dict[str, np.ndarray] = {}
    cj, ci1, ci2 = C[j], C[i1], C[i2]
    dj, di1, di2 = DIF[j], DIF[i1], DIF[i2]
    ej, ei1 = DEA[j], DEA[i1]

    # ---- 族A 金叉三元组时空结构(12 条,全向量化)
    out["GC_GAP_NEAR_BARS"] = (j - i1).astype(np.float64)
    out["GC_GAP_FAR_BARS"] = (i1 - i2).astype(np.float64)
    out["GC_GAP_RATIO"] = _safe_div(j - i1, i1 - i2)
    out["DIF_EVENT_CLOSE_NORM"] = _safe_div(dj, cj)
    out["DIF_PREV_CLOSE_NORM"] = _safe_div(di1, ci1)
    out["DIF_LIFT_CLOSE_NORM"] = _safe_div(dj - di1, cj)
    out["DIF_LIFT_RATE_CLOSE_NORM"] = _safe_div(dj - di1, (j - i1) * cj)
    out["DEA_EVENT_CLOSE_NORM"] = _safe_div(ej, cj)
    out["DEA_LIFT_CLOSE_NORM"] = _safe_div(ej - ei1, cj)
    out["DIF_DEA_GAP_CLOSE_NORM"] = _safe_div(dj - ej, cj)
    out["DIF_PREV2_CLOSE_NORM"] = _safe_div(di2, ci2)
    out["DIF_LIFT_PREV_CLOSE_NORM"] = _safe_div(di1 - di2, ci1)

    # ---- 族B/C/D/E 窗口路径特征(逐事件,窗口 <= 21/61 行)
    dd20 = np.full(m, np.nan)
    b1 = np.full(m, np.nan)
    b3 = np.full(m, np.nan)
    b4 = np.full(m, np.nan)
    b5 = np.full(m, np.nan)
    b6 = np.full(m, np.nan)
    c2 = np.full(m, np.nan)
    c3 = np.full(m, np.nan)
    c4 = np.full(m, np.nan)
    d3 = np.full(m, np.nan)
    d5 = np.full(m, np.nan)
    e1 = np.full(m, np.nan)
    for e in range(m):
        je = int(j[e])
        t0 = max(0, je - 20)
        w = C[t0:je + 1]                       # 急跌窗口 W(含事件日)
        nw = je - t0 + 1
        wmax = float(np.nanmax(w))
        dd = C[je] / wmax - 1.0
        dd20[e] = dd
        peak = t0 + int(np.nanargmax(w))       # 并列取最早(nanargmax 首个最大值)
        b1[e] = float(je - peak)
        if nw > 1:
            rets = C[t0 + 1:je + 1] / C[t0:je] - 1.0
            b3[e] = float(np.mean(C[t0 + 1:je + 1] < C[t0:je]))
            b4[e] = float(np.nanmin(rets))
            absum = float(np.nansum(np.abs(rets)))
            b6[e] = abs(dd) / absum if absum > 0 else np.nan
            cnt = 0                            # B5: 局部低点 t in (t0, j) 开区间
            for t in range(t0 + 1, je):
                if C[t] < C[t - 1] and C[t] < C[t + 1]:
                    cnt += 1
            b5[e] = float(cnt)
        # E1: 急跌窗口均量 / 前六十日基线均量(基线 = [max(0,j-80), t0-1])
        vbase = V[max(0, je - 80):t0]
        den = float(np.nanmean(vbase)) if len(vbase) else np.nan
        num = float(np.nanmean(V[t0:je + 1]))
        e1[e] = num / den if np.isfinite(den) and den > 0 else np.nan
        # 族C 反弹形态(锚点行 a 至事件行 j)
        ae = int(a[e])
        span = je - ae
        if span >= 1:
            seg = C[ae:je + 1]
            run_max = np.maximum.accumulate(seg)
            c2[e] = float(np.nanmin(seg / run_max - 1.0))
            c3[e] = float(np.mean(C[ae + 1:je + 1] > C[ae:je]))
            vnum = float(np.nanmean(V[ae + 1:je + 1]))
        else:  # a == j: 反弹段长度为 0
            c2[e] = 0.0                      # t=a 单点: close[a]/max(close[a..a])-1 = 0
            c3[e] = 0.0                      # 公式分子为空集求和 = 0,分母 max(1, 0) = 1
            vnum = float(V[je])
        vden_seg = V[max(0, ae - 20):ae + 1]
        vden = float(np.nanmean(vden_seg)) if len(vden_seg) else np.nan
        c4[e] = vnum / vden if np.isfinite(vden) and vden > 0 else np.nan
        # 族D 锚点结构(前区间 (i2, i1] 左开右闭)
        i1e, i2e = int(i1[e]), int(i2[e])
        prev_seg = C[i2e + 1:i1e + 1]
        min_prev = float(np.nanmin(prev_seg))
        d3[e] = C[ae] / min_prev - 1.0
        d5[e] = min_prev / float(np.nanmax(prev_seg)) - 1.0

    bounce_arr = _safe_div(cj, C[a]) - 1.0
    out["DD20"] = dd20
    out["DD20_START_BARS"] = b1
    out["DD20_AVG_SPEED"] = _safe_div(dd20, np.maximum(1.0, b1))
    out["DD20_DOWN_DAY_SHARE"] = b3
    out["DD20_MAX_DAY_DROP"] = b4
    out["DD20_LOCAL_LOW_COUNT"] = b5
    out["DD20_PATH_EFFICIENCY"] = b6
    out["BOUNCE"] = bounce_arr
    out["BOUNCE_AVG_SPEED"] = _safe_div(
        bounce_arr, np.maximum(1.0, (j - a).astype(np.float64)))
    out["BOUNCE_MAX_GIVEBACK"] = c2
    out["BOUNCE_UP_DAY_SHARE"] = c3
    out["BOUNCE_VOL_RATIO"] = c4
    out["ANCHOR_DIST_BARS"] = (j - a).astype(np.float64)
    out["ANCHOR_REL_POS"] = _safe_div(a - i1, j - i1)
    out["ANCHOR_BREAK_DEPTH"] = d3
    out["MIN_PREV_DIST_BARS"] = (j - p_low).astype(np.float64)
    out["MIN_PREV_DRAWDOWN"] = d5
    out["DD20_VOL_BASELINE_RATIO"] = e1

    # ---- 族F 种子阈值距离与成员资格(布尔在 float64 上判定,精度守卫)
    out["DD20_DIST_TH15"] = dd20 + 0.15
    out["DD20_DIST_TH20"] = dd20 + 0.20
    out["DD20_DIST_TH25"] = dd20 + 0.25
    out["BOUNCE_DIST_LO"] = bounce_arr - BOUNCE_LO
    out["BOUNCE_DIST_HI"] = BOUNCE_HI - bounce_arr
    in_band = (bounce_arr > BOUNCE_LO) & (bounce_arr <= BOUNCE_HI)
    for col, (th, with_band) in SEED_THRESHOLDS.items():
        sel = dd20 <= th
        if with_band:
            sel = sel & in_band
        out[col] = sel.astype(np.float64)

    assert list(out.keys()) == GEO_COLUMNS, "几何族列序与 GEO_SPECS 不一致"
    return out
