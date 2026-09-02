#!/usr/bin/env python3
"""特征引擎: 按《特征主表》(v3_pipeline/reports/feature_harvest/feature_master_spec.md v1.0) 实现.

输入: 股票日线面板 (stock_data/daily/*.parquet) + 事件表 (divergence_lab m_scan 两池 events.parquet).
输出: 每个事件在信号日 T 的特征行 (parquet), 附 feature_dictionary.csv.

硬性纪律 (主表第 7 章):
  1. 全历史逐股计算 -> 末端按事件行切片; 禁止"先截 N 天再算".
  2. 所有跨日价格比较走 pct_chg 链重构的 CF 复权序列 (主表 1.2 口径):
     R=pct_chg/100; CF=cumprod(1+R); f=CF/C; C~=CF, O~=O*f, H~=H*f, L~=L*f.
     单日截面比率 (K 线形态/RET_ID/AMP1) 与跳空 RET_ON(=O/PC-1) 用原始价.
  3. 列名黑名单 14 条正则 (BLACKLIST_PATTERNS) 对全部输出列 assert.
  4. 标签命名空间隔离: 引擎只读 events.parquet, 绝不读 labels.parquet.
  5. 滚动算子一律 min_periods=window (全窗口才出值, 新股不足窗口 = NaN);
     递推类 (EMA/MACD/RSI/KDJ/OBV/ATR) 全历史自然预热.

生成器层 (主表第 5 章, 1500-2500 列) 本任务不实现, 仅以 GeneratorRegistry 预留注册表结构.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from numpy.lib.stride_tricks import sliding_window_view

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "stock_data" / "daily"
INDEX_CODES = ("000001.SH", "399001.SZ")  # 指数文件, schema 与个股不同
LN2 = float(np.log(2.0))

# ================================================================ 泄漏防线: 列名黑名单 (主表 7.1)
# 14 条规则逐条展开为独立正则 (含分支), 匹配忽略大小写.
BLACKLIST_PATTERNS = [
    r"^stop_loss_",                 # 1  V3 实锤泄漏族
    r"^future_",                    # 2
    r"^next_",                      # 3
    r"^label",                      # 4
    r"^mfr_",                       # 5
    r"^cur_return$",                # 6a
    r"^max_forward_return$",        # 6b
    r"^open_exec_return",           # 6c
    r"^rank_future_",               # 7a
    r"^rank_open_exec_",            # 7b
    r"^ret_h\d+$",                  # 8
    r"^hit_N",                      # 9
    r"^mfe_",                       # 10a
    r"^mae_",                       # 10b
    r"^tmfe",                       # 10c
    r"^tmae",                       # 10d
    r"^dyn_",                       # 11
    r"^entry_date$",                # 12
    r"^rank_",                      # 13 (含 7a/7b; 横截面变体一律用后缀 _csrank/_csz)
]
_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]

# 主表 1.1 字段白名单 (唯一允许的数据源字段)
ALLOWED_INPUTS = frozenset({
    "open", "high", "low", "close", "pre_close", "change", "pct_chg",
    "vol", "amount", "trade_date", "ts_code",           # 个股日线
    "idx_open", "idx_high", "idx_low", "idx_close", "idx_volume",  # 指数日线
    "list_date",                                        # universe_latest
})

# 已知标签命名空间 (主表 7.2-3): 特征矩阵不得包含
LABEL_NAMESPACE_RE = re.compile(r"^(ret_h\d+|hit_N.*|mfe_.*|dyn|cur_return|max_forward_return)$")


def assert_no_blacklisted(columns):
    """特征矩阵任何列命中黑名单正则即构建失败 (CI 硬卡点)."""
    bad = [c for c in columns if any(rx.match(c) for rx in _BLACKLIST_RE)]
    assert not bad, f"特征矩阵列命中黑名单: {bad}"


def assert_no_label_columns(columns):
    bad = [c for c in columns if LABEL_NAMESPACE_RE.match(c)]
    assert not bad, f"特征矩阵混入标签列: {bad}"


# ================================================================ 特征注册表
@dataclass(frozen=True)
class FeatureSpec:
    column: str          # 输出列名
    feature: str         # 主表规范名 (多列特征共享一个规范名)
    layer: str           # 'P0' | 'P1' | 'G'
    family: str          # 主表家族
    formula: str         # 公式 (本地字段口径)
    inputs: tuple        # 输入字段 (ALLOWED_INPUTS 白名单内)
    event_only: bool = False  # 事件结构特征: 只在事件行有定义


FEATURE_REGISTRY: dict[str, FeatureSpec] = {}


def _reg(column, feature, layer, family, formula, inputs, event_only=False):
    spec = FeatureSpec(column, feature, layer, family, formula, tuple(inputs), event_only)
    assert column not in FEATURE_REGISTRY, f"重复注册: {column}"
    FEATURE_REGISTRY[column] = spec
    return spec


def assert_registry_inputs_whitelisted():
    for c, s in FEATURE_REGISTRY.items():
        bad = set(s.inputs) - ALLOWED_INPUTS
        assert not bad, f"{c} 输入字段越出白名单: {bad}"


class GeneratorRegistry:
    """G 层 (程序化生成层, 主表第 5 章) 预留注册表结构.

    生成特征一律 layer='G'、无先验证据, 表达式字符串入册防口径漂移;
    准入走主表 6.3 稳定性三关 (泄漏结构关/分年稳定关/共线性关).
    本任务不实现生成算子, 仅固化注册接口与命名/字段/算子约定.
    """

    BASE_FIELDS = ("O~", "H~", "L~", "C~", "V", "log(V+1)", "A", "VWAP",
                   "R", "RET_ON", "RET_ID", "H/L", "CF", "R_idx")
    OPERATORS = ("Ref", "Delta", "Mean", "Sum", "Std", "Var", "Skew", "Kurt",
                 "Max", "Min", "Med", "Quant", "TsRank", "IdxMax", "IdxMin",
                 "Slope", "Rsquare", "Resi", "EMA", "WMA", "Count", "Corr", "Cov")
    WINDOWS = (3, 5, 10, 20, 30, 60)
    POSTOPS = ("/C~", "-1x", "cs_rank", "zscore_ts")

    def __init__(self):
        self.specs: dict[str, FeatureSpec] = {}

    def register(self, name: str, expression: str, family: str = "generated") -> FeatureSpec:
        """注册一条生成特征; expression 为 qlib/Alpha191 风格表达式字符串."""
        assert_no_blacklisted([name])
        spec = FeatureSpec(name, name, "G", family, expression, tuple(), False)
        self.specs[name] = spec
        return spec


# ---------------------------------------------------------------- P0 注册 (60 条, 主表 2.1)
_OHLC = ("open", "high", "low", "close")
_P0 = [
    ("RET1", "反转", "R (= pct_chg/100)", ("pct_chg",)),
    ("RET5", "反转", "CF/shift(CF,5)-1", ("pct_chg",)),
    ("RET10", "反转", "CF/shift(CF,10)-1", ("pct_chg",)),
    ("RET20", "反转", "CF/shift(CF,20)-1", ("pct_chg",)),
    ("RET_ON", "隔夜/日内", "O/PC - 1", ("open", "pre_close")),
    ("RET_ID", "隔夜/日内", "C/O - 1", ("open", "close")),
    ("CUMON20", "隔夜/日内", "Sum(log1p(RET_ON), 20)", ("open", "pre_close")),
    ("CUMID20", "隔夜/日内", "Sum(log1p(RET_ID), 20)", ("open", "close")),
    ("VOL5", "波动率", "Std(R,5)", ("pct_chg",)),
    ("VOL15", "波动率", "Std(R,15)", ("pct_chg",)),
    ("VOL20", "波动率", "Std(R,20)", ("pct_chg",)),
    ("IVOL60", "波动率", "60 日 R = a+b*R_idx OLS 残差的 std (指数 000001.SH)", ("pct_chg", "idx_close")),
    ("AMP1", "波动率", "(H-L)/PC", ("high", "low", "pre_close")),
    ("AMP20", "波动率", "Mean((H-L)/PC, 20)", ("high", "low", "pre_close")),
    ("ATRN", "波动率", "ATR14(H~,L~,C~)/C~ (Wilder 14)", ("high", "low", "close", "pct_chg")),
    ("ILLIQ20", "流动性", "log(Mean(abs(R)/(A*1000), 20) + 1e-30)", ("pct_chg", "amount")),
    ("AMT20", "流动性", "log(Mean(A,20))", ("amount",)),
    ("VR5_60", "量能", "Mean(V,5)/Mean(V,60)", ("vol",)),
    ("VR1_20", "量能", "V/Mean(V,20)", ("vol",)),
    ("VOL_MOM", "量能", "(Mean(V,5)-Mean(V,20))/Mean(V,20)", ("vol",)),
    ("CPV10", "量价相关", "Corr(C~, V, 10)", ("close", "pct_chg", "vol")),
    ("CPV20", "量价相关", "Corr(C~, V, 20)", ("close", "pct_chg", "vol")),
    ("CPV_VWAP10", "量价相关", "-Corr(A/V, V, 10)", ("amount", "vol")),
    ("HLV_DIV10", "量价相关", "-Corr(H/L, V, 10)", ("high", "low", "vol")),
    ("MAX20", "彩票", "Max(R,20)", ("pct_chg",)),
    ("MIN20", "彩票", "Min(R,20)", ("pct_chg",)),
    ("MKT_RET20", "市场状态", "C_idx/shift(C_idx,20)-1", ("idx_close",)),
    ("MKT_MA60", "市场状态", "C_idx/Mean(C_idx,60)-1", ("idx_close",)),
    ("MKT_AVG_AMP", "市场状态", "两指数 (H-L)/L 当日均值", ("idx_high", "idx_low")),
    ("MKT_SYNC_SCORE", "市场状态", "1 - min(1, |pc_sh-pc_sz|/0.02), pc=(C-O)/O (v2 market_sync_score 口径)",
     ("idx_open", "idx_close")),
    ("LIMITCNT20", "制度", "Count(pct_chg >= limit-0.1, 20), limit 按板块 10/20/30", ("pct_chg", "trade_date", "ts_code")),
    ("LIMITDOWN_CNT20", "制度", "Count(pct_chg <= -(limit-0.1), 20)", ("pct_chg", "trade_date", "ts_code")),
    ("MACD_DIF_NORM", "技术指标", "DIF/(C~*ATRp)", ("close", "pct_chg", "high", "low")),
    ("MACD_HIST", "技术指标", "2*(DIF-DEA)", ("close", "pct_chg")),
    ("MACD_HIST_SLOPE5", "技术指标", "Slope(MACD_HIST, 5)", ("close", "pct_chg")),
    ("MACD_ZERO_CROSS", "技术指标", "DIF 穿零轴标记 (上穿+1/下穿-1, 仅穿越日非零)", ("close", "pct_chg")),
    ("RSI6", "技术指标", "Wilder RSI(C~,6)", ("close", "pct_chg")),
    ("RSI14", "技术指标", "Wilder RSI(C~,14)", ("close", "pct_chg")),
    ("BIAS20", "技术指标", "C~/Mean(C~,20)-1", ("close", "pct_chg")),
    ("BIAS60", "技术指标", "C~/Mean(C~,60)-1", ("close", "pct_chg")),
    ("BOLL_SQUEEZE", "技术指标", "1[(4*Std(C~,20))/Mean(C~,20) < 0.05]", ("close", "pct_chg")),
    ("DIST_SUPPORT20", "位置结构", "(C~-Min(L~,20))/C~", ("close", "low", "pct_chg")),
    ("DIST_RESIST20", "位置结构", "(Max(H~,20)-C~)/C~", ("close", "high", "pct_chg")),
    ("SR_RATIO20", "位置结构", "Mean(H~,20)/Mean(L~,20) (RSRS 思路)", ("high", "low", "pct_chg")),
    ("EFF_RATIO10", "位置结构", "abs(C~-shift(C~,10))/Sum(abs(dC~),10) (Kaufman)", ("close", "pct_chg")),
    ("DIST_HIGH60", "位置结构", "C~/Max(H~,60) (取 high 口径, 并入 drawdown_60)", ("close", "high", "pct_chg")),
    ("CLOSE_VS_HIGH", "K线形态", "(H-C)/(H-L), 一字板置 0.5", ("open", "high", "low", "close")),
    ("PRICE_TREND5", "位置结构", "Slope(C~,5)/C~", ("close", "pct_chg")),
    ("MA_STACK", "趋势", "sign(C~-MA60)+sign(MA60-MA120)+sign(MA120-MA250) (加总)", ("close", "pct_chg")),
    ("TREND_SLOPE_120", "趋势", "Slope(log(C~),120)*120 / (Std(R,120)*sqrt(120)) (t 统计量口径)", ("close", "pct_chg")),
    ("RET20_CSR", "反转×截面", "cs_rank(RET20) 横截面百分位 (全 universe 当日)", ("pct_chg",)),
]
for _name, _fam, _f, _inp in _P0:
    _ev = _name in ("RSI_DIV",)
    _reg(_name, _name, "P0", _fam, _f, _inp, event_only=_ev)

# P0 事件结构 (39, 52-59): 只在事件行有定义
_P0_EVENT = [
    ("RSI_DIV", "1[L~[i2]<L~[i1] 且 RSI14[i2]>RSI14[i1]]"),
    ("DIV_COUNT_120", "过去 120 交易日内同池同向底背离次数 (含本次), 窗口 (T-120, T]"),
    ("DIV_HIST_AREA_SHRINK", "S2/S1, Sk = 第 k 低点前最近绿柱区间 Sum(HIST|HIST<0)"),
    ("DIV_GOLDEN_CROSS_STATE", "三值: 2=T 日金叉; 1=T 前 5 日内已金叉; 0=未金叉"),
    ("REBOUND_FROM_L2", "(C~[T]-L~[i2])/ATR14[T]"),
    ("DIV_PRICE_NEWLOW_DEPTH", "L~[i2]/L~[i1]-1"),
    ("DIV_DIF_LIFT", "(DIF[i2]-DIF[i1])/ATR14[T]"),
    ("DIV_SPAN_BARS", "i2-i1 (= formation)"),
    ("DAYS_SINCE_L2", "T-i2 (= confirm_lag)"),
]
for _name, _f in _P0_EVENT:
    _reg(_name, _name, "P0", "事件结构", _f, ("open", "high", "low", "close", "pct_chg", "vol"),
         event_only=True)


# ---------------------------------------------------------------- P1 注册 (81 条, 主表第 3 章)
_P1 = [
    ("RET60", "RET60", "反转", "CF/shift(CF,60)-1", ("pct_chg",)),
    ("RET120_20", "RET120_20", "动量", "shift(CF,20)/shift(CF,240)-1 (剔除近 1 月的中期动量)", ("pct_chg",)),
    ("GAP_MEAN20", "GAP_MEAN20", "隔夜", "Mean(RET_ON,20)", ("open", "pre_close")),
    ("GAP_VOL20", "GAP_VOL20", "隔夜", "Std(RET_ON,20)", ("open", "pre_close")),
    ("INFO_DISC20", "INFO_DISC20", "反转", "sign(RET20)*(Count(R<0,20)-Count(R>0,20))/20 (Frog-in-the-Pan)", ("pct_chg",)),
    ("T1_GAP", "T1_GAP", "制度", "Mean(RET_ON[t] | RET_ID[t-1] < -2%, 20)", ("open", "close", "pre_close")),
    ("VOL60", "VOL60", "波动率", "Std(R,60)", ("pct_chg",)),
    ("DVOL", "DVOL", "波动率", "Std(R,5)-Std(R,60)", ("pct_chg",)),
    ("PARK20", "PARK20", "波动率", "sqrt(Sum(log(H/L)^2,20)/(4*ln2*20)) (Parkinson)", ("high", "low")),
    ("GK_VOL", "GK_VOL", "波动率", "Mean20( sqrt(max(0.5*ln(H/L)^2-(2ln2-1)*ln(C/O)^2, 0)) ) (Garman-Klass)", _OHLC),
    ("GK_VOL_RATIO", "GK_VOL_RATIO", "波动率", "GK_VOL/Mean(GK_VOL,20)", _OHLC),
    ("VOL_CONTRACTION60", "VOL_CONTRACTION60", "波动率", "ATRp/shift(ATRp,60)", ("high", "low", "close", "pct_chg")),
    ("BBW_PCTILE250", "BBW_PCTILE250", "波动率", "pct_ts(4*Std(C~,20)/Mean(C~,20), 250)", ("close", "pct_chg")),
    ("VMA5", "VMA5", "量能", "V/Mean(V,5) (量比近似)", ("vol",)),
    ("VSTD20", "VSTD20", "量能", "Std(V,20)/(Mean(V,20)+1)", ("vol",)),
    ("VOLUME_TREND10", "VOLUME_TREND10", "量能", "Slope(V,10)/Mean(V,20)", ("vol",)),
    ("OBV_TREND", "OBV_TREND", "量能", "Slope(OBV,5)/Mean(V,20), OBV=sign(R)*V 累积", ("close", "pct_chg", "vol")),
    ("OBV_DIV", "OBV_DIV", "量价背离", "zscore_ts(C~,20)-zscore_ts(OBV,20)", ("close", "pct_chg", "vol")),
    ("PVR20", "PVR20", "量价背离", "Mean(V|R>0,20)/Mean(V|R<0,20) (最少 5 个上涨日否则 NaN)", ("pct_chg", "vol")),
    ("VSHRINK", "VSHRINK", "量价背离", "Mean(V|R<0,近10)/Mean(V|R<0,前10)", ("pct_chg", "vol")),
    ("VOL_DRYUP_EXTREME", "VOL_DRYUP_EXTREME", "量能", "1[Mean(V,5) < Quant(Mean(V,5),250,0.1)]", ("vol",)),
    ("AMT_SHRINK_PEAK", "AMT_SHRINK_PEAK", "量能", "Mean(A,5)/Max(Mean(A,5),120)", ("amount",)),
    ("RVC20", "RVC20", "量价相关", "Corr(R, V/shift(V,1)-1, 20)", ("pct_chg", "vol")),
    ("CPV_TREND", "CPV_TREND", "量价相关", "Slope(Corr(C~,V,10), 10)", ("close", "pct_chg", "vol")),
    ("DSR20", "DSR20", "波动结构", "Sum(min(R,0)^2,20)/Sum(R^2,20)", ("pct_chg",)),
    ("SJV60", "SJV60", "波动结构", "Sum(R^2*1[R>th],60)-Sum(R^2*1[R<-th],60), th=2*Std(R,60)", ("pct_chg",)),
    ("JUMPFREQ60", "JUMPFREQ60", "波动结构", "Count(abs(R)>2*Std(R,60),60)/60", ("pct_chg",)),
    ("MAX5_20", "MAX5_20", "彩票", "20 日内最大 5 个 R 的均值", ("pct_chg",)),
    ("SKEW60", "SKEW60", "高阶矩", "Skew(R,60)", ("pct_chg",)),
    ("KURT60", "KURT60", "高阶矩", "Kurt(R,60)", ("pct_chg",)),
    ("IVOV60", "IVOV60", "高阶矩", "Std(Std(R,5),60)", ("pct_chg",)),
    ("CAL_MONTH_POS", "CAL_MONTH_POS", "日历", "距月末交易日数 (同月内之后的工作日数, 静态工作日近似, 截断不变)", ("trade_date",)),
    ("CAL_HOLIDAY", "CAL_HOLIDAY", "日历", "春节/国庆前 10 个工作日、后 5 个工作日 dummy (静态节假日表+工作日近似, 截断不变)", ("trade_date",)),
    ("MKT_VOL20_PCT", "MKT_VOL20_PCT", "市场状态", "pct_ts(Std(R_idx,20),250)", ("idx_close",)),
    ("MKT_DD120", "MKT_DD120", "市场状态", "C_idx/Max(H_idx,120)-1", ("idx_close", "idx_high")),
    ("MKT_RSI14", "MKT_RSI14", "市场状态", "RSI(C_idx,14)", ("idx_close",)),
    ("BREADTH_ADV5", "BREADTH_ADV5", "市场宽度", "universe 内 RET5>0 占比 (自算)", ("pct_chg",)),
    ("BREADTH_NEWLOW", "BREADTH_NEWLOW", "市场宽度", "universe 内 L~=Min(L~,250) 占比 (自算)", ("low", "pct_chg")),
    ("BREADTH_ABOVE_MA20", "BREADTH_ABOVE_MA20", "市场宽度", "universe 内 C~>Mean(C~,20) 占比 (自算)", ("close", "pct_chg")),
    ("MKT_MEDIAN_RET20", "MKT_MEDIAN_RET20", "市场宽度", "universe 内 RET20 中位数 (自算)", ("pct_chg",)),
    ("MKT_AMT_PCT", "MKT_AMT_PCT", "市场情绪", "全市场 amount 加总 20 日均值的 250 日时序分位", ("amount",)),
    ("REGIME_CODE", "REGIME_CODE", "市场状态",
     "全样本等权日收益累积指数 120 日滚动收益 -> unknown/sideways/up/down = -1/0/1/2 "
     "(divergence_lab build_market_regime 口径, universe 剔除 2 只指数文件)", ("pct_chg",)),
    ("DIST_LIMIT", "DIST_LIMIT", "制度", "clip(pct_chg/limit_pct, -1, 1), 1=收涨停 (近似口径)", ("pct_chg", "trade_date", "ts_code")),
    ("DISPOSAL60", "DISPOSAL60", "制度", "C/(Sum(A,60)/Sum(V,60)) - 1 (现价对 60 日 VWAP 成本偏离)", ("close", "amount", "vol")),
    ("NEW_LISTING", "NEW_LISTING", "制度", "1[T - list_date < 250 交易日] (= 股内行号 < 250)", ("trade_date", "list_date")),
    ("PCTB", "PCTB", "技术指标", "(C~-Mean(C~,20))/(2*Std(C~,20)) (%B)", ("close", "pct_chg")),
    ("RSI_OVERSOLD_DAYS", "RSI_OVERSOLD_DAYS", "技术指标", "RSI14<30 连续天数", ("close", "pct_chg")),
    ("RSV9", "RSV9", "技术指标", "(C~-Min(L~,9))/(Max(H~,9)-Min(L~,9))", ("high", "low", "close", "pct_chg")),
    ("CNTD20", "CNTD20", "技术指标", "(Count(R>0,20)-Count(R<0,20))/20", ("pct_chg",)),
    ("SUMD20", "SUMD20", "技术指标", "(Sum(max(R,0),20)-Sum(max(-R,0),20))/Sum(abs(R),20)", ("pct_chg",)),
    ("TSRANK20", "TSRANK20", "位置", "TsRank(C~,20) (最新值在 20 日内分位)", ("close", "pct_chg")),
    ("IMIN20", "IMIN20", "位置", "IdxMin(L~,20)/20 (最值距今天数/20)", ("low", "pct_chg")),
    ("IMXD20", "IMXD20", "位置", "(IdxMax(H~,20)-IdxMin(L~,20))/20", ("high", "low", "pct_chg")),
    ("MACD_HIST_AMP20", "MACD_HIST_AMP20", "技术指标", "Max(HIST,20)-Min(HIST,20)", ("close", "pct_chg")),
    ("MACD_GOLDEN_CROSS", "MACD_GOLDEN_CROSS", "技术指标", "DIF 上穿 DEA 标记 (日频, 非事件窗)", ("close", "pct_chg")),
    ("UPPER_SHADOW_RATIO", "UPPER_SHADOW_RATIO", "K线形态", "(H-max(O,C))/(H-L), 一字板置 0", _OHLC),
    ("BODY_STRENGTH", "BODY_STRENGTH", "K线形态", "(C-O)/(H-L+1e-12)", _OHLC),
    ("DOJI", "DOJI", "K线形态", "1[abs(C-O) < 0.1*(H-L)]", _OHLC),
    ("ENGULFING", "ENGULFING", "K线形态", "前阴今阳/前阳今阴且实体吞没 (+1/-1/0)", _OHLC),
    ("POS_52W", "POS_52W", "位置", "(C~-Min(L~,250))/(Max(H~,250)-Min(L~,250))", ("high", "low", "close", "pct_chg")),
    ("DIST_52W_LOW", "DIST_52W_LOW", "位置", "C~/Min(L~,250)-1", ("low", "close", "pct_chg")),
    ("DIST_52W_HIGH", "DIST_52W_HIGH", "位置", "C~/Max(H~,250)-1", ("high", "close", "pct_chg")),
    ("AMIHUD_INTRADAY", "AMIHUD_INTRADAY", "流动性", "abs(C/O-1)/(V*C) (日内口径)", ("open", "close", "vol")),
    ("HL_SPREAD", "HL_SPREAD", "流动性", "(H-L)/((H+L)/2)", ("high", "low")),
    ("PRICE_IMPACT", "PRICE_IMPACT", "流动性", "abs(C-O)/sqrt(V)", ("open", "close", "vol")),
    ("LOG_PRICE", "LOG_PRICE", "价格水平", "ln(C) (原始价, 除权失真标注: 仅作彩票/退市代理)", ("close",)),
]
for _col, _name, _fam, _f, _inp in _P1:
    _reg(_col, _name, "P1", _fam, _f, _inp)

# P1 多列展开
for _i in range(5):
    _reg(f"CAL_DOW_{_i}", "CAL_DOW", "P1", "日历", "信号日星期 one-hot (0=Mon..4=Fri)", ("trade_date",))
for _m in range(1, 13):
    _reg(f"CAL_MONTH_{_m}", "CAL_MONTH", "P1", "日历", "月份 one-hot", ("trade_date",))
for _c, _f in (("KDJ_K", "RSV9 递推 K=EMA(RSV,1/3)"), ("KDJ_D", "D=EMA(K,1/3)"),
               ("KDJ_J", "J=3K-2D (输出 K/D/J 三列)")):
    _reg(_c, "KDJ_J", "P1", "技术指标", _f, ("high", "low", "close", "pct_chg"))
_reg("MA200_RATIO", "MA200_RATIO", "P1", "趋势", "C~/Mean(C~,200)-1", ("close", "pct_chg"))
_reg("MA200_ABOVE", "MA200_RATIO", "P1", "趋势", "1[C~ > MA200]", ("close", "pct_chg"))
_reg("HAMMER", "HAMMER", "P1", "K线形态",
     "1[下影>=2*实体 且 上影<=0.5*实体 且 下影>=0.6*全长] (T 日)", _OHLC)
_reg("DOWN_STREAK", "DOWN_STREAK", "P1", "K线形态", "T 日末连续 C<O 天数", ("open", "close"))
_reg("DOWN_DAY_PCT10", "DOWN_STREAK", "P1", "K线形态", "10 日阴线 (C<O) 占比", ("open", "close"))
_reg("LIMITUP_N_MKT", "LIMITUP_N_MKT", "P1", "市场情绪", "全市场涨停家数 (pct_chg>=limit-0.1, 自算)",
     ("pct_chg", "trade_date", "ts_code"))
_reg("LIMITUP_N_MKT_MA20", "LIMITUP_N_MKT", "P1", "市场情绪", "LIMITUP_N_MKT 的 20 日均值",
     ("pct_chg", "trade_date", "ts_code"))

# P1 MKT_IDX_DAILY 族 (v2 §2.13 的 20 列口径; 均值版振幅/同步度由 P0 MKT_AVG_AMP/MKT_SYNC_SCORE 承接)
_IDX_FAM = [
    ("MKT_SH_PRICE_CHANGE", "上证指数 (C-O)/O"), ("MKT_SZ_PRICE_CHANGE", "深证成指 (C-O)/O"),
    ("MKT_SH_AMPLITUDE", "上证指数 (H-L)/L"), ("MKT_SZ_AMPLITUDE", "深证成指 (H-L)/L"),
    ("MKT_SH_VOLUME_RATIO", "上证指数量/20 日均量"), ("MKT_SZ_VOLUME_RATIO", "深证成指量/20 日均量"),
    ("MKT_SH_PRICE_CHANGE_ABS", "|MKT_SH_PRICE_CHANGE|"), ("MKT_SZ_PRICE_CHANGE_ABS", "|MKT_SZ_PRICE_CHANGE|"),
    ("MKT_SH_PRICE_WAVE_ABS", "上证指数振幅绝对值"), ("MKT_SZ_PRICE_WAVE_ABS", "深证成指振幅绝对值"),
    ("MKT_SH_SENTIMENT", "涨>1%且振幅<2%->2; 跌>1%且振幅>3%->0; 否则 1 (上证)"),
    ("MKT_SZ_SENTIMENT", "同上 (深证)"),
    ("MKT_SH_VOLUME_SIGNAL", "1[量比>1.2] (上证)"), ("MKT_SZ_VOLUME_SIGNAL", "1[量比>1.2] (深证)"),
    ("MKT_SH_SZ_SYNC_DIRECTION", "1[两指数 (C-O)/O 同号]"),
    ("MKT_SH_SZ_SYNC_STRENGTH", "|pc_sh - pc_sz|"),
    ("MKT_SENTIMENT", "两指数均值版情绪 (规则同单指数)"),
    ("MKT_AVG_CHANGE", "两指数 (C-O)/O 均值"),
]
for _c, _f in _IDX_FAM:
    _reg(_c, "MKT_IDX_DAILY", "P1", "市场状态", _f, ("idx_open", "idx_high", "idx_low", "idx_close", "idx_volume"))

# P1 事件结构 (72-77, 61/62 事件侧)
_P1_EVENT = [
    ("DIV_SPAN_VS_CYCLE", "(i2-i1)/自 T 起最近一次 DIF 完整正负循环时长 (最后两个已完成异号 run 长度和)"),
    ("DIV_DIF_SLOPE", "(DIF[i2]-DIF[i1])/(i2-i1)/ATR14[T]"),
    ("DIV_ZERO_AXIS_DEPTH", "max(DIF[T],DEA[T])/(C~[T]*ATRp[T])"),
    ("DIV_DIF_LEVEL_L1", "DIF[i1]/(C~[i1]*ATRp[i1])"),
    ("DIV_HIST_TROUGH_SHRINK", "min(HIST@i2 绿柱区)/min(HIST@i1 绿柱区) (负值之比, <1 为收缩)"),
    ("REBOUND_DAY_T", "(C~[T]-O~[T])/ATR14[T] (等价原始价口径 (C-O)/ATR)"),
    ("HAMMER_L2", "HAMMER 形态在 i2 日的取值"),
    ("LOWER_SHADOW_L2", "(min(O~,C~)-L~)[i2]/ATR14[i2]"),
]
for _name, _f in _P1_EVENT:
    _canon = "HAMMER" if _name == "HAMMER_L2" else _name
    _reg(_name, _canon, "P1", "事件结构", _f, ("open", "high", "low", "close", "pct_chg", "vol"),
         event_only=True)

assert_registry_inputs_whitelisted()


# ================================================================ 基础工具
def _to_days(s):
    """trade_date -> int32 日序 (datetime64[D]), 兼容 datetime64 / YYYYMMDD 字符串."""
    arr = s.to_numpy() if hasattr(s, "to_numpy") else np.asarray(s)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]").astype(np.int32)
    return pd.to_datetime(pd.Series(arr).astype(str)).to_numpy("datetime64[D]").astype(np.int32)


def load_stock_df(path):
    """读取单股 parquet 并清洗 (与 divergence_lab.load_stock 完全同口径, 保证 sig_idx 对齐)."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    return df.reset_index(drop=True)


def load_index_df(path):
    """读取指数 parquet (schema 不同: trade_date 为 YYYYMMDD 字符串, 仅 OHLC+volume)."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    df = df.reset_index(drop=True)
    df["_days"] = _to_days(df["trade_date"])
    return df


def _roll(a, w):
    return pd.Series(a, copy=False).rolling(w, min_periods=w)


def _slope(a, w):
    """对时间 OLS 斜率 (x=0..w-1 固定权重相关, 全窗口才出值)."""
    a = np.asarray(a, np.float64)
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    x = np.arange(w, dtype=np.float64)
    k = x - x.mean()
    k /= (k * k).sum()
    valid = np.isfinite(a)
    y0 = np.where(valid, a, 0.0)
    num = np.correlate(y0, k, mode="valid")
    cnt = np.correlate(valid.astype(np.float64), np.ones(w), mode="valid")
    out[w - 1:] = np.where(cnt == w, num, np.nan)
    return out


def _ts_pct_rank(a, w):
    """时序分位: 当前值在过去 w 日 (含当日) 内分位 = (less + 0.5*ties)/w; 全窗口才出值."""
    a = np.asarray(a, np.float64)
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    sw = sliding_window_view(a, w)
    last = sw[:, -1:]
    bad = np.isnan(sw).any(axis=1)
    less = (sw < last).sum(axis=1)
    ties = (sw == last).sum(axis=1)
    val = (less + 0.5 * ties) / w
    out[w - 1:] = np.where(bad, np.nan, val)
    return out


def _idx_extreme(a, w, mode):
    """最值距今天数 (0=今天, w-1=最旧); 全窗口才出值. ties 取最旧出现."""
    a = np.asarray(a, np.float64)
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    sw = sliding_window_view(a, w)
    bad = np.isnan(sw).any(axis=1)
    fill = -np.inf if mode == "max" else np.inf
    sw2 = np.where(np.isnan(sw), fill, sw)
    idx = np.argmax(sw2, axis=1) if mode == "max" else np.argmin(sw2, axis=1)
    dist = (w - 1) - idx
    out[w - 1:] = np.where(bad, np.nan, dist.astype(np.float64))
    return out


def _topk_mean(a, w, k):
    """窗口内最大 k 个值的均值; 全窗口才出值."""
    a = np.asarray(a, np.float64)
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    sw = sliding_window_view(a, w)
    bad = np.isnan(sw).any(axis=1)
    part = np.partition(sw, w - k, axis=1)[:, w - k:]
    out[w - 1:] = np.where(bad, np.nan, part.mean(axis=1))
    return out


def _streak(mask):
    """连续 True 天数 (截至当日, 当日 False 则 0); NaN 比较产生的 False 自然断链."""
    mask = np.asarray(mask, bool)
    idx = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(mask, -1, idx))
    return np.where(mask, idx - last_false, 0).astype(np.float64)


def _safe_div(num, den):
    """逐元素除法, 0/0 与 x/0 -> NaN."""
    num = np.asarray(num, np.float64)
    den = np.asarray(den, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def limit_pct_array(code, days):
    """板块涨停幅度 (%): 688->20; 300/301->2020-08-24 起 20 此前 10; .BJ->30; 其余 10.
    ST 5% 无法识别 (无历史 ST 状态), 误差接受 (主表 2.2 备注)."""
    n = len(days)
    if code.endswith(".BJ"):
        return np.full(n, 30.0)
    if code.startswith("688"):
        return np.full(n, 20.0)
    if code.startswith(("300", "301")):
        cut = np.datetime64("2020-08-24").astype("datetime64[D]").astype(np.int32)
        return np.where(days >= cut, 20.0, 10.0)
    return np.full(n, 10.0)


# ================================================================ 单股特征计算 (全历史, 因果)
def compute_stock_features(df, code, idx_dates=None, idx_r=None):
    """对单股全历史计算全部 P0/P1 股内特征 (主表 2/3 章).

    返回 (feats, ctx):
      feats: DataFrame, 行=股内行号 (与 df 对齐), 列=注册表股内特征 (f64);
      ctx:   dict, 事件特征与面板所需的内部数组 (复权价/指标/面板列等).
    idx_dates/idx_r: 000001.SH 的日序数组与日收益数组 (IVOL60 用; None 则 IVOL60 全 NaN).
    """
    n = len(df)
    days = _to_days(df["trade_date"])
    C = df["close"].to_numpy(np.float64)
    O = df["open"].to_numpy(np.float64)
    H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64)
    if "pct_chg" in df.columns:
        pct = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(np.float64)
    else:  # 指数文件兜底 (与 divergence_lab 一致)
        pct = np.concatenate([[np.nan], C[1:] / C[:-1] * 100.0 - 100.0])
    if "vol" in df.columns:
        V = pd.to_numeric(df["vol"], errors="coerce").to_numpy(np.float64)
    elif "volume" in df.columns:
        V = pd.to_numeric(df["volume"], errors="coerce").to_numpy(np.float64)
    else:
        V = np.full(n, np.nan)
    A = pd.to_numeric(df["amount"], errors="coerce").to_numpy(np.float64) \
        if "amount" in df.columns else np.full(n, np.nan)
    PC = pd.to_numeric(df["pre_close"], errors="coerce").to_numpy(np.float64) \
        if "pre_close" in df.columns else np.full(n, np.nan)

    # ---- CF 复权链 (主表 1.2)
    R = pct / 100.0
    cf = np.cumprod(1.0 + np.where(np.isfinite(R), R, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        f_adj = np.where(C > 0, cf / C, np.nan)
    ca, oa, ha, la = cf, O * f_adj, H * f_adj, L * f_adj

    out: dict[str, np.ndarray] = {}
    S = pd.Series

    # ---- 收益/隔夜 (P0 1-8, P1 1-5)
    out["RET1"] = R
    scf = S(cf)
    for w, name in ((5, "RET5"), (10, "RET10"), (20, "RET20"), (60, "RET60")):
        out[name] = (scf / scf.shift(w) - 1.0).to_numpy()
    out["RET120_20"] = (scf.shift(20) / scf.shift(240) - 1.0).to_numpy()
    ret_on = _safe_div(O, PC) - 1.0
    ret_id = _safe_div(C, O) - 1.0
    out["RET_ON"] = ret_on
    out["RET_ID"] = ret_id
    out["CUMON20"] = _roll(np.log1p(np.where(ret_on > -1, ret_on, np.nan)), 20).sum().to_numpy()
    out["CUMID20"] = _roll(np.log1p(np.where(ret_id > -1, ret_id, np.nan)), 20).sum().to_numpy()
    out["GAP_MEAN20"] = _roll(ret_on, 20).mean().to_numpy()
    out["GAP_VOL20"] = _roll(ret_on, 20).std().to_numpy()
    up20 = _roll((R > 0).astype(np.float64), 20).sum().to_numpy()
    dn20 = _roll((R < 0).astype(np.float64), 20).sum().to_numpy()
    r_nan20 = _roll(np.isfinite(R).astype(np.float64), 20).sum().to_numpy()
    with np.errstate(invalid="ignore"):
        out["INFO_DISC20"] = np.where(r_nan20 == 20,
                                      np.sign(out["RET20"]) * (dn20 - up20) / 20.0, np.nan)
        out["CNTD20"] = np.where(r_nan20 == 20, (up20 - dn20) / 20.0, np.nan)

    # T1_GAP: 前日 RET_ID < -2% 日的 RET_ON 条件均值 (20 日窗)
    cond_gap = (S(ret_id).shift(1) < -0.02).to_numpy()
    cond_gap = np.where(np.isfinite(S(ret_id).shift(1).to_numpy()), cond_gap, False)
    gap_sel = np.where(cond_gap, ret_on, np.nan)
    gap_num = pd.Series(gap_sel).rolling(20, min_periods=1).sum().to_numpy()
    gap_den = pd.Series(cond_gap.astype(np.float64)).rolling(20, min_periods=20).sum().to_numpy()
    out["T1_GAP"] = np.where(gap_den > 0, gap_num / gap_den, np.nan)

    # ---- 波动率 (P0 9-15, P1 7-13)
    for w, name in ((5, "VOL5"), (15, "VOL15"), (20, "VOL20"), (60, "VOL60")):
        out[name] = _roll(R, w).std().to_numpy()
    out["DVOL"] = out["VOL5"] - out["VOL60"]
    amp1 = _safe_div(H - L, PC)
    out["AMP1"] = amp1
    out["AMP20"] = _roll(amp1, 20).mean().to_numpy()
    atr14 = talib.ATR(ha, la, ca, timeperiod=14)
    atrp = _safe_div(atr14, ca)
    out["ATRN"] = atrp
    out["VOL_CONTRACTION60"] = (S(atrp) / S(atrp).shift(60)).to_numpy()
    log_hl = np.log(_safe_div(H, L))
    park_sum = _roll(log_hl * log_hl, 20).sum().to_numpy()
    out["PARK20"] = np.sqrt(np.where(park_sum >= 0, park_sum, np.nan) / (4.0 * LN2 * 20.0))
    log_co = np.log(_safe_div(C, O))
    gk_daily = np.sqrt(np.maximum(0.5 * log_hl**2 - (2.0 * LN2 - 1.0) * log_co**2, 0.0))
    gk_daily[~np.isfinite(log_hl) | ~np.isfinite(log_co)] = np.nan
    gk_vol = _roll(gk_daily, 20).mean()
    out["GK_VOL"] = gk_vol.to_numpy()
    out["GK_VOL_RATIO"] = (gk_vol / _roll(out["GK_VOL"], 20).mean()).to_numpy()
    std_c20 = _roll(ca, 20).std()
    mean_c20 = _roll(ca, 20).mean()
    bbw = _safe_div(4.0 * std_c20.to_numpy(), mean_c20.to_numpy())
    out["BOLL_SQUEEZE"] = np.where(np.isfinite(bbw), (bbw < 0.05).astype(np.float64), np.nan)
    out["BBW_PCTILE250"] = _ts_pct_rank(bbw, 250)

    # IVOL60: 60 日 R ~ R_idx OLS 残差 std (矩法)
    if idx_dates is not None and idx_r is not None and n > 0:
        pos = np.searchsorted(idx_dates, days)
        pos_c = np.clip(pos, 0, len(idx_dates) - 1)
        rx = np.where(idx_dates[pos_c] == days, idx_r[pos_c], np.nan)
        e_r, e_x = _roll(R, 60).mean(), _roll(rx, 60).mean()
        e_rr, e_xx = _roll(R * R, 60).mean(), _roll(rx * rx, 60).mean()
        e_rx = _roll(R * rx, 60).mean()
        var_r = (e_rr - e_r * e_r).to_numpy()
        var_x = (e_xx - e_x * e_x).to_numpy()
        cov = (e_rx - e_r * e_x).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            iv = var_r - np.where(var_x > 0, cov * cov / var_x, np.nan)
        out["IVOL60"] = np.sqrt(np.where(iv >= 0, iv, np.nan))
    else:
        out["IVOL60"] = np.full(n, np.nan)

    # ---- 流动性 (P0 16-17, P1 78-81)
    illiq = _safe_div(np.abs(R), A * 1000.0)
    out["ILLIQ20"] = np.log(_roll(illiq, 20).mean().to_numpy() + 1e-30)
    out["AMT20"] = np.log(_roll(A, 20).mean().to_numpy())
    out["AMIHUD_INTRADAY"] = _safe_div(np.abs(ret_id), V * C)
    out["HL_SPREAD"] = _safe_div(H - L, (H + L) / 2.0)
    out["PRICE_IMPACT"] = _safe_div(np.abs(C - O), np.sqrt(np.where(V > 0, V, np.nan)))
    out["LOG_PRICE"] = np.log(np.where(C > 0, C, np.nan))

    # ---- 量能 (P0 18-20, P1 14-22)
    mv5, mv20, mv60 = _roll(V, 5).mean(), _roll(V, 20).mean(), _roll(V, 60).mean()
    out["VR5_60"] = _safe_div(mv5.to_numpy(), mv60.to_numpy())
    out["VR1_20"] = _safe_div(V, mv20.to_numpy())
    out["VOL_MOM"] = _safe_div((mv5 - mv20).to_numpy(), mv20.to_numpy())
    out["VMA5"] = _safe_div(V, mv5.to_numpy())
    out["VSTD20"] = _safe_div(_roll(V, 20).std().to_numpy(), mv20.to_numpy() + 1.0)
    out["VOLUME_TREND10"] = _safe_div(_slope(V, 10), mv20.to_numpy())
    obv = talib.OBV(ca, V) if n else np.array([])
    out["OBV_TREND"] = _safe_div(_slope(obv, 5), mv20.to_numpy())
    z_c20 = _safe_div((S(ca) - mean_c20).to_numpy(), std_c20.to_numpy())
    obv_s = S(obv)
    z_obv20 = _safe_div((obv_s - _roll(obv, 20).mean()).to_numpy(), _roll(obv, 20).std().to_numpy())
    out["OBV_DIV"] = z_c20 - z_obv20
    v_up = np.where(R > 0, V, np.nan)
    v_dn = np.where(R < 0, V, np.nan)
    su = pd.Series(v_up).rolling(20, min_periods=1).sum().to_numpy()
    cu = pd.Series(v_up).rolling(20, min_periods=1).count().to_numpy()
    sd = pd.Series(v_dn).rolling(20, min_periods=1).sum().to_numpy()
    cd = pd.Series(v_dn).rolling(20, min_periods=1).count().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pvr = (su / cu) / (sd / cd)
    out["PVR20"] = np.where((cu >= 5) & (cd >= 1) & (r_nan20 == 20), pvr, np.nan)
    dn_near = pd.Series(v_dn).rolling(10, min_periods=1).mean()
    dn_prev = pd.Series(v_dn).shift(10).rolling(10, min_periods=1).mean()
    out["VSHRINK"] = _safe_div(dn_near.to_numpy(), dn_prev.to_numpy())
    q10_mv5 = pd.Series(mv5.to_numpy()).rolling(250, min_periods=250).quantile(0.1).to_numpy()
    out["VOL_DRYUP_EXTREME"] = np.where(np.isfinite(q10_mv5),
                                        (mv5.to_numpy() < q10_mv5).astype(np.float64), np.nan)
    ma5 = _roll(A, 5).mean()
    out["AMT_SHRINK_PEAK"] = _safe_div(ma5.to_numpy(), _roll(ma5.to_numpy(), 120).max().to_numpy())

    # ---- 量价相关 (P0 21-24, P1 23-24)
    sca, sv = S(ca), S(V)
    cpv10 = sca.rolling(10, min_periods=10).corr(sv).to_numpy()
    out["CPV10"] = cpv10
    out["CPV20"] = sca.rolling(20, min_periods=20).corr(sv).to_numpy()
    vwap = _safe_div(A, V)
    out["CPV_VWAP10"] = -S(vwap).rolling(10, min_periods=10).corr(sv).to_numpy()
    out["HLV_DIV10"] = -S(_safe_div(H, L)).rolling(10, min_periods=10).corr(sv).to_numpy()
    v_chg = _safe_div(V, S(V).shift(1).to_numpy()) - 1.0
    out["RVC20"] = S(R).rolling(20, min_periods=20).corr(S(v_chg)).to_numpy()
    out["CPV_TREND"] = _slope(cpv10, 10)

    # ---- 彩票/高阶矩/波动结构 (P0 25-26, P1 25-31)
    out["MAX20"] = _roll(R, 20).max().to_numpy()
    out["MIN20"] = _roll(R, 20).min().to_numpy()
    out["MAX5_20"] = _topk_mean(R, 20, 5)
    out["SKEW60"] = _roll(R, 60).skew().to_numpy()
    out["KURT60"] = _roll(R, 60).kurt().to_numpy()
    out["IVOV60"] = _roll(out["VOL5"], 60).std().to_numpy()
    r2 = R * R
    dsr_num = _roll(np.where(R < 0, r2, 0.0), 20).sum().to_numpy()
    dsr_den = _roll(r2, 20).sum().to_numpy()
    out["DSR20"] = _safe_div(dsr_num, dsr_den)
    theta = 2.0 * out["VOL60"]
    jump_up = np.where(R > theta, r2, 0.0)
    jump_dn = np.where(R < -theta, r2, 0.0)
    jump_up[~np.isfinite(theta)] = np.nan
    jump_dn[~np.isfinite(theta)] = np.nan
    out["SJV60"] = _roll(jump_up, 60).sum().to_numpy() - _roll(jump_dn, 60).sum().to_numpy()
    jump_ind = np.where(np.isfinite(theta) & np.isfinite(R),
                        (np.abs(R) > theta).astype(np.float64), np.nan)
    out["JUMPFREQ60"] = _roll(jump_ind, 60).mean().to_numpy()

    # ---- 技术指标 (P0 33-42, P1 50-55, 59-60)
    dif, dea, _ = talib.MACD(ca, fastperiod=12, slowperiod=26, signalperiod=9) if n else (None, None, None)
    if n == 0:
        dif = dea = np.array([])
    hist = 2.0 * (dif - dea)
    out["MACD_DIF_NORM"] = _safe_div(dif, ca * atrp)
    out["MACD_HIST"] = hist
    out["MACD_HIST_SLOPE5"] = _slope(hist, 5)
    dif_prev = S(dif).shift(1).to_numpy()
    zc = np.where((dif > 0) & (dif_prev <= 0), 1.0, np.where((dif < 0) & (dif_prev >= 0), -1.0, 0.0))
    out["MACD_ZERO_CROSS"] = np.where(np.isfinite(dif) & np.isfinite(dif_prev), zc, np.nan)
    rsi6 = talib.RSI(ca, timeperiod=6) if n else np.array([])
    rsi14 = talib.RSI(ca, timeperiod=14) if n else np.array([])
    out["RSI6"] = rsi6
    out["RSI14"] = rsi14
    out["BIAS20"] = _safe_div(ca, mean_c20.to_numpy()) - 1.0
    out["BIAS60"] = _safe_div(ca, _roll(ca, 60).mean().to_numpy()) - 1.0
    out["PCTB"] = _safe_div((S(ca) - mean_c20).to_numpy(), 2.0 * std_c20.to_numpy())
    out["RSI_OVERSOLD_DAYS"] = _streak(rsi14 < 30)
    llv9, hhv9 = _roll(la, 9).min().to_numpy(), _roll(ha, 9).max().to_numpy()
    rsv9 = _safe_div(ca - llv9, hhv9 - llv9)
    out["RSV9"] = rsv9
    kdj_k = pd.Series(rsv9).ewm(alpha=1.0 / 3.0, adjust=False).mean()
    kdj_d = kdj_k.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    out["KDJ_K"] = kdj_k.to_numpy()
    out["KDJ_D"] = kdj_d.to_numpy()
    out["KDJ_J"] = (3.0 * kdj_k - 2.0 * kdj_d).to_numpy()
    out["TSRANK20"] = _ts_pct_rank(ca, 20)
    imin20 = _idx_extreme(la, 20, "min")
    imax20 = _idx_extreme(ha, 20, "max")
    out["IMIN20"] = imin20 / 20.0
    out["IMXD20"] = (imax20 - imin20) / 20.0
    out["MACD_HIST_AMP20"] = (_roll(hist, 20).max() - _roll(hist, 20).min()).to_numpy()
    dea_prev = S(dea).shift(1).to_numpy()
    gc = ((dif > dea) & (dif_prev <= dea_prev)).astype(np.float64)
    out["MACD_GOLDEN_CROSS"] = np.where(np.isfinite(dif) & np.isfinite(dea), gc, np.nan)
    sum_r_abs20 = _roll(np.abs(R), 20).sum().to_numpy()
    sum_pos20 = _roll(np.where(R > 0, R, 0.0), 20).sum().to_numpy()
    sum_neg20 = _roll(np.where(R < 0, -R, 0.0), 20).sum().to_numpy()
    out["SUMD20"] = np.where(r_nan20 == 20, _safe_div(sum_pos20 - sum_neg20, sum_r_abs20), np.nan)

    # ---- 位置结构/趋势 (P0 43-51, P1 68-71)
    min_l20, max_h20 = _roll(la, 20).min().to_numpy(), _roll(ha, 20).max().to_numpy()
    out["DIST_SUPPORT20"] = _safe_div(ca - min_l20, ca)
    out["DIST_RESIST20"] = _safe_div(max_h20 - ca, ca)
    out["SR_RATIO20"] = _safe_div(_roll(ha, 20).mean().to_numpy(), _roll(la, 20).mean().to_numpy())
    dca = S(ca).diff().to_numpy()
    out["EFF_RATIO10"] = _safe_div(np.abs(S(ca) - S(ca).shift(10)).to_numpy(),
                                   _roll(np.abs(dca), 10).sum().to_numpy())
    out["DIST_HIGH60"] = _safe_div(ca, _roll(ha, 60).max().to_numpy())
    out["PRICE_TREND5"] = _safe_div(_slope(ca, 5), ca)
    ma60 = _roll(ca, 60).mean().to_numpy()
    ma120 = _roll(ca, 120).mean().to_numpy()
    ma250 = _roll(ca, 250).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        out["MA_STACK"] = np.sign(ca - ma60) + np.sign(ma60 - ma120) + np.sign(ma120 - ma250)
    log_ca = np.log(np.where(ca > 0, ca, np.nan))
    out["TREND_SLOPE_120"] = _safe_div(_slope(log_ca, 120) * 120.0,
                                       _roll(R, 120).std().to_numpy() * np.sqrt(120.0))
    min_l250, max_h250 = _roll(la, 250).min().to_numpy(), _roll(ha, 250).max().to_numpy()
    out["POS_52W"] = _safe_div(ca - min_l250, max_h250 - min_l250)
    out["DIST_52W_LOW"] = _safe_div(ca, min_l250) - 1.0
    out["DIST_52W_HIGH"] = _safe_div(ca, max_h250) - 1.0
    ma200 = _roll(ca, 200).mean().to_numpy()
    out["MA200_RATIO"] = _safe_div(ca, ma200) - 1.0
    out["MA200_ABOVE"] = np.where(np.isfinite(ma200), (ca > ma200).astype(np.float64), np.nan)

    # ---- 制度 (P0 31-32, P1 47-49)
    lim = limit_pct_array(code, days)
    lu_ind = np.where(np.isfinite(pct), (pct >= lim - 0.1).astype(np.float64), np.nan)
    ld_ind = np.where(np.isfinite(pct), (pct <= -(lim - 0.1)).astype(np.float64), np.nan)
    out["LIMITCNT20"] = _roll(lu_ind, 20).sum().to_numpy()
    out["LIMITDOWN_CNT20"] = _roll(ld_ind, 20).sum().to_numpy()
    out["DIST_LIMIT"] = np.clip(_safe_div(pct, lim), -1.0, 1.0)
    sum_a60 = _roll(A, 60).sum().to_numpy()
    sum_v60 = _roll(V, 60).sum().to_numpy()
    out["DISPOSAL60"] = _safe_div(C, _safe_div(sum_a60, sum_v60)) - 1.0
    out["NEW_LISTING"] = (np.arange(n) < 250).astype(np.float64)

    # ---- K 线形态 (P0 48, P1 61-67; 原始价, 单日截面比率)
    body = np.abs(C - O)
    lower_sh = np.minimum(O, C) - L
    upper_sh = H - np.maximum(O, C)
    full = H - L
    flat = full <= 0  # 一字板
    cvh = _safe_div(H - C, full)
    out["CLOSE_VS_HIGH"] = np.where(flat, 0.5, cvh)
    usr = _safe_div(upper_sh, full)
    out["UPPER_SHADOW_RATIO"] = np.where(flat, 0.0, usr)
    out["BODY_STRENGTH"] = _safe_div(C - O, full + 1e-12)
    hammer = ((lower_sh >= 2.0 * body) & (upper_sh <= 0.5 * body)
              & (lower_sh >= 0.6 * full) & ~flat)
    out["HAMMER"] = hammer.astype(np.float64)
    out["DOJI"] = (body < 0.1 * full).astype(np.float64)
    c_prev, o_prev = S(C).shift(1).to_numpy(), S(O).shift(1).to_numpy()
    bull_eng = (C > O) & (c_prev < o_prev) & (C >= o_prev) & (O <= c_prev)
    bear_eng = (C < O) & (c_prev > o_prev) & (C <= o_prev) & (O >= c_prev)
    eng = np.where(bull_eng, 1.0, np.where(bear_eng, -1.0, 0.0))
    out["ENGULFING"] = np.where(np.isfinite(c_prev), eng, np.nan)
    out["DOWN_STREAK"] = _streak(C < O)
    out["DOWN_DAY_PCT10"] = _roll((C < O).astype(np.float64), 10).mean().to_numpy()

    feats = pd.DataFrame(out)
    # inf 护栏 (主表 7.2: NaN 交模型处理, inf 一律转 NaN;
    # 典型暴露面: 连续一字板致 H/L 等窗口序列常数, rolling.corr 产出 inf)
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # ---- ctx: 事件特征与面板所需的内部量
    r1 = np.where(np.isfinite(R), R, 0.0)
    if n:
        r1[0] = np.nan  # regime 口径 = cf 差分, 首日不存在 (与 divergence_lab 一致)
    mean_ca20 = mean_c20.to_numpy()
    above_ma20 = np.where(np.isfinite(mean_ca20),
                          (ca > mean_ca20).astype(np.float64), np.nan)
    min_l250_s = _roll(la, 250).min().to_numpy()
    newlow250 = np.where(np.isfinite(min_l250_s), (la == min_l250_s).astype(np.float64), np.nan)
    cross_up = ((dif > dea) & (dif_prev <= dea_prev))
    cross_up = np.where(np.isfinite(dif) & np.isfinite(dea) & np.isfinite(dif_prev)
                        & np.isfinite(dea_prev), cross_up, False)
    ctx = dict(
        days=days, ca=ca, oa=oa, ha=ha, la=la, dif=dif, dea=dea, hist=hist,
        rsi14=rsi14, atr14=atr14, atrp=atrp, cross_up=cross_up, hammer=out["HAMMER"],
        lower_sh=lower_sh * f_adj,  # 复权下影线 (事件特征用)
        panel=dict(r1=r1.astype(np.float32), ret5=out["RET5"].astype(np.float32),
                   ret20=out["RET20"].astype(np.float32),
                   above_ma20=above_ma20.astype(np.float32),
                   newlow250=newlow250.astype(np.float32),
                   amount=A.astype(np.float32), limit_up=lu_ind.astype(np.float32)),
    )
    for k, v in ctx["panel"].items():
        ctx["panel"][k] = np.where(np.isinf(v), np.nan, v).astype(np.float32)
    return feats, ctx


# ================================================================ 事件结构特征 (主表 2.2: 只用允许的三套锚点)
def _dif_cycle_length(dif):
    """cyc[t] = 自 t 起最近一次 DIF 完整正负循环时长 = 最后两个已完成异号 run 的长度和.
    run = DIF 符号 (0/NaN 顺延前号) 的极大连续段; t 所在 run 未完成, 不计."""
    n = len(dif)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    s = np.sign(dif)
    runs = []  # (start, end_exclusive)
    cur, prev, start = 0.0, 0.0, 0
    for i in range(n):
        v = s[i] if np.isfinite(s[i]) else 0.0
        if v == 0.0:
            v = prev
        if v == 0.0:
            start = i + 1
            continue
        if prev == 0.0:
            cur, start = v, i
        elif v != cur:
            runs.append((start, i))
            cur, start = v, i
        prev = v
    for k in range(1, len(runs)):
        e0 = runs[k][1]
        e1 = runs[k + 1][1] if k + 1 < len(runs) else n
        out[e0:e1] = (runs[k][1] - runs[k][0]) + (runs[k - 1][1] - runs[k - 1][0])
    return out


def _green_area_trough(hist, i):
    """第 i 根 (含) 之前最近的绿柱 (HIST<0) 连续区间 -> (面积和, 极值 min); 无则 None."""
    j = i
    while j >= 0 and not (hist[j] < 0):
        j -= 1
    if j < 0:
        return None
    k = j
    while k - 1 >= 0 and hist[k - 1] < 0:
        k -= 1
    seg = hist[k:j + 1]
    return float(seg.sum()), float(seg.min())


def compute_event_features(ctx, sig, i1, i2, pool_sigs):
    """事件结构特征 (P0 39/52-59, P1 61/62/72-77).

    sig/i1/i2: 股内行号数组 (确认日 T / 前低 i1 / 次低 i2, 锚点来自 events.parquet);
    pool_sigs: 该股同池全部事件信号日行号 (升序), DIV_COUNT_120 用.
    返回 dict[str, f64 array], 长度 = len(sig); 锚点无效处为 NaN.
    """
    sig = np.asarray(sig, np.int64)
    i1 = np.asarray(i1, np.int64)
    i2 = np.asarray(i2, np.int64)
    m = len(sig)
    ca, oa, la = ctx["ca"], ctx["oa"], ctx["la"]
    dif, dea, hist = ctx["dif"], ctx["dea"], ctx["hist"]
    rsi14, atr14, atrp = ctx["rsi14"], ctx["atr14"], ctx["atrp"]
    n = len(ca)
    valid = (sig >= 0) & (sig < n) & (i1 >= 0) & (i1 < n) & (i2 >= 0) & (i2 < n) & (i2 > i1)

    def g(arr, idx):
        return np.where(valid, arr[np.clip(idx, 0, max(n - 1, 0))], np.nan)

    out: dict[str, np.ndarray] = {}
    rsi_ok = valid & np.isfinite(g(rsi14, i1)) & np.isfinite(g(rsi14, i2))
    out["RSI_DIV"] = np.where(rsi_ok,
                              ((g(la, i2) < g(la, i1)) & (g(rsi14, i2) > g(rsi14, i1))).astype(np.float64),
                              np.nan)
    sigs = np.asarray(pool_sigs, np.int64)
    sigs = np.sort(sigs)
    cnt = np.searchsorted(sigs, sig, "right") - np.searchsorted(sigs, sig - 120, "right")
    out["DIV_COUNT_120"] = np.where(valid, cnt, np.nan).astype(np.float64)
    a1 = g(atr14, sig)
    out["REBOUND_FROM_L2"] = _safe_div(g(ca, sig) - g(la, i2), a1)
    out["DIV_PRICE_NEWLOW_DEPTH"] = _safe_div(g(la, i2), g(la, i1)) - 1.0
    out["DIV_DIF_LIFT"] = _safe_div(g(dif, i2) - g(dif, i1), a1)
    span = np.where(valid, i2 - i1, np.nan)
    out["DIV_SPAN_BARS"] = span
    out["DAYS_SINCE_L2"] = np.where(valid, sig - i2, np.nan)
    cyc = _dif_cycle_length(dif)
    out["DIV_SPAN_VS_CYCLE"] = _safe_div(span, g(cyc, sig))
    out["DIV_DIF_SLOPE"] = _safe_div(_safe_div(g(dif, i2) - g(dif, i1), span), a1)
    out["DIV_ZERO_AXIS_DEPTH"] = _safe_div(np.maximum(g(dif, sig), g(dea, sig)),
                                           g(ca, sig) * g(atrp, sig))
    out["DIV_DIF_LEVEL_L1"] = _safe_div(g(dif, i1), g(ca, i1) * g(atrp, i1))
    out["REBOUND_DAY_T"] = _safe_div(g(ca, sig) - g(oa, sig), a1)
    out["HAMMER_L2"] = np.where(valid, ctx["hammer"][np.clip(i2, 0, max(n - 1, 0))], np.nan)
    out["LOWER_SHADOW_L2"] = _safe_div(g(ctx["lower_sh"], i2), g(atr14, i2))

    # 逐事件: 绿柱面积/极值收缩, 金叉状态
    area_ratio = np.full(m, np.nan)
    trough_ratio = np.full(m, np.nan)
    gcs = np.full(m, np.nan)
    cross = ctx["cross_up"]
    csum = np.concatenate([[0], np.cumsum(cross.astype(np.int64))])
    for e in range(m):
        if not valid[e]:
            continue
        t, a, b = int(sig[e]), int(i1[e]), int(i2[e])
        r1 = _green_area_trough(hist, a)
        r2 = _green_area_trough(hist, b)
        if r1 and r2:
            if r1[0] != 0:
                area_ratio[e] = r2[0] / r1[0]
            if r1[1] != 0:
                trough_ratio[e] = r2[1] / r1[1]
        if np.isfinite(dif[t]) and np.isfinite(dea[t]):
            if cross[t]:
                gcs[e] = 2.0
            else:
                lo = max(t - 5, 0)
                gcs[e] = 1.0 if (csum[t] - csum[lo]) > 0 else 0.0
    out["DIV_HIST_AREA_SHRINK"] = area_ratio
    out["DIV_HIST_TROUGH_SHRINK"] = trough_ratio
    out["DIV_GOLDEN_CROSS_STATE"] = gcs
    return out


EVENT_FEATURE_COLS = [
    "RSI_DIV", "DIV_COUNT_120", "DIV_HIST_AREA_SHRINK", "DIV_GOLDEN_CROSS_STATE",
    "REBOUND_FROM_L2", "DIV_PRICE_NEWLOW_DEPTH", "DIV_DIF_LIFT", "DIV_SPAN_BARS",
    "DAYS_SINCE_L2", "DIV_SPAN_VS_CYCLE", "DIV_DIF_SLOPE", "DIV_ZERO_AXIS_DEPTH",
    "DIV_DIF_LEVEL_L1", "DIV_HIST_TROUGH_SHRINK", "REBOUND_DAY_T", "HAMMER_L2",
    "LOWER_SHADOW_L2",
]


# ================================================================ 市场环境/宽度/日历 (date 级)
CNY_DATES = [  # 春节日期 (正月初一), 1990-2027, 静态表 (确定性、因果)
    "1990-01-27", "1991-02-15", "1992-02-04", "1993-01-23", "1994-02-10", "1995-01-31",
    "1996-02-19", "1997-02-07", "1998-01-28", "1999-02-16", "2000-02-05", "2001-01-24",
    "2002-02-12", "2003-02-01", "2004-01-22", "2005-02-09", "2006-01-29", "2007-02-18",
    "2008-02-07", "2009-01-26", "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10",
    "2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
    "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29",
    "2026-02-17", "2027-02-06",
]


def _calendar_features(dates):
    """日历特征 (date 级): CAL_DOW_0..4 / CAL_MONTH_1..12 / CAL_MONTH_POS / CAL_HOLIDAY."""
    dates = np.asarray(dates, np.int32)
    out = {}
    dow = (dates + 3) % 7  # 1970-01-01 为周四 -> Monday=0
    for i in range(5):
        out[f"CAL_DOW_{i}"] = (dow == i).astype(np.float64)
    months = dates.astype("datetime64[M]").astype(np.int64)
    month_num = (months % 12 + 1).astype(int)
    for mnum in range(1, 13):
        out[f"CAL_MONTH_{mnum}"] = (month_num == mnum).astype(np.float64)
    # 距月末交易日数: 静态工作日近似 (同月内 d 之后的工作日数; 与数据范围无关, 截断不变)
    d64 = dates.astype("datetime64[D]")
    next_month = (d64.astype("datetime64[M]") + 1).astype("datetime64[D]")
    out["CAL_MONTH_POS"] = np.busday_count(d64 + 1, next_month).astype(np.float64)
    # 节假日 dummy: 春节/国庆前 10 个工作日、后 5 个工作日 (静态节假日表 + 工作日近似,
    # 与数据范围无关, 截断不变; 主表 P1-34 的近似口径)
    hol = np.zeros(len(dates), bool)
    yrs = range(int(d64[0].astype("datetime64[Y]").astype(int)) + 1970,
                int(d64[-1].astype("datetime64[Y]").astype(int)) + 1972)
    hol_days = [np.datetime64(d, "D") for d in CNY_DATES]
    hol_days += [np.datetime64(f"{y}-10-01", "D") for y in yrs]
    for h64 in hol_days:
        pre_lo = np.busday_offset(h64, -10, roll="backward")
        hol |= (d64 >= pre_lo) & (d64 < h64)
        post_lo = np.busday_offset(h64 + np.timedelta64(7, "D"), 0, roll="forward")
        post_hi = np.busday_offset(post_lo, 5, roll="forward")
        hol |= (d64 >= post_lo) & (d64 < post_hi)
    out["CAL_HOLIDAY"] = hol.astype(np.float64)
    return out


def _index_daily_family(idx_sh, idx_sz):
    """MKT_IDX_DAILY 族 + P0 的 MKT_AVG_AMP/MKT_SYNC_SCORE (v2 §2.13 口径, 全历史向量化)."""
    def single(df, p):
        o, h, l = df["open"].to_numpy(np.float64), df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64)
        c, v = df["close"].to_numpy(np.float64), df["volume"].to_numpy(np.float64)
        pc = _safe_div(c - o, o)
        amp = _safe_div(h - l, l)
        vr = _safe_div(v, _roll(v, 20).mean().to_numpy())
        sent = np.where((pc > 0.01) & (amp < 0.02), 2.0,
                        np.where((pc < -0.01) & (amp > 0.03), 0.0, 1.0))
        return pd.DataFrame({
            "_days": df["_days"].to_numpy(),
            f"MKT_{p}_PRICE_CHANGE": pc, f"MKT_{p}_AMPLITUDE": amp,
            f"MKT_{p}_VOLUME_RATIO": vr,
            f"MKT_{p}_PRICE_CHANGE_ABS": np.abs(pc), f"MKT_{p}_PRICE_WAVE_ABS": amp,
            f"MKT_{p}_SENTIMENT": np.where(np.isfinite(pc) & np.isfinite(amp), sent, np.nan),
            f"MKT_{p}_VOLUME_SIGNAL": np.where(np.isfinite(vr), (vr > 1.2).astype(np.float64), np.nan),
        }).set_index("_days")

    a, b = single(idx_sh, "SH"), single(idx_sz, "SZ")
    j = a.join(b, how="outer")
    pc_sh, pc_sz = j["MKT_SH_PRICE_CHANGE"], j["MKT_SZ_PRICE_CHANGE"]
    amp_sh, amp_sz = j["MKT_SH_AMPLITUDE"], j["MKT_SZ_AMPLITUDE"]
    avg_pc = (pc_sh + pc_sz) / 2.0
    avg_amp = (amp_sh + amp_sz) / 2.0
    j["MKT_SH_SZ_SYNC_DIRECTION"] = np.where((pc_sh * pc_sz) > 0, 1.0, 0.0)
    j["MKT_SH_SZ_SYNC_STRENGTH"] = (pc_sh - pc_sz).abs()
    sent_m = np.where((avg_pc > 0.01) & (avg_amp < 0.02), 2.0,
                      np.where((avg_pc < -0.01) & (avg_amp > 0.03), 0.0, 1.0))
    j["MKT_SENTIMENT"] = np.where(avg_pc.notna() & avg_amp.notna(), sent_m, np.nan)
    j["MKT_AVG_CHANGE"] = avg_pc
    j["MKT_AVG_AMP"] = avg_amp
    j["MKT_SYNC_SCORE"] = 1.0 - ((pc_sh - pc_sz).abs() / 0.02).clip(upper=1.0)
    return j


def _index_state_features(idx_sh):
    """000001.SH 指数状态特征: MKT_RET20 / MKT_MA60 / MKT_VOL20_PCT / MKT_DD120 / MKT_RSI14."""
    c = idx_sh["close"].to_numpy(np.float64)
    h = idx_sh["high"].to_numpy(np.float64)
    sc = pd.Series(c)
    r_idx = (sc / sc.shift(1) - 1.0).to_numpy()
    vol20 = _roll(r_idx, 20).std().to_numpy()
    return pd.DataFrame({
        "_days": idx_sh["_days"].to_numpy(),
        "MKT_RET20": (sc / sc.shift(20) - 1.0).to_numpy(),
        "MKT_MA60": _safe_div(c, _roll(c, 60).mean().to_numpy()) - 1.0,
        "MKT_VOL20_PCT": _ts_pct_rank(vol20, 250),
        "MKT_DD120": _safe_div(c, _roll(h, 120).max().to_numpy()) - 1.0,
        "MKT_RSI14": talib.RSI(c, timeperiod=14),
    }).set_index("_days"), r_idx


def build_market_frame(panel, idx_sh, idx_sz):
    """由全 universe 面板 + 双指数构建 date 级市场特征框.

    panel: DataFrame [sid, date, r1, ret5, ret20, above_ma20, newlow250, amount, limit_up, is_index]
           (全历史全股票全交易日; 指数文件行 is_index=True, 不参与宽度/成交/涨停聚合与 regime)
    返回 (market_df, calendar): market_df 以 date(i32) 为索引, 含 MKT_*/BREADTH_*/REGIME_CODE/CAL_*.
    """
    st = panel[~panel["is_index"]]
    codes, dates = pd.factorize(st["date"].to_numpy(), sort=True)
    dates = dates.to_numpy(np.int32) if hasattr(dates, "to_numpy") else np.asarray(dates, np.int32)
    D = len(dates)

    def agg_frac(ind):
        ind = np.asarray(ind, np.float64)
        valid = np.isfinite(ind)
        cnt = np.bincount(codes[valid], minlength=D)
        s = np.bincount(codes[valid], weights=ind[valid], minlength=D)
        return _safe_div(s, cnt)

    ret5 = st["ret5"].to_numpy(np.float64)
    adv5_ind = np.where(np.isfinite(ret5), (ret5 > 0).astype(np.float64), np.nan)
    breadth = pd.DataFrame({
        "_days": dates,
        "BREADTH_ADV5": agg_frac(adv5_ind),
        "BREADTH_ABOVE_MA20": agg_frac(st["above_ma20"].to_numpy(np.float64)),
        "BREADTH_NEWLOW": agg_frac(st["newlow250"].to_numpy(np.float64)),
        "LIMITUP_N_MKT": np.bincount(codes, weights=np.nan_to_num(
            st["limit_up"].to_numpy(np.float64)), minlength=D),
    }).set_index("_days")
    breadth["LIMITUP_N_MKT_MA20"] = breadth["LIMITUP_N_MKT"].rolling(20, min_periods=20).mean()

    # 全市场成交额 20 日均值的 250 日分位
    amt = st["amount"].to_numpy(np.float64)
    amt_sum = np.bincount(codes[np.isfinite(amt)], weights=amt[np.isfinite(amt)], minlength=D)
    amt_m20 = pd.Series(amt_sum).rolling(20, min_periods=20).mean().to_numpy()
    breadth["MKT_AMT_PCT"] = _ts_pct_rank(amt_m20, 250)

    # RET20 中位数 (排序法)
    ret20 = st["ret20"].to_numpy(np.float64)
    order = np.lexsort((np.where(np.isfinite(ret20), ret20, np.inf), codes))
    c_sorted = codes[order]
    starts = np.searchsorted(c_sorted, np.arange(D))
    ends = np.searchsorted(c_sorted, np.arange(D), "right")
    r_sorted = ret20[order]
    med = np.full(D, np.nan)
    for d in range(D):
        vals = r_sorted[starts[d]:ends[d]]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            med[d] = np.median(vals)
    breadth["MKT_MEDIAN_RET20"] = med

    # REGIME_CODE (divergence_lab build_market_regime 口径: 等权日收益 cumprod, 120 日滚动 ±10%)
    r1 = st["r1"].to_numpy(np.float64)
    valid_r = np.isfinite(r1)
    cnt_r = np.bincount(codes[valid_r], minlength=D)
    sum_r = np.bincount(codes[valid_r], weights=r1[valid_r], minlength=D)
    mean_r = np.divide(sum_r, cnt_r, out=np.zeros(D), where=cnt_r > 0)
    idx_eq = np.cumprod(1.0 + mean_r)
    regime = np.full(D, -1.0)
    if D > 120:
        roll120 = idx_eq[120:] / idx_eq[:-120] - 1.0
        regime[120:] = np.where(roll120 > 0.10, 1.0, np.where(roll120 < -0.10, 2.0, 0.0))
    breadth["REGIME_CODE"] = regime

    idx_fam = _index_daily_family(idx_sh, idx_sz)
    idx_state, _ = _index_state_features(idx_sh)
    cal = pd.DataFrame({"_days": dates, **_calendar_features(dates)}).set_index("_days")
    market = breadth.join([idx_fam, idx_state, cal], how="left")
    return market, dates


def attach_ret20_csr(panel, q_date, q_ret20):
    """RET20_CSR = cs_rank(RET20): 当日全 universe 横截面百分位 (less+0.5*ties)/n.

    panel: 全 universe 面板 (剔除指数行); q_date/q_ret20: 查询的 (日期, 该股 RET20).
    """
    st = panel[~panel["is_index"]]
    dates = st["date"].to_numpy(np.int32)
    ret20 = st["ret20"].to_numpy(np.float64)
    order = np.lexsort((np.where(np.isfinite(ret20), ret20, np.inf), dates))
    d_sorted = dates[order]
    r_sorted = ret20[order]
    uniq, starts = np.unique(d_sorted, return_index=True)
    bounds = dict(zip(uniq.tolist(), zip(starts.tolist(),
                                         np.append(starts[1:], len(d_sorted)).tolist())))
    q_date = np.asarray(q_date, np.int32)
    q_ret20 = np.asarray(q_ret20, np.float64)
    out = np.full(len(q_date), np.nan)
    for i in range(len(q_date)):
        x = q_ret20[i]
        if not np.isfinite(x):
            continue
        b = bounds.get(int(q_date[i]))
        if b is None:
            continue
        vals = r_sorted[b[0]:b[1]]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        l = int(np.searchsorted(vals, x, "left"))
        r = int(np.searchsorted(vals, x, "right"))
        out[i] = (l + 0.5 * (r - l)) / len(vals)
    return out


# ================================================================ 全 universe 装配 (驱动层共用)
META_COLS = ["event_id", "ts_code", "date", "sig_idx"]


def feature_column_order(frame_cols):
    """最终特征列顺序: 注册表顺序, 过滤到实际存在的列."""
    cols = [c for c in FEATURE_REGISTRY if c in frame_cols]
    missing = [c for c in FEATURE_REGISTRY if c not in frame_cols]
    return cols, missing


def process_stock(args):
    """工作进程: 单股全历史特征 + 事件行切片 + 面板列.

    args = (path, code, sid, events, idx_dates, idx_r)
      events: dict pool -> list of (event_id, sig_idx, sig_day, low_day, prev_low_day)
    返回 dict(panel=DataFrame|None, rows=dict pool -> DataFrame(index=event_id), error=None|str,
              n_anchor_miss=int)
    """
    path, code, sid, events, idx_dates, idx_r = args
    try:
        df = load_stock_df(path)
    except Exception as e:  # noqa: BLE001
        return dict(panel=None, rows={}, error=f"read: {e}", n_anchor_miss=0)
    if len(df) < 30:
        return dict(panel=None, rows={}, error="too_short", n_anchor_miss=0)
    try:
        feats, ctx = compute_stock_features(df, code, idx_dates, idx_r)
    except Exception as e:  # noqa: BLE001
        return dict(panel=None, rows={}, error=f"compute: {e}", n_anchor_miss=0)

    days = ctx["days"]
    is_index = code in INDEX_CODES
    panel = pd.DataFrame({"sid": np.full(len(days), sid, np.int32), "date": days,
                          "is_index": is_index, **ctx["panel"]})

    day2idx = {int(d): i for i, d in enumerate(days)}
    rows: dict[str, pd.DataFrame] = {}
    n_anchor_miss = 0
    f32 = lambda x: x.astype(np.float32)
    for pool, evs in events.items():
        if not evs:
            continue
        eid = np.array([e[0] for e in evs], np.int64)
        sig = np.array([e[1] for e in evs], np.int64)
        sig_day = np.array([e[2] for e in evs], np.int64)
        if not np.array_equal(days[sig], sig_day):
            raise RuntimeError(f"{code} {pool}: sig_idx 与事件日不对齐 (清洗口径漂移)")
        i1 = np.array([day2idx.get(int(e[4]), -1) for e in evs], np.int64)
        i2 = np.array([day2idx.get(int(e[3]), -1) for e in evs], np.int64)
        n_anchor_miss += int(((i1 < 0) | (i2 < 0)).sum())
        ev_feats = compute_event_features(ctx, sig, i1, i2, sig)
        blk = feats.iloc[sig].reset_index(drop=True)
        for c in EVENT_FEATURE_COLS:
            blk[c] = ev_feats[c]
        blk.index = pd.Index(eid, name="event_id")
        rows[pool] = blk.astype(np.float32)
    return dict(panel=panel, rows=rows, error=None, n_anchor_miss=n_anchor_miss)


def assert_no_inf(frame):
    """硬断言: 任何特征列不得含 ±inf (NaN 允许, 交模型处理)."""
    vals = frame.to_numpy(np.float64)
    mask = np.isinf(vals)
    assert not mask.any(), f"特征矩阵含 inf: {list(frame.columns[mask.any(axis=0)])}"


def assemble_pool(events_df, feat_rows, market, panel):
    """装配单池特征矩阵: 事件元数据 + 股内/事件特征 + 市场特征(按日) + RET20_CSR."""
    df = events_df[META_COLS].sort_values("event_id").reset_index(drop=True)
    fr = feat_rows.sort_index()
    assert len(fr) == len(df), f"特征行数 {len(fr)} != 事件数 {len(df)}"
    assert (fr.index.to_numpy() == df["event_id"].to_numpy()).all(), "event_id 未对齐"
    df = pd.concat([df, fr.reset_index(drop=True)], axis=1)
    days = _to_days(df["date"])
    mkt = market.reindex(days.to_numpy() if hasattr(days, "to_numpy") else days)
    mkt.index = df.index
    df = pd.concat([df, mkt.reset_index(drop=True).astype(np.float32)], axis=1)
    df["RET20_CSR"] = attach_ret20_csr(panel, days, df["RET20"].to_numpy(np.float64)).astype(np.float32)
    extra = [c for c in df.columns if c not in META_COLS and c not in FEATURE_REGISTRY]
    assert not extra, f"未注册注入列 (拒绝静默丢弃): {extra}"
    feat_cols, missing = feature_column_order(df.columns)
    assert not missing, f"注册特征未产出: {missing}"
    df = df[META_COLS + feat_cols]
    assert_no_blacklisted(df.columns)
    assert_no_label_columns(df.columns)
    assert_no_inf(df[feat_cols])
    return df


def feature_dictionary():
    """feature_dictionary.csv: 列名/规范名/家族/层/公式/输入字段/事件行限定."""
    rows = []
    for c, s in FEATURE_REGISTRY.items():
        rows.append(dict(column=c, feature=s.feature, layer=s.layer, family=s.family,
                         formula=s.formula, inputs="|".join(s.inputs), event_only=s.event_only))
    return pd.DataFrame(rows)
