#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2-on-v6 复刻线(issue #54) 步骤 4:事件 × 特征主表 master_v2on6.parquet。

预登记军令状 = 同目录 README.md(commit 49b9c5b,冻结);公式唯一来源 = 同目录
anatomy_report.md(解剖报告);v2 原始代码(/tmp/v2_excavation/)只读参考,禁止 import。

合成内容:
  1. 事件行切片(cache/eventrows_run1/,build_panel 产出):197 基础列 + cs_n
     + 195 rankpct + 195 z = 588 列;
  2. 背离事件结构族 16 列(README §四.8 = anatomy §6.3 逐条,直接吃事件表列;
     volume_signal 的 prev_event_vol 取自 build_panel 的 event_aux(P12);
     LabelEncoder 拟合只在 train 段(P12/§九.2),映射全量落台账);
  3. 键与段界:event_id/seg 自 master_v6 按 (ts_code, event_date, event_row) 并入;
  4. dictionary_v2on6.csv:全部 609 列(604 特征 + 5 键)中文全称(§九.6)。

验收断言(README §九,本驱动机检项):
  §九.2 泄漏:无 future_/stop_loss_/label_ 列;LabelEncoder 仅 train 段拟合;
  §九.3 段界:assert_segment_integrity 全量执行;段计数对预登记值;
  §九.4 行数与键:96,577 行守恒、event_id 唯一、(ts_code,event_date) 与事件表互证;
  §九.5 背离结构族:抽样 500 事件自写标量重算逐位一致(seed 42);
  §九.6 词典:每列中文全称非空。

用法:
  python3 build_master_v2on6.py                   # run1
  python3 build_master_v2on6.py --rerun-tag run2  # 第二进程重跑(md5 对账)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import train_eval_pipeline as tep  # noqa: E402  (v3_pipeline 既有断言,禁改)
from build_panel_v2on6 import (  # noqa: E402  列规格常量单一来源
    BASE_FAM1_7, DIVERGENCE_COLS, MASTER_FEATURE_COLS, RPZ_BASE,
)

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
MASTER_V6_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_v6.parquet"
CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"
SH_INDEX_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"

PREREG_SEGMENT_COUNTS = {"train": 50165, "val": 24405, "test": 17902,
                         "embargo": 3063, "pre2001": 1042}
KEY_COLS = ["ts_code", "event_date", "event_row", "event_id", "seg"]
N_STRUCTURE_SAMPLE = 500
SAMPLE_SEED = 42

CN_KEY = {
    "ts_code": "证券代码",
    "event_date": "事件日(相邻金叉对后一金叉日)",
    "event_row": "事件日在个股日线升序帧中的行号",
    "event_id": "事件唯一编号(master_v6 权威)",
    "seg": "段界标签(train/val/test/embargo/pre2001)",
}
CN_DIVERGENCE = {
    "close_current": "背离结构·当前收盘价(=锚点日收盘 anchor_close)",
    "close_previous": "背离结构·前低收盘价(=前低日收盘 min_prev_close)",
    "macd_current": "背离结构·当前金叉 DIF(=cross_dif)",
    "macd_previous": "背离结构·前一金叉 DIF(=cross_prev_dif)",
    "price_decline_pct": "背离结构·价格变动幅度((锚点收盘−前低收盘)/前低收盘)",
    "macd_increase_pct": "背离结构·DIF 抬升幅度(dif_lift/|前一金叉 DIF|,分母 0 → 0)",
    "compare_rank": "背离结构·比较序位(P10:v6 事件恒为相邻金叉对,常量 1)",
    "formation_period": "背离结构·形成周期(两金叉间 K 线根数 = anchor_bars)",
    "is_quick_divergence_y": "背离结构·快速背离标记(形成周期 < 3 → 1)",
    "divergence_strength": "背离结构·背离强度(0.4·min(1,|价格变动|/0.1)+0.6·min(1,DIF抬升/0.1))",
    "volume_signal": "背离结构·量能信号(事件量<前金叉量→bullish;>1.5倍→bearish;否则 neutral;"
                     "LabelEncoder 仅 train 段拟合)",
    "price_macd_ratio": "背离结构·价 DIF 比(|DIF抬升/max(|价格变动|,1e-6)|)",
    "divergence_magnitude": "背离结构·背离量级((|价格变动|+|DIF抬升|)/2)",
    "confirmation_score": "背离结构·确认得分(价格变动<−0.02 且 DIF抬升>0.01 → 1)",
    "divergence_amount": "背离结构·当日全市场事件总数(P11:按事件日计数)",
    "v2j2_macd_increase_pct": "背离结构·前二金叉 DIF 抬升幅度(P10 增补列,"
                              "(当前DIF−前二DIF)/|前二DIF|,分母 0 → 0;命名隔离非 v2 原列)",
}
CN_BASE = {
    "volume": "成交量", "macd": "MACD 快线 DIF(12,26,9)", "macd_signal": "MACD 信号线 DEA",
    "macd_hist": "MACD 柱(DIF−DEA)", "rsi_6": "相对强弱指标 RSI(6)",
    "rsi_14": "相对强弱指标 RSI(14)", "rsi_24": "相对强弱指标 RSI(24)",
    "ma_5": "收盘简单移动平均 MA(5)", "ma_20": "收盘简单移动平均 MA(20)",
    "ma_60": "收盘简单移动平均 MA(60)", "bb_upper": "布林上轨(20,2,2)",
    "bb_middle": "布林中轨(20,2,2)", "bb_lower": "布林下轨(20,2,2)",
    "volume_ma_20": "成交量移动平均 VMA(20,talib)",
    "obv": "能量潮 OBV", "atr": "平均真实波幅 ATR(14)",
    "slowk": "随机指标慢速 K(STOCH 5,3,3)", "slowd": "随机指标慢速 D(STOCH 5,3,3)",
    "price_vs_ma5": "收盘对 MA5 乖离率", "price_vs_ma20": "收盘对 MA20 乖离率",
    "price_vs_ma60": "收盘对 MA60 乖离率",
    "ma_arrangement": "均线排列(多头 1/空头 −1/其他 0)",
    "bb_position": "布林通道内位置", "bb_squeeze": "布林挤压标记(带宽/收盘<0.05)",
    "macd_signal_distance": "DIF−DEA 距离",
    "macd_golden_cross": "MACD 金叉标记(族 3 覆盖值:昨 DIF<昨 DEA 且今 DIF>今 DEA)",
    "volume_ma20": "成交量均值(20,min_periods=1)",
    "volume_ratio": "量比(成交量/20 日均量)",
    "volume_spike": "放量标记(量比>2)", "volume_dryup": "缩量标记(量比<0.5)",
    "atr_ratio": "ATR 占比(ATR/收盘)",
    "hammer_pattern": "锤子线形态标记",
    "downtrend": "下降趋势标记(5 日均线<10 日均线,bool)",
    "hammer_signal": "下降趋势中锤子线信号(bool)",
    "doji_pattern": "十字星形态标记(实体<全幅 0.1)",
    "distance_to_support": "至支撑位距离(20 窗最低价,前 20 行强制 0)",
    "distance_to_resistance": "至阻力位距离(20 窗最高价,前 20 行强制 0)",
    "stoch_oversold": "随机指标超卖标记(慢速 K<20)",
    "stoch_overbought": "随机指标超买标记(慢速 D>80)",
    "macd_percentile": "MACD 百分位(P7:按股截尾 100 行不含当日,rank 法;恰为 50 → NaN)",
    "obv_trend": "OBV 趋势(6 点最小二乘斜率)",
    "engulfing_pattern": "吞没形态标记(首行强制 0)",
    "signed_volume_strength": "带符号量能强度(量比×(收−开)/开)",
    "close_vs_high": "收盘距当日高点相对位置((高−收)/(高−低))",
    "volume_ma_ratio": "量能均线比(5 均量前一日/10 均量前一日,缺 → 1)",
    "rsi_momentum": "RSI(14) 动量(一阶差分,首行 0)",
    "rsi_turning_simple": "RSI 简单转向标记(差分符号翻转)",
    "rsi_turning": "RSI 转向标记(顶/底拐点)",
    "volume_trend_5": "成交量趋势(6 点斜率)", "volume_trend_10": "成交量趋势(11 点斜率)",
    "price_trend_5": "价格趋势(6 点斜率)",
    "price_volume_divergence": "量价背离标记(价/量 6 点斜率异号)",
    "volume_consistency": "成交量一致性(6 窗标准差/均值)",
    "macd_hist_trend_5": "MACD 柱趋势(6 窗 min_periods=2 斜率,输入填 0)",
    "macd_death_cross": "MACD 死叉标记",
    "macd_signal_cross": "MACD 信号交叉(P9 复制列 ≡ macd_golden_cross)",
    "macd_zero_cross_up": "DIF 上穿零轴标记", "macd_zero_cross_down": "DIF 下穿零轴标记",
    "macd_zero_cross": "DIF 零轴交叉标记",
    "macd_hist_amplitude": "MACD 柱振幅(20 窗极差)",
    "macd_hist_direction": "MACD 柱方向(sign)",
    "macd_hist_acceleration": "MACD 柱加速度(二阶差分)",
    "macd_signal_convergence": "DIF−DEA 收敛度(P9 复制列 ≡ macd_signal_distance)",
    "macd_signal_convergence_trend": "DIF−DEA 收敛趋势(5 窗 min_periods=2 斜率)",
    "pct_change": "日收益率", "clv": "收盘位置值((收−低)/(高−低+eps))",
    "upper_shadow_ratio": "上影线占比", "body_strength": "实体强度((收−开)/(高−低+eps))",
    "rank_return": "当日全市场收益率截面秩百分位",
    "rank_volume": "当日全市场成交量截面秩百分位",
    "signed_vol_strength": "带符号成交量(量×sign(收−开))",
    "pv_corr_10": "量价相关(10 窗)",
    "dist_to_high_60": "距 60 日高点比(收盘/(60 窗最高收盘+eps))",
    "vol_divergence": "波动背离(P6:5 日波动/滚动 100 交易日全市场均值)",
    "vol_gk": "Garman-Klass 波动率", "vol_gk_ratio": "GK 波动率 20 日均值比",
    "illiq": "非流动性 Amihud(log(|收益|/(收×量+eps)+1))",
    "efficiency_ratio": "效率系数(10 日净位移/路径长度)",
    "intraday_pos": "日内位置((收−低)/(高−低+eps))",
    "ret_overnight": "隔夜收益(开/昨收−1)", "ret_intraday": "日内收益(收/开−1)",
    "smart_money_diff": "聪明钱差(日内−隔夜)",
    "high_mean_20": "20 日最高价均值", "low_mean_20": "20 日最低价均值",
    "support_resistance_ratio": "支撑阻力比(高均/低均)",
    "log_volume": "对数成交量 log1p",
    "boxcox_atr": "ATR Box-Cox 变换(P5:lambda 按当日全市场截面 MLE)",
    "rsi_robust": "RSI(14) 稳健 Z(逐股全历史 expanding 中位数/IQR,IQR=0 退化)",
    "macd_robust": "DIF 稳健 Z(逐股全历史 expanding 中位数/IQR,IQR=0 退化)",
    "daily_return": "日收益率(P9 复制列 ≡ pct_change)",
    "amplitude": "振幅((高−低)/低)",
    "cs_n": "当日全市场截面股票数",
    "sh_price_change": "上证指数日内涨幅((收−开)/开)",
    "sh_amplitude": "上证指数振幅((高−低)/低)",
    "sh_volume_ratio": "上证指数量比(量/20 日均量,不足或零 → 1)",
    "sh_price_change_abs": "上证指数日内涨幅绝对值",
    "sh_price_wave_abs": "上证指数振幅绝对值(P9 复制列 ≡ sh_amplitude)",
    "sh_sentiment": "上证指数情绪(涨>1%且振幅<2% → 2;跌>1%且振幅>3% → 0;否则 1)",
    "sh_volume_signal": "上证指数量能信号(量比>1.2 → 1)",
    "sz_price_change": "深证成指日内涨幅((收−开)/开)",
    "sz_amplitude": "深证成指振幅((高−低)/低)",
    "sz_volume_ratio": "深证成指量比(量/20 日均量,不足或零 → 1)",
    "sz_price_change_abs": "深证成指日内涨幅绝对值",
    "sz_price_wave_abs": "深证成指振幅绝对值(P9 复制列 ≡ sz_amplitude)",
    "sz_sentiment": "深证成指情绪(规则同上)",
    "sz_volume_signal": "深证成指量能信号(量比>1.2 → 1)",
    "sh_sz_sync_direction": "沪深同向标记(涨幅乘积>0)",
    "sh_sz_sync_strength": "沪深涨幅差绝对值",
    "market_sentiment": "市场情绪(两指数均值,规则同单指数)",
    "market_avg_change": "市场平均涨幅(两指数均值)",
    "market_avg_amplitude": "市场平均振幅(两指数均值)",
    "market_sync_score": "市场同步得分(1−min(1,涨幅差/0.02))",
}
FAMILY_OF = {}
for c in ("volume", "macd", "macd_signal", "macd_hist", "rsi_6", "rsi_14", "rsi_24",
          "ma_5", "ma_20", "ma_60", "bb_upper", "bb_middle", "bb_lower", "volume_ma_20",
          "obv", "atr", "slowk", "slowd"):
    FAMILY_OF[c] = "基础TA族"
for c in ("macd_hist_trend_5", "macd_golden_cross", "macd_death_cross", "macd_signal_cross",
          "macd_zero_cross_up", "macd_zero_cross_down", "macd_zero_cross",
          "macd_hist_amplitude", "macd_hist_direction", "macd_hist_acceleration",
          "macd_signal_convergence", "macd_signal_convergence_trend"):
    FAMILY_OF[c] = "MACD深度族"
for c in ("pct_change", "clv", "upper_shadow_ratio", "body_strength", "rank_return",
          "rank_volume", "signed_vol_strength", "pv_corr_10", "dist_to_high_60",
          "vol_divergence"):
    FAMILY_OF[c] = "alpha族"
for c in ("vol_gk", "vol_gk_ratio", "illiq", "efficiency_ratio", "intraday_pos",
          "ret_overnight", "ret_intraday", "smart_money_diff", "high_mean_20",
          "low_mean_20", "support_resistance_ratio", "log_volume", "boxcox_atr",
          "rsi_robust", "macd_robust"):
    FAMILY_OF[c] = "结构化族"
for c in BASE_FAM1_7:
    if c.startswith(("sh_", "sz_", "market_")):
        FAMILY_OF[c] = "市场大盘族"
    elif "_lag_" in c or c in ("daily_return", "amplitude"):
        FAMILY_OF[c] = "lag族"
FAMILY_OF["cs_n"] = "横截面族"
for c in DIVERGENCE_COLS:
    FAMILY_OF[c] = "背离事件结构族"


def _cn_base(col: str) -> str:
    """lag/volatility 等模式列的中文名补全(模式外列须查 CN_BASE)。"""
    if col in CN_BASE:
        return CN_BASE[col]
    for n in (6, 14, 24):
        if col == f"rsi_oversold_{n}":
            return f"RSI({n}) 超卖标记(<30)"
        if col == f"rsi_overbought_{n}":
            return f"RSI({n}) 超买标记(>70)"
    for n in (3, 5, 10, 15, 20, 25, 30):
        if col == f"volatility_{n}d":
            return f"年化波动率({n} 日日对数收益标准差×√252)"
        for px, nm in (("close", "收盘价"), ("open", "开盘价"), ("high", "最高价"),
                       ("low", "最低价"), ("volume", "成交量")):
            if col == f"{px}_lag_{n}":
                return f"{nm}滞后 {n} 日"
        for n2 in (1, 2, 3, 5, 10, 20):
            if col == f"return_lag_{n2}":
                return f"日收益率滞后 {n2} 日"
    for n3 in (1, 2, 3, 5, 10):
        for base, nm in (("amplitude", "振幅"), ("vol_gk", "Garman-Klass 波动率"),
                         ("vol_gk_ratio", "GK 波动率 20 日均值比"),
                         ("illiq", "非流动性 Amihud"),
                         ("efficiency_ratio", "效率系数"),
                         ("intraday_pos", "日内位置"),
                         ("smart_money_diff", "聪明钱差"),
                         ("ret_overnight", "隔夜收益"),
                         ("ret_intraday", "日内收益"),
                         ("support_resistance_ratio", "支撑阻力比")):
            if col == f"{base}_lag_{n3}":
                return f"{nm}滞后 {n3} 日"
    raise KeyError(f"缺中文全称: {col}")


def build_dictionary() -> pd.DataFrame:
    rows = []
    for c in KEY_COLS:
        rows.append(dict(column=c, family="键", cn_name=CN_KEY[c], note="键/段界列"))
    for c in MASTER_FEATURE_COLS:
        if c.endswith("_rankpct"):
            base = c[:-len("_rankpct")]
            cn = _cn_base(base) + "·当日全市场截面秩百分位"
            fam = "横截面族(rankpct)"
        elif c.endswith("_z"):
            base = c[:-len("_z")]
            cn = _cn_base(base) + "·当日全市场截面Z值((x−中位数)/(标准差+1e-9))"
            fam = "横截面族(z)"
        elif c in CN_DIVERGENCE:
            cn, fam = CN_DIVERGENCE[c], FAMILY_OF[c]
        else:
            cn = _cn_base(c)
            fam = FAMILY_OF.get(c, "lag族" if "_lag_" in c else "基础/进阶TA族")
        rows.append(dict(column=c, family=fam, cn_name=cn, note=""))
    dic = pd.DataFrame(rows)
    assert dic["cn_name"].str.len().gt(0).all(), "存在空中文全称"
    assert not dic["column"].duplicated().any()
    return dic


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [build_master] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def compute_divergence_cols(ev: pd.DataFrame, prev_vol: pd.Series,
                            train_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """背离事件结构族 16 列(anatomy §6.3 逐条;向量化,输入 = 事件表列)。"""
    out = pd.DataFrame(index=ev.index)
    ac = ev["anchor_close"].to_numpy(np.float64)
    mpc = ev["min_prev_close"].to_numpy(np.float64)
    cd = ev["cross_dif"].to_numpy(np.float64)
    cpd = ev["cross_prev_dif"].to_numpy(np.float64)
    cp2d = ev["cross_prev2_dif"].to_numpy(np.float64)
    lift = ev["dif_lift"].to_numpy(np.float64)
    bars = ev["anchor_bars"].to_numpy(np.int64)
    out["close_current"] = ac
    out["close_previous"] = mpc
    out["macd_current"] = cd
    out["macd_previous"] = cpd
    pdp = (ac - mpc) / mpc
    out["price_decline_pct"] = pdp
    with np.errstate(divide="ignore", invalid="ignore"):
        mip = np.where(cpd == 0, 0.0, lift / np.abs(cpd))
    out["macd_increase_pct"] = mip
    out["compare_rank"] = np.int64(1)                      # P10:常量列保留
    out["formation_period"] = bars
    out["is_quick_divergence_y"] = (bars < 3).astype(np.int64)
    out["divergence_strength"] = (0.4 * np.minimum(1.0, np.abs(pdp) / 0.1)
                                  + 0.6 * np.minimum(1.0, mip / 0.1))
    out["price_macd_ratio"] = np.abs(mip / np.maximum(np.abs(pdp), 1e-6))
    out["divergence_magnitude"] = (np.abs(pdp) + np.abs(mip)) / 2.0
    out["confirmation_score"] = ((pdp < -0.02) & (mip > 0.01)).astype(np.int64)
    # volume_signal(P12:prev = cross_prev_date 当日个股成交量;LabelEncoder 仅 train 拟合)
    pv = prev_vol.to_numpy(np.float64)
    evol = ev["event_vol"].to_numpy(np.float64)
    vs = np.where(evol < pv, "bullish",
                  np.where(evol > 1.5 * pv, "bearish", "neutral"))
    le = LabelEncoder()
    le.fit(vs[train_mask])
    unseen = set(np.unique(vs)) - set(le.classes_)
    assert not unseen, f"volume_signal 出现 train 段未见类别 {unseen},停工待处置"
    out["volume_signal"] = le.transform(vs).astype(np.int64)
    enc = {cls: int(i) for i, cls in enumerate(le.classes_)}
    out["divergence_amount"] = ev.groupby("event_date")["ts_code"] \
                                 .transform("size").astype(np.int64)  # P11
    with np.errstate(divide="ignore", invalid="ignore"):
        out["v2j2_macd_increase_pct"] = np.where(
            cp2d == 0, 0.0, (cd - cp2d) / np.abs(cp2d))     # P10 增补列
    out = out[DIVERGENCE_COLS]  # 列序对齐 v2 注册尾序(build_panel DIVERGENCE_COLS)
    assert list(out.columns) == DIVERGENCE_COLS, \
        f"背离结构族列序/列集不符: {list(out.columns)}"
    return out, dict(volume_signal_encoder=enc)


def structure_sample_assert(ev: pd.DataFrame, prev_vol: pd.Series,
                            div: pd.DataFrame, enc: dict) -> dict:
    """§九.5:抽样 500 事件,自写标量重算背离结构族,与向量化结果逐位一致。"""
    rng = np.random.default_rng(SAMPLE_SEED)
    picks = rng.choice(len(ev), size=N_STRUCTURE_SAMPLE, replace=False)
    inv = {v: k for k, v in enc.items()}
    n_checked, mismatches = 0, []
    for i in picks:
        r = ev.iloc[i]
        ac, mpc = float(r.anchor_close), float(r.min_prev_close)
        cd, cpd, cp2d = float(r.cross_dif), float(r.cross_prev_dif), float(r.cross_prev2_dif)
        lift, bars = float(r.dif_lift), int(r.anchor_bars)
        pdp = (ac - mpc) / mpc
        mip = 0.0 if cpd == 0 else lift / abs(cpd)
        exp = {
            "close_current": ac, "close_previous": mpc, "macd_current": cd,
            "macd_previous": cpd, "price_decline_pct": pdp, "macd_increase_pct": mip,
            "compare_rank": 1, "formation_period": bars,
            "is_quick_divergence_y": 1 if bars < 3 else 0,
            "divergence_strength": 0.4 * min(1.0, abs(pdp) / 0.1) + 0.6 * min(1.0, mip / 0.1),
            "price_macd_ratio": abs(mip / max(abs(pdp), 1e-6)),
            "divergence_magnitude": (abs(pdp) + abs(mip)) / 2.0,
            "confirmation_score": 1 if (pdp < -0.02 and mip > 0.01) else 0,
            "v2j2_macd_increase_pct": 0.0 if cp2d == 0 else (cd - cp2d) / abs(cp2d),
        }
        pvv = float(prev_vol.iloc[i])
        vs = ("bullish" if float(r.event_vol) < pvv
              else "bearish" if float(r.event_vol) > 1.5 * pvv else "neutral")
        exp["volume_signal"] = enc[vs]
        n_checked += 1
        for coln, e in exp.items():
            g = div.iloc[i][coln]
            if isinstance(e, float):
                if not (float(g) == e or (np.isnan(float(g)) and np.isnan(e))):
                    mismatches.append(f"行{i} {coln}: {g!r} != {e!r}")
            elif int(g) != int(e):
                mismatches.append(f"行{i} {coln}: {g!r} != {e!r}")
    return dict(n_checked=n_checked, mismatches=mismatches)


def main() -> None:
    ap = argparse.ArgumentParser(description="v2on6 事件×特征主表合成")
    ap.add_argument("--rerun-tag", default="run1")
    args = ap.parse_args()
    tag = args.rerun_tag
    t_all = time.time()
    log(f"MASTER BUILD START | tag={tag}")

    # ---- 输入装载 ----
    ev = pd.read_parquet(EVENTS_PATH)
    assert len(ev) == 96577 and not ev.duplicated(["ts_code", "event_date"]).any()
    # 输入目录随 tag 走(2026-09-11 四轮裁定:原硬编码 eventrows_run1 会在修复链静默吃旧数据)
    ev_files = sorted((CACHE_DIR / f"eventrows_{tag}").glob("chunk_*.parquet"))
    assert ev_files, f"eventrows_{tag} 缺失,先跑 build_panel --full --rerun-tag {tag}"
    evrows = pd.concat([pd.read_parquet(f) for f in ev_files], ignore_index=True)
    assert len(evrows) == 96577, f"事件行切片 {len(evrows)} != 96,577"
    aux_files = sorted((CACHE_DIR / f"event_aux_{tag}").glob("*.parquet"))
    assert aux_files, f"event_aux_{tag} 缺失"
    aux = pd.concat([pd.read_parquet(f).assign(
        ts_code=f.stem) for f in aux_files], ignore_index=True)

    # ---- 键对齐:eventrows → 事件表行序 ----
    evrows = evrows.rename(columns={"trade_date": "event_date"})
    base = ev.merge(evrows, on=["ts_code", "event_date"], how="left", validate="1:1")
    assert len(base) == 96577
    n_missing_feat = int(base["cs_n"].isna().sum())
    assert n_missing_feat == 0, f"{n_missing_feat} 事件无特征行,停工待处置"
    prev_vol = base[["ts_code", "event_date"]].merge(
        aux, on=["ts_code", "event_date"], how="left", validate="1:1")["prev_cross_vol"]
    assert prev_vol.notna().all(), "存在缺 prev_cross_vol 的事件,停工待处置"

    # ---- event_id/seg 自 master_v6 并入((ts_code, event_date, event_row) 互证)----
    m6 = pd.read_parquet(MASTER_V6_PATH, columns=["event_id", "ts_code", "date",
                                                  "event_row", "seg"])
    assert m6["event_id"].is_unique
    base = base.merge(m6.rename(columns={"date": "event_date"}),
                      on=["ts_code", "event_date", "event_row"],
                      how="left", validate="1:1")
    assert base["event_id"].notna().all(), "存在对不上 master_v6 的事件,停工待处置"
    base["event_id"] = base["event_id"].astype(np.int64)

    # ---- 背离结构族(§四.8;train 段 LabelEncoder)----
    train_mask = (base["seg"] == "train").to_numpy()
    div, div_info = compute_divergence_cols(base, prev_vol, train_mask)
    for c in DIVERGENCE_COLS:
        base[c] = div[c].to_numpy()

    # ---- 主表定型:键 + 604 特征,行序 = event_id 升序 ----
    panel_cols = [c for c in MASTER_FEATURE_COLS if c not in DIVERGENCE_COLS]
    feat_present = [c for c in panel_cols if c in base.columns]
    assert feat_present == panel_cols, \
        f"面板列缺失: {sorted(set(panel_cols) - set(feat_present))[:10]}"
    master = base[KEY_COLS + MASTER_FEATURE_COLS] \
        .sort_values("event_id", kind="mergesort").reset_index(drop=True)
    assert len(master) == 96577 and master["event_id"].is_unique     # §九.4
    # §九.2:泄漏列扫描
    bad = [c for c in master.columns
           if c.startswith(("future_", "stop_loss_", "label_"))]
    assert not bad, f"特征表混入标签列: {bad}"
    # §九.3:段界
    idx_sh = pd.read_parquet(SH_INDEX_PATH, columns=["trade_date"])
    sh_cal = pd.to_datetime(idx_sh["trade_date"]).sort_values().unique()
    tep.assert_segment_integrity(
        master.rename(columns={"event_date": "date"}), sh_cal)
    seg_counts = master["seg"].value_counts().to_dict()
    assert {k: seg_counts.get(k, 0) for k in PREREG_SEGMENT_COUNTS} \
        == PREREG_SEGMENT_COUNTS, f"段计数不符预登记: {seg_counts}"

    # ---- §九.5:背离结构族抽样 500 自写重算 ----
    t0 = time.time()
    chk = structure_sample_assert(base, prev_vol, div, div_info["volume_signal_encoder"])
    assert not chk["mismatches"], f"背离结构族抽检不一致: {chk['mismatches'][:5]}"
    log(f"背离结构族抽检 {chk['n_checked']} 事件全过({time.time() - t0:.0f}s)")

    # ---- 词典(§九.6)----
    dic = build_dictionary()
    assert set(dic["column"]) == set(master.columns), \
        f"词典列集与主表不符: {sorted(set(dic['column']) ^ set(master.columns))[:10]}"
    dic.to_csv(SCRIPT_DIR / "dictionary_v2on6.csv", index=False)

    out_path = CACHE_DIR / f"master_v2on6_{tag}.parquet"
    master.to_parquet(out_path, index=False)
    h = hashlib.md5()
    with open(out_path, "rb") as fh:
        for chunk_b in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk_b)
    ledger = dict(tag=tag, sec=round(time.time() - t_all, 1), n_rows=int(len(master)),
                  n_feature_cols=len(MASTER_FEATURE_COLS),
                  n_cols_total=int(master.shape[1]), md5=h.hexdigest(),
                  seg_counts={k: int(v) for k, v in seg_counts.items()},
                  structure_sample=dict(n_checked=chk["n_checked"],
                                        n_mismatch=len(chk["mismatches"])),
                  **div_info)
    with open(CACHE_DIR / f"master_ledger_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    log(f"MASTER BUILD DONE | tag={tag} | {master.shape} | 特征 {len(MASTER_FEATURE_COLS)} | "
        f"md5 {h.hexdigest()} | {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
