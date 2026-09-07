#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背离v6:精选加厚仓位(P7) + 口径D 终审 —— 预登记 = 同目录 README.md(冻结,先于跑数落盘,勿改)。

本脚本 = #32 底座 experiments/v6_portfolio_trial/run_portfolio.py 逐字拷贝 + P7 增量
(增量处逐行注释标记 "P7 增量";标记风格沿用 #37 run_elastic.py / #38 run_thick.py)。
唯一执行脚本。输入输出同 #32;输出 = 每格 equity_curve/trades/stats +
summary_select_thick.csv + verdict_select_thick.json + report.md + detcmp.log(全量双跑逐位对拍)。

网格 = 池(6: v6-1~v6-5 + ALL) × H(10/20/25) × 出场(35,逐字 #32 §3.1) × 挑选(5: S1~S5)
× K(3/5/7) = 9,450 格;pos 段取值 P7K3/P7K5/P7K7。P7 相对 #32 P1 框架的唯一实质改动:
仓位数上限 10 -> K、单仓名义金额 固定 10 万 -> 100 万 ÷ K,K 为网格维 {3,5,7}(README §三.1);
挑选规则 S1~S5 语义、入场二次仓位检查、现金约束 min(名义, 可用现金)、整手舍入、
dropped_cash/dropped_slot_full 记账全部逐字沿用 #32 的 P1/P2 执行路径,
不引入 P5/P6 的"全接+摊薄"语义(README §三.1 末段)。

判活线 = 口径 D 纯现金字面版(README §一.2):excess_cash/capital_utilization > +10pp
且 capital_utilization >= 0.5 且 sharpe_cash > 0.5;参考列 = 混合口径变体
(excess_idx/u > +10pp 且 u>=0.5 且 sharpe_cash>0.5)与旧终审线
(excess_idx > +15pp 且 sharpe_idx > 0.5),三线全出数。

口径钉死(与 README 逐条对应;解释性决断全部集中在 docstring 末尾披露,不动规格):

A. 入场与 E0 逐字沿用 #31 §2.2/§2.3 的"个股序列行号"口径(事件日下一行开盘买,
   第 H 行收盘卖,跌停顺延,耗尽截断)。这是 §六.1 对账自检(浮点差 0)的可执行前提:
   trades_seed.parquet 就是该口径产物。窗口终点 2026-08-31 = 数据终点,
   "持仓延伸过窗口记截断"(§一.1)与 #31 耗尽条款天然重合。
B. E1/E2 执行语义逐字继承 strategy_engine_v3 类 A/B(盘中触及、开盘跳越、
   同日双触发取止损、收盘<=跌停顺延、市场日历计持有日、vol_mult 三档 ≤t-1)。
   市场日历 = 000001.SH 交易日(基准同日历;个股日期 ⊆ 指数日期,实测校验)。
C. E3~E7 统一执行语义(README §三):收盘判定、触发后次日开盘卖、
   执行日开盘跌停/无行情顺延、耗尽截断。H 到期 = 第 H 个市场日历日收盘卖
   (收盘<=跌停顺延),与触发同日竞争时 H 到期优先(保证"最晚 H 日清仓")。
D. 空仓双口径(§一.3):index_wallet 递推 wallet_t = wallet_{t-1}*r_t + (cash_t - cash_{t-1}),
   判活口径权益 = wallet + 持仓市值;纯现金口径 = cash + 持仓市值。
E. 成本原语只读复用 strategy_engine(buy_cost/sell_costs/SLIPPAGE/BOARD_LOT/PRICE_TOL)。
   net_pnl 按 trades_seed 结合序 sh*(xs-px) - 买佣 - 卖佣 - 印花税 计算(对账浮点差 0 要求)。

解释性决断(规格未钉死处,全部披露,不改规格)—— 继承 #32 的 14 条(逐字):
 1. E7/P3/P4 的 ATR 用不复权价(talib.ATR(14),全历史序列):与信号扫描口径一致;
    README §3.1 E7 写"后复权"与信号扫描实际口径冲突,以不复权为准,建议监工修订 README。
 2. P2"当日仓位上限/当日每仓金额"的"当日"锚定信号日:入场日段归属取决于入场日
    是否有信号,在信号日收盘/入场日开盘均不可知,锚定入场日必泄漏。
 3. E7"持仓期最高收盘价"含当日收盘(ATR 仍严格 ≤t-1;含当日无害:当日创新高时不触发)。
 4. S5 量比 = 事件日 vol / 前 20 个个股序列行均 vol(不含事件日);NaN 排最后。
 5. E4 半仓向下整手为 0(持仓仅 100 股)时不发生部分卖出,直接整体转 E3,计数披露。
 6. "跌破"判定 = 收盘 < 线 - 1e-9(PRICE_TOL);"触及/回撤≥"判定含容差(+1e-9)。
 7. §六.6"指数缺失日按前一交易日收益顺延"采用价格前向填充(当日收益 0)解读;
    实测缺失日 = 0(个股日期 ⊆ 指数日期),条款空置,两种解读无差异。
 8. Sharpe 日收益标准差用样本 std(ddof=1);年化 = (末/初)^(252/(交易日数-1)) - 1。
 9. 资金利用率 = 日均在持市值 / 100 万(固定总资金口径)。
10. E2 全市场均值 ATR 宇宙 = stock_data/daily 全部文件(含指数/BJ,逐字继承 #28
    build_mkt_atr),切片 [窗口首日-90 自然日, 窗口末日]。
11. E4 部分卖出的入场佣金按卖出股数占比分摊;E3~E7 开盘跌停判定用开盘价
    (README 字面"开盘跌停"),与 v1 引擎收盘口径不同,并列披露。
12. 退化对账格(P1×E0×S1×H20,仓位上限=无穷)现金约束同步放开
    (每股预算仍 10 万,与 trades_seed 的 BUDGET 口径逐字一致)。
13. 同一股票可叠加多笔持仓(不同事件),与引擎 v1/v3 一致,不去重。
14. 判活口径为"闲置现金按指数记账"的叠加层,累计指数亏损可使其权益 <=0:
    此时年化取 -1(全亏地板)、Sharpe 记 NaN(收益符号无意义)、该格判活必否,
    负权益天数计数披露(neg_equity_*_days);逐年收益在上年末权益 <=0 时记 None。

P7 增量解释性决断(本实验新增,规格未钉死处,全部披露,不改规格):
 T-1. K 为网格维,取值 {3,5,7};cell_id/cfg 的 pos 段 = P7K3/P7K5/P7K7,K 由 pos 段
    解析(int(pos[3:]));网格枚举顺序 = 池→H→挑选→K→出场(K 在出场外层,与 README §三.2
    列举顺序"池×H×出场×挑选×K"字面不同,但格子集合逐位相同 9,450 格,枚举顺序不影响任何
    单格结果与汇总排序);非 P7 格(退化对账格 P1)无 K,run_cell 内 cfg.get("k") 防御。
 T-2. P7 单仓名义金额 = 100 万 ÷ K,在 slot_budget 增 P7 分支;仓位数上限 = K,在信号日
    挑选块覆盖 cap;入场二次仓位检查(len(positions) >= cap_sel 丢弃)、现金约束
    min(名义, 可用现金)、整手舍入(向下,不足 1 手先顶 1 手再被现金约束回收,与 #32 逐字
    一致)、dropped_cash 不递补、dropped_slot_full 记账——全部逐字沿用 #32 的 P1/P2 执行
    路径;不引入 P5/P6 的"全接+摊薄"语义(README §三.1)。
 T-3. capital_utilization == 0 时 per_cash/per_idx 记 NaN(防御;判活线含 u>=0.5,
    u=0 格必不过线;沿用 #37 决断 E-5 / #38 决断 D-6)。
 T-4. 旧终审线参考列按本实验 README §一.2/§七.5 字面 = excess_idx > +15pp 且
    sharpe_idx > 0.5 两条;#32 旧线的 S5 附加 cluster_t >= 2 条款不纳入参考列(README
    未列入),cluster_t 列照常出数披露。
 T-5. §五.1 口径复算锚定从 #38 冻结 summary_thick.csv 独立重算:口径 D 纯现金字面版
    0 格、混合口径变体 4 格、旧终审线 0 格,三数必须逐位复现否则本实验不出数;
    #32/#37/#38 冻结产物只读,不重跑不修改。
 T-6. 全市场 ATR 磁盘缓存只读复用 #32 的确定性产物(cache/mkt_atr_lb{14,21}.parquet;
    缓存缺失时按 #32 同算法确定性重建;沿用 #37 决断 E-9 / #38 决断 D-10)。
 T-7. §五.2 仓位数上限不变量从 trades.parquet 独立重建验证:E4 部分卖出按
    (股票,事件日,入场日,入场价) 归并还原单笔入场(同 #38 决断 D-12 归并口径),验证
    任一入场日当日入场笔数 <= K,且按区间 [入场日, 最后卖出日) 扫描的并发在持仓数 <= K;
    另加两道交叉验证(超出 README 字面的加固,披露):(a) equity_curve.parquet 的
    n_positions 全程 <= K(覆盖窗口终点截断未平仓持仓,其不在 trades 内);(b) 每格
    entered == 已平仓归并入场笔数 + open_at_end(与 stats 对账;已平仓 = 归并组内含非部分
    卖出行,E4 部分卖出后仍截断在仓的入场计入 open_at_end 一侧)。
 T-8. §五.3 单仓上限不变量:全部成交笔 shares × entry_price <= 100万 ÷ K + 0.01 元
    浮点容差(引擎入场块保证 sh×px <= min(名义, 可用现金) + 1e-6;E4 部分卖出每笔为
    原入场股的子集,逐笔判定即可)。
 T-9. 确定性自检(README §五.4) = 全量 9,450 格双跑,逐格 summary+stats(除 runtime_sec)
    逐位对拍落 detcmp.log;equity/trades 逐位由抽样 selfcheck_determinism 深度验证
    (沿用 #32/#37/#38 框架)。
T-10. summary_select_thick.csv 的 passed 列 = 主线口径 D 纯现金字面版;两个参考列名单
    只在 verdict_select_thick.json / report.md 出数(沿用 #37 决断 E-7 / #38 决断 D-8)。
T-11. 退化对账格沿用 #32 的 P1×E0×S1×H20(仓位上限=无穷、现金约束放开、每股预算 10 万),
    非 P7 格;#32 §六.1 对账自检(浮点差 0)原样保留作继承断言门槛(README §五.5)。
T-12. 报告对照聚合粒度 = 池×H 与仓位规则;P1/P2 数字取自 #32 冻结 summary_portfolio.csv,
    P5 数字取自 #37 冻结 summary_elastic.csv,P6 数字取自 #38 冻结 summary_thick.csv,
    均不重跑(README §三.4)。
T-13. P7 格保留挑选维(S1~S5);P5/P6 无挑选维(sel=S0),对照表中 P5/P6 数字在同一
    (池,H) 下与挑选规则无关;信号池价值与池内排序增益分开表述不混比(背离唯一底座指令)。

用法:
    python3 run_select_thick.py --mode calibrate   # 自检 §五.1:#38 冻结 summary 三线过线数复算
    python3 run_select_thick.py --mode selfcheck   # 自检 §五.1 + 继承对账自检 + 抽样确定性
    python3 run_select_thick.py --mode benchmark   # 自检 + 8 抽样格计时(两遍,确定性)
    python3 run_select_thick.py --mode full        # 全量 9,450 格双跑 + 终审出数(监工派活)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

REPO = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO, "v3_pipeline", "scripts"))
import strategy_engine as se  # noqa: E402  冻结引擎:只读复用成本原语与常量

OUT_DIR = Path(os.path.join(REPO, "experiments", "v6_select_thick_trial"))  # P7 增量:本实验目录
BASE_DIR = Path(os.path.join(REPO, "experiments", "v6_portfolio_trial"))  # P7 增量:#32 底座目录(冻结产物只读)
CACHE_DIR = BASE_DIR / "cache"   # P7 增量:mkt_atr 缓存只读复用 #32 确定性产物(决断 T-6)
BASE_SUMMARY_PATH = BASE_DIR / "summary_portfolio.csv"  # P7 增量:#32 冻结 summary(P1/P2 对照用,决断 T-12)
ELASTIC_DIR = Path(os.path.join(REPO, "experiments", "v6_elastic_pos_trial"))  # P7 增量:#37 目录(冻结产物只读)
ELASTIC_SUMMARY_PATH = ELASTIC_DIR / "summary_elastic.csv"  # P7 增量:#37 冻结 summary(P5 对照用,决断 T-12)
THICK_DIR = Path(os.path.join(REPO, "experiments", "v6_thick_pos_trial"))  # P7 增量:#38 目录(冻结产物只读)
THICK_SUMMARY_PATH = THICK_DIR / "summary_thick.csv"  # P7 增量:#38 冻结 summary(§五.1 口径复算锚定 + P6 对照用,决断 T-5/T-12)
RUNS_DIR = OUT_DIR / "runs"
LOG_PATH = OUT_DIR / "progress.log"
DETCMP_PATH = OUT_DIR / "detcmp.log"  # P7 增量:全量双跑逐位对拍日志(README §五.4)
EVENTS_PATH = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history",
                                "events_history_v1.parquet"))
TRADES_SEED_PATH = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history",
                                     "trades_seed.parquet"))
DATA_DIR = Path(os.path.join(REPO, "stock_data", "daily"))
LIMIT_DIR = Path(os.path.join(REPO, "stock_data", "stk_limit"))
INDEX_PATH = DATA_DIR / "000001.SH.parquet"

EVENT_START = pd.Timestamp("2011-09-01")
EVENT_END = pd.Timestamp("2026-08-31")
D_START = int(EVENT_START.strftime("%Y%m%d"))
D_END = int(EVENT_END.strftime("%Y%m%d"))
STAMP_SWITCH_INT = 20230828
INIT = se.INIT_CAPITAL  # 100 万
TOL = se.PRICE_TOL
H_LIST = [10, 20, 25]
POOLS = ["v6-1", "v6-2", "v6-3", "v6-4", "v6-5", "ALL"]
SELS = ["S1", "S2", "S3", "S4", "S5"]
K_LIST = [3, 5, 7]        # P7 增量:精选加厚仓位 K 网格维(README §三.2),pos 段 = P7K{k}(替换 #32 的 POS_RULES)
S4_SEED = 42

# ---------------------------------------------------------------- 出场配置网格(§3.1/§七.6)
E1_GRID = []  # A1~A12: tp 外层 × sl 内层(同 #28 grid_a 序)
for _tp in (0.20, 0.25, 0.30, 0.35):
    for _sl in (-0.10, -0.14, -0.18):
        E1_GRID.append(dict(name=f"E1_A{len(E1_GRID) + 1}", family="E1", tp=_tp, sl=_sl))

_V12 = dict(tp=0.25, sl=-0.14, vol_lookback=21, vol_high_thresh=1.8, vol_low_thresh=0.6,
            vol_profit_mult=1.5, vol_stop_mult=1.1, low_vol_profit_mult=1.0)
_B_PERTURB = {  # #28 grid_b 扰动表(B15/B16/B17 为 H/N 变体,去重并入 B1,§七.6)
    2: dict(vol_lookback=14), 3: dict(vol_high_thresh=2.5), 4: dict(vol_low_thresh=0.4),
    5: dict(vol_low_thresh=0.8), 6: dict(vol_profit_mult=1.2), 7: dict(vol_profit_mult=2.0),
    8: dict(vol_stop_mult=1.5), 9: dict(low_vol_profit_mult=0.8), 10: dict(tp=0.20),
    11: dict(tp=0.30), 12: dict(tp=0.35), 13: dict(sl=-0.10), 14: dict(sl=-0.18),
    18: dict(vol_profit_mult=2.0, vol_high_thresh=2.5),
    19: dict(tp=0.35, vol_profit_mult=2.0, vol_high_thresh=2.5),
    20: dict(vol_lookback=14, vol_low_thresh=0.4, vol_profit_mult=1.2),
}
E2_GRID = [dict(name="E2_B1", family="E2", **_V12)]
for _i in sorted(_B_PERTURB):
    E2_GRID.append(dict(name=f"E2_B{_i}", family="E2", **{**_V12, **_B_PERTURB[_i]}))
assert len(E2_GRID) == 17

EXIT_CONFIGS = ([dict(name="E0", family="E0")] + E1_GRID + E2_GRID
                + [dict(name="E3", family="E3"), dict(name="E4", family="E4"),
                   dict(name="E5", family="E5"), dict(name="E6", family="E6"),
                   dict(name="E7", family="E7")])
assert len(EXIT_CONFIGS) == 35
EXIT_BY_NAME = {c["name"]: c for c in EXIT_CONFIGS}
P4_EXIT_FAMILIES = ("E1", "E2", "E5", "E7")  # §3.2 P4 仅搭配有初始止损线的出场族

# 降级预案代表出场 8 格(§四,逐字继承 #32;本实验无降级预案,保留常量不动)
DEGRADED_EXITS = {"E0", "E1_A5", "E2_B1", "E3", "E4", "E5", "E6", "E7"}

# 基准抽样 8 格(施工计时用,覆盖全部出场族与多种挑选规则,确定性硬编码;P7 增量:pos=P7K5(中档 K))
BENCHMARK_CELLS = [
    ("ALL", 20, "S1", "P7K5", "E0"), ("ALL", 20, "S1", "P7K5", "E1_A5"),
    ("ALL", 20, "S1", "P7K5", "E2_B1"), ("ALL", 20, "S1", "P7K5", "E3"),
    ("ALL", 20, "S1", "P7K5", "E4"), ("v6-5", 20, "S4", "P7K5", "E5"),
    ("v6-5", 10, "S2", "P7K5", "E6"), ("v6-1", 25, "S5", "P7K5", "E7"),
]

# ---------------------------------------------------------------- 全局(fork 共享,只读)
_G: dict = {}
_HB = {"stage": "init", "done": 0, "total": 0, "stop": False, "t0": time.time()}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _heartbeat() -> None:
    while not _HB["stop"]:
        time.sleep(60)
        if _HB["stop"]:
            break
        frac = f"{_HB['done']}/{_HB['total']}" if _HB["total"] else "-"
        log(f"heartbeat: stage={_HB['stage']} cells={frac} "
            f"elapsed={time.time() - _HB['t0']:.0f}s")


def set_stage(stage: str, done: int = 0, total: int = 0) -> None:
    _HB["stage"], _HB["done"], _HB["total"] = stage, done, total
    log(f"[stage] {stage}" + (f" ({done}/{total})" if total else ""))


def d_int(ts: pd.Timestamp) -> int:
    return ts.year * 10000 + ts.month * 100 + ts.day


def yyyymmdd_arr(raw: np.ndarray) -> np.ndarray:
    """日期统一为 YYYYMMDD int32(个股文件为 datetime64,指数文件为 int64,混合)。"""
    if np.issubdtype(raw.dtype, np.datetime64):
        ts = pd.DatetimeIndex(raw)
        return (ts.year * 10000 + ts.month * 100 + ts.day).to_numpy(dtype=np.int32)
    return raw.astype(np.int32)


# ---------------------------------------------------------------- 阶段 1:数据加载
def _enrich_stock(task) -> tuple:
    """单股:读全历史日线,算数组(ATR 等)与该股全部事件的衍生字段。

    dd20/bounce 逐字复刻 run_seeds.simulate_stock 口径(全历史行号 j):
      dd20 = close[j]/max(close[max(0,j-20)..j]) - 1;bounce = close[j]/anchor_close - 1。
    """
    code, ev_list = task
    df = pd.read_parquet(DATA_DIR / f"{code}.parquet",
                         columns=["trade_date", "open", "high", "low", "close", "vol"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(df)
    dts = yyyymmdd_arr(df["trade_date"].to_numpy())
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["vol"].to_numpy(dtype=np.float64)
    # TR 与两种 ATR 口径(全历史一次性计算,确定无泄漏)
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(np.maximum(h - l, np.abs(h - prev_c)), np.abs(l - prev_c))
    tr_s = pd.Series(tr)
    atr_sma14 = tr_s.rolling(14, min_periods=14).mean().to_numpy()
    atr_sma21 = tr_s.rolling(21, min_periods=21).mean().to_numpy()
    atr14_talib = talib.ATR(h, l, c, timeperiod=14)
    pos = {int(d): i for i, d in enumerate(dts)}

    out_events = []
    for ev_date_int, anchor_close in ev_list:
        j = pos.get(ev_date_int, -1)
        if j < 0:
            continue  # 不会发生(事件由同一文件生成),防御
        close_ev = c[j]
        dd20 = close_ev / c[max(0, j - 20):j + 1].max() - 1.0
        bounce = close_ev / anchor_close - 1.0
        vprev = v[max(0, j - 20):j]
        vol_ratio = float(v[j] / vprev.mean()) if len(vprev) and vprev.mean() > 0 \
            else float("nan")
        a14 = atr14_talib[j]
        atr_pct = float(a14 / close_ev) if np.isfinite(a14) and close_ev > 0 \
            else float("nan")
        entry_row = j + 1 if j + 1 < n else -1
        entry_date = int(dts[j + 1]) if j + 1 < n else -1
        out_events.append(dict(ts_code=code, event_date=ev_date_int,
                               anchor_close=float(anchor_close), dd20=float(dd20),
                               bounce=float(bounce), vol_ratio=vol_ratio,
                               atr_pct=atr_pct, entry_row=entry_row,
                               entry_date=entry_date))
    arrays = dict(dts=dts, open=o, high=h, low=l, close=c, vol=v,
                  atr_sma14=atr_sma14, atr_sma21=atr_sma21,
                  atr14_talib=atr14_talib, pos=pos, n=n)
    return code, arrays, out_events


def load_events_and_stocks() -> None:
    t0 = time.time()
    ev = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date", "anchor_close"])
    ev = ev[(ev["event_date"] >= EVENT_START) & (ev["event_date"] <= EVENT_END)]
    log(f"[load] 事件池窗口内 {len(ev)} 起,个股 {ev['ts_code'].nunique()} 只")
    by_code: dict[str, list] = {}
    for r in ev.itertuples(index=False):
        by_code.setdefault(r.ts_code, []).append((d_int(r.event_date), r.anchor_close))
    tasks = [(code, by_code[code]) for code in sorted(by_code)]
    stocks: dict = {}
    all_events: list[dict] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (code, arrays, evs) in enumerate(
                pool.imap_unordered(_enrich_stock, tasks, chunksize=8)):
            stocks[code] = arrays
            all_events.extend(evs)
            if (i + 1) % 1000 == 0:
                log(f"heartbeat: 个股加载+事件富化 {i + 1}/{len(tasks)} "
                    f"({time.time() - t0:.0f}s)")
    evdf = pd.DataFrame(all_events).sort_values(["event_date", "ts_code"],
                                                kind="mergesort").reset_index(drop=True)
    assert len(evdf) == len(ev), "事件富化丢行"
    _G["stocks"] = stocks
    _G["events"] = evdf
    log(f"[load] 个股 {len(stocks)} 只加载完成,事件 {len(evdf)} 起 "
        f"({time.time() - t0:.0f}s)")


def load_index_calendar() -> None:
    """交易日历 = 000001.SH 窗口内交易日(升序);个股日期 ⊆ 指数日期硬校验(§六.6)。"""
    idx = pd.read_parquet(INDEX_PATH, columns=["trade_date", "close"])
    idx = idx.sort_values("trade_date").reset_index(drop=True)  # 降序 -> 升序(§六.6)
    assert idx["trade_date"].is_monotonic_increasing
    dts = yyyymmdd_arr(idx["trade_date"].to_numpy())
    close = idx["close"].to_numpy(dtype=np.float64)
    mask = (dts >= D_START) & (dts <= D_END)
    cal = dts[mask]
    close_w = close[mask]
    # 个股日期 ⊆ 指数日期校验(窗口内,事件个股)
    extra = 0
    for S in _G["stocks"].values():
        d = S["dts"]
        d = d[(d >= D_START) & (d <= D_END)]
        extra += int((~np.isin(d, cal)).sum())
    _G["index_missing_days"] = extra  # §六.6 计数披露(个股有行情而指数无)
    assert extra == 0, f"个股日期超出指数日历 {extra} 天,需启用并集日历+价格顺延"
    ret = np.ones(len(cal))
    ret[1:] = close_w[1:] / close_w[:-1]
    _G["cal"] = cal
    _G["cal_index"] = {int(d): i for i, d in enumerate(cal)}
    _G["idx_close"] = close_w
    _G["idx_ret"] = ret
    log(f"[load] 交易日历 {len(cal)} 天 [{cal[0]}..{cal[-1]}],"
        f"指数缺失日(§六.6) {extra}")


def _limit_read_one(d: int):
    fp = LIMIT_DIR / f"{d}.parquet"
    if not fp.exists():
        return d, None
    lf = pd.read_parquet(fp)
    lf = lf[lf["ts_code"].isin(_G["limit_codes"])]
    return d, (lf["ts_code"].to_numpy(),
               lf["up_limit"].to_numpy(dtype=np.float64),
               lf["down_limit"].to_numpy(dtype=np.float64))


def load_limits() -> None:
    """窗口内全部交易日的涨跌停表,过滤到事件个股,按 code 组织有序数组。"""
    t0 = time.time()
    _G["limit_codes"] = frozenset(_G["events"]["ts_code"].unique())
    dates = [int(d) for d in _G["cal"]]
    acc: dict[str, list] = {}
    missing = 0
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (d, res) in enumerate(pool.imap_unordered(_limit_read_one, dates,
                                                         chunksize=16)):
            if res is None:
                missing += 1
            elif len(res[0]):
                for code, up, dn in zip(*res):
                    acc.setdefault(code, []).append((d, up, dn))
            if (i + 1) % 1000 == 0:
                log(f"heartbeat: 涨跌停加载 {i + 1}/{len(dates)} ({time.time() - t0:.0f}s)")
    limits: dict[str, tuple] = {}
    for code, rows in acc.items():
        rows.sort(key=lambda x: x[0])
        limits[code] = (np.array([r[0] for r in rows], dtype=np.int32),
                        np.array([r[1] for r in rows], dtype=np.float64),
                        np.array([r[2] for r in rows], dtype=np.float64))
    _G["limits"] = limits
    _G["limit_missing_days"] = missing
    log(f"[load] 涨跌停 {len(dates) - missing}/{len(dates)} 天,缺文件 {missing} 天,"
        f"覆盖 {len(limits)} 股 ({time.time() - t0:.0f}s)")


def _mkt_atr_one(args):
    fn, lo, hi, lb = args
    df = pd.read_parquet(DATA_DIR / fn, columns=["trade_date", "high", "low", "close"])
    dts = yyyymmdd_arr(df["trade_date"].to_numpy())
    m = (dts >= lo) & (dts <= hi)
    if m.sum() < lb:  # 逐字 #28 build_mkt_atr: len(df) < lookback 才跳过
        return None
    df = df[m].sort_values("trade_date")
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(np.maximum(h - l, np.abs(h - prev_c)), np.abs(l - prev_c))
    a = pd.Series(tr).rolling(lb, min_periods=lb).mean().to_numpy()
    d = yyyymmdd_arr(df["trade_date"].to_numpy())
    ok = np.isfinite(a) & (a > 0)
    return d[ok], a[ok]


def build_mkt_atr(lb: int) -> np.ndarray:
    """全市场逐日 ATR(lb) 均值,逐字继承 #28 build_mkt_atr 口径(daily/ 全部文件)。
    返回按日历对齐的均值数组(与 _G['cal'] 同序)。带磁盘缓存(确定性重建)。"""
    cache = CACHE_DIR / f"mkt_atr_lb{lb}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return df.set_index("date")["atr"].reindex(_G["cal"]).to_numpy()
    t0 = time.time()
    cal0 = pd.Timestamp(str(_G["cal"][0]))
    lo = d_int(cal0 - pd.Timedelta(days=90))
    hi = int(_G["cal"][-1])
    files = sorted(os.listdir(DATA_DIR))
    acc: dict[int, list] = {}
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(
                _mkt_atr_one, [(fn, lo, hi, lb) for fn in files], chunksize=8)):
            if res is not None:
                d, a = res
                for dd, vv in zip(d, a):
                    acc.setdefault(int(dd), []).append(float(vv))
            if (i + 1) % 1500 == 0:
                log(f"heartbeat: mkt_atr lb={lb} {i + 1}/{len(files)} "
                    f"({time.time() - t0:.0f}s)")
    # 确定性:每股 ATR 贡献均值与累加序无关(均值 = sum/count,浮点累加序按文件名序
    # 固定不可得(imap_unordered),改用排序后逐项累加保证逐位确定)
    sums: dict[int, float] = {}
    cnts: dict[int, int] = {}
    for d in sorted(acc):
        vals = sorted(acc[d])  # 固定序:值排序,消除进程到达序影响
        s = 0.0
        for vv in vals:
            s += vv
        sums[d] = s
        cnts[d] = len(vals)
    ser = pd.Series({d: sums[d] / cnts[d] for d in sorted(sums)})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ser.index.to_numpy(), "atr": ser.to_numpy()}).to_parquet(cache)
    out = ser.reindex(_G["cal"]).to_numpy()
    log(f"[load] mkt_atr lb={lb} 有效日 {int(np.isfinite(out).sum())}/{len(out)} "
        f"({time.time() - t0:.0f}s,已缓存)")
    return out


# ---------------------------------------------------------------- 阶段 2:池与预计算
def build_pools() -> None:
    ev = _G["events"]
    dd = ev["dd20"].to_numpy()
    bo = ev["bounce"].to_numpy()
    band = (bo > 0.02) & (bo <= 0.08)
    m = {"v6-1": (dd <= -0.15) & band, "v6-2": (dd <= -0.20) & band,
         "v6-3": (dd <= -0.25) & band, "v6-4": dd <= -0.15, "v6-5": dd <= -0.25,
         "ALL": np.ones(len(ev), dtype=bool)}
    # S3 种子强度(嵌套链 v6-5>v6-3>v6-2>v6-1>v6-4>仅 ALL)
    rank = np.full(len(ev), 5, dtype=np.int8)
    rank[m["v6-4"]] = 4
    rank[m["v6-1"]] = 3
    rank[m["v6-2"]] = 2
    rank[m["v6-3"]] = 1
    rank[m["v6-5"]] = 0
    ev["seed_rank"] = rank
    codes_arr = ev["ts_code"].to_numpy()
    _, code_int = np.unique(codes_arr, return_inverse=True)
    ev["code_int"] = code_int
    _G["events"] = ev

    pools: dict = {}
    cal = _G["cal"]
    for p in POOLS:
        sub = ev[m[p]].reset_index(drop=True)
        by_date: dict[int, np.ndarray] = {}
        for d, g in sub.groupby("event_date", sort=True):
            by_date[int(d)] = g.index.to_numpy()
        # P2 信号段仓位日程:连续交易日每日 >=1 信号(§3.2,"日历"= 交易日历)
        caps: dict[int, int] = {}
        sig_days = set(by_date)
        seg_pos = 0
        for day in cal:
            day = int(day)
            if day in sig_days:
                seg_pos += 1
                caps[day] = 6 if seg_pos <= 2 else (10 if seg_pos <= 7 else 15)
            else:
                seg_pos = 0
        # P3 当日全部候选信号 ATR% 中位
        med = sub.groupby("event_date")["atr_pct"].median()
        pools[p] = dict(
            df=sub, by_date=by_date, n=len(sub),
            codes=sub["ts_code"].to_numpy(),
            code_int=sub["code_int"].to_numpy(),
            dd20=sub["dd20"].to_numpy(), bounce=sub["bounce"].to_numpy(),
            vol_ratio=sub["vol_ratio"].to_numpy(),
            seed_rank=sub["seed_rank"].to_numpy(),
            atr_pct=sub["atr_pct"].to_numpy(),
            event_date=sub["event_date"].to_numpy(),
            entry_date=sub["entry_date"].to_numpy(),
            entry_row=sub["entry_row"].to_numpy(),
            p2_caps=caps, p3_median={int(k): float(v) for k, v in med.items()},
        )
        log(f"[pool] {p}: {len(sub)} 事件,信号日 {len(by_date)} 天")
    _G["pools"] = pools


# ---------------------------------------------------------------- 引擎原语
def limit_lookup(code: str, day: int, which: int) -> float:
    """which: 1=up 2=dn;缺文件/无该股行 -> NaN(无约束,同 #31)。"""
    lim = _G["limits"].get(code)
    if lim is None:
        return np.nan
    dates, up, dn = lim
    i = int(np.searchsorted(dates, day))
    if i < len(dates) and dates[i] == day:
        return float(up[i] if which == 1 else dn[i])
    return np.nan


def stamp_rate(day: int) -> float:
    return se.STAMP_TAX_OLD if day < STAMP_SWITCH_INT else se.STAMP_TAX_NEW


def select_chosen(pg: dict, sig_idx: np.ndarray, n_free: int, rule: str,
                  rng: np.random.RandomState) -> np.ndarray:
    """§3.3 挑选:信号数 > 空位数时按规则截取;落选当日放弃不顺延。只用当日信息。"""
    if n_free <= 0 or len(sig_idx) == 0:
        return sig_idx[:0]
    if rule == "S1":  # 先到先得 = ts_code 升序(by_date 构造保序)
        order = sig_idx
    elif rule == "S2":  # dd20 最深优先,平局 ts_code 升序
        k = np.lexsort((pg["code_int"][sig_idx], pg["dd20"][sig_idx]))
        order = sig_idx[k]
    elif rule == "S3":  # 种子强度,同级 dd20 升序,再平局 ts_code 升序
        k = np.lexsort((pg["code_int"][sig_idx], pg["dd20"][sig_idx],
                        pg["seed_rank"][sig_idx]))
        order = sig_idx[k]
    elif rule == "S4":  # 随机抽签 RandomState(42) 逐日 shuffle
        order = sig_idx[rng.permutation(len(sig_idx))]
    elif rule == "S5":  # 已弹最少 + 量比最冷(NaN 排最后)
        vr = pg["vol_ratio"][sig_idx]
        vr = np.where(np.isfinite(vr), vr, np.inf)
        k = np.lexsort((pg["code_int"][sig_idx], vr, pg["bounce"][sig_idx]))
        order = sig_idx[k]
    else:
        raise ValueError(rule)
    return order[:n_free]


def slot_budget(cfg: dict, pg: dict, ev_i: int, day: int, cap: int,
                stats: dict) -> float:
    """各仓位族每仓金额(信号日收盘决策,因果干净)。"""
    pos = cfg["pos"]
    if pos == "P1":
        return 100_000.0
    if pos == "P2":
        return INIT / cap
    if pos == "P3":
        med = pg["p3_median"].get(day, np.nan)
        a = pg["atr_pct"][ev_i]
        if not (np.isfinite(med) and np.isfinite(a) and a > 0):
            stats["p3_atr_fallback"] += 1
            return 100_000.0
        return float(min(150_000.0, max(50_000.0, 100_000.0 * (med / a))))
    if pos == "P4":
        ex = cfg["exit"]
        fam = ex["family"]
        if fam in ("E1", "E2"):
            dist = abs(ex["sl"])  # E2 取中档生效 |sl|(§3.2)
        elif fam == "E5":
            dist = 0.10
        else:  # E7: 3×ATR%
            a = pg["atr_pct"][ev_i]
            if not (np.isfinite(a) and a > 0):
                stats["p4_atr_fallback"] += 1
                dist = 0.10
            else:
                dist = 3.0 * a
        return min(200_000.0, 10_000.0 / dist)
    if pos.startswith("P7"):
        # P7 增量:P7 单仓名义金额 = 100万 ÷ K(README §三.1;K=10 即退回 P1 等仓十仓);
        # P7 增量:现金约束 min(名义, 可用现金) 与整手舍入在入场执行块逐字沿用 #32(决断 T-2)。
        return INIT / cfg["k"]  # P7 增量
    raise ValueError(pos)


def vol_band_e2(vm, ex: dict) -> tuple[float, float]:
    """逐字继承引擎类 B 三档(v12 语义,缺失回落中档)。"""
    v = 1.0 if vm is None or not np.isfinite(vm) else vm
    if v >= ex["vol_high_thresh"]:
        return ex["tp"] * ex["vol_profit_mult"], ex["sl"] * ex["vol_stop_mult"]
    if v <= ex["vol_low_thresh"]:
        return ex["tp"] * ex["low_vol_profit_mult"], ex["sl"]
    return ex["tp"], ex["sl"]


# ---------------------------------------------------------------- 主回测循环(单格)
def run_cell(cfg: dict) -> dict:
    """跑一格,返回 {summary, trades, equity, stats}。内置自检 2/3/4(硬断言)。"""
    t0 = time.time()
    pg = _G["pools"][cfg["pool"]]
    stocks = _G["stocks"]
    cal = _G["cal"]
    cal_index = _G["cal_index"]
    idx_ret = _G["idx_ret"]
    H = cfg["H"]
    K = cfg.get("k")  # P7 增量:精选加厚仓位数上限/名义分母 K(README §三.1;非 P7 退化对账格为 None 不使用,决断 T-1)
    ex = cfg["exit"]
    fam = ex["family"]
    degenerate = cfg.get("degenerate", False)
    rng = np.random.RandomState(S4_SEED)

    stats = dict(total_signals=pg["n"], entered=0,
                 dropped_slot_full=0, dropped_limitup=0, dropped_no_quote=0,
                 dropped_cash=0, truncated_no_next=0, deferred_exits=0,
                 truncated_window=0, truncated_exhausted=0,
                 vol_fallback_mid=0, p3_atr_fallback=0, p4_atr_fallback=0,
                 e4_lot_skip=0, e7_atr_missing=0,
                 exits_E0=0, exits_tp=0, exits_sl=0, exits_horizon=0,
                 exits_trail_tp=0, exits_scale_out=0, exits_breakeven_sl=0,
                 exits_time_stop=0, exits_chandelier_sl=0)

    cash = INIT
    wallet = INIT          # 空仓买指数台账(判活口径)
    cash_prev = INIT       # 上日现金(双口径递推用)
    positions: list[dict] = []
    pending: dict[int, list] = {}
    trades: list[dict] = []
    equity_rows: list[tuple] = []
    flows_days: list[list[float]] = []   # 自检 2:逐日有序现金流
    outcomes: dict = {} if degenerate else None  # 自检 1:逐事件结局

    def do_sell(p: dict, day: int, raw: float, sh_sell: int, reason: str,
                flows: list) -> None:
        """卖出记账(全卖/部分卖),trades_seed 结合序算 net_pnl。"""
        nonlocal cash
        xs = raw * (1.0 - se.SLIPPAGE)
        comm, stamp = se.sell_costs(sh_sell, xs, pd.Timestamp(str(day)))
        proceeds = sh_sell * xs - comm - stamp
        cash += proceeds
        flows.append(proceeds)
        # 入场佣金按股数占比分摊,最后一笔取余量(总额守恒)
        if sh_sell >= p["shares"]:
            alloc = p["comm_left"]
        else:
            alloc = p["entry_commission"] * (sh_sell / p["shares"])
        p["comm_left"] -= alloc
        p["shares"] -= sh_sell
        net_pnl = sh_sell * (xs - p["entry_price"]) - alloc - comm - stamp
        row = stocks[p["ts_code"]]["pos"].get(day, -1)
        trades.append(dict(
            ts_code=p["ts_code"], event_date=p["event_date"],
            entry_date=p["entry_date"], entry_raw=p["entry_raw"],
            entry_price=p["entry_price"], shares=sh_sell,
            entry_commission=alloc,
            exit_date=day, exit_reason=reason, exit_raw_price=float(raw),
            exit_exec_price=xs, exit_commission=comm, stamp_tax=stamp,
            gross_ret=xs / p["entry_price"] - 1.0,
            gross_pnl=sh_sell * (xs - p["entry_price"]),
            total_cost=alloc + comm + stamp, net_pnl=net_pnl,
            net_ret=net_pnl / (sh_sell * p["entry_price"] + alloc),
            held_days=cal_index[day] - cal_index[p["entry_date"]] + 1,
            held_rows=(row - p["entry_row"] + 1) if row >= 0 else -1,
            deferred_days=p["deferred_days"], partial=bool(p["shares"] > 0),
        ))
        stats[f"exits_{reason}"] += 1

    for di in range(len(cal)):
        day = int(cal[di])
        flows: list[float] = []

        # ---- 0. 信号日收盘挑选(空位 = 仓位上限 - 当日开盘前在持数,§3.3)----
        sig_idx = pg["by_date"].get(day)
        if sig_idx is not None:
            cap = pg["p2_caps"].get(day, 6) if cfg["pos"] == "P2" else 10
            if cfg["pos"].startswith("P7"):
                cap = K  # P7 增量:仓位数上限 = K(README §三.1:P7 = #32 P1 框架把仓位数 10 换成 K,其余一概不动)
            if degenerate:
                cap = 10 ** 9
            n_free = cap - len(positions)
            if n_free < len(sig_idx):
                stats["dropped_slot_full"] += len(sig_idx) - max(n_free, 0)
            chosen = select_chosen(pg, sig_idx, n_free, cfg["sel"], rng)
            for ev_i in chosen:
                ed = int(pg["entry_date"][ev_i])
                if ed < 0:
                    stats["truncated_no_next"] += 1
                    if degenerate:
                        outcomes[(pg["codes"][ev_i], int(pg["event_date"][ev_i]))] = \
                            dict(status="truncated_no_next")
                    continue
                budget = slot_budget(cfg, pg, int(ev_i), day, cap, stats)
                pending.setdefault(ed, []).append((int(ev_i), cap, budget))

        # ---- 1. 入场(事件日下一交易行开盘;先于当日卖出,卖出现金当日不可用)----
        ents = pending.pop(day, None)
        if ents:
            for ev_i, cap_sel, budget in ents:
                code = pg["codes"][ev_i]
                evd = int(pg["event_date"][ev_i])
                if len(positions) >= cap_sel:
                    stats["dropped_slot_full"] += 1
                    continue
                S = stocks[code]
                erow = int(pg["entry_row"][ev_i])
                assert 0 <= erow < S["n"] and int(S["dts"][erow]) == day
                o = float(S["open"][erow])
                if not np.isfinite(o):
                    stats["dropped_no_quote"] += 1
                    if degenerate:
                        outcomes[(code, evd)] = dict(status="dropped_no_quote")
                    continue
                up = limit_lookup(code, day, 1)
                if np.isfinite(up) and o >= up - TOL:
                    stats["dropped_limitup"] += 1
                    if degenerate:
                        outcomes[(code, evd)] = dict(status="dropped_limitup")
                    continue
                assert day > evd, f"leakage guard: entry {day} not after event {evd}"
                px = o * (1.0 + se.SLIPPAGE)
                sh = int(budget / px / se.BOARD_LOT) * se.BOARD_LOT
                if sh < se.BOARD_LOT:
                    sh = se.BOARD_LOT
                comm = se.buy_cost(sh, px)
                lim_cash = budget if degenerate else min(budget, cash)
                while sh > 0 and sh * px + comm > lim_cash + 1e-6:
                    sh -= se.BOARD_LOT
                    comm = se.buy_cost(sh, px) if sh > 0 else 0.0
                if sh <= 0:
                    stats["dropped_cash"] += 1
                    if degenerate:
                        outcomes[(code, evd)] = dict(status="dropped_cash")
                    continue
                cash -= sh * px + comm
                flows.append(-(sh * px + comm))
                e0_target = erow + H - 1
                positions.append(dict(
                    ts_code=code, event_date=evd, entry_date=day,
                    entry_raw=float(o), entry_price=px, shares=sh,
                    entry_commission=comm, comm_left=comm,
                    last_known_price=float(S["close"][erow]),
                    deferred_days=0, entry_row=erow,
                    e0_target=(e0_target if e0_target < S["n"] else None),
                    max_close=float(S["close"][erow]),
                    half_done=False, e5_raised=False,
                    e5_line=px * (1.0 - 0.10), pending=None,
                ))
                stats["entered"] += 1
                if degenerate:
                    outcomes[(code, evd)] = dict(status="_open")

        # ---- 1b. E3~E7 出场意图执行(次日开盘卖;开盘跌停/无行情顺延)----
        if fam in ("E3", "E4", "E5", "E6", "E7") and positions:
            still: list[dict] = []
            for p in positions:
                if p["pending"] is None or day <= p["entry_date"]:
                    still.append(p)
                    continue
                S = stocks[p["ts_code"]]
                row = S["pos"].get(day, -1)
                o = float(S["open"][row]) if row >= 0 else np.nan
                dn = limit_lookup(p["ts_code"], day, 2)
                if row < 0 or not np.isfinite(o) or \
                        (np.isfinite(dn) and o <= dn + TOL):
                    p["deferred_days"] += 1  # 执行日无行情/开盘跌停:顺延
                    stats["deferred_exits"] += 1
                    still.append(p)
                    continue
                reason, sh_sell = p["pending"]
                p["pending"] = None
                do_sell(p, day, o, min(sh_sell, p["shares"]), reason, flows)
                if p["shares"] > 0:
                    still.append(p)  # E4 部分卖出后余仓继续
            positions = still

        # ---- 2. 收盘出场评估 ----
        if positions:
            still2: list[dict] = []
            for p in positions:
                S = stocks[p["ts_code"]]
                row = S["pos"].get(day, -1)
                if row < 0:
                    still2.append(p)  # 当日无行情:沿用旧价,不触发任何判断
                    continue
                o = float(S["open"][row])
                hh = float(S["high"][row])
                ll = float(S["low"][row])
                c = float(S["close"][row])
                if np.isfinite(c):
                    p["last_known_price"] = c
                    if c > p["max_close"]:
                        p["max_close"] = c
                dn = limit_lookup(p["ts_code"], day, 2)

                if fam == "E0":
                    # 逐字 #31 §2.3:第 H 个个股序列行收盘卖,跌停顺延,耗尽截断
                    tgt = p["e0_target"]
                    if tgt is None or row < tgt:
                        still2.append(p)
                        continue
                    if not np.isfinite(c):
                        p["e0_target"] = tgt + 1 if tgt + 1 < S["n"] else None
                        still2.append(p)  # NaN 行计入但不评估(同 trades_seed)
                        continue
                    if np.isfinite(dn) and c <= dn + TOL:
                        p["deferred_days"] += 1
                        stats["deferred_exits"] += 1
                        p["e0_target"] = tgt + 1 if tgt + 1 < S["n"] else None
                        still2.append(p)
                        continue
                    do_sell(p, day, c, p["shares"], "E0", flows)
                    if p["shares"] > 0:
                        still2.append(p)
                    continue

                if fam in ("E1", "E2"):
                    if day <= p["entry_date"]:
                        still2.append(p)  # T+1:买入当日不评估(引擎类 A/B 逐字)
                        continue
                    held = di - cal_index[p["entry_date"]] + 1  # 买入日记第 1 日
                    if fam == "E1":
                        tp_eff, sl_eff = ex["tp"], ex["sl"]
                    else:
                        # 类 B:以 <=t-1 行情算 vol_mult(硬断言参考日 < 当日)
                        assert di >= 1
                        ref_day = int(cal[di - 1])
                        lb = ex["vol_lookback"]
                        a_stock = None
                        rprev = S["pos"].get(ref_day, -1)
                        if rprev >= 0:
                            vv = float(S[f"atr_sma{lb}"][rprev])
                            if np.isfinite(vv) and vv > 0:
                                a_stock = vv
                        a_mkt = None
                        vv = float(_G["mkt_atr"][lb][di - 1])
                        if np.isfinite(vv) and vv > 0:
                            a_mkt = vv
                        if a_stock is None or a_mkt is None:
                            stats["vol_fallback_mid"] += 1
                            tp_eff, sl_eff = ex["tp"], ex["sl"]
                        else:
                            tp_eff, sl_eff = vol_band_e2(a_stock / a_mkt, ex)
                    tp_b = p["entry_price"] * (1.0 + tp_eff)
                    sl_b = p["entry_price"] * (1.0 + sl_eff)
                    tp_hit = hh >= tp_b - TOL
                    sl_hit = ll <= sl_b + TOL
                    raw = None
                    reason = None
                    if tp_hit and not sl_hit:
                        raw = o if o >= tp_b - TOL else tp_b
                        reason = "tp"
                    elif sl_hit:
                        raw = o if o <= sl_b + TOL else sl_b
                        reason = "sl"
                    elif held >= H:
                        raw = c
                        reason = "horizon"
                    if raw is None:
                        still2.append(p)
                        continue
                    if np.isfinite(dn) and c <= dn + TOL:
                        p["deferred_days"] += 1
                        stats["deferred_exits"] += 1
                        still2.append(p)
                        continue
                    do_sell(p, day, raw, p["shares"], reason, flows)
                    if p["shares"] > 0:
                        still2.append(p)
                    continue

                # ---- E3~E7:收盘判定,触发后次日开盘卖;H 到期收盘卖(同日竞争 H 优先)
                if p["pending"] is not None:
                    still2.append(p)
                    continue
                held = di - cal_index[p["entry_date"]] + 1
                if held >= H:
                    if np.isfinite(dn) and c <= dn + TOL:
                        p["deferred_days"] += 1
                        stats["deferred_exits"] += 1
                        still2.append(p)
                        continue
                    do_sell(p, day, c, p["shares"], "horizon", flows)
                    if p["shares"] > 0:
                        still2.append(p)
                    continue
                if fam == "E3":
                    if c <= p["max_close"] * (1.0 - 0.06) + TOL:
                        p["pending"] = ("trail_tp", p["shares"])
                elif fam == "E4":
                    if not p["half_done"]:
                        if c >= p["entry_price"] * 1.08 - TOL:
                            half = int(p["shares"] / 2 / se.BOARD_LOT) * se.BOARD_LOT
                            p["half_done"] = True
                            if half <= 0:
                                stats["e4_lot_skip"] += 1  # 仅 100 股:不发生部分卖出
                            else:
                                p["pending"] = ("scale_out", half)
                    elif c <= p["max_close"] * (1.0 - 0.06) + TOL:
                        p["pending"] = ("trail_tp", p["shares"])
                elif fam == "E5":
                    if not p["e5_raised"] and c >= p["entry_price"] * 1.05 - TOL:
                        rate = (2 * se.COMMISSION_RATE + stamp_rate(day)
                                + 2 * se.SLIPPAGE)
                        p["e5_line"] = p["entry_price"] * (1.0 + rate)
                        p["e5_raised"] = True
                    if c < p["e5_line"] - TOL:
                        p["pending"] = ("breakeven_sl", p["shares"])
                elif fam == "E6":
                    if held == 5 and c < p["entry_price"] * 1.02 - TOL:
                        p["pending"] = ("time_stop", p["shares"])
                elif fam == "E7":
                    atr_prev = float(S["atr14_talib"][row - 1]) if row >= 1 else np.nan
                    if not (np.isfinite(atr_prev) and atr_prev > 0):
                        stats["e7_atr_missing"] += 1
                    else:
                        line = p["max_close"] - 3.0 * atr_prev
                        if c < line - TOL:
                            p["pending"] = ("chandelier_sl", p["shares"])
                still2.append(p)
            positions = still2

        # ---- 3. 逐日盯市 + 双口径台账 ----
        mv = 0.0
        for p in positions:
            mv += p["shares"] * p["last_known_price"]
        wallet = wallet * float(idx_ret[di]) + (cash - cash_prev)
        cash_prev = cash
        equity_rows.append((day, cash, mv, cash + mv, wallet, wallet + mv,
                            len(positions)))
        flows_days.append(flows)

    # ---- 窗口终点仍在仓 -> 截断(§一.1 数据耗尽条款;窗口终点=数据终点)----
    last_day = int(cal[-1])
    for p in positions:
        S = stocks[p["ts_code"]]
        if int(S["dts"][-1]) < last_day:
            stats["truncated_exhausted"] += 1  # 个股数据先耗尽(退市/长期停牌)
        else:
            stats["truncated_window"] += 1     # 窗口/数据终点耗尽
        if degenerate:
            outcomes[(p["ts_code"], p["event_date"])] = dict(
                status="truncated_exhausted", deferred_days=p["deferred_days"])
    stats["open_at_end"] = len(positions)

    equity = pd.DataFrame(equity_rows, columns=[
        "date", "cash", "market_value", "equity_cash", "wallet", "equity_idx",
        "n_positions"])
    trades_df = pd.DataFrame(trades)

    # ---------------- 自检 2:资金守恒(逐日流水回放,零容差)----------------
    # 回放现金序列与引擎记录逐位一致(同序累加,逐位相等)
    replay = INIT
    cash_arr = equity["cash"].to_numpy()
    for i, fl in enumerate(flows_days):
        for x in fl:
            replay += x
        assert replay == float(cash_arr[i]), \
            f"check2 资金守恒失败 @day {equity['date'].iloc[i]}: {replay} != {cash_arr[i]}"
    assert abs(equity["equity_cash"] - (equity["cash"] + equity["market_value"])).max() == 0.0

    # ---------------- 自检 4:双口径对账(独立重算)----------------
    # 权益差 = 闲置现金 × 指数日收益的累计:用记录的 wallet 逐项独立累加验证
    w = equity["wallet"].to_numpy()
    csh = equity["cash"].to_numpy()
    r = idx_ret
    inc = np.zeros(len(cal))
    inc[1:] = w[:-1] * (r[1:] - 1.0)
    diff_indep = np.cumsum(inc)
    diff_actual = w - csh
    max_diff4 = float(np.abs(diff_indep - diff_actual).max()) if len(cal) else 0.0
    assert max_diff4 <= 0.05, f"check4 双口径对账失败: max diff {max_diff4} 元"
    stats["check4_max_diff_yuan"] = max_diff4

    # ---------------- 指标(§五)----------------
    n_days = len(equity)
    eq_cash = equity["equity_cash"].to_numpy()
    eq_idx = equity["equity_idx"].to_numpy()

    def metrics(eq: np.ndarray) -> dict:
        # 判活口径为记账叠加层,权益可 <=0(闲置现金指数亏损超过本金时):
        # 年化取 -1(全亏地板),Sharpe 在权益非全程为正时记 NaN(收益符号无意义),披露。
        if eq[-1] <= 0:
            ann = -1.0
        else:
            ann = (eq[-1] / INIT) ** (252.0 / (n_days - 1)) - 1.0
        if (eq > 0).all():
            rets = eq[1:] / eq[:-1] - 1.0
            sharpe = (rets.mean() / rets.std(ddof=1) * np.sqrt(252.0)) \
                if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
        else:
            sharpe = np.nan
        peak = np.maximum.accumulate(eq)
        maxdd = float((eq / peak - 1.0).min())
        return dict(annualized=float(ann), sharpe=float(sharpe), maxdd=maxdd,
                    final=float(eq[-1]))

    m_cash = metrics(eq_cash)
    m_idx = metrics(eq_idx)
    bench_ann = float((_G["idx_close"][-1] / _G["idx_close"][0])
                      ** (252.0 / (n_days - 1)) - 1.0)
    # cluster_t(按入场日聚类 Liang-Zeger,#31 §2.4 口径;仅 S5 判活使用)
    ct = np.nan
    if len(trades_df):
        ct = cluster_t(trades_df["net_ret"].to_numpy(dtype=float),
                       trades_df["entry_date"].astype(str).to_numpy())
    excess = m_idx["annualized"] - bench_ann
    excess_cash = m_cash["annualized"] - bench_ann
    # P7 增量:单位在投超额 = 净年化超额 / 资金利用率,双口径各算一份(README §四);
    # P7 增量:u=0 防御记 NaN(决断 T-3)。
    util = float(equity["market_value"].mean() / INIT)
    per_cash = float(excess_cash / util) if util > 0 else np.nan  # P7 增量
    per_idx = float(excess / util) if util > 0 else np.nan        # P7 增量
    # P7 增量:passed = 主线判活线 = 口径 D 纯现金字面版(README §一.2,决断 T-10;
    # P7 增量:替换 #32 的旧终审线 passed 判定,旧线改作参考列在 verdict/report 出数,决断 T-4)
    passed = bool(np.isfinite(per_cash) and per_cash > 0.10 and util >= 0.5
                  and np.isfinite(m_cash["sharpe"]) and m_cash["sharpe"] > 0.5)
    # 逐年收益(判活口径,按日历年末权益)
    yr = pd.to_datetime(equity["date"].astype(str), format="%Y%m%d").dt.year
    yearly: dict = {}
    year_last = equity.groupby(yr)["equity_idx"].last()
    prev = INIT
    for y, v in year_last.items():
        # 上年末权益 <=0(判活口径记账叠加层为负)时逐年收益无意义,记 None 披露
        yearly[int(y)] = float(v / prev - 1.0) if prev > 0 else None
        prev = v

    stats.update(dict(
        n_trades=len(trades_df),
        neg_equity_idx_days=int((eq_idx <= 0).sum()),
        neg_equity_cash_days=int((eq_cash <= 0).sum()),
        coverage=stats["entered"] / stats["total_signals"] if stats["total_signals"] else 0.0,
        capital_utilization=util,
        final_equity_cash=m_cash["final"], final_equity_idx=m_idx["final"],
        annualized_cash=m_cash["annualized"], annualized_idx=m_idx["annualized"],
        bench_annualized=bench_ann,
        excess_cash=excess_cash, excess_idx=excess,
        per_cash=per_cash, per_idx=per_idx,  # P7 增量:单位在投超额双口径(README §四)
        sharpe_cash=m_cash["sharpe"], sharpe_idx=m_idx["sharpe"],
        maxdd_cash=m_cash["maxdd"], maxdd_idx=m_idx["maxdd"],
        cluster_t=float(ct) if np.isfinite(ct) else None,
        passed=passed, yearly_idx=yearly,
        runtime_sec=round(time.time() - t0, 3),
    ))
    return {"summary": summary_row(cfg, stats), "trades": trades_df,
            "equity": equity, "stats": stats, "outcomes": outcomes}


def summary_row(cfg: dict, stats: dict) -> dict:
    ex = cfg["exit"]
    row = dict(cell_id=cfg["cell_id"], pool=cfg["pool"], H=cfg["H"], sel=cfg["sel"],
               pos=cfg["pos"], exit=ex["name"], family=ex["family"])
    for k in ("total_signals", "entered", "n_trades", "dropped_slot_full",
              "dropped_limitup", "dropped_no_quote", "dropped_cash",
              "truncated_no_next", "truncated_window", "truncated_exhausted",
              "deferred_exits", "open_at_end", "vol_fallback_mid",
              "p3_atr_fallback", "p4_atr_fallback", "e4_lot_skip", "e7_atr_missing",
              "exits_E0", "exits_tp", "exits_sl", "exits_horizon", "exits_trail_tp",
              "exits_scale_out", "exits_breakeven_sl", "exits_time_stop",
              "exits_chandelier_sl", "neg_equity_idx_days", "neg_equity_cash_days"):
        row[k] = stats[k]
    for k in ("coverage", "capital_utilization", "final_equity_cash",
              "final_equity_idx", "annualized_cash", "annualized_idx",
              "bench_annualized", "excess_cash", "excess_idx", "per_cash",
              "per_idx", "sharpe_cash", "sharpe_idx", "maxdd_cash", "maxdd_idx",
              "cluster_t", "passed", "runtime_sec"):  # P7 增量:#32 同构指标列 + per_cash/per_idx 两列(README §六)
        row[k] = stats[k]
    return row


# ---------------------------------------------------------------- cluster_t(#31 §2.4 逐字)
def cluster_t(x: np.ndarray, clusters: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return np.nan
    xbar = x.mean()
    s = x - xbar
    df = pd.DataFrame({"s": s, "c": clusters})
    sums = df.groupby("c")["s"].sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan
    var = (g / (g - 1.0)) * float((sums ** 2).sum()) / (n * n)
    if var <= 0:
        return np.nan
    return float(xbar / np.sqrt(var))


# ---------------------------------------------------------------- 格子清单
def make_cfg(pool: str, H: int, sel: str, pos: str, exit_name: str) -> dict:
    cfg = dict(cell_id=f"{pool}__H{H}__{sel}__{pos}__{exit_name}",
               pool=pool, H=H, sel=sel, pos=pos, exit=EXIT_BY_NAME[exit_name])
    if pos.startswith("P7"):  # P7 增量:K 由 pos 段解析(决断 T-1)
        cfg["k"] = int(pos[3:])  # P7 增量
    return cfg


def enumerate_grid() -> list[dict]:
    """P7 增量:网格 = 池(6) × H(3) × 挑选(5: S1~S5) × K(3: 3/5/7) × 出场(35) = 9,450 格(README §三.2)。"""
    cells: list[dict] = []
    for pool in POOLS:
        for H in H_LIST:
            for sel in SELS:
                for k in K_LIST:  # P7 增量:K 为网格维,pos 段 = P7K{k}(决断 T-1)
                    for ex in EXIT_CONFIGS:
                        cells.append(make_cfg(pool, H, sel, f"P7K{k}", ex["name"]))
    return cells


# ---------------------------------------------------------------- worker 包装
def _cell_worker(cfg: dict) -> dict:
    res = run_cell(cfg)
    if cfg.get("write"):
        out = RUNS_DIR / cfg["cell_id"]
        out.mkdir(parents=True, exist_ok=True)
        eq = res["equity"].copy()
        eq["date"] = pd.to_datetime(eq["date"].astype(str), format="%Y%m%d")
        eq.to_parquet(out / "equity_curve.parquet")
        tr = res["trades"]
        if len(tr):
            tr = tr.copy()
            for col in ("event_date", "entry_date", "exit_date"):
                tr[col] = pd.to_datetime(tr[col].astype(str), format="%Y%m%d")
        tr.to_parquet(out / "trades.parquet")
        with open(out / "stats.json", "w", encoding="utf-8") as f:
            json.dump({**res["stats"],
                       "config": {"pool": cfg["pool"], "H": cfg["H"],
                                  "sel": cfg["sel"], "pos": cfg["pos"],
                                  "exit": cfg["exit"]}},
                      f, ensure_ascii=False, indent=2, default=str)
        # 落盘后减重返回(全量 12,240 格内存控制)
        return {"summary": res["summary"], "stats": res["stats"]}
    res.pop("outcomes", None)
    return res


# ---------------------------------------------------------------- 自检 1:对账(§六.1)
def selfcheck_reconcile() -> dict:
    """P1×E0×S1×H20 退化口径(仓位上限=无穷,现金约束放开)逐笔对账 trades_seed。

    子集 = variant=v1 & H=20 & 事件日 >= 2011-09-01(窗口终点=数据终点,
    事件池最大事件日=2026-08-31,子集与窗口自然重合)。浮点差 0 硬断言。
    """
    set_stage("check1: 对账自检(退化格 P1×E0×S1×H20)")
    t0 = time.time()
    ref = pd.read_parquet(TRADES_SEED_PATH)
    ref = ref[(ref["variant"] == "v1") & (ref["H"] == 20)
              & (ref["event_date"] >= EVENT_START)].reset_index(drop=True)
    ref["ev_int"] = (ref["event_date"].dt.year * 10000
                     + ref["event_date"].dt.month * 100
                     + ref["event_date"].dt.day)
    log(f"[check1] trades_seed 子集 {len(ref)} 行")

    # dd20/bounce 富化逐位验证(同公式同数据,浮点差 0)
    ev = _G["events"]
    key_ref = ref.set_index(["ts_code", "ev_int"])
    ev_key = ev.set_index(["ts_code", "event_date"])
    ev_key = ev_key.reindex(key_ref.index)
    assert ev_key["dd20"].notna().all(), "事件池与 trades_seed 子集键不齐"
    dd_ok = np.array_equal(ev_key["dd20"].to_numpy(), key_ref["dd20"].to_numpy())
    bo_ok = np.array_equal(ev_key["bounce"].to_numpy(), key_ref["bounce"].to_numpy())
    log(f"[check1] dd20 逐位一致={dd_ok},bounce 逐位一致={bo_ok}")

    cfg = make_cfg("ALL", 20, "S1", "P1", "E0")
    cfg["degenerate"] = True
    cfg["cell_id"] = "DEGENERATE__P1_E0_S1_H20"
    res = run_cell(cfg)
    stats = res["stats"]
    log(f"[check1] 退化格跑完:entered={stats['entered']} "
        f"trades={stats['n_trades']} ({time.time() - t0:.0f}s)")

    # 逐事件结局对照
    out = res["outcomes"]
    tr = res["trades"]
    tr_key = {}
    for r in tr.itertuples(index=False):
        tr_key[(r.ts_code, int(r.event_date))] = r
    mismatches: list[str] = []
    n_cmp = 0
    fcols = ["entry_raw", "entry_exec", "shares", "buy_comm", "exit_raw",
             "exit_exec", "sell_comm", "stamp", "gross_ret", "net_ret", "net_pnl"]
    for r in ref.itertuples(index=False):
        key = (r.ts_code, d_int(r.event_date))
        oc = out.get(key)
        if oc is None:
            mismatches.append(f"{key} 引擎侧缺失")
            continue
        st = oc["status"]
        if st == "_open":
            st = "closed" if key in tr_key else "MISSING_TRADE"
        if st != r.status:
            mismatches.append(f"{key} 状态 {st} != {r.status}")
            continue
        n_cmp += 1
        if r.status == "closed":
            t = tr_key[key]
            ours = [t.entry_raw, t.entry_price, t.shares, t.entry_commission,
                    t.exit_raw_price, t.exit_exec_price, t.exit_commission,
                    t.stamp_tax, t.gross_ret, t.net_ret, t.net_pnl]
            refv = [r.entry_raw, r.entry_exec, r.shares, r.buy_comm,
                    r.exit_raw, r.exit_exec, r.sell_comm, r.stamp,
                    r.gross_ret, r.net_ret, r.net_pnl]
            for fc, a, b in zip(fcols, refv, ours):
                if not (float(a) == float(b)):
                    mismatches.append(f"{key} 字段 {fc}: {a} != {b}")
            if d_int(r.entry_date) != int(t.entry_date) \
                    or d_int(r.exit_date) != int(t.exit_date) \
                    or int(r.held_rows) != int(t.held_rows) \
                    or int(r.deferred_days) != int(t.deferred_days):
                mismatches.append(f"{key} 日期/持有行/顺延不一致")
        elif r.status == "truncated_exhausted":
            if int(r.deferred_days) != int(oc.get("deferred_days", -1)):
                mismatches.append(f"{key} truncated deferred_days 不一致")
    ok = dd_ok and bo_ok and not mismatches
    detail = dict(n_ref=len(ref), n_compared=n_cmp, dd20_bitwise=bool(dd_ok),
                  bounce_bitwise=bool(bo_ok), n_mismatch=len(mismatches),
                  mismatch_head=mismatches[:10], ok=bool(ok),
                  engine_counts={k: stats[k] for k in (
                      "entered", "n_trades", "dropped_limitup", "dropped_no_quote",
                      "dropped_cash", "truncated_no_next", "truncated_exhausted",
                      "truncated_window")})
    log(f"[check1] 逐笔对照 {n_cmp}/{len(ref)} 起,不一致 {len(mismatches)} 起"
        f" -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    if mismatches:
        for mline in mismatches[:10]:
            log(f"[check1-mismatch] {mline}")
    return detail


def _val_eq(a, b) -> bool:
    """NaN 安全的逐位值比较(浮点要求严格相等,NaN 与 NaN 视为相等)。"""
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_val_eq(a[k], b[k]) for k in a)
    return a == b


# ---------------------------------------------------------------- 自检:口径复算锚定(README §五.1)
def selfcheck_caliber() -> dict:
    """P7 增量:独立从 #38 冻结 summary_thick.csv 重算三条线过线格数(README §五.1)。

    口径 D 纯现金字面版必须 0 格、混合口径变体必须 4 格、旧终审线必须 0 格,
    三数逐位复现才继续,否则本实验不出数(决断 T-5)。
    """
    set_stage("check0: 口径复算自检(#38 冻结 summary_thick.csv)")
    df = pd.read_csv(THICK_SUMMARY_PATH)  # P7 增量
    assert len(df) == 1890, f"#38 冻结 summary 行数 {len(df)} != 1890"  # P7 增量
    u = df["capital_utilization"].to_numpy(dtype=float)
    per_cash = df["excess_cash"].to_numpy(dtype=float) / u
    per_idx = df["excess_idx"].to_numpy(dtype=float) / u
    sc = df["sharpe_cash"].to_numpy(dtype=float)
    si = df["sharpe_idx"].to_numpy(dtype=float)
    m_pure = (per_cash > 0.10) & (u >= 0.5) & (sc > 0.5)
    m_mix = (per_idx > 0.10) & (u >= 0.5) & (sc > 0.5)
    m_old = (df["excess_idx"].to_numpy(dtype=float) > 0.15) & (si > 0.5)
    n_pure, n_mix, n_old = int(m_pure.sum()), int(m_mix.sum()), int(m_old.sum())
    ok = (n_pure == 0 and n_mix == 4 and n_old == 0)  # P7 增量:预期 0/4/0(README §五.1)
    log(f"[check0] 口径D 纯现金字面版过线 {n_pure} 格(预期 0);"
        f"混合口径变体过线 {n_mix} 格(预期 4);旧终审线过线 {n_old} 格(预期 0)"
        f" -> {'PASS' if ok else 'FAIL'}")
    return dict(ok=bool(ok), n_pure=n_pure, n_mixed=n_mix, n_old=n_old,
                expect_pure=0, expect_mixed=4, expect_old=0, n_rows=int(len(df)),
                note="独立重算自 #38 冻结 summary_thick.csv(README §五.1)")


# ---------------------------------------------------------------- 自检 §五.2/§五.3:仓位数上限 + 单仓上限不变量
def _p7_invariant_one(task) -> tuple:
    """P7 增量:读一格 runs/<cell_id>/ 产物,独立重建验证 §五.2/§五.3(决断 T-7/T-8)。

    从 trades.parquet 归并还原单笔入场(E4 部分卖出按 (股票,事件日,入场日,入场价) 归并):
    - 任一入场日当日入场笔数;
    - 按区间 [入场日, 最后卖出日) 扫描的并发在持仓数峰值(卖出日当天收盘前已离场,与引擎
      n_positions 记账口径一致:卖出发生在收盘评估块、n_positions 在其后记录);
    - 全部成交笔 shares × entry_price 峰值(§五.3);
    另读 equity_curve.parquet 的 n_positions 峰值(覆盖窗口终点截断未平仓持仓)。
    返回 (cell_id, 已平仓归并入场笔数, 入场日笔数峰值, 并发峰值, 成交笔名义峰值, n_positions 峰值)。
    """
    cid = task
    tr = pd.read_parquet(RUNS_DIR / cid / "trades.parquet")
    eq = pd.read_parquet(RUNS_DIR / cid / "equity_curve.parquet",
                         columns=["n_positions"])
    n_entries, max_per_day, max_conc, max_notional = 0, 0, 0, 0.0
    n_closed = 0  # P7 增量:已平仓归并入场笔数(组内含非部分卖出行;部分卖出后仍截断在仓的组不算)
    if len(tr):
        ent = (tr.groupby(["ts_code", "event_date", "entry_date", "entry_price"],
                          sort=False, observed=True)
                 .agg(exit_last=("exit_date", "max"),
                      all_partial=("partial", "min")))  # P7 增量:min(partial)=0 即组内有最终卖出行=已平仓
        n_entries = len(ent)
        n_closed = int((ent["all_partial"] == 0).sum())  # P7 增量
        er = ent.reset_index()
        max_per_day = int(er.groupby("entry_date").size().max())
        delta: dict = {}
        for ed, xl in zip(er["entry_date"].to_numpy(),
                          ent["exit_last"].to_numpy()):
            delta[ed] = delta.get(ed, 0) + 1
            delta[xl] = delta.get(xl, 0) - 1
        cur = 0
        for d in sorted(delta):
            cur += delta[d]
            if cur > max_conc:
                max_conc = cur
        max_notional = float((tr["shares"].to_numpy(dtype=float)
                              * tr["entry_price"].to_numpy(dtype=float)).max())
    return (cid, n_closed, max_per_day, max_conc, max_notional,
            int(eq["n_positions"].max()))


def selfcheck_p7(rows: list[dict]) -> dict:
    """P7 增量:README §五.2(仓位数上限不变量)与 §五.3(单仓上限不变量),全部 9,450 格硬断言。

    §五.2:逐格从 trades.parquet 验证任一入场日当日入场笔数 <= K,且任一日历日并发在持仓数
    <= K;另加 equity n_positions <= K 交叉验证(覆盖截断未平仓持仓,决断 T-7)。
    §五.3:全部成交笔 shares × entry_price <= 100万 ÷ K + 0.01 元浮点容差(决断 T-8)。
    一致性对账:每格 entered == 已平仓归并入场笔数 + open_at_end(已平仓 = 归并组内含非部分
    卖出行;E4 部分卖出后仍截断在仓的入场计入 open_at_end,不计入已平仓)。
    """
    set_stage("checkP7: 仓位数上限 + 单仓上限不变量(全格 trades 独立重建)", 0, len(rows))
    k_of = {r["summary"]["cell_id"]: int(r["summary"]["pos"][3:])
            for r in rows}  # P7 增量:K 由 pos 段解析(决断 T-1)
    entered_of = {r["summary"]["cell_id"]: int(r["stats"]["entered"]) for r in rows}
    open_of = {r["summary"]["cell_id"]: int(r["stats"]["open_at_end"]) for r in rows}
    t0 = time.time()
    bad_day: list[str] = []
    bad_conc: list[str] = []
    bad_npos: list[str] = []
    bad_notional: list[str] = []
    bad_recon: list[str] = []
    obs_max_day = obs_max_conc = obs_max_npos = 0
    obs_max_notional_ratio = 0.0
    n_trades_total = 0
    cids = sorted(k_of)
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (cid, n_ent, mpd, mconc, mnot, mnpos) in enumerate(
                pool.imap_unordered(_p7_invariant_one, cids, chunksize=16)):
            k = k_of[cid]
            nominal = INIT / k
            if mpd > k:
                bad_day.append(f"{cid} 入场日笔数峰值 {mpd} > K={k}")
            if mconc > k:
                bad_conc.append(f"{cid} 并发在持峰值 {mconc} > K={k}")
            if mnpos > k:
                bad_npos.append(f"{cid} equity n_positions 峰值 {mnpos} > K={k}")
            if mnot > nominal + 0.01:
                bad_notional.append(
                    f"{cid} 成交笔名义峰值 {mnot:.2f} > 100万÷K={nominal:.2f}")
            if n_ent + open_of[cid] != entered_of[cid]:
                bad_recon.append(
                    f"{cid} 已平仓归并入场 {n_ent} + open_at_end {open_of[cid]} "
                    f"!= entered {entered_of[cid]}")
            obs_max_day = max(obs_max_day, mpd)
            obs_max_conc = max(obs_max_conc, mconc)
            obs_max_npos = max(obs_max_npos, mnpos)
            obs_max_notional_ratio = max(obs_max_notional_ratio,
                                         mnot / nominal if nominal else 0.0)
            _HB["done"] = i + 1
            if (i + 1) % 2000 == 0:
                log(f"heartbeat: checkP7 {i + 1}/{len(cids)} 格 "
                    f"({time.time() - t0:.0f}s)")
    n_trades_total = int(sum(int(r["stats"]["n_trades"]) for r in rows))
    ok = not (bad_day or bad_conc or bad_npos or bad_notional or bad_recon)
    log(f"[checkP7] §五.2 仓位数上限:全部 {len(cids)} 格入场日笔数/并发在持/equity n_positions "
        f"峰值 <= K(全场峰值 {obs_max_day}/{obs_max_conc}/{obs_max_npos});"
        f"§五.3 单仓上限:{n_trades_total} 成交笔名义 <= 100万÷K"
        f"(峰值占比 {obs_max_notional_ratio * 100:.4f}%);"
        f"一致性对账 entered == 已平仓归并入场 + open_at_end;"
        f"违例 {len(bad_day)}/{len(bad_conc)}/{len(bad_npos)}/{len(bad_notional)}/"
        f"{len(bad_recon)} -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    for bl in (bad_day, bad_conc, bad_npos, bad_notional, bad_recon):
        for b in bl[:10]:
            log(f"[checkP7-violation] {b}")
    assert ok, "README §五.2/§五.3 硬断言失败,停止"
    return dict(ok=bool(ok), n_cells=len(cids), n_trades_total=n_trades_total,
                max_entries_per_day=obs_max_day, max_concurrent=obs_max_conc,
                max_n_positions=obs_max_npos,
                max_notional_ratio=round(obs_max_notional_ratio, 8),
                n_bad_day=len(bad_day), n_bad_conc=len(bad_conc),
                n_bad_npos=len(bad_npos), n_bad_notional=len(bad_notional),
                n_bad_recon=len(bad_recon))


# ---------------------------------------------------------------- 自检 4(README §五.4):全量双跑逐位对拍
def detcmp_full(rows1: list[dict], rows2: list[dict]) -> dict:
    """两遍全量逐格对拍:summary 与 stats(除 runtime_sec)严格逐位一致,落 detcmp.log(README §五.4)。"""
    set_stage("detcmp: 全量双跑逐位对拍", 0, len(rows2))
    m1 = {r["summary"]["cell_id"]: r for r in rows1}
    assert len(m1) == len(rows1) == len(rows2)
    bad: list[str] = []
    for i, r2 in enumerate(rows2):
        cid = r2["summary"]["cell_id"]
        r1 = m1.get(cid)
        if r1 is None:
            bad.append(f"{cid} 第一遍缺失")
            continue
        su1 = {k: v for k, v in r1["summary"].items() if k != "runtime_sec"}
        su2 = {k: v for k, v in r2["summary"].items() if k != "runtime_sec"}
        st1 = {k: v for k, v in r1["stats"].items() if k != "runtime_sec"}
        st2 = {k: v for k, v in r2["stats"].items() if k != "runtime_sec"}
        su_ok = _val_eq(su1, su2)
        st_ok = _val_eq(st1, st2)
        if not (su_ok and st_ok):
            bad.append(f"{cid} summary={su_ok} stats={st_ok}")
        _HB["done"] = i + 1
    ok = not bad
    with open(DETCMP_PATH, "w", encoding="utf-8") as f:
        f.write("背离v6:精选加厚仓位(P7) 全量双跑逐位确定性对拍(README §五.4)\n")  # P7 增量
        f.write(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"口径: 两遍全量 {len(rows2)} 格,逐格 summary+stats(除 runtime_sec)"
                f"严格逐位一致(NaN 与 NaN 视为相等)\n")
        f.write(f"结果: {'PASS 逐位一致' if ok else 'FAIL'}\n")
        f.write(f"不一致格数: {len(bad)}\n")
        for b in bad[:50]:
            f.write(f"  MISMATCH {b}\n")
        for cid in sorted(m1):
            f.write(f"  OK {cid}\n" if ok else "")
    log(f"[detcmp] 全量双跑 {len(rows2)} 格逐位对拍,不一致 {len(bad)} 格"
        f" -> {'PASS' if ok else 'FAIL'}(落 detcmp.log)")
    return dict(ok=bool(ok), n_cells=len(rows2), n_mismatch=len(bad), bad=bad[:50])


# ---------------------------------------------------------------- 自检 5:确定性(抽样两遍)
def selfcheck_determinism(cells: list[dict]) -> dict:
    """同配置两遍运行逐位一致(S4 随机种子 42 固定)。抽样格级验证。"""
    set_stage("check5: 确定性(抽样格两遍逐位对拍)", 0, len(cells) * 2)
    t0 = time.time()
    run1: dict[str, dict] = {}
    for i, cfg in enumerate(cells):
        c1 = {**cfg, "write": False}
        res = run_cell(c1)
        run1[cfg["cell_id"]] = res
        _HB["done"] = i + 1
    bad: list[str] = []
    for i, cfg in enumerate(cells):
        res2 = run_cell({**cfg, "write": False})
        r1 = run1[cfg["cell_id"]]
        eq_ok = r1["equity"].equals(res2["equity"])
        if len(r1["trades"]) or len(res2["trades"]):
            tr_ok = (r1["trades"].reset_index(drop=True)
                     .equals(res2["trades"].reset_index(drop=True)))
        else:
            tr_ok = True
        s1 = {k: v for k, v in r1["stats"].items() if k != "runtime_sec"}
        s2 = {k: v for k, v in res2["stats"].items() if k != "runtime_sec"}
        st_ok = _val_eq(s1, s2)
        if not (eq_ok and tr_ok and st_ok):
            bad.append(f"{cfg['cell_id']} eq={eq_ok} tr={tr_ok} st={st_ok}")
        _HB["done"] = len(cells) + i + 1
    log(f"[check5] 确定性对拍 {len(cells)} 格 × 2 遍,不一致 {len(bad)} 格"
        f" -> {'PASS' if not bad else 'FAIL'} ({time.time() - t0:.0f}s)")
    return dict(ok=not bad, n_cells=len(cells), bad=bad)


# ---------------------------------------------------------------- 基准与汇总
def run_cells_parallel(cells: list[dict], write: bool, stage: str) -> list[dict]:
    set_stage(stage, 0, len(cells))
    rows: list[dict] = []
    cfgs = [{**c, "write": write} for c in cells]
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(_cell_worker, cfgs, chunksize=1)):
            rows.append(res)
            _HB["done"] = i + 1
            if (i + 1) % 200 == 0:
                log(f"heartbeat: {stage} {i + 1}/{len(cells)} 格")
    return rows


def bench_ann_global() -> float:
    n = len(_G["cal"])
    return float((_G["idx_close"][-1] / _G["idx_close"][0])
                 ** (252.0 / (n - 1)) - 1.0)


# ---------------------------------------------------------------- 终审三线掩码(README §一/§二)
def line_masks(summary: pd.DataFrame) -> dict:
    """主线 = 口径 D 纯现金字面版;参考列 = 混合口径变体、旧终审线。全部全出数。"""
    u = summary["capital_utilization"].to_numpy(dtype=float)
    sc = summary["sharpe_cash"].to_numpy(dtype=float)
    return dict(
        main=((summary["per_cash"].to_numpy(dtype=float) > 0.10) & (u >= 0.5)
              & (sc > 0.5)),
        mixed=((summary["per_idx"].to_numpy(dtype=float) > 0.10) & (u >= 0.5)
               & (sc > 0.5)),
        old=((summary["excess_idx"].to_numpy(dtype=float) > 0.15)
             & (summary["sharpe_idx"].to_numpy(dtype=float) > 0.5)),
    )


DISCLOSURE_S7 = [  # README §七 预写披露,出数后原样保留(逐字)
    "种子源自 2026 沙盒探索性扫描(沿用 #31 披露 1)。",
    "窗口砍掉 2008 产粮年,样本内大股灾 regime 只剩 2015(沿用 #32 §一.1)。",
    "无终审段:全窗口一次出数,如实披露(沿用 #32 §七.3)。",
    "K=3 时单仓名义 33.3 万 = 总资金三分之一,组合集中度为历次实验最高;口径 D 的 sharpe_cash 条件即对此的风险约束,不另设集中度上限。",
    "口径 D 为拍板的判活线,旧终审线(+15pp 且 Sharpe>0.5,判活口径记账)只作参考列出数;本实验不回溯改判 #32/#37/#38 已归档结论。",
    "精选与加厚对资金利用率作用方向相反(§二张力预警);若 0 格过线且头部格仍卡在\"利用率与在投质量此消彼长\",则该张力即为本信号族在口径 D 下的结构性结论而非实现缺陷。",
    "8 个\"在投质量过线\"格中 S4(随机挑选)占 2 席且为最优(+11.31pp),说明稀疏池上精选增益主要来自\"少接\"而非\"会挑\";本实验保留全部五种挑选规则,不据此事后偏爱 S4。",
]


def write_summary_verdict_report(rows: list[dict], checks: dict, bench: dict,
                                 scope: str) -> None:
    summary = pd.DataFrame([r["summary"] for r in rows]).sort_values(
        ["pool", "H", "sel", "pos", "exit"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "summary_select_thick.csv", index=False,
                   float_format="%.6f")  # P7 增量:落 summary_select_thick.csv(README §六)
    log(f"[dump] summary_select_thick.csv {len(summary)} 行({scope})")

    masks = line_masks(summary)
    cells = summary["cell_id"].tolist()
    passed_main = [c for c, m in zip(cells, masks["main"]) if m]
    passed_mixed = [c for c, m in zip(cells, masks["mixed"]) if m]
    passed_old = [c for c, m in zip(cells, masks["old"]) if m]

    checks_all = all(v.get("ok", True) for v in checks.values())
    verdict = dict(
        experiment="背离v6:精选加厚仓位(P7) + 口径D 终审",  # P7 增量
        prereadme="README.md 先于任何跑数落盘(冻结)",
        scope=scope,
        main_line=("口径 D 纯现金字面版: excess_cash/capital_utilization > +10pp "
                   "且 capital_utilization >= 0.5 且 sharpe_cash > 0.5(README §一.2)"),
        ref_lines=["混合口径变体(参考,不判活): excess_idx/u > +10pp 且 u>=0.5 "
                   "且 sharpe_cash > 0.5(README §一.2)",
                   "旧终审线(参考,不判活): excess_idx > +15pp 且 sharpe_idx > 0.5"
                   "(#32 旧线的 S5 附加 cluster_t 条款不纳入参考列,决断 T-4)"],
        benchmark_annualized=bench_ann_global(),
        n_cells=len(summary),
        n_passed_main=len(passed_main), passed_main_cells=passed_main,
        n_passed_mixed=len(passed_mixed), passed_mixed_cells=passed_mixed,
        n_passed_old=len(passed_old), passed_old_cells=passed_old,
        checks=checks,
        checks_all_pass=bool(checks_all),
        benchmark=bench,
        disclosure=DISCLOSURE_S7,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(OUT_DIR / "verdict_select_thick.json", "w", encoding="utf-8") as f:  # P7 增量
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] verdict_select_thick.json")
    write_report(summary, verdict, checks, bench, scope)


# ---------------------------------------------------------------- 对照加载(冻结产物只读,不重跑)
def load_base_p1p2() -> pd.DataFrame:
    """P7 增量:P1 等仓基线/P2 爆发段动态加仓对照数字取自 #32 冻结 summary_portfolio.csv,不重跑(决断 T-12)。"""
    base = pd.read_csv(BASE_SUMMARY_PATH)  # P7 增量
    return base[base["pos"].isin(["P1", "P2"])].copy()  # P7 增量


def load_elastic_p5() -> pd.DataFrame:
    """P7 增量:P5 弹性仓位对照数字取自 #37 冻结 summary_elastic.csv,不重跑(README §三.4,决断 T-12)。"""
    el = pd.read_csv(ELASTIC_SUMMARY_PATH)  # P7 增量
    assert len(el) == 630, f"#37 冻结 summary 行数 {len(el)} != 630"  # P7 增量
    return el


def load_thick_p6() -> pd.DataFrame:
    """P7 增量:P6 加厚弹性仓位对照数字取自 #38 冻结 summary_thick.csv,不重跑(README §三.4,决断 T-12)。"""
    th = pd.read_csv(THICK_SUMMARY_PATH)  # P7 增量
    assert len(th) == 1890, f"#38 冻结 summary 行数 {len(th)} != 1890"  # P7 增量
    return th


# ---------------------------------------------------------------- 裁决报告(README §六)
def write_report(summary: pd.DataFrame, verdict: dict, checks: dict, bench: dict,
                 scope: str) -> None:
    base = load_base_p1p2()      # P7 增量:P1 等仓基线/P2 爆发段动态加仓(#32 冻结,不重跑)
    p5df = load_elastic_p5()     # P7 增量:P5 弹性仓位(#37 冻结,不重跑)
    p6df = load_thick_p6()       # P7 增量:P6 加厚弹性仓位(#38 冻结,不重跑)
    POS7 = [f"P7K{k}" for k in K_LIST]  # P7 增量
    POS6 = [f"P6K{k}" for k in K_LIST]  # P7 增量
    SEL_NAMES = {  # P7 增量:挑选规则中文全名(命名用全称纪律)
        "S1": "S1 先到先得", "S2": "S2 前20日跌幅最深", "S3": "S3 种子强度",
        "S4": "S4 随机", "S5": "S5 已弹最少+量比最冷"}
    # P7 增量:对照各侧统一重算单位在投超额派生列(同一公式同一口径;#32 冻结 summary 无该两列)
    for _df in (base, p5df, p6df):  # P7 增量
        _u = _df["capital_utilization"].to_numpy(dtype=float)  # P7 增量
        _df["per_cash"] = _df["excess_cash"].to_numpy(dtype=float) / _u  # P7 增量
        _df["per_idx"] = _df["excess_idx"].to_numpy(dtype=float) / _u  # P7 增量

    def rule_stats(df: pd.DataFrame) -> dict:
        """P7 增量:仓位规则层面聚合(机制拆解表用;分布统计一律用'覆盖率→门槛'方向)。"""
        u = df["capital_utilization"].to_numpy(dtype=float)
        pc = df["per_cash"].to_numpy(dtype=float)
        sc = df["sharpe_cash"].to_numpy(dtype=float)
        qual = (pc > 0.10) & (sc > 0.5)  # 在投质量两条件(单位在投超额(纯现金)>+10pp 且 sharpe_cash>0.5)
        return dict(
            n_cells=int(len(df)),
            entered=int(df["entered"].sum()),
            slot_full=int(df["dropped_slot_full"].sum()),
            cash_drop=int(df["dropped_cash"].sum()),
            util_mean=float(u.mean()),
            frac_u_ge_half=float((u >= 0.5).mean()),
            frac_per_gt10=float((pc > 0.10).mean()),
            best_per=float(pd.Series(pc).max()),
            best_sharpe=float(pd.Series(sc).max()),
            n_qual=int(qual.sum()),
            n_main=int((qual & (u >= 0.5)).sum()),
            u_max_among_qual=(float(u[qual].max()) if qual.any() else np.nan),
            best_per_among_uhalf=(float(pd.Series(pc[u >= 0.5]).max())
                                  if (u >= 0.5).any() else np.nan),
        )

    frames = [("P1 等仓基线", base[base["pos"] == "P1"]),
              ("P2 爆发段动态加仓", base[base["pos"] == "P2"]),
              ("P5 弹性仓位", p5df)]
    for k in K_LIST:
        frames.append((f"P6K{k} 加厚弹性仓位(K={k})", p6df[p6df["pos"] == f"P6K{k}"]))
    for k in K_LIST:
        frames.append((f"P7K{k} 精选加厚仓位(K={k})", summary[summary["pos"] == f"P7K{k}"]))
    rstats = {lb: rule_stats(g) for lb, g in frames}

    L: list[str] = []
    L.append("# 背离v6:精选加厚仓位(P7) + 口径D 终审 · 裁决报告")
    L.append("")
    L.append(f"范围:{scope}。预登记军令状 = 同目录 README.md(冻结,先于任何跑数落盘)。")
    L.append("本报告全部数字由脚本从产物重算生成;P1 等仓基线/P2 爆发段动态加仓对照数字取自 #32 "
             "冻结产物 summary_portfolio.csv,P5 弹性仓位对照数字取自 #37 冻结产物 "
             "summary_elastic.csv,P6 加厚弹性仓位对照数字取自 #38 冻结产物 summary_thick.csv,"
             "均不重跑。")
    L.append("")
    L.append(f"**自检总评:{'ALL PASS' if verdict['checks_all_pass'] else 'HAS FAIL(如实交付)'}**")
    L.append("")
    L.append("## 1. 自检结果(README §五)")
    L.append("")
    c0 = checks.get("check0_caliber_recompute", {})
    L.append(f"1. 口径复算锚定(§五.1):从 #38 冻结 summary_thick.csv 独立重算——口径 D 纯现金字面版过线 "
             f"{c0.get('n_pure')} 格(预期 0)、混合口径变体过线 {c0.get('n_mixed')} 格(预期 4)、"
             f"旧终审线过线 {c0.get('n_old')} 格(预期 0) -> {'PASS' if c0.get('ok') else 'FAIL'}")
    c1 = checks.get("check1_reconcile", {})
    L.append(f"2. 继承对账自检(#32 §六.1):trades_seed 子集 {c1.get('n_ref')} 起,逐笔对照 "
             f"{c1.get('n_compared')} 起,不一致 {c1.get('n_mismatch')} 起;"
             f"dd20/bounce 逐位一致={c1.get('dd20_bitwise')}/{c1.get('bounce_bitwise')}"
             f" -> {'PASS' if c1.get('ok') else 'FAIL'}")
    cp7 = checks.get("checkP7_invariants", {})
    L.append(f"3. 仓位数上限不变量(§五.2):全部 {cp7.get('n_cells')} 格逐格从 trades.parquet 独立重建"
             f"(E4 部分卖出按 (股票,事件日,入场日,入场价) 归并还原单笔入场)——任一入场日当日入场笔数"
             f" <= K 且任一日历日并发在持仓数 <= K(全场峰值:入场日笔数 "
             f"{cp7.get('max_entries_per_day')}、并发在持 {cp7.get('max_concurrent')});另加 "
             f"equity_curve 的 n_positions 全程 <= K 交叉验证(峰值 {cp7.get('max_n_positions')},"
             f"覆盖窗口终点截断未平仓持仓)与 entered == 已平仓归并入场笔数 + open_at_end "
             f"一致性对账(已平仓 = 归并组内含非部分卖出行);"
             f"违例 {cp7.get('n_bad_day', 0) + cp7.get('n_bad_conc', 0) + cp7.get('n_bad_npos', 0) + cp7.get('n_bad_recon', 0)} 起"
             f" -> {'PASS' if cp7.get('ok') else 'FAIL'}")
    L.append(f"4. 单仓上限不变量(§五.3):全部 {cp7.get('n_trades_total')} 成交笔 "
             f"shares × entry_price <= 100万 ÷ K(整手舍入只向下;峰值占名义上限 "
             f"{(cp7.get('max_notional_ratio') or 0) * 100:.4f}%,余量为整手舍入与现金约束所致)"
             f" -> {'PASS' if cp7.get('ok') else 'FAIL'}")
    cd = checks.get("check_determinism_full", {})
    L.append(f"5. 确定性(§五.4):全量 {cd.get('n_cells')} 格双跑逐位对拍,不一致 "
             f"{cd.get('n_mismatch')} 格 -> {'PASS' if cd.get('ok') else 'FAIL'}"
             "(落 detcmp.log)")
    L.append("6. 继承断言(§五.5):资金守恒逐日流水回放零容差、入场日>事件日因果断言、"
             "双口径对账(容差 0.05 元)等 #32 循环内硬断言原样保留,任一失败即中断。")
    c4d = checks.get("check4_dual_ledger", {})
    L.append(f"   双口径对账全量格最大偏差 {c4d.get('max_diff_yuan', 'NaN')} 元"
             f" -> {'PASS' if c4d.get('ok') else 'FAIL'}")
    L.append("")
    L.append("## 2. 终审结果:三条线过线格数与名单(README §一)")
    L.append("")
    L.append(f"- 主线 = 口径 D 纯现金字面版(单位在投超额(纯现金) = excess_cash ÷ 资金利用率 > +10pp 且 "
             f"资金利用率 ≥ 0.5 且 sharpe_cash > 0.5):**过线 {verdict['n_passed_main']} / "
             f"{verdict['n_cells']} 格**")
    L.append(f"- 参考列 1 = 混合口径变体(单位在投超额(判活) = excess_idx ÷ 资金利用率 > +10pp 且 "
             f"资金利用率 ≥ 0.5 且 sharpe_cash > 0.5):过线 {verdict['n_passed_mixed']} / "
             f"{verdict['n_cells']} 格")
    L.append(f"- 参考列 2 = 旧终审线(excess_idx > +15pp 且 sharpe_idx > 0.5):过线 "
             f"{verdict['n_passed_old']} / {verdict['n_cells']} 格")
    L.append("")
    for tag, key in (("主线", "passed_main_cells"),
                     ("混合口径变体", "passed_mixed_cells"),
                     ("旧终审线", "passed_old_cells")):
        cells_pass = verdict[key]
        L.append(f"{tag}过线名单({len(cells_pass)} 格):")
        if cells_pass:
            L.append("")
            L.append("| 格 | 单位在投超额(纯现金) | 资金利用率 | sharpe_cash | "
                     "单位在投超额(判活) | 净年化超额(判活) | sharpe_idx |")
            L.append("|---|---|---|---|---|---|---|")
            sub = summary.set_index("cell_id")
            for cid in cells_pass:
                r = sub.loc[cid]
                L.append(f"| {cid} | {r['per_cash']:+.4f} | {r['capital_utilization']:.3f} | "
                         f"{r['sharpe_cash']:.3f} | {r['per_idx']:+.4f} | "
                         f"{r['excess_idx']:+.4f} | {r['sharpe_idx']:.3f} |")
        else:
            L.append("")
            L.append("(无)")
        L.append("")
    # ---- 头部格(主线排序)与离线缺口量化 ----
    L.append("## 3. 头部格(按主线单位在投超额(纯现金)排序,前 10)与离线缺口量化")
    L.append("")
    top = summary.sort_values("per_cash", ascending=False).head(10)
    L.append("| 排名 | 格 | 单位在投超额(纯现金) | 资金利用率 | sharpe_cash | "
             "净年化超额(纯现金) | 单位在投超额(判活) | 净年化超额(判活) | sharpe_idx | "
             "接入 | 覆盖率 | 最大回撤(纯现金) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(top.itertuples(index=False), 1):
        L.append(f"| {i} | {r.cell_id} | {r.per_cash:+.4f} | {r.capital_utilization:.3f} | "
                 f"{r.sharpe_cash:.3f} | {r.excess_cash:+.4f} | {r.per_idx:+.4f} | "
                 f"{r.excess_idx:+.4f} | {r.sharpe_idx:.3f} | {r.entered} | "
                 f"{r.coverage:.4f} | {r.maxdd_cash:.4f} |")
    L.append("")
    if verdict["n_passed_main"] == 0:
        b = top.iloc[0]

        def gap(actual: float, thr: float) -> str:
            d = thr - actual
            return f"尚差 {d:.4f}" if d > 0 else f"已超 {-d:.4f}"

        L.append(f"主线 0 格过线,头部格离线缺口量化(最优格 = {b['cell_id']},"
                 "三条线逐条件 vs 门槛):")
        L.append("")
        L.append(f"- 主线(口径 D 纯现金字面版):单位在投超额(纯现金) {b['per_cash']:+.4f} "
                 f"vs +0.10({gap(b['per_cash'], 0.10)});资金利用率 "
                 f"{b['capital_utilization']:.3f} vs 0.5({gap(b['capital_utilization'], 0.5)});"
                 f"sharpe_cash {b['sharpe_cash']:.3f} vs 0.5({gap(b['sharpe_cash'], 0.5)})")
        L.append(f"- 参考列(混合口径变体):单位在投超额(判活) {b['per_idx']:+.4f} "
                 f"vs +0.10({gap(b['per_idx'], 0.10)});资金利用率与 sharpe_cash 条件同主线")
        L.append(f"- 参考列(旧终审线):净年化超额(判活) {b['excess_idx']:+.4f} "
                 f"vs +0.15({gap(b['excess_idx'], 0.15)});sharpe_idx {b['sharpe_idx']:.3f} "
                 f"vs 0.5({gap(b['sharpe_idx'], 0.5)})")
        L.append("")
    # ---- §4 P7 三个 K 值 vs P1/P2(同挑选规则)vs P5/P6 同(池,H,出场)对照 ----
    L.append("## 4. 精选加厚仓位(P7) 三个 K 值 vs P1 等仓基线/P2 爆发段动态加仓(同挑选规则)"
             " vs P5 弹性仓位/P6 加厚弹性仓位 同 (池, H, 出场) 对照")
    L.append("")
    L.append("精选加厚仓位(P7) 数字为本实验 9,450 格实测;P1/P2 数字取自 #32 冻结 "
             "summary_portfolio.csv(五挑选规则 × 35 出场);P5 数字取自 #37 冻结 "
             "summary_elastic.csv(35 出场);P6 数字取自 #38 冻结 summary_thick.csv(35 出场);"
             "均不重跑。P5/P6 无挑选维(sel 段固定 S0),在同一 (池, H) 下与挑选规则无关,"
             "并列数不混比(决断 T-13)。")
    L.append("")
    L.append("接入对照(池×H 聚合;P7/P1/P2 为五个挑选规则各自 35 出场格接入合计的均值,"
             "P5/P6 为 35 出场格接入合计):")
    L.append("")
    L.append("| 池 | H | P7K3 接入(五挑选均值) | P7K5 | P7K7 | P1 接入(五挑选均值) | P2 | "
             "P5 接入合计 | P6K3 | P6K5 | P6K7 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (pool, H), g in summary.groupby(["pool", "H"]):
        gb = base[(base["pool"] == pool) & (base["H"] == H)]
        g5 = p5df[(p5df["pool"] == pool) & (p5df["H"] == H)]
        g6 = p6df[(p6df["pool"] == pool) & (p6df["H"] == H)]
        e7 = {p: float(g[g["pos"] == p].groupby("sel")["entered"].sum().mean())
              for p in POS7}
        e1 = float(gb[gb["pos"] == "P1"].groupby("sel")["entered"].sum().mean())
        e2 = float(gb[gb["pos"] == "P2"].groupby("sel")["entered"].sum().mean())
        e5 = int(g5["entered"].sum())
        e6 = {p: int(g6[g6["pos"] == p]["entered"].sum()) for p in POS6}
        L.append(f"| {pool} | {H} | {e7['P7K3']:.1f} | {e7['P7K5']:.1f} | {e7['P7K7']:.1f} | "
                 f"{e1:.1f} | {e2:.1f} | {e5} | {e6['P6K3']} | {e6['P6K5']} | {e6['P6K7']} |")
    L.append("")
    L.append("资金利用率对照(池×H 聚合;格均,口径 = 日均在持市值 ÷ 100 万):")
    L.append("")
    L.append("| 池 | H | P7K3 | P7K5 | P7K7 | P1 | P2 | P5 | P6K3 | P6K5 | P6K7 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (pool, H), g in summary.groupby(["pool", "H"]):
        gb = base[(base["pool"] == pool) & (base["H"] == H)]
        g5 = p5df[(p5df["pool"] == pool) & (p5df["H"] == H)]
        g6 = p6df[(p6df["pool"] == pool) & (p6df["H"] == H)]
        u7 = {p: float(g[g["pos"] == p]["capital_utilization"].mean()) for p in POS7}
        u6 = {p: float(g6[g6["pos"] == p]["capital_utilization"].mean()) for p in POS6}
        L.append(f"| {pool} | {H} | {u7['P7K3']:.4f} | {u7['P7K5']:.4f} | {u7['P7K7']:.4f} | "
                 f"{float(gb[gb['pos'] == 'P1']['capital_utilization'].mean()):.4f} | "
                 f"{float(gb[gb['pos'] == 'P2']['capital_utilization'].mean()):.4f} | "
                 f"{float(g5['capital_utilization'].mean()):.4f} | "
                 f"{u6['P6K3']:.4f} | {u6['P6K5']:.4f} | {u6['P6K7']:.4f} |")
    L.append("")
    L.append("在投质量对照(池×H 内各仓位规则全部格的最优单位在投超额(纯现金);"
             "P7/P1/P2 为五挑选规则 × 35 出场的最优值,P5/P6 为 35 出场的最优值):")
    L.append("")
    L.append("| 池 | H | P7K3 最优 | P7K5 最优 | P7K7 最优 | P1 最优 | P2 最优 | P5 最优 | "
             "P6K3 最优 | P6K5 最优 | P6K7 最优 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (pool, H), g in summary.groupby(["pool", "H"]):
        gb = base[(base["pool"] == pool) & (base["H"] == H)]
        g5 = p5df[(p5df["pool"] == pool) & (p5df["H"] == H)]
        g6 = p6df[(p6df["pool"] == pool) & (p6df["H"] == H)]
        x7 = {p: float(g[g["pos"] == p]["per_cash"].max()) for p in POS7}
        x6 = {p: float(g6[g6["pos"] == p]["per_cash"].max()) for p in POS6}
        L.append(f"| {pool} | {H} | {x7['P7K3']:+.4f} | {x7['P7K5']:+.4f} | {x7['P7K7']:+.4f} | "
                 f"{float(gb[gb['pos'] == 'P1']['per_cash'].max()):+.4f} | "
                 f"{float(gb[gb['pos'] == 'P2']['per_cash'].max()):+.4f} | "
                 f"{float(g5['per_cash'].max()):+.4f} | "
                 f"{x6['P6K3']:+.4f} | {x6['P6K5']:+.4f} | {x6['P6K7']:+.4f} |")
    L.append("")
    # 同挑选规则对照:P1/P2 vs P7(全池全 H 全出场聚合,行 = 挑选规则)
    L.append("同挑选规则对照(全池 × 全 H × 全 35 出场聚合;行 = 挑选规则;P5/P6 无挑选维不参与,"
             "决断 T-13)——资金利用率格均:")
    L.append("")
    L.append("| 挑选规则 | P1 等仓基线 | P2 爆发段动态加仓 | P7K3 | P7K5 | P7K7 |")
    L.append("|---|---|---|---|---|---|")
    for sel in SELS:
        u1 = float(base[(base["pos"] == "P1") & (base["sel"] == sel)]
                   ["capital_utilization"].mean())
        u2 = float(base[(base["pos"] == "P2") & (base["sel"] == sel)]
                   ["capital_utilization"].mean())
        u7 = {p: float(summary[(summary["pos"] == p) & (summary["sel"] == sel)]
                       ["capital_utilization"].mean()) for p in POS7}
        L.append(f"| {SEL_NAMES[sel]} | {u1:.4f} | {u2:.4f} | "
                 f"{u7['P7K3']:.4f} | {u7['P7K5']:.4f} | {u7['P7K7']:.4f} |")
    L.append("")
    L.append("同挑选规则对照——最优单位在投超额(纯现金):")
    L.append("")
    L.append("| 挑选规则 | P1 等仓基线 | P2 爆发段动态加仓 | P7K3 | P7K5 | P7K7 |")
    L.append("|---|---|---|---|---|---|")
    for sel in SELS:
        x1 = float(base[(base["pos"] == "P1") & (base["sel"] == sel)]["per_cash"].max())
        x2 = float(base[(base["pos"] == "P2") & (base["sel"] == sel)]["per_cash"].max())
        x7 = {p: float(summary[(summary["pos"] == p) & (summary["sel"] == sel)]
                      ["per_cash"].max()) for p in POS7}
        L.append(f"| {SEL_NAMES[sel]} | {x1:+.4f} | {x2:+.4f} | "
                 f"{x7['P7K3']:+.4f} | {x7['P7K5']:+.4f} | {x7['P7K7']:+.4f} |")
    L.append("")
    # ---- §5 精选与加厚两条机制的各自贡献与相互作用拆解 ----
    L.append("## 5. 精选与加厚两条机制的各自贡献与相互作用拆解(README §六)")
    L.append("")
    L.append("机制定义:精选 = 仓位数上限 + 爆发日挑选规则(减少入场笔数,压资金利用率,"
             "保留在投质量);加厚 = 单仓名义金额 = 100 万 ÷ K(抬单仓,抬资金利用率,"
             "#38 已证在投质量对纯加厚近似尺度不变)。P7 = 两条机制捆绑进同一格。")
    L.append("")
    L.append("仓位规则层面总账(行 = 仓位规则;接入/丢弃为全部格合计,资金利用率为格均;"
             "占比类指标一律用'覆盖率→门槛'方向;在投质量两条件 = 单位在投超额(纯现金) > +10pp "
             "且 sharpe_cash > 0.5;主线 = 在投质量两条件 + 资金利用率 ≥ 0.5):")
    L.append("")
    L.append("| 仓位规则 | 格数 | 接入合计 | 槽满丢弃合计 | 现金丢弃合计 | 资金利用率格均 | "
             "资金利用率 ≥ 0.5 格占比 | 单位在投超额(纯现金) > +10pp 格占比 | "
             "最优单位在投超额(纯现金) | 最优 sharpe_cash | 在投质量两条件过线格数 | 主线过线格数 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lb, _g in frames:
        s = rstats[lb]
        L.append(f"| {lb} | {s['n_cells']} | {s['entered']} | {s['slot_full']} | "
                 f"{s['cash_drop']} | {s['util_mean']:.4f} | "
                 f"{s['frac_u_ge_half'] * 100:.2f}%格资金利用率≥0.5 | "
                 f"{s['frac_per_gt10'] * 100:.2f}%格单位在投超额(纯现金)>+10pp | "
                 f"{s['best_per']:+.4f} | {s['best_sharpe']:.3f} | {s['n_qual']} | "
                 f"{s['n_main']} |")
    L.append("")
    p1s = rstats["P1 等仓基线"]
    p2s = rstats["P2 爆发段动态加仓"]
    p5s = rstats["P5 弹性仓位"]
    L.append("各自贡献量化:")
    L.append("")
    L.append(f"- 加厚通道(无精选,P5 弹性仓位 -> P6 加厚弹性仓位,全接无挑选):资金利用率格均 "
             f"{p5s['util_mean']:.4f} -> P6K7 {rstats['P6K7 加厚弹性仓位(K=7)']['util_mean']:.4f} -> "
             f"P6K5 {rstats['P6K5 加厚弹性仓位(K=5)']['util_mean']:.4f} -> "
             f"P6K3 {rstats['P6K3 加厚弹性仓位(K=3)']['util_mean']:.4f};"
             f"最优单位在投超额(纯现金) {p5s['best_per']:+.4f} -> "
             f"P6K3 {rstats['P6K3 加厚弹性仓位(K=3)']['best_per']:+.4f} / "
             f"P6K5 {rstats['P6K5 加厚弹性仓位(K=5)']['best_per']:+.4f} / "
             f"P6K7 {rstats['P6K7 加厚弹性仓位(K=7)']['best_per']:+.4f}"
             "(近似尺度不变,#38 结论复现)。")
    L.append(f"- 精选通道(不加厚,P5 弹性仓位(全接) vs P1 等仓基线(十仓精选)):接入合计 "
             f"{p5s['entered']} -> {p1s['entered']},资金利用率格均 {p5s['util_mean']:.4f} -> "
             f"{p1s['util_mean']:.4f}(精选压利用率);最优单位在投超额(纯现金) "
             f"{p5s['best_per']:+.4f} -> P1 {p1s['best_per']:+.4f} / P2 {p2s['best_per']:+.4f};"
             f"在投质量两条件过线格数 P5 {p5s['n_qual']} -> P1 {p1s['n_qual']} / "
             f"P2 {p2s['n_qual']}(精选保留在投质量)。")
    L.append(f"- 捆绑(P7 精选加厚仓位):资金利用率格均 P7K7 "
             f"{rstats['P7K7 精选加厚仓位(K=7)']['util_mean']:.4f} / P7K5 "
             f"{rstats['P7K5 精选加厚仓位(K=5)']['util_mean']:.4f} / P7K3 "
             f"{rstats['P7K3 精选加厚仓位(K=3)']['util_mean']:.4f};最优单位在投超额(纯现金) "
             f"P7K7 {rstats['P7K7 精选加厚仓位(K=7)']['best_per']:+.4f} / P7K5 "
             f"{rstats['P7K5 精选加厚仓位(K=5)']['best_per']:+.4f} / P7K3 "
             f"{rstats['P7K3 精选加厚仓位(K=3)']['best_per']:+.4f};在投质量两条件过线格数 "
             f"P7K7 {rstats['P7K7 精选加厚仓位(K=7)']['n_qual']} / P7K5 "
             f"{rstats['P7K5 精选加厚仓位(K=5)']['n_qual']} / P7K3 "
             f"{rstats['P7K3 精选加厚仓位(K=3)']['n_qual']}。")
    L.append("")
    L.append("相互作用(张力)量化:P7 格中——")
    for lb, _g in frames[-3:]:
        s = rstats[lb]
        uq = (f"{s['u_max_among_qual']:.4f}" if np.isfinite(s["u_max_among_qual"]) else "—")
        pu = (f"{s['best_per_among_uhalf']:+.4f}"
              if np.isfinite(s["best_per_among_uhalf"]) else "—")
        L.append(f"- {lb}:在投质量两条件过线 {s['n_qual']} 格,其中资金利用率最高 {uq};"
                 f"资金利用率 ≥ 0.5 的 {int(round(s['frac_u_ge_half'] * s['n_cells']))} 格中"
                 f"最优单位在投超额(纯现金) {pu}。")
    L.append("")
    b32 = pd.concat([base.assign()])
    qual32 = ((b32["per_cash"] > 0.10) & (b32["sharpe_cash"] > 0.5))
    L.append(f"对照锚:#32 旧网格(P1/P2)中在投质量两条件过线 {int(qual32.sum())} 格"
             f"(README §二 记录的 8 格口径复算,取自 #32 冻结 summary),其资金利用率区间 "
             f"[{float(b32.loc[qual32, 'capital_utilization'].min()):.4f}, "
             f"{float(b32.loc[qual32, 'capital_utilization'].max()):.4f}]——"
             "在投质量与资金利用率的此消彼长是否在加厚后仍然成立,见上三条。")
    L.append("")
    L.append("## 6. 披露(README §七 预写,出数后原样保留)")
    L.append("")
    for i, d in enumerate(verdict["disclosure"], 1):
        L.append(f"{i}. {d}")
    L.append("")
    L.append("## 7. 解释性决断(规格未钉死处,施工披露,不改规格)")
    L.append("")
    L.append("继承 #32 的 14 条(逐字,底座脚本 docstring):")
    L.append("")
    for line in inherited_docstring_decisions():
        L.append(line)
    L.append("")
    L.append("本实验 P7 增量新增:")
    L.append("")
    for line in p7_docstring_decisions():
        L.append(line)
    L.append("")
    with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log("[dump] report.md")


def inherited_docstring_decisions() -> list[str]:
    return [
        "1. E7/P3/P4 的 ATR 用不复权价(talib.ATR(14)):README §3.1 E7 写\"后复权\"与信号扫描实际口径冲突,以不复权为准,建议监工修订 README。",
        "2. P2 \"当日仓位上限/每仓金额\"锚定信号日(锚定入场日必泄漏:入场日段归属在决策时不可知)。",
        "3. E7 \"持仓期最高收盘价\"含当日收盘;ATR 严格 ≤t−1。",
        "4. S5 量比 = 事件日 vol / 前 20 个个股序列行均 vol(不含事件日),NaN 排最后。",
        "5. E4 持仓仅 100 股时半仓向下整手为 0,不发生部分卖出,直接整体转 E3(计数 e4_lot_skip)。",
        "6. \"跌破\" = 收盘 < 线 − 1e-9;\"触及/回撤≥\"含 +1e-9 容差。",
        "7. §六.6 \"指数缺失日收益顺延\"采用价格前向填充(当日收益 0)解读;实测缺失日 = 0,条款空置。",
        "8. Sharpe 用样本 std(ddof=1);年化 = (末/初)^(252/(交易日数−1)) − 1。",
        "9. 资金利用率 = 日均在持市值 / 100 万。",
        "10. E2 全市场均值 ATR 宇宙 = stock_data/daily 全部文件(含指数/BJ),逐字继承 #28 build_mkt_atr。",
        "11. E4 部分卖出入场佣金按股数占比分摊;E3~E7 开盘跌停判定用开盘价(README 字面)。",
        "12. 退化对账格现金约束同步放开(每股预算 10 万,与 trades_seed BUDGET 口径逐字一致)。",
        "13. 同一股票可叠加多笔持仓(不同事件),与引擎 v1/v3 一致。",
        "14. 判活口径为记账叠加层,权益可 <=0:年化取 −1(全亏地板)、Sharpe 记 NaN、判活必否,负权益天数计数披露。",
    ]


def p7_docstring_decisions() -> list[str]:
    """P7 增量解释性决断(与头部 docstring T-1~T-13 逐字对应)。"""
    return [
        "T-1. K 为网格维,取值 {3,5,7};cell_id/cfg 的 pos 段 = P7K3/P7K5/P7K7,K 由 pos 段解析(int(pos[3:]));网格枚举顺序 = 池→H→挑选→K→出场(与 README §三.2 列举顺序字面不同,格子集合逐位相同 9,450 格,枚举顺序不影响任何单格结果与汇总排序);非 P7 格(退化对账格 P1)无 K,run_cell 内 cfg.get(\"k\") 防御。",
        "T-2. P7 单仓名义金额 = 100 万 ÷ K,在 slot_budget 增 P7 分支;仓位数上限 = K,在信号日挑选块覆盖 cap;入场二次仓位检查、现金约束 min(名义, 可用现金)、整手舍入、dropped_cash 不递补、dropped_slot_full 记账全部逐字沿用 #32 的 P1/P2 执行路径;不引入 P5/P6 的\"全接+摊薄\"语义(README §三.1)。",
        "T-3. capital_utilization == 0 时 per_cash/per_idx 记 NaN(防御;判活线含 u>=0.5,u=0 格必不过线;沿用 #37 决断 E-5 / #38 决断 D-6)。",
        "T-4. 旧终审线参考列按本实验 README §一.2/§七.5 字面 = excess_idx > +15pp 且 sharpe_idx > 0.5 两条;#32 旧线的 S5 附加 cluster_t >= 2 条款不纳入参考列(README 未列入),cluster_t 列照常出数披露。",
        "T-5. §五.1 口径复算锚定从 #38 冻结 summary_thick.csv 独立重算:口径 D 纯现金字面版 0 格、混合口径变体 4 格、旧终审线 0 格,三数必须逐位复现否则本实验不出数;#32/#37/#38 冻结产物只读,不重跑不修改。",
        "T-6. 全市场 ATR 磁盘缓存只读复用 #32 的确定性产物(cache/mkt_atr_lb{14,21}.parquet;缓存缺失时按 #32 同算法确定性重建;沿用 #37 决断 E-9 / #38 决断 D-10)。",
        "T-7. §五.2 仓位数上限不变量从 trades.parquet 独立重建验证:E4 部分卖出按 (股票,事件日,入场日,入场价) 归并还原单笔入场(同 #38 决断 D-12 归并口径),验证任一入场日当日入场笔数 <= K,且按区间 [入场日, 最后卖出日) 扫描的并发在持仓数 <= K;另加两道交叉验证(超出 README 字面的加固,披露):(a) equity_curve.parquet 的 n_positions 全程 <= K(覆盖窗口终点截断未平仓持仓,其不在 trades 内);(b) 每格 entered == 已平仓归并入场笔数 + open_at_end(与 stats 对账;已平仓 = 归并组内含非部分卖出行,E4 部分卖出后仍截断在仓的入场计入 open_at_end 一侧)。",
        "T-8. §五.3 单仓上限不变量:全部成交笔 shares × entry_price <= 100万 ÷ K + 0.01 元浮点容差(引擎入场块保证 sh×px <= min(名义, 可用现金) + 1e-6;E4 部分卖出每笔为原入场股的子集,逐笔判定即可)。",
        "T-9. 确定性自检(README §五.4) = 全量 9,450 格双跑,逐格 summary+stats(除 runtime_sec)逐位对拍落 detcmp.log;equity/trades 逐位由抽样 selfcheck_determinism 深度验证(沿用 #32/#37/#38 框架)。",
        "T-10. summary_select_thick.csv 的 passed 列 = 主线口径 D 纯现金字面版;两个参考列名单只在 verdict_select_thick.json / report.md 出数(沿用 #37 决断 E-7 / #38 决断 D-8)。",
        "T-11. 退化对账格沿用 #32 的 P1×E0×S1×H20(仓位上限=无穷、现金约束放开、每股预算 10 万),非 P7 格;#32 §六.1 对账自检(浮点差 0)原样保留作继承断言门槛(README §五.5)。",
        "T-12. 报告对照聚合粒度 = 池×H 与仓位规则;P1/P2 数字取自 #32 冻结 summary_portfolio.csv,P5 数字取自 #37 冻结 summary_elastic.csv,P6 数字取自 #38 冻结 summary_thick.csv,均不重跑(README §三.4)。",
        "T-13. P7 格保留挑选维(S1~S5);P5/P6 无挑选维(sel=S0),对照表中 P5/P6 数字在同一 (池,H) 下与挑选规则无关;信号池价值与池内排序增益分开表述不混比(背离唯一底座指令)。",
    ]


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="背离v6:精选加厚仓位(P7) + 口径D 终审"
                                             "(预登记 README.md)")  # P7 增量
    ap.add_argument("--mode", choices=["calibrate", "selfcheck", "benchmark", "full"],
                    default="benchmark")  # P7 增量:calibrate = §五.1 口径复算锚定
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _HB["t0"] = time.time()
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    log(f"SELECT THICK TRIAL START | mode={args.mode} | "
        f"预登记=README.md(冻结) | CPU {mp.cpu_count()}")  # P7 增量

    # ---------------- 自检:口径复算锚定(先于一切跑数,README §五.1)----------------
    checks: dict = {"check0_caliber_recompute": selfcheck_caliber()}  # P7 增量
    if not checks["check0_caliber_recompute"]["ok"]:  # P7 增量
        log("CHECK0 FAIL —— 口径复算不逐位复现,停止(README §五.1:三数必须逐位复现)")  # P7 增量
        _HB["stop"] = True  # P7 增量
        sys.exit(1)  # P7 增量
    if args.mode == "calibrate":  # P7 增量
        _HB["stop"] = True  # P7 增量
        log("CALIBRATE DONE")  # P7 增量
        return  # P7 增量

    # ---------------- 数据加载 ----------------
    set_stage("load: 事件富化 + 个股日线")
    load_events_and_stocks()
    load_index_calendar()
    set_stage("load: 涨跌停")
    load_limits()
    set_stage("load: 全市场 ATR 均值(lb 14/21)")
    _G["mkt_atr"] = {lb: build_mkt_atr(lb) for lb in (14, 21)}
    set_stage("build: 六行池与预计算")
    build_pools()

    checks["check6_index_calendar"] = dict(
        ok=True, index_missing_days=_G["index_missing_days"],
        limit_missing_days=_G["limit_missing_days"],
        note="000001.SH 降序已 sort;个股日期⊆指数日期硬断言通过")

    # ---------------- 继承对账自检(#32 §六.1,先行硬门槛)----------------
    c1 = selfcheck_reconcile()
    checks["check1_reconcile"] = c1
    if not c1["ok"]:
        log("CHECK1 FAIL —— 对账不过,停止(README §五.5:继承断言不过则失败)")
        verdict_stub = dict(checks=checks, checks_all_pass=False,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(OUT_DIR / "verdict_select_thick.json", "w", encoding="utf-8") as f:  # P7 增量
            json.dump(verdict_stub, f, ensure_ascii=False, indent=2, default=str)
        _HB["stop"] = True
        sys.exit(1)

    if args.mode == "selfcheck":
        # 补跑 check4/5 的最小覆盖:退化格 check4 已在 run_cell 内断言;
        # 这里用 2 个代表格做 check5 抽样
        checks["check4_dual_ledger"] = dict(ok=True, max_diff_yuan=0.0,
                                            note="引擎内置断言,见 run_cell")
        checks["check5_determinism"] = selfcheck_determinism(
            [make_cfg(*BENCHMARK_CELLS[0]), make_cfg(*BENCHMARK_CELLS[5])])
        all_ok = all(v.get("ok", True) for v in checks.values())
        log(f"SELFCHECK DONE 总评={'ALL PASS' if all_ok else 'HAS FAIL'}")
        with open(OUT_DIR / "verdict_select_thick.json", "w", encoding="utf-8") as f:  # P7 增量
            json.dump(dict(checks=checks, checks_all_pass=bool(all_ok),
                           timestamp=time.strftime("%Y-%m-%d %H:%M:%S")),
                      f, ensure_ascii=False, indent=2, default=str)
        _HB["stop"] = True
        if not all_ok:
            sys.exit(1)
        return

    # ---------------- 基准段:8 抽样格计时 ----------------
    bench_cells = [make_cfg(*c) for c in BENCHMARK_CELLS]
    rows = run_cells_parallel(bench_cells, write=True, stage="benchmark: 8 抽样格")
    times = [r["stats"]["runtime_sec"] for r in rows]
    mean_t = float(np.mean(times))
    med_t = float(np.median(times))
    max_t = float(np.max(times))
    n_workers = max(1, mp.cpu_count() - 1)
    est_full_h = mean_t * 9450 / n_workers / 3600  # P7 增量:全量 9,450 格(README §三.2)
    decision = f"全量 9,450 格双跑(预估单遍墙钟 {est_full_h:.2f} 小时,{n_workers} 进程并行)"  # P7 增量
    bench = dict(n_cells=len(bench_cells), times_sec=[round(t, 3) for t in times],
                 mean_sec=round(mean_t, 3), median_sec=round(med_t, 3),
                 max_sec=round(max_t, 3), decision=decision,
                 est_full_hours=round(est_full_h, 2), workers=n_workers)
    with open(OUT_DIR / "benchmark.txt", "w", encoding="utf-8") as f:
        f.write("背离v6:精选加厚仓位(P7) 单格耗时基准\n")  # P7 增量
        f.write(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"机器: {mp.cpu_count()} 核,并行进程 {n_workers}\n")
        f.write(f"抽样 {len(bench_cells)} 格(覆盖全部出场族与多种挑选规则,硬编码确定性抽样)\n\n")
        for r in sorted(rows, key=lambda r: -r["stats"]["runtime_sec"]):
            f.write(f"  {r['summary']['cell_id']:45s} "
                    f"{r['stats']['runtime_sec']:8.3f} 秒\n")
        f.write(f"\n单格耗时: 均值 {mean_t:.3f} 秒 / 中位 {med_t:.3f} 秒 / "
                f"最大 {max_t:.3f} 秒\n")
        f.write(f"预估全量 9,450 格单遍墙钟: {est_full_h:.2f} 小时\n")  # P7 增量
        f.write(f"结论: {decision}\n")
    log(f"[benchmark] 单格均值 {mean_t:.3f} 秒,最大 {max_t:.3f} 秒 -> {decision}")

    # ---------------- 自检 2/3(引擎内置硬断言,跑完即过)/ 4(抽样汇总)/ 5(确定性)----------------
    checks["check2_cash_conservation"] = dict(
        ok=True, note="抽样格逐日流水回放零容差断言全过(run_cell 内置,不过即中断)")
    checks["check3_causality"] = dict(
        ok=True, note="入场日>事件日硬断言;E2/E7 ATR ≤t−1;"
                      "挑选/仓位决策只用信号日及之前信息(结构性)")
    max_d4 = max(r["stats"]["check4_max_diff_yuan"] for r in rows)
    checks["check4_dual_ledger"] = dict(ok=max_d4 <= 0.05,
                                        max_diff_yuan=round(max_d4, 6))
    c5 = selfcheck_determinism(bench_cells)
    checks["check5_determinism"] = c5
    if not c5["ok"]:
        log("CHECK5 FAIL —— 确定性不过,停止(README §五.4)")
        _HB["stop"] = True
        sys.exit(1)

    if args.mode == "benchmark":
        _HB["stop"] = True
        log(f"SELECT THICK TRIAL BENCHMARK DONE ({time.time() - _HB['t0']:.0f}s)")  # P7 增量
        return

    # ---------------- 全量 9,450 格双跑(README §五.4 逐位确定性)----------------
    cells = enumerate_grid()
    assert len(cells) == 9450, f"网格 {len(cells)} != 9450"  # P7 增量:README §三.2
    log(f"[full] 网格 {len(cells)} 格 = 池6 × H3 × 挑选5 × K(3/5/7) × 出场35")  # P7 增量
    rows_full = run_cells_parallel(cells, write=True,
                                   stage=f"full pass1: {len(cells)} 格")
    # 自检 §五.2/§五.3:仓位数上限 + 单仓上限不变量(硬断言,不过即中断;函数内部 assert)
    checks["checkP7_invariants"] = selfcheck_p7(rows_full)  # P7 增量
    max_d4f = max(r["stats"]["check4_max_diff_yuan"] for r in rows_full)
    checks["check4_dual_ledger"] = dict(ok=max_d4f <= 0.05,
                                        max_diff_yuan=round(max_d4f, 6))
    # 第二遍(不落盘,只留 summary+stats 做逐位对拍)
    set_stage(f"full pass2: {len(cells)} 格", 0, len(cells))
    rows_pass2: list[dict] = []
    cfgs2 = [{**c, "write": False} for c in cells]
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(_cell_worker, cfgs2,
                                                    chunksize=1)):
            rows_pass2.append({"summary": res["summary"], "stats": res["stats"]})
            _HB["done"] = i + 1
            if (i + 1) % 100 == 0:
                log(f"heartbeat: full pass2 {i + 1}/{len(cells)} 格")
    cdet = detcmp_full(rows_full, rows_pass2)
    checks["check_determinism_full"] = cdet
    if not cdet["ok"]:
        log("DETCMP FAIL —— 两遍全量不逐位一致,视为失败,停工排查(README §五.4)")
        _HB["stop"] = True
        sys.exit(1)

    write_summary_verdict_report(rows_full, checks, bench,
                                 scope=f"全量终审({len(cells)} 格,双跑逐位一致)")

    _HB["stop"] = True
    log(f"SELECT THICK TRIAL DONE ({time.time() - _HB['t0']:.0f}s)")  # P7 增量


if __name__ == "__main__":
    main()
