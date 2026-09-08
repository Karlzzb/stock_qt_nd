#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背离v6:长持审判(H 拉长至 60/120 交易日) + 口径D 终审 —— 预登记 = 同目录 README.md(冻结,先于跑数落盘,勿改)。

本脚本 = #39 底座 experiments/v6_select_thick_trial/run_select_thick.py 逐字拷贝
+ P5/P6 移植(P5 逐字移植自 #37 experiments/v6_elastic_pos_trial/run_elastic.py,
P6 逐字移植自 #38 experiments/v6_thick_pos_trial/run_thick.py;移植块逐行注释标记
"P5/P6 移植")+ H 网格增量(H ∈ {60,120},新代码逐行注释标记 "H 拉长增量")。
唯一执行脚本。输入输出同 #39;输出 = 每格 equity_curve/trades/stats +
summary_long_hold.csv + verdict_long_hold.json + report.md + detcmp.log(全量双跑逐位对拍)。

主线网格 = 池(6: v6-1~v6-5 + ALL) × H(60/120,H 拉长增量) × 出场(35,逐字 #32 §3.1)
× [P1/P2 × 挑选(S1~S5);P5 固定 S0;P6 K(3/5/7) 固定 S0;P7 K(3/5/7) × 挑选(S1~S5)]
= 4,200 + 420 + 1,260 + 6,300 = 12,180 格(README §三.5)。
cell_id 段序沿用 #39:pool__H{H}__{sel}__{pos}__{exit}。
P1/P2/P5/P6/P7 五族仓位规则语义逐字沿用 #32 §3.2 / #37 §三 / #38 §三 / #39 §三,一概不动;
H 是统一出场维度、对全部 35 出场族生效(E1/E2 内嵌 horizon 被 H 网格统一覆盖的既有机制不变,
README §三.1)。

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

本实验解释性决断(规格未钉死处,全部披露,不改规格)—— L 系列(P5/P6 移植与 H 拉长增量):
 L-1. H 网格二分(H 拉长增量):H_LIST_MAIN = [60, 120](README §三.1 冻结,主线出数用);
    H_LIST_REG = [10, 20, 25](README §五.2 全量回归锚定用,复跑 #37/#38/#39 全网格与 #32
    P1/P2 子网格)。主线网格 12,180 格;回归网格 630+1,890+9,450+6,300 = 18,270 格。
    H 是统一出场维度、对全部 35 出场族生效;E1/E2 内嵌 horizon 被 H 网格统一覆盖的既有
    机制逐字不动(README §三.1,沿用 #32 §一.5)。
 L-2. P5/P6 移植来源(P5/P6 移植):P5 块逐字移植自 #37 run_elastic.py(全接、
    名义金额 = 100万/max(10,S)、现金再摊薄 min(名义, cash/S)、p5_days_diluted/
    p5_days_cashbound 信号日计数);P6 块逐字移植自 #38 run_thick.py(名义金额 =
    100万/max(K,S),p6_days_diluted/p6_days_cashbound 信号日计数;S = 当日池内全部
    信号数,含 truncated_no_next,沿用 #37 决断 E-1 / #38 决断 D-1;cash = 信号日收盘
    决策时点现金,沿用决断 E-2/D-3;遍历顺序 = ts_code 升序,沿用决断 E-4/D-5)。
    施工前逐段比对结论:#37/#38/#39 三份底座在共享代码(信号日挑选块框架、入场执行块、
    出场族执行、双口径记账、统计指标口径)上逐字一致,无分歧情形需要裁决。
 L-3. slot_budget 增 P5/P6 防御性 raise 分支(P5/P6 移植;结构性不可达——P5/P6 单仓金额
    在信号日挑选块内联计算并随 pending 传入入场执行块,沿用 #37 决断 E-3 / #38 决断 D-4);
    make_cfg 对 P6/P7 均解析 K(int(pos[3:]));P5 无 K。
 L-4. 主线网格枚举顺序 = 池→H→仓位族(P1,P2→P5→P6→P7)→挑选→K→出场;格子集合 =
    README §三.5 的 12,180 格(硬断言),枚举顺序不影响任何单格结果与汇总排序
    (summary 落盘前按 pool/H/sel/pos/exit 排序)。
 L-5. summary_long_hold.csv 列 = #39 同构指标列 + 摊薄统计列 signal_days/signal_days_burst/
    signal_max(全行出数,池级常量,格内同值)+ p5_days_diluted/p5_days_cashbound
    (仅 P5 行有值)+ p6_days_diluted/p6_days_cashbound(仅 P6 行有值),非适用行留空;
    摊薄统计列构造与 #37/#38 同构(README §四;burst 阈值 S>10 沿用 #37 定义)。
 L-6. §五.1 口径复算锚定从 #39 冻结 summary_select_thick.csv 独立重算:口径 D 纯现金字面版
    0 格、混合口径变体 32 格、旧终审线 0 格,三数必须逐位复现否则本实验不出数;
    #32/#37/#38/#39 冻结产物只读,不重跑不修改。
 L-7. §五.2 回归方法:H∈{10,20,25} 上复跑 #37 全网格 630 格、#38 全网格 1,890 格、
    #39 全网格 9,450 格,外加 #32 P1/P2 子网格 6,300 格作 P1/P2 路径的额外锚定(超出
    README 字面的加固,披露);比对 = 以冻结 summary 的列集合与行序为准,逐格逐字段按
    落盘渲染规则重渲染文本(整数列整型断言后原样、布尔列 True/False、浮点 %.6f、
    空值空串)与冻结 CSV 原文逐位对拍;runtime_sec 全豁免;#32 对拍额外豁免 passed 列
    (#32 的 passed = 旧终审线,自 #37 起 passed = 口径 D 主线,语义有意变更,披露)。
    任一格不一致即失败,本实验不出数。
 L-8. §五.3"出场触发判定日距入场 ≤ H 个交易日"统一按个股行情行口径作硬断言:全部 35
    出场族、全部成交笔验证 held_rows − deferred_days ≤ H(E0 原生按个股序列行计日;
    对 E1/E2/E3~E7 可证明成立:horizon/tp/sl 首个触发日 = 市场日序上首个 held ≥ H 或
    满足触发条件的有个股行情日,其个股行情行距不超过 H;开盘跌停/无行情顺延只移执行日
    不移触发判定日)。施工期实证修正披露:预登记时拟按"各族原生计日口径"(E1/E2/E3~E7
    按市场日历日 held_days − deferred_days ≤ H)落断言,首跑实测 10,271/12,180 格出现
    市场日口径超出(峰值达 H 的 1080%),根因 = 引擎"当日无个股行情:不触发任何判断"
    原生语义(#32~#39 全部冻结实验同此,从未设 H 不变量检查故未暴露)——持有期内的
    停牌/无行情间隙不计顺延、不评估,复牌后首个有行情日触发判定,市场日距离可远超 H
    而个股行情行距离恒 ≤ H(首跑后六格采样与解析推导双证:行口径违例 0)。因此修订
    为统一行口径硬断言,并把"市场日口径 > H 而行口径 ≤ H"的成交笔列为第三类例外
    (停牌/无行情间隙顺延)逐格计数披露;README §五.3 字面只命名两类例外,系预登记时
    未预见该现象,此处如实披露修订。前两类例外逐格计数披露不变:开盘跌停/无行情顺延笔
    (deferred_days>0,其触发判定日折算后同样 ≤ H);数据耗尽截断持仓(truncated_window +
    truncated_exhausted,无触发判定日、非成交笔,按 stats 逐格披露,README §七.4)。
    E0 的 NaN 收盘行顺移(#31 §2.3"NaN 行计入但不评估"原生语义)若发生将使
    held_rows − deferred_days > H,计为违例;按 #39 冻结产物抽检实证(50 格 21,088 笔
    E0 成交中 held_rows>H 的 455 笔全部 deferred_days>0、无一 NaN 顺移),预期 0,
    非 0 即 FAIL 停止。
 L-9. §五.4 落位:P7 格 6,300 格复用 #39 selfcheck_p7 全量硬断言(任一入场日入场笔数
    <= K、任一日历日并发在持 <= K、全部成交笔单仓金额 <= 100万÷K、entered ==
    已平仓归并入场 + open_at_end 对账);P5/P6 格 1,680 格 dropped_slot_full == 0
    硬断言(README §五.4);P2 爆发段动态上限语义不变由 §五.2 的 #32 P1/P2 子网格
    6,300 格逐位对拍证明(语义未动 => 逐位一致),不再单设断言(披露)。
L-10. 基准抽样 8 格(H 拉长增量:全部落在 H∈{60,120},覆盖五个仓位族与全部七个出场族,
    硬编码确定性)。
L-11. 退化对账格沿用 #32 的 P1×E0×S1×H20(H20 属回归档 H_LIST_REG,非主线格;仓位上限=
    无穷、现金约束放开、每股预算 10 万);#32 §六.1 对账自检(浮点差 0)原样保留作继承
    断言门槛(README §五.6)。
L-12. 截断披露(H 拉长增量):truncated_window/truncated_exhausted 逐格入 summary 并在
    §五.3 自检中汇总;H=120 时约 2026-03 起的事件持仓被数据耗尽截断、截断占比高于既往
    实验,属机制预期(README §二预警④ / §七.4)。
L-13. 报告聚合粒度(README §六):H 响应表以配置为行、同配置五档 H 轨迹(H10/20/25 取自
    #32/#37/#38/#39 冻结产物,H60/120 本实验实测,冻结实验不重跑);出场族分层按族聚合
    对照屏障族(E1/E2,29 个出场配置)与其余六族(E0/E3/E4/E5/E6/E7)的 H 敏感性;
    信号池价值与池内排序增益分开表述不混比(背离唯一底座指令);分布统计一律用
    "覆盖率→门槛"方向,不用裸 P-分位列。
L-14. 确定性自检(README §五.5) = 主线 12,180 格全量双跑,逐格 summary+stats(除
    runtime_sec)逐位对拍落 detcmp.log;equity/trades 逐位由抽样 selfcheck_determinism
    深度验证(沿用 #32/#37/#38/#39 框架)。

用法:
    python3 run_long_hold.py --mode calibrate   # 自检 §五.1:#39 冻结 summary 三线过线数复算
    python3 run_long_hold.py --mode selfcheck   # 自检 §五.1 + 继承对账自检 + 抽样确定性
    python3 run_long_hold.py --mode benchmark   # 自检 + 8 抽样格计时(两遍,确定性)
    python3 run_long_hold.py --mode full        # §五.2 回归 18,270 格 + 主线 12,180 格双跑 + 终审出数
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

OUT_DIR = Path(os.path.join(REPO, "experiments", "v6_long_hold_trial"))  # H 拉长增量:本实验目录
BASE_DIR = Path(os.path.join(REPO, "experiments", "v6_portfolio_trial"))  # #32 底座目录(冻结产物只读)
CACHE_DIR = BASE_DIR / "cache"   # mkt_atr 缓存只读复用 #32 确定性产物(沿用 #39 决断 T-6)
BASE_SUMMARY_PATH = BASE_DIR / "summary_portfolio.csv"  # #32 冻结 summary(§五.2 P1/P2 回归锚定 + 对照用,决断 L-7/L-13)
ELASTIC_DIR = Path(os.path.join(REPO, "experiments", "v6_elastic_pos_trial"))  # #37 目录(冻结产物只读)
ELASTIC_SUMMARY_PATH = ELASTIC_DIR / "summary_elastic.csv"  # #37 冻结 summary(§五.2 P5 回归锚定 + 对照用,决断 L-7/L-13)
THICK_DIR = Path(os.path.join(REPO, "experiments", "v6_thick_pos_trial"))  # #38 目录(冻结产物只读)
THICK_SUMMARY_PATH = THICK_DIR / "summary_thick.csv"  # #38 冻结 summary(§五.2 P6 回归锚定 + 对照用,决断 L-7/L-13)
SELECT_THICK_DIR = Path(os.path.join(REPO, "experiments", "v6_select_thick_trial"))  # H 拉长增量:#39 底座目录(冻结产物只读)
SELECT_THICK_SUMMARY_PATH = SELECT_THICK_DIR / "summary_select_thick.csv"  # H 拉长增量:#39 冻结 summary(§五.1 口径复算锚定 + §五.2 P7 回归锚定 + 对照用,决断 L-6/L-7/L-13)
RUNS_DIR = OUT_DIR / "runs"
LOG_PATH = OUT_DIR / "progress.log"
DETCMP_PATH = OUT_DIR / "detcmp.log"  # 全量双跑逐位对拍日志(README §五.5)
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
H_LIST_MAIN = [60, 120]   # H 拉长增量:主线网格 H 档(README §三.1 冻结,决断 L-1)
H_LIST_REG = [10, 20, 25]  # H 拉长增量:全量回归锚定 H 档(README §五.2,决断 L-1)
POOLS = ["v6-1", "v6-2", "v6-3", "v6-4", "v6-5", "ALL"]
SELS = ["S1", "S2", "S3", "S4", "S5"]
SEL_FIXED = "S0"          # P5/P6 移植:P5/P6 无挑选对象,sel 段固定写 S0(沿用 #37/#38,README §三.4)
K_LIST = [3, 5, 7]        # 精选加厚仓位(P7)/加厚弹性仓位(P6) K 网格维(README §三.2),pos 段 = P7K{k}/P6K{k}
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

# 基准抽样 8 格(施工计时用;H 拉长增量:全部落在 H∈{60,120},覆盖五个仓位族与全部七个
# 出场族,确定性硬编码,决断 L-10)
BENCHMARK_CELLS = [
    ("ALL", 60, "S1", "P1", "E0"), ("ALL", 60, "S1", "P7K5", "E1_A5"),
    ("ALL", 60, "S0", "P6K5", "E2_B1"), ("ALL", 120, "S0", "P5", "E3"),
    ("ALL", 60, "S1", "P2", "E4"), ("v6-5", 120, "S4", "P7K3", "E5"),
    ("v6-5", 60, "S0", "P6K7", "E6"), ("v6-1", 120, "S5", "P7K7", "E7"),
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
    if pos == "P5":
        # P5/P6 移植:P5 单仓金额 = min(100万/max(10,S), cash/S),cash 在信号日挑选块内
        # P5/P6 移植:才拿得到,故金额在挑选块内联计算(沿用 #37 决断 E-3);本分支结构性不可达。
        raise AssertionError("P5 单仓金额在信号日挑选块内联计算,slot_budget 不可达")  # P5/P6 移植
    if pos.startswith("P6"):
        # P5/P6 移植:P6 单仓金额 = min(100万/max(K,S), cash/S),cash 在信号日挑选块内
        # P5/P6 移植:才拿得到,故金额在挑选块内联计算(沿用 #38 决断 D-4);本分支结构性不可达。
        raise AssertionError("P6 单仓金额在信号日挑选块内联计算,slot_budget 不可达")  # P5/P6 移植
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
    K = cfg.get("k")  # 加厚弹性仓位(P6)/精选加厚仓位(P7)名义分母与仓位数上限 K(README §三.2;P1/P2/P5 与退化对账格为 None 不使用,决断 L-3)
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
    # P5/P6 移植:爆发日接入与摊薄统计(逐字自 #37/#38;signal_* 为池级常量,格内同值,全行出数,决断 L-5)
    _sizes = [len(v) for v in pg["by_date"].values()]  # P5/P6 移植
    stats["signal_days"] = len(_sizes)                 # P5/P6 移植
    stats["signal_days_burst"] = int(sum(1 for s in _sizes if s > 10))  # P5/P6 移植:S>10 阈值沿用 #37 定义(README §四)
    stats["signal_max"] = int(max(_sizes)) if _sizes else 0             # P5/P6 移植
    stats["p5_days_diluted"] = 0     # P5/P6 移植:P5 名义摊薄(S>10)信号日计数
    stats["p5_days_cashbound"] = 0   # P5/P6 移植:P5 现金再摊薄(cash/S 紧于名义)信号日计数
    stats["p6_days_diluted"] = 0     # P5/P6 移植:P6 名义摊薄(S>K)信号日计数
    stats["p6_days_cashbound"] = 0   # P5/P6 移植:P6 现金再摊薄(cash/S 紧于名义)信号日计数

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
                cap = K  # 精选加厚仓位(P7) 仓位数上限 = K(README §三.2:P7 = #32 P1 框架把仓位数 10 换成 K,其余一概不动)
            if degenerate:
                cap = 10 ** 9
            if cfg["pos"] == "P5" or cfg["pos"].startswith("P6"):  # P5/P6 移植
                cap = 10 ** 9  # P5/P6 移植:全部接入,无仓位数上限;入场时二次检查同此 cap,恒放开(dropped_slot_full 恒 0)
            n_free = cap - len(positions)
            if n_free < len(sig_idx):
                stats["dropped_slot_full"] += len(sig_idx) - max(n_free, 0)
            if cfg["pos"] == "P5":
                # P5/P6 移植:无挑选对象,当日池内信号全部接入;遍历顺序 = ts_code 升序
                # P5/P6 移植:(by_date 由 (event_date, ts_code) 排序的事件表 groupby 保序构造,沿用 #37 决断 E-4)
                chosen = sig_idx
                # P5/P6 移植:单仓金额 = min(100万/max(10,S), 当前 cash/S),当日所有信号同一金额。
                # P5/P6 移植:S = 当日池内全部信号数(含 truncated_no_next,沿用 #37 决断 E-1);
                # P5/P6 移植:cash = 信号日收盘后可用现金(上一交易行收盘值,沿用 #37 决断 E-2);
                # P5/P6 移植:slot_budget 拿不到 cash,此处内联(沿用 #37 决断 E-3)。
                S_n = len(sig_idx)  # P5/P6 移植
                nominal_p5 = INIT / max(10, S_n)  # P5/P6 移植:名义金额 = 100万/max(10,S)
                budget_p5 = min(nominal_p5, cash / S_n)  # P5/P6 移植:现金再摊薄
                if S_n > 10:
                    stats["p5_days_diluted"] += 1  # P5/P6 移植:名义摊薄日
                if cash / S_n < nominal_p5:
                    stats["p5_days_cashbound"] += 1  # P5/P6 移植:现金再摊薄日
            elif cfg["pos"].startswith("P6"):
                # P5/P6 移植:无挑选对象,当日池内信号全部接入;遍历顺序 = ts_code 升序
                # P5/P6 移植:(by_date 由 (event_date, ts_code) 排序的事件表 groupby 保序构造,沿用 #38 决断 D-5)
                chosen = sig_idx
                # P5/P6 移植:单仓金额 = min(100万/max(K,S), 当前 cash/S),当日所有信号同一金额。
                # P5/P6 移植:名义金额 = 100万/max(K,S) 为 P6 相对 P5 的唯一实质改动(分母 10 -> K,README §三.2);
                # P5/P6 移植:S = 当日池内全部信号数(含 truncated_no_next,沿用 #38 决断 D-1);
                # P5/P6 移植:cash = 信号日收盘后可用现金(上一交易行收盘值,沿用 #38 决断 D-3);
                # P5/P6 移植:slot_budget 拿不到 cash,此处内联(沿用 #38 决断 D-4)。
                S_n = len(sig_idx)  # P5/P6 移植
                nominal_p6 = INIT / max(K, S_n)  # P5/P6 移植:名义金额 = 100万/max(K,S)
                budget_p6 = min(nominal_p6, cash / S_n)  # P5/P6 移植:现金再摊薄(逐字沿用 P5 语义)
                if S_n > K:
                    stats["p6_days_diluted"] += 1  # P5/P6 移植:名义摊薄日(S>K)
                if cash / S_n < nominal_p6:
                    stats["p6_days_cashbound"] += 1  # P5/P6 移植:现金再摊薄日
            else:
                chosen = select_chosen(pg, sig_idx, n_free, cfg["sel"], rng)
            for ev_i in chosen:
                ed = int(pg["entry_date"][ev_i])
                if ed < 0:
                    stats["truncated_no_next"] += 1
                    if degenerate:
                        outcomes[(pg["codes"][ev_i], int(pg["event_date"][ev_i]))] = \
                            dict(status="truncated_no_next")
                    continue
                if cfg["pos"] == "P5":
                    budget = budget_p5  # P5/P6 移植:当日所有信号同一金额,随 pending 传入场执行块
                elif cfg["pos"].startswith("P6"):
                    budget = budget_p6  # P5/P6 移植:当日所有信号同一金额,随 pending 传入场执行块
                else:
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
    # 摊薄统计列(P5/P6 移植,与 #37/#38 同构,README §四,决断 L-5):signal_* 全行出数
    # (池级常量,格内同值);p5_* 仅 P5 行、p6_* 仅 P6 行有值,非适用行 None(落盘空串)
    row["signal_days"] = stats["signal_days"]  # P5/P6 移植
    row["signal_days_burst"] = stats["signal_days_burst"]  # P5/P6 移植
    row["signal_max"] = stats["signal_max"]  # P5/P6 移植
    row["p5_days_diluted"] = stats["p5_days_diluted"] if cfg["pos"] == "P5" else None  # P5/P6 移植
    row["p5_days_cashbound"] = stats["p5_days_cashbound"] if cfg["pos"] == "P5" else None  # P5/P6 移植
    row["p6_days_diluted"] = stats["p6_days_diluted"] if cfg["pos"].startswith("P6") else None  # P5/P6 移植
    row["p6_days_cashbound"] = stats["p6_days_cashbound"] if cfg["pos"].startswith("P6") else None  # P5/P6 移植
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
    if pos.startswith("P6") or pos.startswith("P7"):  # K 由 pos 段解析(决断 L-3;P6 移植自 #38,P7 沿用 #39)
        cfg["k"] = int(pos[3:])
    return cfg


def enumerate_grid() -> list[dict]:
    """H 拉长增量:主线网格 = 池(6) × H(60/120) × [P1/P2 × 挑选(5) × 出场(35);
    P5 × 出场(35) 固定 S0;P6 K(3) × 出场(35) 固定 S0;P7 K(3) × 挑选(5) × 出场(35)]
    = 4,200 + 420 + 1,260 + 6,300 = 12,180 格(README §三.5,决断 L-4)。"""
    cells: list[dict] = []
    for pool in POOLS:
        for H in H_LIST_MAIN:  # H 拉长增量
            for pos in ("P1", "P2"):
                for sel in SELS:
                    for ex in EXIT_CONFIGS:
                        cells.append(make_cfg(pool, H, sel, pos, ex["name"]))
            for ex in EXIT_CONFIGS:
                cells.append(make_cfg(pool, H, SEL_FIXED, "P5", ex["name"]))  # P5/P6 移植
            for k in K_LIST:
                for ex in EXIT_CONFIGS:
                    cells.append(make_cfg(pool, H, SEL_FIXED, f"P6K{k}", ex["name"]))  # P5/P6 移植
            for k in K_LIST:
                for sel in SELS:
                    for ex in EXIT_CONFIGS:
                        cells.append(make_cfg(pool, H, sel, f"P7K{k}", ex["name"]))
    assert len(cells) == 12180, f"主线网格 {len(cells)} != 12180(README §三.5)"
    assert len({c["cell_id"] for c in cells}) == 12180, "主线网格 cell_id 重复"
    return cells


def enumerate_regression() -> dict[str, list[dict]]:
    """H 拉长增量:§五.2 全量回归锚定网格,H∈{10,20,25}(决断 L-1/L-7)。

    R37 = #37 全网格 630 格(池×H×出场×P5,sel=S0);R38 = #38 全网格 1,890 格
    (池×H×出场×K×P6,sel=S0);R39 = #39 全网格 9,450 格(池×H×挑选×K×P7);
    R32 = #32 P1/P2 子网格 6,300 格(池×H×挑选×(P1,P2)×出场,超出 README 字面的
    加固锚定,证明 P2 爆发段动态上限语义未动,决断 L-9)。
    """
    reg: dict[str, list[dict]] = {"R37": [], "R38": [], "R39": [], "R32": []}
    for pool in POOLS:
        for H in H_LIST_REG:
            for ex in EXIT_CONFIGS:
                reg["R37"].append(make_cfg(pool, H, SEL_FIXED, "P5", ex["name"]))
            for k in K_LIST:
                for ex in EXIT_CONFIGS:
                    reg["R38"].append(make_cfg(pool, H, SEL_FIXED, f"P6K{k}", ex["name"]))
            for sel in SELS:
                for k in K_LIST:
                    for ex in EXIT_CONFIGS:
                        reg["R39"].append(make_cfg(pool, H, sel, f"P7K{k}", ex["name"]))
            for pos in ("P1", "P2"):
                for sel in SELS:
                    for ex in EXIT_CONFIGS:
                        reg["R32"].append(make_cfg(pool, H, sel, pos, ex["name"]))
    assert len(reg["R37"]) == 630 and len(reg["R38"]) == 1890
    assert len(reg["R39"]) == 9450 and len(reg["R32"]) == 6300
    return reg


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
    """独立从 #39 冻结 summary_select_thick.csv 重算三条线过线格数(README §五.1)。

    口径 D 纯现金字面版必须 0 格、混合口径变体必须 32 格、旧终审线必须 0 格,
    三数逐位复现才继续,否则本实验不出数(决断 L-6)。
    """
    set_stage("check0: 口径复算自检(#39 冻结 summary_select_thick.csv)")
    df = pd.read_csv(SELECT_THICK_SUMMARY_PATH)  # H 拉长增量
    assert len(df) == 9450, f"#39 冻结 summary 行数 {len(df)} != 9450"  # H 拉长增量
    u = df["capital_utilization"].to_numpy(dtype=float)
    per_cash = df["excess_cash"].to_numpy(dtype=float) / u
    per_idx = df["excess_idx"].to_numpy(dtype=float) / u
    sc = df["sharpe_cash"].to_numpy(dtype=float)
    si = df["sharpe_idx"].to_numpy(dtype=float)
    m_pure = (per_cash > 0.10) & (u >= 0.5) & (sc > 0.5)
    m_mix = (per_idx > 0.10) & (u >= 0.5) & (sc > 0.5)
    m_old = (df["excess_idx"].to_numpy(dtype=float) > 0.15) & (si > 0.5)
    n_pure, n_mix, n_old = int(m_pure.sum()), int(m_mix.sum()), int(m_old.sum())
    ok = (n_pure == 0 and n_mix == 32 and n_old == 0)  # H 拉长增量:预期 0/32/0(README §五.1)
    log(f"[check0] 口径D 纯现金字面版过线 {n_pure} 格(预期 0);"
        f"混合口径变体过线 {n_mix} 格(预期 32);旧终审线过线 {n_old} 格(预期 0)"
        f" -> {'PASS' if ok else 'FAIL'}")
    return dict(ok=bool(ok), n_pure=n_pure, n_mixed=n_mix, n_old=n_old,
                expect_pure=0, expect_mixed=32, expect_old=0, n_rows=int(len(df)),
                note="独立重算自 #39 冻结 summary_select_thick.csv(README §五.1)")


# ---------------------------------------------------------------- 自检 §五.2:全量回归锚定(P5/P6/P7 移植的硬验证)
_REG_STR_COLS = frozenset({"cell_id", "pool", "sel", "pos", "exit", "family"})
_REG_INT_COLS = frozenset({  # H 拉长增量:整数列(整型断言后原样渲染,决断 L-7)
    "H", "total_signals", "entered", "n_trades", "dropped_slot_full",
    "dropped_limitup", "dropped_no_quote", "dropped_cash", "truncated_no_next",
    "truncated_window", "truncated_exhausted", "deferred_exits", "open_at_end",
    "vol_fallback_mid", "p3_atr_fallback", "p4_atr_fallback", "e4_lot_skip",
    "e7_atr_missing", "exits_E0", "exits_tp", "exits_sl", "exits_horizon",
    "exits_trail_tp", "exits_scale_out", "exits_breakeven_sl",
    "exits_time_stop", "exits_chandelier_sl", "neg_equity_idx_days",
    "neg_equity_cash_days", "signal_days", "signal_days_burst", "signal_max",
    "p5_days_diluted", "p5_days_cashbound", "p6_days_diluted",
    "p6_days_cashbound"})
_REG_BOOL_COLS = frozenset({"passed"})


def _reg_render_cell(val, col: str) -> str:
    """H 拉长增量:按落盘渲染规则把 summary 字段重渲染为 CSV 原文文本(决断 L-7)。"""
    if col in _REG_STR_COLS:
        return str(val)
    if col in _REG_BOOL_COLS:
        assert isinstance(val, (bool, np.bool_)), f"{col} 非布尔: {val!r}"
        return "True" if val else "False"
    if col in _REG_INT_COLS:
        assert val is not None and not (isinstance(val, float) and np.isnan(val)), \
            f"{col} 回归格整数列为空: {val!r}"
        assert float(val) == int(val), f"{col} 回归格整数列非整: {val!r}"
        return str(int(val))
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return "%.6f" % float(val)


def _reg_compare_subset(name: str, frozen_path: Path, rows: list[dict],
                        extra_drop: list[str], pos_filter: tuple | None) -> dict:
    """H 拉长增量:以冻结 summary 的列集合与行序为准,逐格逐字段重渲染文本逐位对拍(决断 L-7)。

    runtime_sec 全豁免;extra_drop 列额外豁免(#32 对拍豁免 passed:#32 的 passed = 旧终审线,
    自 #37 起 passed = 口径 D 主线,语义有意变更,披露)。
    """
    frozen = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
    if pos_filter is not None:
        frozen = frozen[frozen["pos"].isin(pos_filter)].reset_index(drop=True)
    cols = [c for c in frozen.columns if c != "runtime_sec" and c not in extra_drop]
    got = {r["summary"]["cell_id"]: r["summary"] for r in rows}
    frozen_ids = frozen["cell_id"].tolist()
    assert len(got) == len(rows), f"{name} 回归产出 cell_id 重复"
    mismatch: list[str] = []
    if set(frozen_ids) != set(got):
        only_frozen = sorted(set(frozen_ids) - set(got))[:5]
        only_got = sorted(set(got) - set(frozen_ids))[:5]
        mismatch.append(f"cell_id 集合不一致: 冻结独有 {only_frozen} 产出独有 {only_got}")
    else:
        for r in frozen.itertuples(index=False):
            cid = r.cell_id
            summ = got[cid]
            bad_fields = []
            for col in cols:
                frozen_txt = getattr(r, col)
                got_txt = _reg_render_cell(summ[col], col)
                if frozen_txt != got_txt:
                    bad_fields.append(f"{col}: 冻结={frozen_txt!r} 产出={got_txt!r}")
            if bad_fields:
                mismatch.append(f"{cid} 字段不一致 {len(bad_fields)} 个: "
                                + "; ".join(bad_fields[:5]))
    ok = not mismatch
    for mline in mismatch[:20]:
        log(f"[checkR-{name}-mismatch] {mline}")
    log(f"[checkR] {name}: {len(rows)} 格 vs 冻结 {frozen_path.name} "
        f"{len(frozen)} 行(豁免列 runtime_sec{''.join('/' + c for c in extra_drop)})"
        f"逐位对拍,不一致 {len(mismatch)} 格 -> {'PASS' if ok else 'FAIL'}")
    return dict(ok=bool(ok), n_cells=len(rows), n_frozen=int(len(frozen)),
                n_mismatch=len(mismatch), mismatch_head=mismatch[:20],
                exempt_cols=["runtime_sec"] + list(extra_drop),
                frozen=str(frozen_path))


def run_cells_light(cells: list[dict], stage: str) -> list[dict]:
    """H 拉长增量:与 run_cells_parallel 同路径但只留 summary+stats(回归大网格内存控制)。"""
    set_stage(stage, 0, len(cells))
    rows: list[dict] = []
    cfgs = [{**c, "write": False} for c in cells]
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(_cell_worker, cfgs, chunksize=1)):
            rows.append({"summary": res["summary"], "stats": res["stats"]})
            _HB["done"] = i + 1
            if (i + 1) % 200 == 0:
                log(f"heartbeat: {stage} {i + 1}/{len(cells)} 格")
    return rows


def selfcheck_regression() -> dict:
    """H 拉长增量:README §五.2 全量回归锚定(决断 L-7)——合并脚本在 H∈{10,20,25} 上
    复跑 #37 全网格 630 格、#38 全网格 1,890 格、#39 全网格 9,450 格,与三份冻结 summary
    逐位对拍一致(runtime_sec 豁免);外加 #32 P1/P2 子网格 6,300 格额外锚定(豁免
    runtime_sec 与 passed,决断 L-7)。任一格不一致即失败,本实验不出数。"""
    set_stage("checkR: 全量回归锚定(§五.2, H∈{10,20,25})")
    t0 = time.time()
    grids = enumerate_regression()
    specs = [
        ("R37", ELASTIC_SUMMARY_PATH, None, []),
        ("R38", THICK_SUMMARY_PATH, None, []),
        ("R39", SELECT_THICK_SUMMARY_PATH, None, []),
        ("R32", BASE_SUMMARY_PATH, ("P1", "P2"), ["passed"]),
    ]
    subsets: dict = {}
    for name, path, pos_filter, extra_drop in specs:
        cells = grids[name]
        rows = run_cells_light(cells, stage=f"regression {name}")
        res = _reg_compare_subset(name, path, rows, extra_drop, pos_filter)
        res["n_cells_expected"] = len(cells)
        subsets[name] = res
        if not res["ok"]:
            log(f"[checkR] {name} FAIL —— 回归锚定不逐位一致,停止(README §五.2:任一格不一致即失败)")
            return dict(ok=False, subsets=subsets,
                        elapsed_sec=round(time.time() - t0, 1))
    n_total = int(sum(s["n_cells"] for s in subsets.values()))
    log(f"[checkR] 四路回归全部逐位一致(R37 630 + R38 1890 + R39 9450 + R32 6300 = "
        f"{n_total} 格) -> PASS ({time.time() - t0:.0f}s)")
    return dict(ok=True, subsets=subsets, n_cells_total=n_total,
                elapsed_sec=round(time.time() - t0, 1))


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
    """P7 子集全量硬断言(逐字沿用 #39 自检;本实验 README §五.4 的 P7 部分 = #39 §五.2/§五.3 同构)。

    逐格从 trades.parquet 验证任一入场日当日入场笔数 <= K,且任一日历日并发在持仓数
    <= K;另加 equity n_positions <= K 交叉验证(覆盖截断未平仓持仓,沿用 #39 决断 T-7)。
    全部成交笔 shares × entry_price <= 100万 ÷ K + 0.01 元浮点容差(沿用 #39 决断 T-8)。
    一致性对账:每格 entered == 已平仓归并入场笔数 + open_at_end(已平仓 = 归并组内含非部分
    卖出行;E4 部分卖出后仍截断在仓的入场计入 open_at_end,不计入已平仓)。
    """
    set_stage("checkP7: P7 仓位数上限 + 单仓上限不变量(全格 trades 独立重建)", 0, len(rows))
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
    log(f"[checkP7] P7 格:全部 {len(cids)} 格入场日笔数/并发在持/equity n_positions "
        f"峰值 <= K(全场峰值 {obs_max_day}/{obs_max_conc}/{obs_max_npos});"
        f"单仓上限:{n_trades_total} 成交笔名义 <= 100万÷K"
        f"(峰值占比 {obs_max_notional_ratio * 100:.4f}%);"
        f"一致性对账 entered == 已平仓归并入场 + open_at_end;"
        f"违例 {len(bad_day)}/{len(bad_conc)}/{len(bad_npos)}/{len(bad_notional)}/"
        f"{len(bad_recon)} -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    for bl in (bad_day, bad_conc, bad_npos, bad_notional, bad_recon):
        for b in bl[:10]:
            log(f"[checkP7-violation] {b}")
    assert ok, "README §五.4(P7 部分)硬断言失败,停止"
    return dict(ok=bool(ok), n_cells=len(cids), n_trades_total=n_trades_total,
                max_entries_per_day=obs_max_day, max_concurrent=obs_max_conc,
                max_n_positions=obs_max_npos,
                max_notional_ratio=round(obs_max_notional_ratio, 8),
                n_bad_day=len(bad_day), n_bad_conc=len(bad_conc),
                n_bad_npos=len(bad_npos), n_bad_notional=len(bad_notional),
                n_bad_recon=len(bad_recon))


# ---------------------------------------------------------------- 自检 §五.3:H 上限不变量(全部成交笔触发判定日距入场 ≤ H)
def _h_invariant_one(task) -> tuple:
    """H 拉长增量:读一格 runs/<cell_id>/trades.parquet,独立验证 §五.3(决断 L-8)。

    统一硬断言(全部 35 族):held_rows - deferred_days <= H —— 触发判定发生在入场后
    第 H 个个股行情行之内(E0 的原生计日口径;对 E1/E2/E3~E7 可证明成立: horizon/tp/sl
    首个触发日 = 市场日序上首个 held >= H 或满足触发条件的有个股行情日,其个股行情行距
    不超过 H;跌停/无行情顺延只移执行日不移触发判定日)。
    附带按市场日历日口径(held_days - deferred_days)复核:超出 H 的笔 = 持有期内个股
    停牌/无行情间隙所致(引擎"当日无行情:不触发任何判断"原生语义,#32~#39 同此),
    计为第三类例外逐格计数披露。
    返回 (cell_id, 成交笔数, 顺延笔数, 个股行口径违例笔数, 市场日口径违例笔数, 触发距入场峰值(行口径))。
    """
    cid, H, fam = task
    tr = pd.read_parquet(RUNS_DIR / cid / "trades.parquet")
    n_tr = len(tr)
    n_defer = n_viol_row = n_viol_mkt = max_trig = 0
    if n_tr:
        hd = tr["held_days"].to_numpy(dtype=np.int64)
        hr = tr["held_rows"].to_numpy(dtype=np.int64)
        dd = tr["deferred_days"].to_numpy(dtype=np.int64)
        assert (hr >= 0).all(), f"{cid} 存在 held_rows=-1 成交笔(出场日无个股行)"
        n_defer = int((dd > 0).sum())
        trig_row = hr - dd  # 决断 L-8:统一硬断言口径(个股行情行)
        trig_mkt = hd - dd  # 决断 L-8:市场日历日复核口径(无行情间隙会超出 H,披露)
        n_viol_row = int((trig_row > H).sum())
        n_viol_mkt = int((trig_mkt > H).sum())
        max_trig = int(trig_row.max())
    return cid, n_tr, n_defer, n_viol_row, n_viol_mkt, max_trig


def selfcheck_h_invariant(rows: list[dict]) -> dict:
    """H 拉长增量:README §五.3 —— 全部 12,180 格、全部成交笔的出场触发判定日距入场 <= H
    (统一硬断言:held_rows - deferred_days <= H,决断 L-8);开盘跌停/无行情顺延、数据耗尽
    截断、持有期内停牌/无行情间隙三类例外分别逐格计数披露。

    顺延例外笔(deferred_days>0)的触发判定日折算后同样 <= H;截断持仓无触发判定日、
    非成交笔,按 stats(truncated_window + truncated_exhausted)逐格计数披露;
    无行情间隙例外笔 = 市场日口径 held_days - deferred_days > H 但个股行口径 <= H 的成交笔
    (引擎"当日无行情不触发任何判断"原生语义,#32~#39 各实验同此,决断 L-8);
    E0 的 NaN 收盘行顺移若发生将使个股行口径 > H,计为违例(实证预期 0,非 0 即 FAIL)。
    """
    set_stage("checkH: H 上限不变量(全格 trades 独立重建)", 0, len(rows))
    tasks = [(r["summary"]["cell_id"], int(r["summary"]["H"]),
              EXIT_BY_NAME[r["summary"]["exit"]]["family"]) for r in rows]
    trunc_of = {r["summary"]["cell_id"]: int(r["stats"]["truncated_window"])
                + int(r["stats"]["truncated_exhausted"]) for r in rows}
    H_of = {t[0]: t[1] for t in tasks}
    t0 = time.time()
    bad: list[str] = []
    n_trades_total = n_defer_total = n_gap_total = 0
    gap_cells: dict = {}
    obs_max_ratio = 0.0
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (cid, n_tr, n_defer, n_viol_row, n_viol_mkt, max_trig) in enumerate(
                pool.imap_unordered(_h_invariant_one, tasks, chunksize=16)):
            n_trades_total += n_tr
            n_defer_total += n_defer
            n_gap_total += n_viol_mkt
            if n_viol_mkt:
                gap_cells[cid] = n_viol_mkt
            if n_viol_row:
                bad.append(f"{cid} 违例 {n_viol_row} 笔(个股行情行口径:触发距入场 > H)")
            H = H_of[cid]
            obs_max_ratio = max(obs_max_ratio, max_trig / H if H else 0.0)
            _HB["done"] = i + 1
            if (i + 1) % 2000 == 0:
                log(f"heartbeat: checkH {i + 1}/{len(tasks)} 格 "
                    f"({time.time() - t0:.0f}s)")
    trunc_cells = {c: v for c, v in trunc_of.items() if v > 0}
    trunc_total = int(sum(trunc_of.values()))
    ok = not bad
    log(f"[checkH] §五.3 H 上限不变量:全部 {len(tasks)} 格 {n_trades_total} 成交笔,"
        f"个股行情行口径(held_rows-deferred_days <= H)违例 {len(bad)} 格;"
        f"开盘跌停/无行情顺延例外笔 {n_defer_total} 笔(触发判定日折算后同样 <= H);"
        f"数据耗尽截断例外持仓 {trunc_total} 笔(涉及 {len(trunc_cells)} 格,"
        f"无触发判定日、非成交笔,逐格计数披露);"
        f"停牌/无行情间隙例外笔 {n_gap_total} 笔(涉及 {len(gap_cells)} 格,"
        f"市场日口径 > H 但个股行情行口径 <= H,披露,决断 L-8);"
        f"触发距入场峰值占 H 比例(行口径) {obs_max_ratio * 100:.2f}%"
        f" -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    for b in bad[:10]:
        log(f"[checkH-violation] {b}")
    assert ok, "README §五.3 硬断言失败,停止"
    return dict(ok=bool(ok), n_cells=len(tasks), n_trades_total=n_trades_total,
                n_deferred_trades=n_defer_total, n_truncated_positions=trunc_total,
                n_truncated_cells=len(trunc_cells), n_violation_cells=len(bad),
                n_gap_trades=n_gap_total, n_gap_cells=len(gap_cells),
                max_trigger_ratio=round(obs_max_ratio, 6),
                violation_head=bad[:10],
                note=("统一硬断言口径 = 个股行情行(held_rows-deferred_days<=H,全部 35 族);"
                      "市场日历日复核口径(held_days-deferred_days)超出 H 者 = 持有期内"
                      "停牌/无行情间隙所致,逐格计数披露(决断 L-8)"))


# ---------------------------------------------------------------- 自检 §五.4:仓位规则不变量继承
def selfcheck_pos_invariants(rows: list[dict]) -> dict:
    """H 拉长增量:README §五.4(决断 L-9)—— P7 格复用 #39 selfcheck_p7 全量硬断言
    (任一入场日入场笔数 <= K、并发在持 <= K、单仓金额 <= 100万÷K);P5/P6 格
    dropped_slot_full == 0 硬断言;P2 爆发段动态上限语义不变由 §五.2 的 #32 P1/P2
    子网格逐位对拍证明(见 checkR.R32),不再单设断言(披露)。"""
    p7_rows = [r for r in rows if r["summary"]["pos"].startswith("P7")]
    p56_rows = [r for r in rows
                if r["summary"]["pos"] == "P5"
                or r["summary"]["pos"].startswith("P6")]
    assert len(p7_rows) == 6300 and len(p56_rows) == 1680
    res7 = selfcheck_p7(p7_rows)  # P7 子集:逐字复用 #39 自检(本实验 §五.4 = #39 §五.2/§五.3 同构)
    bad56 = [r["summary"]["cell_id"] for r in p56_rows
             if int(r["stats"]["dropped_slot_full"]) != 0]
    for cid in bad56[:10]:
        log(f"[checkPOS-violation] {cid} dropped_slot_full != 0")
    assert not bad56, f"README §五.4 失败:P5/P6 格 dropped_slot_full != 0 共 {len(bad56)} 格"
    log(f"[checkPOS] §五.4:P5/P6 全部 {len(p56_rows)} 格 dropped_slot_full == 0 -> PASS;"
        f"P2 爆发段动态上限语义不变由 checkR.R32(6,300 格逐位对拍)证明")
    return dict(ok=bool(res7["ok"] and not bad56), p7=res7,
                p56_cells=len(p56_rows), p56_slot_full_violations=len(bad56),
                p2_note="P2 动态上限语义不变由 #32 P1/P2 子网格回归逐位对拍证明(决断 L-9)")


# ---------------------------------------------------------------- 自检 4(README §五.4):全量双跑逐位对拍
def detcmp_full(rows1: list[dict], rows2: list[dict]) -> dict:
    """两遍全量逐格对拍:summary 与 stats(除 runtime_sec)严格逐位一致,落 detcmp.log(README §五.5)。"""
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
        f.write("背离v6:长持审判(H 60/120) 全量双跑逐位确定性对拍(README §五.5)\n")  # H 拉长增量
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
    "H=120 时尾部截断加重:约 2026-03 起的事件持仓被数据耗尽截断;截断笔数与占比逐格披露;截断按末日收盘估值记账(沿用 #32 截断条款),偏乐观方向沿用 #31 披露 3。",
    "口径 D 为拍板的判活线,旧终审线(+15pp 且 Sharpe>0.5,判活口径记账)只作参考列出数;本实验不回溯改判 #32/#37/#38/#39 已归档结论。",
    "H 拉长对 E1/E2 屏障族预期钝感(§二预警②);若出数显示这两族指标几乎不随 H 变化,属机制预期内而非实现缺陷。",
    "利用率抬升伴随现金耗尽日增多、接入集合随之改变;若「利用率过线格的在投质量」与「在投质量过线格的利用率」仍互斥,则该张力在持有期维度上复现,即为本信号族在口径 D 下的结构性结论而非实现缺陷。",
    "H 只取 60/120 两档、不做 40/90 等中间档扫描,避免事后挑档;两档对齐阶段底语义(前后约 60~120 交易日尺度的中期低点)。",
]


def write_summary_verdict_report(rows: list[dict], checks: dict, bench: dict,
                                 scope: str) -> None:
    summary = pd.DataFrame([r["summary"] for r in rows]).sort_values(
        ["pool", "H", "sel", "pos", "exit"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "summary_long_hold.csv", index=False,
                   float_format="%.6f")  # H 拉长增量:落 summary_long_hold.csv(README §六)
    log(f"[dump] summary_long_hold.csv {len(summary)} 行({scope})")

    masks = line_masks(summary)
    cells = summary["cell_id"].tolist()
    passed_main = [c for c, m in zip(cells, masks["main"]) if m]
    passed_mixed = [c for c, m in zip(cells, masks["mixed"]) if m]
    passed_old = [c for c, m in zip(cells, masks["old"]) if m]

    checks_all = all(v.get("ok", True) for v in checks.values())
    verdict = dict(
        experiment="背离v6:长持审判(H 拉长至 60/120 交易日) + 口径D 终审",  # H 拉长增量
        prereadme="README.md 先于任何跑数落盘(冻结)",
        scope=scope,
        main_line=("口径 D 纯现金字面版: excess_cash/capital_utilization > +10pp "
                   "且 capital_utilization >= 0.5 且 sharpe_cash > 0.5(README §一.2)"),
        ref_lines=["混合口径变体(参考,不判活): excess_idx/u > +10pp 且 u>=0.5 "
                   "且 sharpe_cash > 0.5(README §一.2)",
                   "旧终审线(参考,不判活): excess_idx > +15pp 且 sharpe_idx > 0.5"
                   "(#32 旧线的 S5 附加 cluster_t 条款不纳入参考列,沿用 #39 决断 T-4)"],
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
    with open(OUT_DIR / "verdict_long_hold.json", "w", encoding="utf-8") as f:  # H 拉长增量
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] verdict_long_hold.json")
    write_report(summary, verdict, checks, bench, scope)


# ---------------------------------------------------------------- 对照加载(冻结产物只读,不重跑)
def load_frozen_h_reg() -> dict:
    """H 拉长增量:同配置 H10/20/25 对照数字取自冻结产物,不重跑(README §三.7,决断 L-13)。

    返回 {"P1P2": #32 P1/P2 子集, "P5": #37, "P6": #38, "P7": #39} 四份冻结 summary;
    #32 冻结 summary 无 per_cash/per_idx 两列,按同一公式补算(与 #37 起逐字一致的派生列)。
    """
    base = pd.read_csv(BASE_SUMMARY_PATH)
    base = base[base["pos"].isin(["P1", "P2"])].copy()
    p5 = pd.read_csv(ELASTIC_SUMMARY_PATH)
    assert len(p5) == 630, f"#37 冻结 summary 行数 {len(p5)} != 630"
    p6 = pd.read_csv(THICK_SUMMARY_PATH)
    assert len(p6) == 1890, f"#38 冻结 summary 行数 {len(p6)} != 1890"
    p7 = pd.read_csv(SELECT_THICK_SUMMARY_PATH)
    assert len(p7) == 9450, f"#39 冻结 summary 行数 {len(p7)} != 9450"
    for _df in (base, p5, p6, p7):
        _u = _df["capital_utilization"].to_numpy(dtype=float)
        _df["per_cash"] = _df["excess_cash"].to_numpy(dtype=float) / _u
        _df["per_idx"] = _df["excess_idx"].to_numpy(dtype=float) / _u
    return {"P1P2": base, "P5": p5, "P6": p6, "P7": p7}


# ---------------------------------------------------------------- 裁决报告(README §六)
POS_FULL_NAMES = {  # 仓位规则中文全名(命名用全称纪律)
    "P1": "P1 等仓十仓", "P2": "P2 爆发段动态加仓", "P5": "P5 R1 弹性仓位",
    "P6K3": "P6K3 加厚弹性仓位(K=3)", "P6K5": "P6K5 加厚弹性仓位(K=5)",
    "P6K7": "P6K7 加厚弹性仓位(K=7)", "P7K3": "P7K3 精选加厚仓位(K=3)",
    "P7K5": "P7K5 精选加厚仓位(K=5)", "P7K7": "P7K7 精选加厚仓位(K=7)"}
SEL_FULL_NAMES = {  # 挑选规则中文全名(命名用全称纪律)
    "S1": "S1 先到先得", "S2": "S2 前20日跌幅最深", "S3": "S3 种子强度",
    "S4": "S4 随机", "S5": "S5 已弹最少+量比最冷"}
H_ALL = [10, 20, 25, 60, 120]  # H 拉长增量:五档 H 轨迹(H10/20/25 冻结 + H60/120 实测)
FAMILIES = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
FAM_N_EXITS = {"E0": 1, "E1": 12, "E2": 17, "E3": 1, "E4": 1, "E5": 1,
               "E6": 1, "E7": 1}
POS_VALUES = ["P1", "P2", "P5", "P6K3", "P6K5", "P6K7", "P7K3", "P7K5", "P7K7"]


def _frozen_src(frozen: dict, pos: str) -> pd.DataFrame:
    """仓位规则 -> 对应冻结 summary(H10/20/25 对照用,不重跑)。"""
    if pos in ("P1", "P2"):
        return frozen["P1P2"]
    if pos == "P5":
        return frozen["P5"]
    if pos.startswith("P6"):
        return frozen["P6"]
    return frozen["P7"]


def _trace_h(frozen: dict, cur: pd.DataFrame, pool: str, sel: str, pos: str,
             exit_name: str, metric: str) -> dict:
    """H 拉长增量:同配置五档 H 轨迹(H10/20/25 冻结产物 + H60/120 本实验实测)。"""
    vals: dict = {}
    fr = _frozen_src(frozen, pos)
    for H in H_ALL:
        src = fr if H in (10, 20, 25) else cur
        hit = src[(src["pool"] == pool) & (src["H"] == H) & (src["sel"] == sel)
                  & (src["pos"] == pos) & (src["exit"] == exit_name)]
        vals[H] = float(hit[metric].iloc[0]) if len(hit) else np.nan
    return vals


def write_report(summary: pd.DataFrame, verdict: dict, checks: dict, bench: dict,
                 scope: str) -> None:
    frozen = load_frozen_h_reg()  # H 拉长增量:H10/20/25 对照(#32/#37/#38/#39 冻结,不重跑)

    def rule_stats(df: pd.DataFrame) -> dict:
        """仓位规则×H 聚合(总账表用;分布统计一律用'覆盖率→门槛'方向)。"""
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

    def fam_h_mean(pos: str, fam: str, H: int, metric: str) -> float:
        src = _frozen_src(frozen, pos) if H in (10, 20, 25) else summary
        sub = src[(src["pos"] == pos) & (src["family"] == fam) & (src["H"] == H)]
        return float(sub[metric].mean()) if len(sub) else np.nan

    def fam_delta(pos: str, fams, metric: str) -> float:
        """H 钝感量化:同配置逐格 |H120 − H10| 的均值(配置集合内)。"""
        a = _frozen_src(frozen, pos)
        a = a[(a["pos"] == pos) & (a["family"].isin(fams)) & (a["H"] == 10)]
        b = summary[(summary["pos"] == pos) & (summary["family"].isin(fams))
                    & (summary["H"] == 120)]
        m = a.merge(b, on=["pool", "sel", "pos", "exit"], suffixes=("_h10", "_h120"))
        return float((m[f"{metric}_h120"] - m[f"{metric}_h10"]).abs().mean())

    L: list[str] = []
    L.append("# 背离v6:长持审判(H 拉长至 60/120 交易日) + 口径D 终审 · 裁决报告")
    L.append("")
    L.append(f"范围:{scope}。预登记军令状 = 同目录 README.md(冻结,先于任何跑数落盘)。")
    L.append("本报告全部数字由脚本从产物重算生成;同配置 H10/20/25 对照数字取自 #32/#37/#38/#39 "
             "冻结产物 summary_portfolio.csv / summary_elastic.csv / summary_thick.csv / "
             "summary_select_thick.csv,均不重跑(README §三.7)。")
    L.append("")
    L.append(f"**自检总评:{'ALL PASS' if verdict['checks_all_pass'] else 'HAS FAIL(如实交付)'}**")
    L.append("")
    L.append("## 1. 自检结果(README §五)")
    L.append("")
    c0 = checks.get("check0_caliber_recompute", {})
    L.append(f"1. 口径复算锚定(§五.1):从 #39 冻结 summary_select_thick.csv 独立重算——口径 D "
             f"纯现金字面版过线 {c0.get('n_pure')} 格(预期 0)、混合口径变体过线 "
             f"{c0.get('n_mixed')} 格(预期 32)、旧终审线过线 {c0.get('n_old')} 格(预期 0)"
             f" -> {'PASS' if c0.get('ok') else 'FAIL'}")
    cr = checks.get("checkR_regression", {})
    crs = cr.get("subsets", {})
    n_mis_main = sum(int(crs.get(k, {}).get("n_mismatch", -1)) for k in ("R37", "R38", "R39")) if crs else -1
    L.append(f"2. 全量回归锚定(§五.2):合并脚本在 H∈{{10,20,25}} 复跑 #37 全网格 "
             f"{crs.get('R37', {}).get('n_cells')} 格、#38 全网格 {crs.get('R38', {}).get('n_cells')} 格、"
             f"#39 全网格 {crs.get('R39', {}).get('n_cells')} 格,与三份冻结 summary 逐位对拍"
             f"(runtime_sec 豁免)不一致合计 {n_mis_main} 格;外加 #32 P1/P2 子网格 "
             f"{crs.get('R32', {}).get('n_cells')} 格额外锚定(豁免 runtime_sec 与 passed,"
             f"决断 L-7)不一致 {crs.get('R32', {}).get('n_mismatch')} 格"
             f" -> {'PASS' if cr.get('ok') else 'FAIL'}")
    ch = checks.get("checkH_invariant", {})
    L.append(f"3. H 上限不变量(§五.3):全部 {ch.get('n_cells')} 格 {ch.get('n_trades_total')} 成交笔,"
             f"出场触发判定日距入场 <= H(统一个股行情行口径 held_rows−deferred_days ≤ H,"
             f"全部 35 出场族,决断 L-8),违例格 {ch.get('n_violation_cells')};开盘跌停/无行情顺延例外笔 "
             f"{ch.get('n_deferred_trades')} 笔(触发判定日折算后同样 <= H);数据耗尽截断例外持仓 "
             f"{ch.get('n_truncated_positions')} 笔(涉及 {ch.get('n_truncated_cells')} 格,"
             f"无触发判定日、非成交笔,逐格计数披露,README §七.4);停牌/无行情间隙例外笔 "
             f"{ch.get('n_gap_trades')} 笔(涉及 {ch.get('n_gap_cells')} 格,市场日历日口径超出 H "
             f"但个股行情行口径 <= H,引擎\"当日无行情不触发任何判断\"原生语义所致,"
             f"#32~#39 全部冻结实验同此,施工期实证修正后逐格计数披露,决断 L-8)"
             f" -> {'PASS' if ch.get('ok') else 'FAIL'}")
    cp = checks.get("checkPOS_invariants", {})
    cp7 = cp.get("p7", {})
    L.append(f"4. 仓位规则不变量(§五.4):P7 全部 {cp7.get('n_cells')} 格任一入场日入场笔数 <= K、"
             f"任一日历日并发在持 <= K(全场峰值:入场日笔数 {cp7.get('max_entries_per_day')}、"
             f"并发在持 {cp7.get('max_concurrent')}、equity n_positions "
             f"{cp7.get('max_n_positions')}),全部 {cp7.get('n_trades_total')} 成交笔单仓金额 "
             f"<= 100万÷K(峰值占名义上限 {(cp7.get('max_notional_ratio') or 0) * 100:.4f}%);"
             f"P5/P6 全部 {cp.get('p56_cells')} 格 dropped_slot_full == 0;P2 爆发段动态上限"
             f"语义不变由 #32 P1/P2 子网格回归逐位对拍证明(决断 L-9)"
             f" -> {'PASS' if cp.get('ok') else 'FAIL'}")
    cd = checks.get("check_determinism_full", {})
    L.append(f"5. 确定性(§五.5):主线全量 {cd.get('n_cells')} 格双跑逐位对拍,不一致 "
             f"{cd.get('n_mismatch')} 格 -> {'PASS' if cd.get('ok') else 'FAIL'}(落 detcmp.log)")
    L.append("6. 继承断言(§五.6):资金守恒逐日流水回放零容差、入场日>事件日因果断言、"
             "双口径对账(容差 0.05 元)等 #32 循环内硬断言原样保留,任一失败即中断。")
    c4d = checks.get("check4_dual_ledger", {})
    L.append(f"   双口径对账全量格最大偏差 {c4d.get('max_diff_yuan', 'NaN')} 元"
             f" -> {'PASS' if c4d.get('ok') else 'FAIL'}")
    c1 = checks.get("check1_reconcile", {})
    L.append(f"   继承对账自检(#32 §六.1,退化对账格 P1×E0×S1×H20,决断 L-11):trades_seed 子集 "
             f"{c1.get('n_ref')} 起,逐笔对照 {c1.get('n_compared')} 起,不一致 "
             f"{c1.get('n_mismatch')} 起;dd20/bounce 逐位一致="
             f"{c1.get('dd20_bitwise')}/{c1.get('bounce_bitwise')}"
             f" -> {'PASS' if c1.get('ok') else 'FAIL'}")
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
        L.append("张力量化(「资金利用率 ≥ 0.5 且单位在投超额 >10pp」联立不可达是否在持有期维度复现,"
                 "README §二/§七.7):")
        L.append("")
        for _H in (60, 120):
            subh = summary[summary["H"] == _H]
            u_h = subh["capital_utilization"].to_numpy(dtype=float)
            pc_h = subh["per_cash"].to_numpy(dtype=float)
            sc_h = subh["sharpe_cash"].to_numpy(dtype=float)
            qual_h = (pc_h > 0.10) & (sc_h > 0.5)
            uhalf_h = u_h >= 0.5
            joint_h = (pc_h > 0.10) & uhalf_h
            if qual_h.any():
                iq = int(np.argmax(np.where(qual_h, u_h, -np.inf)))
                rq = subh.iloc[iq]
                L.append(f"- H{_H}:在投质量两条件过线(单位在投超额(纯现金) > +10pp 且 "
                         f"sharpe_cash > 0.5)格 {int(qual_h.sum())} 个,其中资金利用率最高 = "
                         f"{rq['capital_utilization']:.4f}(格 {rq['cell_id']}),"
                         f"vs 0.5({gap(float(rq['capital_utilization']), 0.5)})")
            else:
                L.append(f"- H{_H}:在投质量两条件过线格 0 个")
            if uhalf_h.any():
                iu = int(np.argmax(np.where(uhalf_h, pc_h, -np.inf)))
                ru = subh.iloc[iu]
                L.append(f"- H{_H}:资金利用率 ≥ 0.5 格 {int(uhalf_h.sum())} 个,其中单位在投超额"
                         f"(纯现金)最优 = {ru['per_cash']:+.4f}(格 {ru['cell_id']}),"
                         f"vs +0.10({gap(float(ru['per_cash']), 0.10)});该格 sharpe_cash = "
                         f"{ru['sharpe_cash']:.3f}")
            else:
                L.append(f"- H{_H}:资金利用率 ≥ 0.5 格 0 个")
            n_joint = int(joint_h.sum())
            if n_joint:
                ij = int(np.argmax(np.where(joint_h, sc_h, -np.inf)))
                rj = subh.iloc[ij]
                L.append(f"- H{_H}:两条件联立(资金利用率 ≥ 0.5 且单位在投超额(纯现金) > +10pp)"
                         f"可达 {n_joint} 格——#39 主线证的联立不可达在 H{_H} 不复现;"
                         f"但三线联立仍 0 格,绑死条件转为 sharpe_cash(联立格中最高 = "
                         f"{rj['sharpe_cash']:.3f}(格 {rj['cell_id']}),"
                         f"vs 0.5({gap(float(rj['sharpe_cash']), 0.5)}))")
            else:
                L.append(f"- H{_H}:两条件联立(资金利用率 ≥ 0.5 且单位在投超额(纯现金) > +10pp)"
                         f"可达 0 格,#39 的联立不可达张力在 H{_H} 复现")
        L.append("")
    # ---- §4 H 响应表 ----
    L.append("## 4. H 响应表(同配置五档 H 轨迹;配置为行)")
    L.append("")
    L.append("行 = 各仓位规则本实验最优单位在投超额(纯现金)的头部配置 + #39 头部格同配置锚;"
             "H10/20/25 数字取自 #32/#37/#38/#39 冻结产物(不重跑),H60/120 为本实验实测;"
             "'—' = 该档无此配置。")
    trace_rows: list[tuple] = []
    for pv in POS_VALUES:
        subp = summary[summary["pos"] == pv]
        hrow = subp.sort_values("per_cash", ascending=False).iloc[0]
        trace_rows.append((f"{POS_FULL_NAMES[pv]}头部(本实验 H{int(hrow['H'])})",
                           hrow["pool"], hrow["sel"], pv, hrow["exit"]))
    trace_rows.append(("#39 头部格同配置锚", "v6-3", "S1", "P7K3", "E1_A6"))
    for metric, label, fmt in (("capital_utilization", "资金利用率", ".4f"),
                               ("per_cash", "单位在投超额(纯现金)", "+.4f"),
                               ("sharpe_cash", "sharpe_cash", ".3f")):
        L.append("")
        L.append(f"{label}随 H 轨迹(配置为行):")
        L.append("")
        L.append("| 配置 | H10 | H20 | H25 | H60 | H120 |")
        L.append("|---|---|---|---|---|---|")
        for lb, pool, sel, pv, exn in trace_rows:
            vals = _trace_h(frozen, summary, pool, sel, pv, exn, metric)
            cells_txt = [("—" if not np.isfinite(vals[H]) else format(vals[H], fmt))
                         for H in H_ALL]
            L.append(f"| {lb} {pool}/{sel}/{pv}/{exn} | " + " | ".join(cells_txt) + " |")
    L.append("")
    # ---- §5 出场族分层 ----
    L.append("## 5. 出场族分层:屏障族(E1/E2,29 个出场配置) vs 其余六族(E0/E3/E4/E5/E6/E7)的 H 敏感性对照")
    L.append("")
    L.append("P1 等仓十仓与 P5 R1 弹性仓位在五档 H 上均有同配置格(H10/20/25 取自 #32/#37 冻结产物,"
             "H60/120 本实验实测),按出场族聚合格均;其余仓位规则仅在两档 H 有数,不参与本表。")
    for metric, label, fmt in (("capital_utilization", "格均资金利用率", ".4f"),
                               ("per_cash", "格均单位在投超额(纯现金)", "+.4f")):
        L.append("")
        L.append(f"{label}(行 = 出场族):")
        L.append("")
        L.append("| 出场族 | 出场配置数 | P1@H10 | P1@H20 | P1@H25 | P1@H60 | P1@H120 | "
                 "P5@H10 | P5@H20 | P5@H25 | P5@H60 | P5@H120 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for fam in FAMILIES:
            vals = []
            for pos in ("P1", "P5"):
                for H in H_ALL:
                    v = fam_h_mean(pos, fam, H, metric)
                    vals.append("—" if not np.isfinite(v) else format(v, fmt))
            L.append(f"| {fam} | {FAM_N_EXITS[fam]} | " + " | ".join(vals) + " |")
    L.append("")
    L.append("H 钝感量化(同配置逐格 H120 对 H10 之差的绝对值,族内均值;越小越钝感;行 = 出场族):")
    L.append("")
    L.append("| 出场族 | P1 资金利用率变化 | P1 单位在投超额(纯现金)变化 | "
             "P5 资金利用率变化 | P5 单位在投超额(纯现金)变化 |")
    L.append("|---|---|---|---|---|")
    for fam in FAMILIES:
        L.append(f"| {fam} | {fam_delta('P1', [fam], 'capital_utilization'):.4f} | "
                 f"{fam_delta('P1', [fam], 'per_cash'):.4f} | "
                 f"{fam_delta('P5', [fam], 'capital_utilization'):.4f} | "
                 f"{fam_delta('P5', [fam], 'per_cash'):.4f} |")
    L.append("")
    bar = {"E1", "E2"}
    oth = sorted(set(FAMILIES) - bar)
    L.append(f"屏障族(E1/E2,29 个出场配置)平均变化:P1 单位在投超额(纯现金) "
             f"{fam_delta('P1', bar, 'per_cash'):.4f}、资金利用率 "
             f"{fam_delta('P1', bar, 'capital_utilization'):.4f};P5 单位在投超额(纯现金) "
             f"{fam_delta('P5', bar, 'per_cash'):.4f}、资金利用率 "
             f"{fam_delta('P5', bar, 'capital_utilization'):.4f}。")
    L.append(f"其余六族(E0/E3/E4/E5/E6/E7,6 个出场配置)平均变化:P1 单位在投超额(纯现金) "
             f"{fam_delta('P1', oth, 'per_cash'):.4f}、资金利用率 "
             f"{fam_delta('P1', oth, 'capital_utilization'):.4f};P5 单位在投超额(纯现金) "
             f"{fam_delta('P5', oth, 'per_cash'):.4f}、资金利用率 "
             f"{fam_delta('P5', oth, 'capital_utilization'):.4f}。")
    L.append("屏障族指标若几乎不随 H 变化,属机制预期内(README §二预警② / §七.6)而非实现缺陷。")
    L.append("")
    # ---- §6 现金耗尽形态 ----
    L.append("## 6. 现金耗尽形态随 H 变化(行 = 仓位规则 × H;合计口径)")
    L.append("")
    L.append("H10/20/25 数字取自 #32/#37/#38/#39 冻结产物(不重跑),H60/120 为本实验实测;"
             "名义摊薄日与现金再摊薄日两列仅 P5/P6 有定义(其余规则记 '—');"
             "截断持仓 = truncated_window + truncated_exhausted(README §七.4)。")
    L.append("")
    L.append("| 仓位规则 | H | 接入合计 | 现金丢弃合计 | 槽满丢弃合计 | 名义摊薄日合计 | "
             "现金再摊薄日合计 | 截断持仓合计 | 顺延出场合计 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for pv in POS_VALUES:
        fr = _frozen_src(frozen, pv)
        for H in H_ALL:
            src = fr if H in (10, 20, 25) else summary
            sub = src[(src["pos"] == pv) & (src["H"] == H)]
            if not len(sub):
                continue
            if pv == "P5":
                diluted = f"{int(sub['p5_days_diluted'].sum())}"
                cashb = f"{int(sub['p5_days_cashbound'].sum())}"
            elif pv.startswith("P6"):
                diluted = f"{int(sub['p6_days_diluted'].sum())}"
                cashb = f"{int(sub['p6_days_cashbound'].sum())}"
            else:
                diluted = cashb = "—"
            trunc = int(sub["truncated_window"].sum() + sub["truncated_exhausted"].sum())
            L.append(f"| {POS_FULL_NAMES[pv]} | {H} | {int(sub['entered'].sum())} | "
                     f"{int(sub['dropped_cash'].sum())} | {int(sub['dropped_slot_full'].sum())} | "
                     f"{diluted} | {cashb} | {trunc} | {int(sub['deferred_exits'].sum())} |")
    L.append("")
    # ---- §7 信号池价值与池内排序增益(分开表述) ----
    L.append("## 7. 信号池价值与池内排序增益(分开表述,不混比)")
    L.append("")
    L.append("信号池价值(不涉池内排序的全接口径 = P5 R1 弹性仓位,本实验实测;行 = 池 × H;"
             "指标 = 35 出场格内最优单位在投超额(纯现金)、最优 sharpe_cash、格均资金利用率):")
    L.append("")
    L.append("| 池 | H | 最优单位在投超额(纯现金) | 最优 sharpe_cash | 格均资金利用率 |")
    L.append("|---|---|---|---|---|")
    for (pool, H), g in summary[summary["pos"] == "P5"].groupby(["pool", "H"]):
        L.append(f"| {pool} | {H} | {float(g['per_cash'].max()):+.4f} | "
                 f"{float(g['sharpe_cash'].max()):.3f} | "
                 f"{float(g['capital_utilization'].mean()):.4f} |")
    L.append("")
    L.append("池内排序增益(P1 等仓十仓口径:同 (池, H, 出场) 内五挑选规则最优单位在投超额(纯现金) "
             "− S1 先到先得;H10/20/25 取自 #32 冻结产物,H60/120 本实验实测;正 = 挑选优于先到先得):")
    L.append("")
    L.append("| H | 格均排序增益 | 最优排序增益 | 增益为正配置占比 |")
    L.append("|---|---|---|---|")
    for H in H_ALL:
        src = frozen["P1P2"] if H in (10, 20, 25) else summary
        sub = src[(src["pos"] == "P1") & (src["H"] == H)]
        piv = sub.pivot_table(index=["pool", "exit"], columns="sel", values="per_cash")
        gain = piv[["S2", "S3", "S4", "S5"]].max(axis=1) - piv["S1"]
        L.append(f"| {H} | {float(gain.mean()):+.4f} | {float(gain.max()):+.4f} | "
                 f"{float((gain > 0).mean()) * 100:.2f}%配置增益为正 |")
    L.append("")
    # ---- §8 仓位规则总账 ----
    L.append("## 8. 仓位规则总账(本实验 12,180 格实测;行 = 仓位规则 × H)")
    L.append("")
    L.append("分布统计一律用'覆盖率→门槛'方向;在投质量两条件 = 单位在投超额(纯现金) > +10pp "
             "且 sharpe_cash > 0.5;主线 = 在投质量两条件 + 资金利用率 ≥ 0.5。")
    L.append("")
    L.append("| 仓位规则 | H | 格数 | 接入合计 | 槽满丢弃合计 | 现金丢弃合计 | 资金利用率格均 | "
             "资金利用率 ≥ 0.5 格占比 | 单位在投超额(纯现金) > +10pp 格占比 | "
             "最优单位在投超额(纯现金) | 最优 sharpe_cash | 在投质量两条件过线格数 | 主线过线格数 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pv in POS_VALUES:
        for H in (60, 120):
            s = rule_stats(summary[(summary["pos"] == pv) & (summary["H"] == H)])
            L.append(f"| {POS_FULL_NAMES[pv]} | {H} | {s['n_cells']} | {s['entered']} | "
                     f"{s['slot_full']} | {s['cash_drop']} | {s['util_mean']:.4f} | "
                     f"{s['frac_u_ge_half'] * 100:.2f}%格资金利用率≥0.5 | "
                     f"{s['frac_per_gt10'] * 100:.2f}%格单位在投超额(纯现金)>+10pp | "
                     f"{s['best_per']:+.4f} | {s['best_sharpe']:.3f} | {s['n_qual']} | "
                     f"{s['n_main']} |")
    L.append("")
    L.append("## 9. 披露(README §七 预写,出数后原样保留)")
    L.append("")
    for i, d in enumerate(verdict["disclosure"], 1):
        L.append(f"{i}. {d}")
    L.append("")
    L.append("## 10. 解释性决断(规格未钉死处,施工披露,不改规格)")
    L.append("")
    L.append("继承 #32 的 14 条(逐字,底座脚本 docstring):")
    L.append("")
    for line in inherited_docstring_decisions():
        L.append(line)
    L.append("")
    L.append("本实验 L 系列(P5/P6 移植与 H 拉长增量,与头部 docstring L-1~L-14 逐字对应):")
    L.append("")
    for line in long_hold_docstring_decisions():
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


def long_hold_docstring_decisions() -> list[str]:
    """本实验解释性决断(与头部 docstring L-1~L-14 逐字对应)。"""
    return [
        "L-1. H 网格二分(H 拉长增量):H_LIST_MAIN = [60, 120](README §三.1 冻结,主线出数用);H_LIST_REG = [10, 20, 25](README §五.2 全量回归锚定用,复跑 #37/#38/#39 全网格与 #32 P1/P2 子网格)。主线网格 12,180 格;回归网格 630+1,890+9,450+6,300 = 18,270 格。H 是统一出场维度、对全部 35 出场族生效;E1/E2 内嵌 horizon 被 H 网格统一覆盖的既有机制逐字不动(README §三.1,沿用 #32 §一.5)。",
        "L-2. P5/P6 移植来源(P5/P6 移植):P5 块逐字移植自 #37 run_elastic.py(全接、名义金额 = 100万/max(10,S)、现金再摊薄 min(名义, cash/S)、p5_days_diluted/p5_days_cashbound 信号日计数);P6 块逐字移植自 #38 run_thick.py(名义金额 = 100万/max(K,S),p6_days_diluted/p6_days_cashbound 信号日计数;S = 当日池内全部信号数,含 truncated_no_next,沿用 #37 决断 E-1 / #38 决断 D-1;cash = 信号日收盘决策时点现金,沿用决断 E-2/D-3;遍历顺序 = ts_code 升序,沿用决断 E-4/D-5)。施工前逐段比对结论:#37/#38/#39 三份底座在共享代码(信号日挑选块框架、入场执行块、出场族执行、双口径记账、统计指标口径)上逐字一致,无分歧情形需要裁决。",
        "L-3. slot_budget 增 P5/P6 防御性 raise 分支(P5/P6 移植;结构性不可达——P5/P6 单仓金额在信号日挑选块内联计算并随 pending 传入入场执行块,沿用 #37 决断 E-3 / #38 决断 D-4);make_cfg 对 P6/P7 均解析 K(int(pos[3:]));P5 无 K。",
        "L-4. 主线网格枚举顺序 = 池→H→仓位族(P1,P2→P5→P6→P7)→挑选→K→出场;格子集合 = README §三.5 的 12,180 格(硬断言),枚举顺序不影响任何单格结果与汇总排序(summary 落盘前按 pool/H/sel/pos/exit 排序)。",
        "L-5. summary_long_hold.csv 列 = #39 同构指标列 + 摊薄统计列 signal_days/signal_days_burst/signal_max(全行出数,池级常量,格内同值)+ p5_days_diluted/p5_days_cashbound(仅 P5 行有值)+ p6_days_diluted/p6_days_cashbound(仅 P6 行有值),非适用行留空;摊薄统计列构造与 #37/#38 同构(README §四;burst 阈值 S>10 沿用 #37 定义)。",
        "L-6. §五.1 口径复算锚定从 #39 冻结 summary_select_thick.csv 独立重算:口径 D 纯现金字面版 0 格、混合口径变体 32 格、旧终审线 0 格,三数必须逐位复现否则本实验不出数;#32/#37/#38/#39 冻结产物只读,不重跑不修改。",
        "L-7. §五.2 回归方法:H∈{10,20,25} 上复跑 #37 全网格 630 格、#38 全网格 1,890 格、#39 全网格 9,450 格,外加 #32 P1/P2 子网格 6,300 格作 P1/P2 路径的额外锚定(超出 README 字面的加固,披露);比对 = 以冻结 summary 的列集合与行序为准,逐格逐字段按落盘渲染规则重渲染文本(整数列整型断言后原样、布尔列 True/False、浮点 %.6f、空值空串)与冻结 CSV 原文逐位对拍;runtime_sec 全豁免;#32 对拍额外豁免 passed 列(#32 的 passed = 旧终审线,自 #37 起 passed = 口径 D 主线,语义有意变更,披露)。任一格不一致即失败,本实验不出数。",
        "L-8. §五.3\"出场触发判定日距入场 ≤ H 个交易日\"统一按个股行情行口径作硬断言:全部 35 出场族、全部成交笔验证 held_rows − deferred_days ≤ H(E0 原生按个股序列行计日;对 E1/E2/E3~E7 可证明成立:horizon/tp/sl 首个触发日 = 市场日序上首个 held ≥ H 或满足触发条件的有个股行情日,其个股行情行距不超过 H;开盘跌停/无行情顺延只移执行日不移触发判定日)。施工期实证修正披露:预登记时拟按\"各族原生计日口径\"(E1/E2/E3~E7 按市场日历日 held_days − deferred_days ≤ H)落断言,首跑实测 10,271/12,180 格出现市场日口径超出(峰值达 H 的 1080%),根因 = 引擎\"当日无个股行情:不触发任何判断\"原生语义(#32~#39 全部冻结实验同此,从未设 H 不变量检查故未暴露)——持有期内的停牌/无行情间隙不计顺延、不评估,复牌后首个有行情日触发判定,市场日距离可远超 H 而个股行情行距离恒 ≤ H(首跑后六格采样与解析推导双证:行口径违例 0)。因此修订为统一行口径硬断言,并把\"市场日口径 > H 而行口径 ≤ H\"的成交笔列为第三类例外(停牌/无行情间隙顺延)逐格计数披露;README §五.3 字面只命名两类例外,系预登记时未预见该现象,此处如实披露修订。前两类例外逐格计数披露不变:开盘跌停/无行情顺延笔(deferred_days>0,其触发判定日折算后同样 ≤ H);数据耗尽截断持仓(truncated_window + truncated_exhausted,无触发判定日、非成交笔,按 stats 逐格披露,README §七.4)。E0 的 NaN 收盘行顺移(#31 §2.3\"NaN 行计入但不评估\"原生语义)若发生将使 held_rows − deferred_days > H,计为违例;按 #39 冻结产物抽检实证(50 格 21,088 笔 E0 成交中 held_rows>H 的 455 笔全部 deferred_days>0、无一 NaN 顺移),预期 0,非 0 即 FAIL 停止。",
        "L-9. §五.4 落位:P7 格 6,300 格复用 #39 selfcheck_p7 全量硬断言(任一入场日入场笔数 <= K、任一日历日并发在持 <= K、全部成交笔单仓金额 <= 100万÷K、entered == 已平仓归并入场 + open_at_end 对账);P5/P6 格 1,680 格 dropped_slot_full == 0 硬断言(README §五.4);P2 爆发段动态上限语义不变由 §五.2 的 #32 P1/P2 子网格 6,300 格逐位对拍证明(语义未动 => 逐位一致),不再单设断言(披露)。",
        "L-10. 基准抽样 8 格(H 拉长增量:全部落在 H∈{60,120},覆盖五个仓位族与全部七个出场族,硬编码确定性)。",
        "L-11. 退化对账格沿用 #32 的 P1×E0×S1×H20(H20 属回归档 H_LIST_REG,非主线格;仓位上限=无穷、现金约束放开、每股预算 10 万);#32 §六.1 对账自检(浮点差 0)原样保留作继承断言门槛(README §五.6)。",
        "L-12. 截断披露(H 拉长增量):truncated_window/truncated_exhausted 逐格入 summary 并在 §五.3 自检中汇总;H=120 时约 2026-03 起的事件持仓被数据耗尽截断、截断占比高于既往实验,属机制预期(README §二预警④ / §七.4)。",
        "L-13. 报告聚合粒度(README §六):H 响应表以配置为行、同配置五档 H 轨迹(H10/20/25 取自 #32/#37/#38/#39 冻结产物,H60/120 本实验实测,冻结实验不重跑);出场族分层按族聚合对照屏障族(E1/E2,29 个出场配置)与其余六族(E0/E3/E4/E5/E6/E7)的 H 敏感性;信号池价值与池内排序增益分开表述不混比(背离唯一底座指令);分布统计一律用\"覆盖率→门槛\"方向,不用裸 P-分位列。",
        "L-14. 确定性自检(README §五.5) = 主线 12,180 格全量双跑,逐格 summary+stats(除 runtime_sec)逐位对拍落 detcmp.log;equity/trades 逐位由抽样 selfcheck_determinism 深度验证(沿用 #32/#37/#38/#39 框架)。",
    ]


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="背离v6:长持审判(H 拉长至 60/120 交易日) + 口径D 终审"
                                             "(预登记 README.md)")  # H 拉长增量
    ap.add_argument("--mode", choices=["calibrate", "selfcheck", "benchmark", "full"],
                    default="benchmark")  # calibrate = §五.1 口径复算锚定
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _HB["t0"] = time.time()
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    log(f"LONG HOLD TRIAL START | mode={args.mode} | "
        f"预登记=README.md(冻结) | CPU {mp.cpu_count()}")  # H 拉长增量

    # ---------------- 自检:口径复算锚定(先于一切跑数,README §五.1)----------------
    checks: dict = {"check0_caliber_recompute": selfcheck_caliber()}
    if not checks["check0_caliber_recompute"]["ok"]:
        log("CHECK0 FAIL —— 口径复算不逐位复现,停止(README §五.1:三数必须逐位复现)")
        _HB["stop"] = True
        sys.exit(1)
    if args.mode == "calibrate":
        _HB["stop"] = True
        log("CALIBRATE DONE")
        return

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
        log("CHECK1 FAIL —— 对账不过,停止(README §五.6:继承断言不过则失败)")
        verdict_stub = dict(checks=checks, checks_all_pass=False,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(OUT_DIR / "verdict_long_hold.json", "w", encoding="utf-8") as f:  # H 拉长增量
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
        with open(OUT_DIR / "verdict_long_hold.json", "w", encoding="utf-8") as f:  # H 拉长增量
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
    est_full_h = mean_t * 12180 / n_workers / 3600  # H 拉长增量:主线全量 12,180 格(README §三.5)
    est_reg_h = mean_t * 18270 / n_workers / 3600  # H 拉长增量:回归 18,270 格(README §五.2 + 决断 L-7 加锚)
    decision = (f"回归 18,270 格(预估墙钟 {est_reg_h:.2f} 小时)+ 主线 12,180 格双跑"
                f"(预估单遍墙钟 {est_full_h:.2f} 小时),{n_workers} 进程并行")  # H 拉长增量
    bench = dict(n_cells=len(bench_cells), times_sec=[round(t, 3) for t in times],
                 mean_sec=round(mean_t, 3), median_sec=round(med_t, 3),
                 max_sec=round(max_t, 3), decision=decision,
                 est_full_hours=round(est_full_h, 2), workers=n_workers)
    with open(OUT_DIR / "benchmark.txt", "w", encoding="utf-8") as f:
        f.write("背离v6:长持审判(H 60/120) 单格耗时基准\n")  # H 拉长增量
        f.write(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"机器: {mp.cpu_count()} 核,并行进程 {n_workers}\n")
        f.write(f"抽样 {len(bench_cells)} 格(H∈{{60,120}},覆盖五个仓位族与全部七个出场族,"
                f"硬编码确定性抽样,决断 L-10)\n\n")
        for r in sorted(rows, key=lambda r: -r["stats"]["runtime_sec"]):
            f.write(f"  {r['summary']['cell_id']:45s} "
                    f"{r['stats']['runtime_sec']:8.3f} 秒\n")
        f.write(f"\n单格耗时: 均值 {mean_t:.3f} 秒 / 中位 {med_t:.3f} 秒 / "
                f"最大 {max_t:.3f} 秒\n")
        f.write(f"预估回归 18,270 格墙钟: {est_reg_h:.2f} 小时\n")  # H 拉长增量
        f.write(f"预估主线 12,180 格单遍墙钟: {est_full_h:.2f} 小时\n")  # H 拉长增量
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
        log("CHECK5 FAIL —— 确定性不过,停止(README §五.5)")
        _HB["stop"] = True
        sys.exit(1)

    if args.mode == "benchmark":
        _HB["stop"] = True
        log(f"LONG HOLD TRIAL BENCHMARK DONE ({time.time() - _HB['t0']:.0f}s)")  # H 拉长增量
        return

    # ---------------- 自检 §五.2:全量回归锚定(先于主线出数,不过则不出数)----------------
    creg = selfcheck_regression()
    checks["checkR_regression"] = creg
    if not creg["ok"]:
        log("CHECKR FAIL —— 回归锚定不逐位一致,停止"
            "(README §五.2:任一格不一致即失败,本实验不出数)")
        verdict_stub = dict(checks=checks, checks_all_pass=False,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(OUT_DIR / "verdict_long_hold.json", "w", encoding="utf-8") as f:
            json.dump(verdict_stub, f, ensure_ascii=False, indent=2, default=str)
        _HB["stop"] = True
        sys.exit(1)

    # ---------------- 主线 12,180 格双跑(README §五.5 逐位确定性)----------------
    cells = enumerate_grid()
    log(f"[full] 主线网格 {len(cells)} 格 = 池6 × H2(60/120) × [P1/P2 × 挑选5 × 出场35 × 2族;"
        f"P5 × 35;P6 × K3 × 35;P7 × K3 × 挑选5 × 35]")  # H 拉长增量
    rows_full = run_cells_parallel(cells, write=True, stage="full pass1")
    # 自检 §五.3:H 上限不变量(硬断言,不过即中断;函数内部 assert)
    checks["checkH_invariant"] = selfcheck_h_invariant(rows_full)  # H 拉长增量
    # 自检 §五.4:仓位规则不变量(硬断言,不过即中断;函数内部 assert)
    checks["checkPOS_invariants"] = selfcheck_pos_invariants(rows_full)  # H 拉长增量
    max_d4f = max(r["stats"]["check4_max_diff_yuan"] for r in rows_full)
    checks["check4_dual_ledger"] = dict(ok=max_d4f <= 0.05,
                                        max_diff_yuan=round(max_d4f, 6))
    # 第二遍(不落盘,只留 summary+stats 做逐位对拍)
    rows_pass2 = run_cells_light(cells, stage="full pass2")
    cdet = detcmp_full(rows_full, rows_pass2)
    checks["check_determinism_full"] = cdet
    if not cdet["ok"]:
        log("DETCMP FAIL —— 两遍全量不逐位一致,视为失败,停工排查(README §五.5)")
        _HB["stop"] = True
        sys.exit(1)

    write_summary_verdict_report(rows_full, checks, bench,
                                 scope=f"全量终审({len(cells)} 格,双跑逐位一致)")

    _HB["stop"] = True
    log(f"LONG HOLD TRIAL DONE ({time.time() - _HB['t0']:.0f}s)")  # H 拉长增量


if __name__ == "__main__":
    main()
