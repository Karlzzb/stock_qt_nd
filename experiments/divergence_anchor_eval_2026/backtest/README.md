# 背离信号两变体 × 三出场 预登记回测（2026-01-01..2026-08-31）

本文件为预登记：先于任何回测结果落盘写成，逐条钉死配置与口径。
任务性质：预登记一次性回测，全程禁网络，仓库内除本目录（experiments/divergence_anchor_eval_2026/backtest/）外一律只读。

## 1. 引擎与完好性硬闸门

回测引擎：v3_pipeline/scripts/strategy_engine_v3.py（冻结候选，一字不改）。
底层机制复用 v1（v3_pipeline/scripts/strategy_engine.py，冻结，一字不改）。
先行硬闸门：`.venv/bin/python -m pytest v3_pipeline/tests/test_strategy_engine_v3.py -x -q` 必须全过；
不通过则停止任务并原样报告失败输出，不修引擎。
闸门结果登记于最终报告。

## 2. 信号源（两个原始背离事件表，无模型分数）

- 变体1（区间最低价锚定）：experiments/divergence_anchor_eval_2026/events_v1.parquet，5646 事件。
- 变体2（右侧确认精确锚定）：experiments/divergence_anchor_eval_2026/events_v2.parquet，5194 事件。
- event_date 为信号可知晓日（因果日），入场为 event_date 的下一交易日开盘（引擎 T+1 机制）。
- 两表实测 event_date 范围均为 2026-01-05..2026-08-31，(ts_code, event_date) 无重复。

## 3. 配置矩阵（6 行 = 2 信号变体 × 3 出场）

全部配置 n_slots=3。
出场规格对应 T8 报告（v3_pipeline/reports/strategy_tuning/t8_report.md）最优配置参数：

| 配置名 | 出场规格 |
|:---|:---|
| A13 | ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=12) |
| B15 | ExitSpec.vol_adaptive(tp=0.25, sl=-0.14, horizon=12, vol_lookback=21, vol_high_thresh=1.8, vol_low_thresh=0.6, vol_profit_mult=1.5, vol_stop_mult=1.1, low_vol_profit_mult=1.0) |
| E1-H12 | ExitSpec.horizon_only(horizon=12)（引擎 E1 模式，裸信号暴露对照，无屏障，持有满 12 个交易日收盘卖） |

## 4. 信号取舍规则

引擎原生取舍为分数优先：当日信号按 prob 降序、ts_code 升序截取空位数。
适配器将事件表 prob 列填为 dif_lift（两变体实测最小值 > 0.001，全部为正值，排序语义与概率分数一致）。
即：信号强度（DIF 抬升幅度）优先，ts_code 升序平局。
被截取信号 T+1 涨停/无行情被放弃不递补（引擎固有行为，同 v1）。

## 5. 回测窗与窗口末端未平仓约定

回测窗：2026-01-01..2026-08-31（引擎交易日历取自 000905.SH 指数交易日）。
窗口末端未平仓约定 = T9 同款，出处：v3_pipeline/reports/test_adjudication/t9_report.md
「工程正确性缺陷修复登记」节与「验证证据」第 3 条，及引擎实现：
- 窗口末端不强制平仓。
- 期末权益对未平仓头寸按最后已知收盘价盯市（逐日盯市循环中 last_known_price 沿用最近可得收盘价；停牌僵尸仓沿用其最后行情日收盘价，可能陈旧）。
- 未平仓头寸归档 open_positions.parquet，并在 stats.open_at_end 计数披露。
- 全部指标（含年化）基于含盯市价值的期末权益计算。

## 6. 基准

基准：中证500（000905.SH），与 T8/T9 同源。
加载方式复用 v3_pipeline/scripts/run_strategy_family.py 的 index_metrics：
读 stock_data/index/000905.SH.parquet（trade_date/close），窗口内首末收盘的几何年化与最大回撤。
该序列实测覆盖 2004-12-31..2026-08-31，完整覆盖本回测窗，无需兜底。
超额年化(pp) = 策略净年化 - 基准年化。

## 7. 本金 / 成本 / 整手 / 滑点口径（引擎内置默认值，逐条抄录自 strategy_engine.py）

- 初始本金 INIT_CAPITAL = 1,000,000 元。
- 整手 BOARD_LOT = 100 股；买入预算 = 前一日收盘权益 / n_slots，向下取整到手，不足一手买一手（现金够时），现金不足逐手递减。
- 滑点 SLIPPAGE = 0.001（单边 0.1%，买入加价、卖出减价）。
- 佣金 COMMISSION_RATE = 0.00025（双边万 2.5），单笔最低 COMMISSION_MIN = 5 元。
- 印花税（仅卖出）：2023-08-28 起 STAMP_TAX_NEW = 0.0005；本窗口全部落在新税率区间（0.05%）。
- 价格比较容差 PRICE_TOL = 1e-9。
- T+1：买入当日不可卖；涨停拒买（开盘价 >= 涨停价 - 容差则放弃）；跌停顺延（收盘价 <= 跌停价 + 容差则顺延至首个可卖日）。
- 逐日盯市：每日权益 = 现金 + 持仓市值（最后已知收盘价）。

## 8. 类 B 全市场 ATR 均值序列（mkt_atr）

复用 T8/T9 构建路径：v3_pipeline/scripts/run_strategy_tuning.py 的 build_mkt_atr。
口径：独立轻量加载 stock_data/daily 全部个股日线（窗口起点前 90 自然日起至日历末端），
逐股 ATR(21)（TR = max(h-l, |h-c'|, |l-c'|)，简单均值窗口，不足 21 个 TR 记缺失），
市场均值 = 当日全部有有效 ATR 个股的算术平均，再 reindex 到回测日历。
覆盖至日历末端 2026-08-31，满足「覆盖到 2026-08」要求。
个股 ATR 序列（stock_atr）与 T9 同法：对 MarketData.daily 逐股 atr_series(LB=21)
（load_market_data 日线向前多取 60 自然日，窗口早期个股 ATR 不足 21 窗口时引擎回落中档并计 stats.vol_fallback_mid）。

## 9. 指标口径

复用 run_strategy_family.compute_metrics（T8/T9 同口径）：
年化 = (期末权益/本金)^(1/年数)-1（年数 = 日历天数/365.25）；
Sharpe = 日收益均值/标准差 × √252；最大回撤基于逐日权益；Calmar = 年化/|最大回撤|；
年换手 = 年卖出笔数 × 平均入场名义 / 平均权益；仓位利用率 = 日均市值/权益（引擎 stats.capital_utilization）。
出场分解取自引擎 stats：exits_tp / exits_sl / exits_horizon（E1 全部计入 exits_horizon）。

## 10. 反泄漏断言

- 引擎内置断言全开：买入日 > 事件日（run_backtest_v3 入场处硬断言）；类 B ATR 只用 <=t-1 行情。
- 适配器侧额外硬断言：所有成交 trades.entry_date > trades.event_date 全量成立（逐配置断言，失败即中止）。
- 适配器断言 prob（=dif_lift）无缺失且全部 > 0；(ts_code, event_date) 无重复。
- 不调用 load_score_events（其信号源为 scores_final.parquet，与本任务无关），事件 DataFrame 由适配器直接构造。

## 11. 自检项（执行后登记于最终报告）

1. 每个 run 的 trades 非空且 entry_date > event_date 全量成立（断言进 run_matrix.py）。
2. 抽查 1 笔 events_v2 × A13 交易手工重放：从 stock_data/daily 原始行情独立重算入场日开盘价（×1.001 滑点）、屏障触发扫描（tp=入场价×1.25 / sl=入场价×0.86、同日双触发保守取止损、到期第 12 日收盘卖）、佣金/印花税/净盈亏，与引擎明细逐字段比对（replay_check.py）。
3. 全程禁网络。

## 12. 产物布局

- runs/<signal>__<config>/（equity_curve.parquet / trades.parquet / open_positions.parquet / stats.json），signal ∈ {events_v1, events_v2}，config ∈ {A13, B15, E1-H12}。
- summary.csv（6 行汇总：净年化、超额年化pp、Sharpe、最大回撤、Calmar、交易笔数、换手、仓位利用率、出场分解、基准年化、期末未平仓数）。
- run_matrix.py（适配器 + 执行脚本）、replay_check.py（手工重放抽查）、run_matrix.log（阶段日志与心跳）。

## 预登记增补 2026-09-04

本节为预登记增补：先于本轮任何回测结果落盘写成。
用户新指令：资金池 100 万不变（真实有限散户量级），仓位数 N=3→10（每仓约 10 万）；新增全池事件级研究，解决组合层样本功效不足。
引擎回归闸门已于跑数前执行：`.venv/bin/python -m pytest v3_pipeline/tests/test_strategy_engine_v3.py -x -q` → 17 passed in 0.23s（原文登记于最终报告）。
全程禁网络；仓库内除本目录外一律只读；v3_pipeline/ 一字不改。

### A. 组合层 N=10 矩阵

矩阵：{events_v1, events_v2} × {A13, B15, E1-H12} × n_slots=10，共 6 配置。
其余口径与上文第 1~11 节逐条相同，仅 n_slots 由 3 改为 10，逐条确认：
- 引擎与硬闸门同第 1 节（v3 冻结引擎，一字不改）。
- 信号源同第 2 节（events_v1 / events_v2 原表，prob←dif_lift）。
- 出场规格同第 3 节（A13 / B15 / E1-H12 参数一字不动）。
- 取舍规则同第 4 节（prob=dif_lift 降序、ts_code 升序截取空位；被截取信号 T+1 涨停/无行情放弃不递补）。
- 回测窗 2026-01-01..2026-08-31 与末端未平仓约定同第 5 节（期末按最后已知收盘盯市，open_at_end 披露，不做事件层式剔除）。
- 基准 000905.SH 同源加载同第 6 节。
- 本金 1,000,000 / 整手 / 滑点 / 佣金 / 印花税 / T+1 / 涨跌停口径同第 7 节（INIT_CAPITAL 不变，单仓预算 = 前一日收盘权益 / 10）。
- 类 B mkt_atr / stock_atr 同第 8 节。
- 指标口径同第 9 节。
- 反泄漏断言同第 10 节。
产物：runs_n10/<signal>__<config>/（四件套同上轮）、summary_n10.csv（列同上轮 summary.csv）、run_matrix_n10.py、run_matrix_n10.log。

### B. 事件层全池研究（event_study.py）

范围：两变体全部事件 × 三档出场（A13 / B15 / E1-H12，参数同第 3 节）逐笔独立模拟。
每笔名义本金 100,000 元（与组合层 N=10 单仓量级一致），单事件现金池 = 100,000 元。
逐笔模拟独立实现，不调用组合引擎主循环；屏障/顺延/成本/整手规则逐条对齐引擎口径（对齐清单见 D 节）。

入场与剔除口径：
- 入场日 = event_date 的次一交易日（日历同引擎：000905.SH 交易日）。
- event_date 落在日历最后一日 → 无入场日，记 dropped_no_next_day，剔除出统计并披露数量。
- 入场日无行情 → dropped_no_quote；开盘价 ≥ 涨停价 − 1e-9 → dropped_limitup（不递补）；event_date 收盘缺失 → dropped_no_close_T（引擎同款存在性校验）。
- 股数 = int(100000 / 执行价 / 100) × 100；不足一手时买一手（现金够），现金不足逐手递减至 0 → dropped_cash（逐条复刻引擎 while 递减逻辑，现金池固定 100,000）。
- 持仓期跌停顺延（触发日收盘 ≤ 跌停价 + 1e-9 → 顺延，deferred_days 计数）；无行情日不评估。
- 日历末端（2026-08-31）仍无法完整出场的事件记 incomplete，剔除出统计并披露数量，不做盯市（与组合层第 5 节盯市口径不同，此处为事件层预登记选择）。

统计指标（每 变体 × 出场 一组）：
- n、净收益（ret，口径同引擎 trades.ret）均值 / 中位数、胜率（ret > 0）。
- 相对同期基准超额：bench_ret = 000905.SH 收盘(entry_date) → 收盘(exit_date) 简单收益，excess = ret − bench_ret，出超额均值。
- t 统计量：单样本 t = mean / (std/√n)，分别对 ret 与 excess 各出一个（t_ret / t_excess）。
- E1-H12 组按 dif_lift 组内三分位分三档（低/中/高），出各档 n、均值收益、胜率，检验 dif_lift 排序有效性。

### C. 对拍锚点（组合层 N=10 vs 事件层）

组合层 N=10 成交的每一笔交易（trades.parquet），与事件层同 (signal, config, ts_code, event_date) 的逐笔结果对拍：
- 第一层（交易机制逐位断言）：以组合层实际成交股数钉住事件层重放（shares_override），entry_date / exit_date / exit_reason / entry_price / exit_raw_price / exit_exec_price / entry_commission / exit_commission / stamp_tax / net_pnl 全字段逐位一致（容差 1e-9），全量断言，任一不符即停止并报告。
- 第二层（名义本金口径披露）：事件层头条统计用固定 100,000 预算，组合层单仓预算 = 前日收盘权益 / 10（随权益漂移），两者股数可能不同；披露股数不一致笔数，并在股数恰好一致的子集上断言固定预算重放与组合层 net_pnl 亦逐位一致。
- 反向披露：事件层 dropped_limitup / dropped_no_quote / dropped_cash 而组合层同事件却成交的笔数（预算口径差异所致）逐笔列出。

### D. 事件层与引擎规则对齐清单（逐条核对自 strategy_engine.py / strategy_engine_v3.py 源码）

1. 交易日历：stock_data/index/000905.SH 交易日（load_market_data 同源）。
2. 入场日 = event_date 次一交易日；硬断言 entry_date > event_date（反泄漏）。
3. 入场执行价 = 当日 open × (1 + 0.001)。
4. 涨停拒买：open ≥ up_limit − 1e-9 → dropped_limitup，不递补；stk_limit 当日文件缺失视为无约束并计数。
5. T 日（event_date）收盘存在性校验，缺失记 dropped_no_close_T。
6. 整手：shares = int(budget / exec / 100) × 100；< 100 则置 100；while 现金不足逐手递减；归 0 记 dropped_cash。引擎买入现金 = 组合现金池，事件层现金 = 100,000（唯一结构性差异，见 C 节第二层）。
7. 买入佣金 = max(5, shares × exec × 0.00025)。
8. T+1：买入当日不可卖，出场评估仅对 day > entry_date。
9. held_days = 日历下标差 + 1（买入日记第 1 日）。
10. A13 屏障：tp = entry × 1.25，sl = entry × 0.86；tp_hit = high ≥ tp − 1e-9，sl_hit = low ≤ sl + 1e-9；tp 独中 → open 越屏障按 open 否则屏障价；sl 中（含同日双触发）→ open 破屏障按 open 否则屏障价；held ≥ 12 → 当日收盘 horizon。
11. E1-H12：无屏障，held ≥ 12 当日收盘卖。
12. B15 分带：评估日 day 取 ref_day = 前一交易日；a_stock = stock_atr[ts_code].loc[ref_day]（se3.atr_series，LB=21，TR = max(h−l, |h−c′|, |l−c′|) 简单均值，窗口不足 21 记 NaN；个股日线与引擎同源 = load_market_data 窗口前多取 60 自然日）；a_mkt = mkt_atr.loc[ref_day]（run_strategy_tuning.build_mkt_atr 产物，日历起点前 90 自然日起）；任一缺失/非正 → 回落中档 (tp, sl) 并计 vol_fallback_mid；vol_mult ≥ 1.8 → (tp×1.5, sl×1.1)；≤ 0.6 → (tp×1.0, sl)；其余 → (tp, sl)；屏障与触发口径同第 10 条。
13. 跌停顺延：触发日 close ≤ down_limit + 1e-9 → 顺延至首个可卖日，deferred_days 计数。
14. 无行情日：不评估、不触发、持仓沿用。
15. 卖出执行价 = raw × (1 − 0.001)；卖出佣金 = max(5, 成交额 × 0.00025)；印花税 = 成交额 × 0.0005（窗口全在 2023-08-28 后）。
16. net_pnl = shares × (exec_sell − entry_price) − (买入佣金 + 卖出佣金 + 印花税)；ret = net_pnl / (shares × entry_price + 买入佣金)。
17. 事件层无仓位竞争、无信号截取（每事件独立占满单仓），这是与组合层的结构性差异；对拍仅覆盖组合层实际成交笔（C 节）。
18. 末端处理：组合层期末盯市并披露 open_at_end；事件层记 incomplete 剔除（B 节）。

### E. 本轮自检项（执行后登记于最终报告）

1. 引擎回归 pytest 17 项全过（跑数前已执行）。
2. 组合层 N=10 每个 run：trades 非空且 entry_date > event_date 全量成立（断言进 run_matrix_n10.py）。
3. 事件层：entry_date > event_date 全量断言；对拍第一层全量 1e-9 断言（C 节），不通过则停止并报告。
4. 手工重放 2 笔（v1 一笔、v2 一笔，优先有跌停顺延或涨停拒买路径的事件），从 stock_data/daily 原始行情 + stk_limit 原始表独立重算，与事件层明细逐字段比对（replay_event_check.py，不复用 event_study.py 的模拟函数）。
5. 全程禁网络。

### F. 本轮新增产物布局

- run_matrix_n10.py / run_matrix_n10.log / runs_n10/<signal>__<config>/ / summary_n10.csv。
- event_study.py / event_study.log / event_study.parquet（每行 = 一事件 × 一出场：signal/config/ts_code/event_date/entry/exit/exit_reason/shares/净收益/ret/超额/持有天数/status 含 dropped_* 与 incomplete 行）/ event_study_summary.csv / event_study_terciles.csv（E1-H12 dif_lift 三档）。
- replay_event_check.py（事件层手工重放抽查）。
