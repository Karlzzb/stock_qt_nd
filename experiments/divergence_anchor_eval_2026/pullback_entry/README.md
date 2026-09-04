# 事件后回踩挂单入场(实验A)预登记 —— 信号定义不动,只改入场规则

本文件为预登记:先于任何跑数结果落盘写成,逐条钉死假设、机制、成本、出场、网格、指标与判活线。
任务性质:预登记一次性回测,全程禁网络;仓库内除本目录
(experiments/divergence_anchor_eval_2026/pullback_entry/)外一律只读;v3_pipeline/ 一字不改。

## 1. 假设

金叉对金叉背离信号的锚点(低点)质量很好,但事件日(信号可知晓日)比锚点晚中位 8~9 天,
事件日次日开盘入场的前瞻收益均值≈0 或为负(反弹肉在信号发出前被吃完)。
本实验检验:信号发出后不追价,改为在锚点价附近挂限价买单等回踩,
吃"第二段/双底"是否可行。
信号定义(events_v1 / events_v2 事件表)一字不动,唯一变量是入场规则。

## 2. 信号源与基线行

- 变体1:experiments/divergence_anchor_eval_2026/events_v1.parquet,5646 事件。
- 变体2:experiments/divergence_anchor_eval_2026/events_v2.parquet,5194 事件。
- 回测窗:2026-01-01..2026-08-31;交易日历 = 000905.SH 指数交易日(与事件层研究同源同窗)。
- 基线行:同信号、同出场的"事件日 T+1 开盘入场"结果,来自
  experiments/divergence_anchor_eval_2026/backtest/event_study.parquet。
  本实验内以对照变体 m=+∞(T+1 开盘必成交)在自管线内复算该基线,
  并以逐笔对拍(见第 8 节自检闸门)证明自管线与事件层语义逐位一致;
  判活比较一律用自管线对照行(对拍通过后它与 event_study 逐笔等价)。

## 3. 挂单成交机制

- 挂单起始:事件日的次一交易日(T+1)起挂限价买单。
- 限价 L = anchor_close × (1+m),m ∈ {0.00, 0.03}。
- 有效期:自 T+1 起 20 个交易日(含 T+1 当日,即日历下标 [事件日+1, 事件日+20]);
  过期未成交记 status=no_fill("未成交"),不是亏损,按信号分母计 0(见第 6 节)。
- 成交判定:某交易日 low ≤ L 则成交;成交原始价 = min(当日 open, L)
  (开盘跳空低于限价按开盘价成交,更保守真实);执行价 = 原始价 × (1+0.001) 滑点。
- 涨停日不可买:当日 open ≥ 涨停价 − 1e-9(涨停价取自 stock_data/stk_limit 当日表,
  与引擎完全同源)则该日不成交,挂单留效顺延至有效期内后续交易日。
  任务书所给代理规则(open ≥ 前收×1.098 或 一字板 low==high 且涨幅≥9.8%)
  仅在 stk_limit 当日表缺失时作为兜底启用;本窗口 stk_limit 缺日 = 0
  (event_study 披露 limit_missing_days=0),兜底实际不触发,登记于此备查。
- 无行情日(该股当日无日线):不成交,挂单留效,有效期照常流逝。
- 事件日收盘缺失(dropped_no_close_T)、事件日为日历最后一日(dropped_no_next_day):
  与事件层同口径剔除并披露。
- 整手与现金:每笔固定名义本金 100,000 元;
  股数 = int(100000 / 执行价 / 100) × 100;不足一手买一手(现金够时);
  现金不足逐手递减,归 0 记 dropped_cash。逐条复刻引擎与事件层口径。
- 对照变体 m=+∞:T+1 开盘必成交(成交原始价 = 当日 open,与 min(open, +∞) 同义),
  T+1 涨停(open ≥ 涨停价 − 1e-9)记 dropped_limitup 不递补,
  T+1 无行情记 dropped_no_quote —— 即逐条复刻事件层 simulate_event 的入场段。

## 4. 成本(与 event_study 逐位一致,直接调用冻结引擎函数)

- 滑点双边各 0.1%:买入执行价 = 原始价 × 1.001;卖出执行价 = 原始价 × 0.999。
- 佣金双边万 2.5,单笔最低 5 元(strategy_engine.buy_cost / sell_costs 原函数)。
- 印花税卖出 0.05%(本窗口全在 2023-08-28 新税率区间)。
- 净收益 ret = 净盈亏 / (股数 × 买入执行价 + 买入佣金),与引擎 trades.ret 同口径。

## 5. 出场(自成交日 f 起算,成交日记持有第 1 日,T+1 起方可卖)

三种出场,日内触及语义与 event_study 完全一致
(high ≥ 屏障 − 1e-9 触止盈;low ≤ 屏障 + 1e-9 触止损;同日双触发保守取止损;
open 越过屏障按 open 否则按屏障价;到期当日收盘卖;触发日收盘 ≤ 跌停价 + 1e-9 顺延至首个可卖日;
无行情日不评估不触发、持有天数按日历下标差照常累计):

| 出场名 | 规格 |
|:---|:---|
| H12裸持 | 无屏障,持有满 12 个交易日当日收盘卖(引擎 E1 模式,等价 event_study 的 E1-H12 行) |
| A13同款 | 止盈 = 买入执行价 × 1.25;止损 = 买入执行价 × 0.86;最长 12 日 |
| 锚止损款 | 止盈 = 买入执行价 × 1.25;止损价 = anchor_close × 0.97(绝对价,不随成交价浮动);最长 12 日 |

窗口末端仍无法完整出场的记 status=incomplete,从收益统计剔除并披露(与事件层口径一致)。

## 6. 指标口径(每行)

记"有效信号分母" = 信号总数 − dropped_no_next_day − dropped_no_close_T − incomplete
(incomplete 分子分母同步剔除,与事件层剔除口径一致,占比单独披露);
no_fill / dropped_limitup / dropped_no_quote / dropped_cash 计入分母且收益贡献 0
(这些是"策略真实没买上"的结果,防选择性偏差的关键)。

- 信号数、成交笔数、成交率 = 成交笔数 / 有效信号分母;
- 成交笔均 ret 均值 / 中位数 / 胜率(ret > 0);
- 超额:bench_ret = 000905.SH 收盘(成交日) → 收盘(出场日) 简单收益,
  excess = ret − bench_ret,出成交笔超额均值;
- 按信号分母的期望(主指标):exp_ret = Σ(成交 ret,未成交记 0) / 有效信号分母;
  同法出 exp_excess(超额版,次要);
- 聚类稳健 t(唯一显著性口径,Liang-Zeger):
  对按信号分母序列 x(成交 = ret,未成交 = 0)计算;
  聚类键 = 成交者的成交日(fill_date),未成交者取事件日(event_date);
  得分 s_i = x_i − x̄,簇得分和 S_c = Σ_{i∈c} s_i,
  var(x̄) = [G/(G−1)] × Σ_c S_c² / n²(G = 簇数,n = 序列长度),
  t_聚类 = x̄ / √var(x̄)。
  去年同日事件聚集导致独立 t 虚高已有教训,本次预登记即钉死聚类 t 为唯一显著性口径,
  不再出独立 t 作为判活依据。

## 7. 网格与判活线

网格:2 信号(events_v1 / events_v2)× 2 档位(m=0.00 / 0.03)× 3 出场 = 12 行,不扩网格。
另有 6 行对照(m=+∞ × 2 信号 × 3 出场)作为基线行与自检闸门载体。

判活:某行"有戏" ⟺ 同时满足:
1. 按信号分母期望 exp_ret > 0;
2. 聚类稳健 t_聚类 ≥ 2;
3. 比同信号同出场的基线行(m=+∞ 对照行)的 exp_ret 高 ≥ +1 个百分点。
全部 12 行不过线 = 判死,原样报告,阴性结果是合法交付。
严禁为出阳性而事后加码:不改网格、不改口径、不改判活线、不补跑规格外配置。

## 8. 自检闸门(不过则停并报告,不进入判读)

对照变体 m=+∞ 配 H12裸持 出场,其逐笔成交明细必须与
backtest/event_study.parquet 中同信号的 E1-H12 行逐笔一致:
同 (signal, ts_code, event_date) 下 status 全量一致,closed 笔的 ret 容差 1e-9。
附加对拍:m=+∞ 配 A13同款 对 event_study 的 A13 行,同标准。
另断言:全部成交笔 fill_date > event_date(反泄漏)。
任何不一致 = 管线 bug,停止并原样报告,不进入判读。

## 9. 产物布局

- README.md(本文件,先于跑数)。
- run_pullback.py(模拟与汇总脚本)、run_pullback.log(阶段日志与心跳)。
- trades_pullback.parquet(逐笔:signal / m / exit / ts_code / event_date / fill_date /
  fill_price(原始成交价)/ entry_exec_price / exit_date / exit_price(卖出执行价)/ ret /
  excess / held_days / status(含 no_fill 未成交标记及 dropped_* / incomplete)/ 股数与成本明细)。
- summary_pullback.csv(12 行 + 6 对照行,配置为行)。
- verdict.json(自检闸门结果、逐行判活判定、总体结论、耗时)。
