# 信号后追强势入场（实验B）预登记 —— 信号定义不动，只改入场规则

本文件为预登记：先于任何跑数结果落盘写成，逐条钉死假设、机制、成本、出场、网格、指标与判活线。
任务性质：预登记一次性回测，全程禁网络；仓库内除本目录
（experiments/divergence_anchor_eval_2026/strength_follow/）外一律只读；v3_pipeline/ 一字不改。

## 1. 假设与动机

背离v6 金叉对金叉事件池上，两个事实已测死：
事件日 T+1 开盘无脑买（E1-H12 裸持 12 日）笔均 −1.280%（v1）/ −0.849%（v2）；
回踩限价挂单死于逆向选择——跌回来成交的是弱者（成交子集均值 −2.67%），没跌回来的反而约 +1.7%。
本实验把该发现反过来用：不挂低价单，等信号后的走势自己证明强势再追入。
信号定义（events_v1 / events_v2 事件表）一字不动，唯一变量是入场规则。

## 2. 信号源、日历与基线行

- 变体1：experiments/divergence_anchor_eval_2026/events_v1.parquet，5646 事件。
- 变体2：experiments/divergence_anchor_eval_2026/events_v2.parquet，5194 事件。
- 回测窗：2026-01-01..2026-08-31；交易日历与日线源 = strategy_engine.load_market_data
  （000905.SH 指数交易日，stock_data/daily 未复权日线，stock_data/stk_limit 涨跌停表），
  与事件层研究 backtest/event_study.py、实验A pullback_entry/run_pullback.py 同源同窗。
- 本窗口 stk_limit 缺日 = 0（事件层与实验A 均披露为零），脚本内硬断言；不做代理兜底。
- 基线行（判活线③用，任务书登记值，写死不动）：
  同信号"事件日 T+1 开盘买 + E1-H12"扣成本笔均 = v1 −1.280%、v2 −0.849%。
  交叉披露：event_study.parquet 内复算的精确值为 v1 −1.2869%、v2 −0.8536%
  （event_study_summary.csv，n=5164/4814）；判活一律用任务书登记值，精确值仅供对照。
- 本实验内另设 k=0 对照行（信号后 T+1 开盘直接买、E1-H12 出场）逐条复刻事件层入场段，
  作为自检闸门载体（见第 8 节），证明自管线与事件层语义逐位一致。

## 3. 入场规则（本实验唯一变量）

记号：事件日 e（金叉日，收盘后信号已知）；entry_close_ref = e 日收盘价（取自日线）；
日历下标 di_ev = e 在交易日历中的位置；k 为观察窗交易日数。

### B1 不破位确认

- 观察窗 = 日历下标 [di_ev+1, di_ev+k]（k 个交易日）。
- 窗内最低低价 W_low = min(low)，仅统计当日有日线的日子（无行情日不评估，与引擎哲学一致）。
- 破位判定：W_low < entry_close_ref × (1−d) − 1e-9 即记破位（容差方向钉死：
  恰好触及阈值不算破位）；窗内零个有行情日 → 无法确认，记 skipped_no_window_data。
- 未破位 → 入场日 = 日历下标 di_ev+k+1，当日开盘买入。
- 破位 → 本信号不交易，记 skipped_breach。
- 日历不足（di_ev+k+1 ≥ 日历长度，即观察窗不完整或入场日不存在）→ dropped_no_next_day。

### B2 突破确认

- 参考窗 = 日历下标 [di_ev+1, di_ev+k]；H_ref = max(high)，仅统计有行情日；
  零个有行情日 → skipped_no_window_data；参考窗被日历末端截断（di_ev+k ≥ 日历长度）→ dropped_no_next_day。
- 触发窗 = 日历下标 [di_ev+k+1, di_ev+k+5] 与日历的交集（其后 5 个交易日内）。
- 触发判定：触发窗内首个有行情且 收盘价 > H_ref + 1e-9 的交易日触发
  （容差方向钉死：恰好触及 H_ref 不算突破）；入场日 = 触发日的次一交易日，当日开盘买入。
- 5 日内未触发 → 本信号不交易，记 skipped_no_trigger
  （触发窗被日历末端截断仍按未触发计，截断笔数以 n_trig_window_truncated 单独披露）。
- 触发日为日历最后一日（无次一交易日）→ dropped_no_next_day。

### 对照 k=0（自检闸门载体）

- 逐条复刻事件层 simulate_event 入场段，判定次序完全一致：
  dropped_no_next_day（e 为日历最后一日）→ dropped_no_quote（T+1 无行情）
  → dropped_limitup（T+1 开盘 ≥ 涨停价 − 1e-9，不递补）→ dropped_no_close_T（e 日收盘缺失）
  → 整手现金段。入场日 = di_ev+1，开盘买入，出场仅配 E1-H12。

### 入场执行（三 flavor 共用，逐条复刻冻结口径）

- 入场日必须当日有日线，否则 dropped_no_quote，不顺延、不递补。
- 当日开盘 ≥ 涨停价 − 1e-9（涨停价取 stk_limit 当日表，与引擎同源）→ dropped_limitup，不递补。
- 买入执行价 = 当日开盘 × 1.001（滑点单边 0.1%）。
- 每笔固定名义本金 100,000 元；股数 = int(100000 / 执行价 / 100) × 100；
  不足一手买一手；现金（含佣金）不足逐手递减；归 0 记 dropped_cash。

## 4. 成本（与 event_study 逐位一致，直接调用 v1 冻结引擎 strategy_engine 原函数）

- 滑点双边各 0.1%：买入执行价 = 原始价 × 1.001；卖出执行价 = 原始价 × 0.999。
- 佣金双边万 2.5，单笔最低 5 元（strategy_engine.buy_cost / sell_costs 原函数）。
- 印花税卖出 0.05%（本窗口全在 2023-08-28 新税率区间）。
- 净收益 ret = 净盈亏 /（股数 × 买入执行价 + 买入佣金），与引擎 trades.ret 同口径。

## 5. 出场（自入场日 f 起算，入场日记持有第 1 日，T+1 起方可卖）

每格各配两种出场，日内触及语义与 event_study / strategy_engine_v3 fixed_tp_sl 逐字一致
（high ≥ 屏障 − 1e-9 触止盈；low ≤ 屏障 + 1e-9 触止损；同日双触发保守取止损；
开盘价越过屏障按开盘价、否则按屏障价；到期当日收盘卖；
触发日收盘 ≤ 跌停价 + 1e-9 顺延至首个可卖日；无行情日不评估不触发、持有天数按日历下标差照常累计）：

| 出场名 | 规格 |
|:---|:---|
| E1-H12 | 无屏障，持有满 12 个交易日当日收盘卖（引擎 E1 模式，等价 event_study 的 E1-H12 行） |
| A13 | 止盈 = 买入执行价 × 1.25；止损 = 买入执行价 × 0.86；最长 12 日（strategy_engine_v3 ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=12) 冻结定义） |

窗口末端仍无法完整出场的记 status=incomplete，从收益统计剔除并披露（与事件层口径一致）。

## 6. 指标口径（每行）

- n_signals = 事件表内经日历过滤后的信号总数（与事件层同口径）。
- n_trades = status ∈ {closed, incomplete} 的笔数（真实入场了的）；
  trade_rate = n_trades / n_signals（任务书口径，分母为全部信号）。
- ret_mean / ret_median / win_rate（ret > 0）：仅对 status=closed 笔计算（扣成本笔均）。
- 超额（披露用，不作判活依据）：bench_ret = 000905.SH 收盘(入场日) → 收盘(出场日) 简单收益，
  excess = ret − bench_ret，出成交笔超额均值。
- excess_vs_baseline = ret_mean − 基线（v1 −0.01280 / v2 −0.00849），判活线③的度量。
- cluster_t（唯一显著性口径，Liang-Zeger，与 ranking_probe/pullback 预登记同式）：
  对 closed 笔的 ret 序列 x 计算；聚类键 = 入场日 entry_date；
  得分 s_i = x_i − x̄，簇得分和 S_c = Σ_{i∈c} s_i，
  var(x̄) = [G/(G−1)] × Σ_c S_c² / n²（G = 簇数，n = 序列长度），
  cluster_t = x̄ / √var(x̄)；n < 2 或 G < 2 或 var ≤ 0 记 NaN（该行判活线②自动不过）。
  不出独立 t 作为判活依据。
- 被跳过子集刻画：skipped_subset_fwd_mean = 规则跳过子集
  （B1 取 skipped_breach，B2 取 skipped_no_trigger；skipped_no_window_data 属数据缺陷不计入，
  笔数单独披露）在 event_study.parquet 同信号 E1-H12 status=closed 行上的 ret 均值
  （只测量、不交易；即"若无脑 T+1 买，这些被跳过的信号会赚多少"），
  并披露跳过笔数 skipped_n 与命中 event_study closed 行的笔数 skipped_matched_n。
  该指标用于验证"跳过的是不是果然是弱者"，不进判活线。
- 全部状态计数原样披露：closed / incomplete / skipped_breach / skipped_no_trigger /
  skipped_no_window_data / dropped_no_next_day / dropped_no_quote / dropped_limitup /
  dropped_no_close_T / dropped_cash，及 n_window_gap（观察/参考窗内存在无行情日但仍完成评估的笔数）。

## 7. 网格与判活线（写死，不许移动球门）

网格（不扩网格）：2 信号 ×（B1：k ∈ {3,5,8} × d ∈ {0.00, 0.03} 共 6 格；
B2：k ∈ {3,5,8} 共 3 格）× 2 出场（E1-H12 / A13）= 每信号 18 格，两信号共 36 行；
另有每信号 1 行 k=0 对照（仅自检与展示，不参与判活）。全部 36 格原样出数，死格也报。

某格判活需同时满足：
1. 扣成本笔均 ret_mean > 0；
2. cluster_t ≥ 2；
3. ret_mean ≥ 同信号 E1-H12 基线 + 2 个百分点（基线写死：v1 −1.280%、v2 −0.849%）。
否则判死；36 格全死则 B 方向判死，阴性结果原样交付。
严禁为出阳性而事后加码：不改网格、不改口径、不改判活线、不补跑规格外配置。

## 8. 自检闸门（零容差，不过则停并报告，不进入判读）

每信号的 k=0 对照行（T+1 开盘买、E1-H12 出场）必须逐笔复现
backtest/event_study.parquet 中同信号 E1-H12 的结果：
同 (signal, ts_code, event_date) 下事件集合一致、status 全量一致、
status=closed 笔数一致、closed 笔的 ret 与 excess 最大绝对差 < 1e-9、入场/出场日期全等。
另断言：全部入场笔 entry_date > event_date（反泄漏）；
B1/B2 全部入场笔 entry_date 日历下标 ≥ di_ev + k + 1（确认窗在入场前，反泄漏）。
任何不一致 = 管线 bug，停止并原样报告，不出任何结果数。

## 9. 产物布局

- README.md（本文件，先于跑数）。
- run_strength.py（模拟与汇总脚本）、progress.log（阶段日志与心跳，每主步骤一行带时间戳，
  开头写预计总时长）。
- trades_strength.parquet（逐笔全量：signal / flavor / k / d / exit_rule / ts_code / event_date /
  entry_date / entry_price_raw / entry_exec_price / shares / 成本明细 / exit_date / exit_reason /
  exit_raw_price / exit_exec_price / ret / bench_ret / excess / held_days / deferred_days /
  status 全分类 / 诊断字段 W_low、breach_threshold、H_ref、trigger_date）。
- summary_strength.csv（配置为行，36 格 + 2 对照行，列含任务书指定全列 + 披露列）。
- verdict.json（自检闸门结果、逐格判活判定、总体结论、耗时）。

## 10. 预计耗时

参照实验A（同机同数据，18 配置 22 秒）：本实验 38 配置 + 每信号一次数据加载，
预计总时长约 5 分钟，进度以 progress.log 心跳为准。
