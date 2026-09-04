# 急跌背离种子全历史审判 —— 预登记(先于任何跑数落盘)

实验日期:2026-09-04。
性质:把 2026 沙盒地形扫描中挑出的"优秀种子"(信号条件组合)冻死,原封不动拿到 2007-01~2026-08 全历史上审判。
种子集合在本文件冻结,出数后不许增删改。

## 0. 显著披露(先于一切设计)

1. **种子来自 2026 沙盒的探索性扫描,是"看图挑出来的"**。
   本实验的价值在于:种子在见到全历史数据之前冻结,全历史是它们从没见过的考场。
   5 颗种子 + 1 个全池对照,多重比较范围冻结为这 6 行,全部出数,不许只报过的。
2. **数据加载自写,成本原语用冻结引擎**。
   冻结引擎 load_market_data 按全市场窗口加载,20 年 × 6000 股内存不可行。
   个股日线直接从 stock_data/daily 逐股 parquet 读(与引擎同源同口径,不复权)。
   成本、滑点、整手仍调 strategy_engine 的 buy_cost / sell_costs / SLIPPAGE / BOARD_LOT。
3. **退市股处理**:stock_data/daily 含已退市股票,全历史扫描天然含它们(反幸存者偏差)。
   若事件后数据不足 +H 个交易日(含退市),记 truncated,计数披露,不进收益统计。
   这是已知近似:真实退市可能伴随无法卖出的损失,本实验按"没买成"处理,结论偏乐观方向,披露于此。
4. **涨跌停判定用 stock_data/stk_limit 逐日文件**(2007-01-04 起),缺文件日视为无涨跌停约束,缺文件天数披露。

## 1. 研究问题

2026 沙盒里"信号前急跌 + 信号日刚起弹"的种子组合,在 2007~2026 全历史(十几轮牛熊)上,
扣完成本后,短线(10/20/25 交易日)持有还赚不赚钱?

## 2. 事件扫描(背离v6,全历史)

- 变体1(主判):复用 experiments/divergence_anchor_eval_2026/scan_v1.py 的口径,逐字不改,
  唯一改动=去掉事件日 2026 窗口过滤,保留全历史全部事件。
  自检硬断言:全历史事件中 event_date ∈ 2026-01-01~2026-08-31 的子集,必须与既有 events_v1.parquet 逐行完全一致(5646 行,全字段)。
- 变体2(参照):复用 scan_v2.py 同理,2026 子集须与 events_v2.parquet 逐行一致(5194 行)。
- 个股过滤沿用扫描器既有口径:含 ts_code 且含 vol 列、行数 ≥100。
- 输出 events_history_v1.parquet / events_history_v2.parquet。

## 3. 种子条件(事件日收盘含之前可知,冻结)

记号:j = 事件日在该股全历史序列下标,close_ev = close[j]。
窗口语义与 2026 地形扫描逐字一致:前20日窗口 = close[j−20 .. j] 共 21 根 K 线(含事件日)。

- dd20 = close_ev / max(close[j−20 .. j]) − 1(≤0,信号前急跌深度)。
- bounce = close_ev / anchor_close − 1(≥−1,信号日已弹幅度;anchor_close 为事件表列,锚点在事件日之前,因果干净)。

种子集合(5 颗 + 对照,每颗对事件输出接受/拒绝):

| 种子 | 条件 |
|---|---|
| S1 | dd20 ≤ −0.15 且 0.02 < bounce ≤ 0.08 |
| S2 | dd20 ≤ −0.20 且 0.02 < bounce ≤ 0.08 |
| S3 | dd20 ≤ −0.25 且 0.02 < bounce ≤ 0.08 |
| S4 | dd20 ≤ −0.15(无 bounce 条件,用户原始假说) |
| S5 | dd20 ≤ −0.25(无 bounce 条件) |
| ALL | 全池对照,不筛选 |

## 4. 交易机制(与已复核的 2026 口径逐条一致)

- 事件日下一交易日(个股序列下一行)开盘买。
- 开盘无报价 → dropped_no_quote;开盘价 ≥ 当日涨停价 − 1e-9 → dropped_limitup(计数不递补)。
- 10 万本金:px = open×(1+SLIPPAGE);sh = int(100000/px/BOARD_LOT)×BOARD_LOT,不足一手取一手;
  while sh>0 且 sh×px + buy_cost(sh,px) > 100000+1e-6:减一手;减到 0 → dropped_cash。
- 入场日记持有第 1 日;第 H 个交易日(H ∈ {10, 20, 25},按个股序列行号计,停牌日行计入但不评估)收盘卖。
- 收盘 ≤ 当日跌停价 + 1e-9 → 顺延至下一非跌停收盘日;数据耗尽 → truncated(计数,不进收益统计)。
- 净收益 = (sh×(xs−px) − 买佣 − 卖佣 − 印花税)/(sh×px + 买佣),xs = 卖收盘×(1−SLIPPAGE)。
- 模拟与选择解耦:每事件每 H 模拟一次,再按种子 join 出每格。

## 5. 格子(配置为行,全出数)

变体 {v1, v2} × 种子 {S1..S5, ALL} × H {10, 20, 25} = 36 行。
每行:n_universe, n_selected, n_closed, n_dropped(limitup/no_quote/cash), n_truncated,
净笔均, 净中位, 胜率, cluster_t(cluster=入场日,Liang-Zeger:G<2 记 NaN),
逐年净笔均(每年一行进 report,summary 里给 盈利年占比 = 有≥30 笔成交的年份中净笔均>0 的占比),
日期集中度(top5 入场日净盈亏合计 / 全部净盈亏合计,总净盈亏≤0 时记 NaN)。

## 6. 判活线(预登记,不移动球门)

每种子只在 v1 上宣判(v2 同跑参照):
过线 = 同一 H 下五条全满:

1. n_closed ≥ 300;
2. 净笔均 > 0;
3. cluster_t ≥ 2;
4. 盈利年占比 ≥ 60%(防单一年份/单一行情撑起);
5. 日期集中度 ≤ 50%(排序探针教训:防 top 几天撑起全部利润)。

任一种子在任一 H 过线 → "该种子活",给出定版候选;
全部不过 → "2026 沙盒的肉是全历史不存在的运气",该方向封档。
v2 行全出数但只作参照披露。

## 7. 自检(全过才可出报告)

1. 扫描一致性:全历史事件的 2026 子集与 events_v1/v2.parquet 逐行一致(硬断言)。
2. 交易因果:全部成交 entry_date > event_date,exit_date > entry_date。
3. 计数守恒:每格 n_selected ≤ n_universe;ALL 行 n_selected = n_universe;
   n_selected = n_closed + n_dropped + n_truncated。
4. 随机抽 5 笔成交(固定种子 20260904,不同年代)打印完整生命周期供人工抽查。
5. stk_limit 缺文件天数披露;truncated 按变体×H 计数披露。

## 8. 纪律

- 阴性结果原样交付;所有格子全出数,配置为行。
- 全程 progress.log 心跳,每大步一行带时间戳。
- 禁网络、禁 gh、禁 git;写文件仅限本目录,仓库其余只读。

## 9. 交付物(本目录内)

README.md(本文件)、progress.log、run_seeds.py、
events_history_v1.parquet、events_history_v2.parquet、
trades_seed.parquet(每事件×变体×H 一行 + 各种子选择布尔列)、
summary_seed.csv(36 行)、verdict.json(逐种子宣判)、report.md(全表+逐年表+自检+宣判,阴性原样)。
