# 阶段底收益天花板测量 —— 预登记（先于任何跑数落盘）

日期：2026-09-04。
性质：预登记纯测量实验，A 方案（背离信号降级为状态标签、吃 60~120 日阶段反转）的生死闸门。
不改任何信号定义，不做策略调优，不做参数搜索。

## 显著警告（防误读，必读）

**本实验不是可交易策略的回测。**
分组标签（near/far）是事后信息：一个事件的锚点是否贴着阶段底，要用事件日之后 W 个交易日的数据才能确认。
入场、持有、出场完全因果（事件日下一交易日开盘买、持满 H 个交易日收盘卖、跌停顺延），
但分组本身不可因果获得。
因此本实验测的是 A 方案的**收益天花板上界**：假设存在完美识别器能在事件日认出"锚点贴阶段底"，
吃因果入场持有能赚多少。
**若完美识别（上界）扣完成本都不赚钱，则任何因果识别规则都不可能赚钱，A 方案封档。**
反之，天花板存在仅是必要条件，不是 A 可行的充分条件。

## 研究问题

若能在事件日完美认出"锚点贴阶段底"的事件，从因果入场点买入并持有 60/120 个交易日，
扣完成本（佣金、印花税、双边滑点、整手、10 万本金约束）赚不赚钱？

## 输入（全部已存在并经过独立复核，只读使用）

- 事件表（背离v6 金叉对金叉）：
  - `../events_v1.parquet`：变体1（区间最低价锚定），5646 行。
  - `../events_v2.parquet`：变体2（右侧确认精确锚定），5194 行。
  - 列含 ts_code, event_date, anchor_date, anchor_close 等；event_date/anchor_date 为日期。
- 阶段底基准（±W 交易日窗口收盘最低且严格低于前 W 日，并列取最早；数据末尾 2026-08-31）：
  - `../stage_bottom_eval/stage_gt_w60.parquet`（1008 低点 / 1006 股）。
  - `../stage_bottom_eval/stage_gt_w120.parquet`（37 低点）。
  - 列：ts_code, low_date, low_close。
- 冻结引擎 `v3_pipeline/scripts/strategy_engine.py`（只读 import 原语）：
  `load_market_data` / `buy_cost` / `sell_costs` / `SLIPPAGE=0.001` / `BOARD_LOT=100` / `PRICE_TOL=1e-9`。
  回测循环逻辑本实验自己写，不复用任何既有回测实现。
- 行情窗口：load_market_data(codes, "2026-01-01", "2026-08-31", REPO)。
  注意引擎日线向前多取 60 个自然日（实际自 2025-11-02 起），该股序列含此缓冲段。

## 分组规则（每股日线序列 = load_market_data 返回的该股 DataFrame 索引序列，按位置计下标）

- 可评分性：事件日对应下标 i_ev 满足 i_ev < len(该股序列) − W
  （与 stage_bottom_eval 截断同口径：事件日 < dates[len−W]）。
  不满足 → unscorable_truncated，单列披露，不进格子。
- i_anchor = anchor_date 在该股序列中的下标。
  anchor_date 不在序列中（如序列起点 2025-11-02 之前的锚）→ unscorable_anchor，单列披露，不进格子。
- 事件日不在该股序列中 → unscorable_no_event_row；该股无序列 → unscorable_no_series；均单列披露。
- dist = min over 该股该窗口（W ∈ {60, 120}）全部阶段底低点 |i_anchor − i_low|。
- dist ≤ 10 → near 组；dist > 10 → far 组；pool 行 = 全部可评分事件（near+far 合并，参照行）。
- 该股在该窗口无任何阶段底低点 → no_gt 披露，不进 near/far/pool。

## 入场（因果，与冻结引擎既有口径一致）

- 事件日下一交易日（日历）开盘买。
- 开盘无报价 → dropped_no_quote；开盘价 ≥ 涨停价 − 1e-9 → dropped_limitup（均计数披露，不递补）。
- 仓位：px = open×(1+SLIPPAGE)；sh = int(100000/px/BOARD_LOT)×BOARD_LOT；
  sh < BOARD_LOT 则取一手；
  再 while sh > 0 且 sh×px + buy_cost(sh, px) > 100000 + 1e-6：sh 减一手；
  sh = 0 → dropped_cash。

## 出场（因果，与冻结引擎持有日口径一致）

- 入场日记持有第 1 日；持有至第 H 个交易日（H ∈ {60, 120}，按日历交易日计，停牌日计入持有日但不评估）收盘卖。
- 若当日收盘 ≤ 跌停价 + 1e-9 → 顺延至下一非跌停收盘日（连续跌停连续顺延）。
- 数据耗尽仍无法卖 → incomplete（计数披露，不进收益统计）。
- 净收益 = (sh×(xs−px) − 买佣 − 卖佣 − 印花税) / (sh×px + 买佣)，xs = 卖出收盘×(1−SLIPPAGE)。
- 毛收益 = xs/px − 1（仅披露用）。

## 格子（配置为行，全出数）

变体 {v1, v2} × 窗口 W {60, 120} × 组 {near, far, pool} × H {60, 120} = 24 行。

每行：n_scorable, n_traded（= 成交含 incomplete）, n_dropped（分因：limitup / no_quote / cash）,
n_incomplete, 净笔均, 净中位, 胜率, cluster_t。

cluster_t：Liang-Zeger 聚类稳健 t，cluster = 入场日。
score = 各笔 (ret − 全组均值) 按入场日求和得 S_c；
var(均值) = [G/(G−1)] × ΣS_c² / n²；t = 均值 / √var。G < 2 记 NaN。

## 主判据（预登记宣判，出数后不许移动球门）

- 主格 = 变体 × W60 × near × H60（两变体各宣判一次）：
  净笔均 > 0 且 cluster_t ≥ 2 → "天花板存在"（A 值得进入识别规则设计）；
  否则 → "天花板不存在"（A 封档依据）。
- 次要佐证：W60 × near × H120（样本进一步截断，n 披露）。
- 参考读数：W120 全部格（可评分事件仅约 19/17，不作判据）。
- far 组与 near−far 差值作为对照披露；pool 行作为全池参照披露。

## 自检（全部通过才可出报告，写进 report.md）

1. 因果断言：全部成交笔 entry_date > event_date，exit_date > entry_date；违反数必须为 0。
2. 计数守恒：每（变体 × W）下 near + far 的可评分事件数 = pool 行可评分数；
   并与 stage_bottom_eval 既有数字对齐披露（W60：v1 可评分 410、v2 可评分 402；W120：v1 19、v2 17；
   允许因 anchor 不在序列、该股序列过滤口径差异等有小幅出入，但必须解释每一笔差异来源）。
3. 随机抽 5 笔成交（固定种子 20260904）打印完整生命周期
   （事件日/锚日/分组/dist/入场日/入场价/出场日/出场价/毛收益/各项成本/净收益）供人工抽查。
4. 引擎原语之外不复用任何既有回测实现；本脚本独立成文。

## 纪律

- 阴性结果原样交付，不许粉饰；所有格子全出数，配置为行。
- 全程 progress.log 心跳日志，每主要步骤一行带时间戳。
- 禁网络、禁 gh、禁 git 操作；写文件仅限本目录。

## 交付物（全部在本目录）

README.md（本文件）、progress.log、run_ceiling.py、
trades_ceiling.parquet（每事件 × 变体 × W × H 一行：含状态、分组、dist、出入场日期价格、毛/净收益）、
summary_ceiling.csv（24 行配置为行）、verdict.json（主格宣判 + 数字 + 理由）、
report.md（全表 + 自检结果 + 宣判，阴性原样）。
