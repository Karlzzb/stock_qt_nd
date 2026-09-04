# 预判金叉挂买（两档设计）事件级模拟 —— 预登记

本文件为预登记：先于任何跑数落盘写成，逐条钉死口径。
任务性质：预登记一次性实验，全程禁网络；仓库内除本目录（experiments/divergence_anchor_eval_2026/predictive_entry/）外一律只读。
执行解释器：.venv/bin/python。

## 1. 背景与核心数学

背离事件 = 相邻 DIF×DEA 金叉对（金叉日 t：DIF[t-1] ≤ DEA[t-1] 且 DIF[t] > DEA[t]，两者均非 NaN），dif_lift ≥ 0.001 + 价格新低条件（两变体口径见 scan_v1.py / scan_v2.py，冻结不改）。
MACD 只用收盘价（talib 默认 12/26/9），因此 t 日收盘后可精确解出"明日收盘价站上多少即金叉"的临界价 c*，全部背离条件可在前夜审完，唯一未知量是明日收盘价。

数学推导（钉死）：
- 明日收盘价 c 满足 DIF(c) = A + B·c，其中 A = EMA12_t×(11/13) − EMA26_t×(25/27)，B = 2/13 − 2/27 = 28/351 > 0。
- DEA(c) = 0.8×DEA_t + 0.2×DIF(c)。
- 金叉 ⟺ DIF(c) > DEA(c) ⟺ DIF(c) > DEA_t ⟺ c > c*，c* = (DEA_t − A)/B。
- 推论：若明日金叉，金叉处 DIF = 今晚 DEA_t（记 dif_star）。
- 明日金叉 ⟺ DIF_t ≤ DEA_t 且 close[t+1] > c*。

### 1.1 EMA 种子口径（关键工程事实，跑数前已数值验证）

talib.MACD 的内部快慢 EMA 种子与 talib.EMA 不同：
- 内部慢 EMA26 在行号 25 处以 close[0:26] 的 SMA 播种；
- 内部快 EMA12 同在行号 25 处播种，种子为 close[14:26] 的 SMA（而非行号 11 起递归）；
- DEA 在行号 33 处以 DIF_int[25:34] 的 SMA 播种；DIF 对外输出自行号 33 起（25..32 行内部参与 DEA 递推但不输出）。
本实验以该口径自行复算内部 EMA12/EMA26（scipy lfilter 实现，zi 携带种子），并以"复算 DIF/DEA 与 talib 输出之差 < 1e-9"作为全市场硬断言的一部分。
600283.SH / 000001.SZ / 300750.SZ / 688981.SH / 002230.SZ / 601212.SH 六票预验证：复算 vs talib 最大差 ≤ 5.2e-13。

## 2. 硬不变量断言（全市场全历史逐条验证，不过即停）

对每只股票全历史每一日 t（EMA/DIF/DEA 有效处）：
1. 对每个真实金叉日 t+1：用 t 日状态算的 c* 满足 close[t+1] > c*；
2. 对每个非金叉日 t+1（且 DIF_t ≤ DEA_t）：close[t+1] ≤ c* + 1e-9；
3. 线性重算 DIF_lin = A_t + B·close[t+1] 与 talib 实际 DIF[t+1] 之差 < 1e-9。
任一股票任一断言失败即停止并原样报告；验证总量与失败数登记于最终报告。

## 3. 前夜候选枚举（预判扫描器，反泄漏核心）

每股全历史因果推演。
对 eve 日 t（trade_date ∈ [2026-01-01, 2026-08-31]），若同时满足：
- DIF_t ≤ DEA_t（尚未金叉），且 t+1 在该股数据内；
- 已有最近两个金叉 C(k−2)、C(k−1)（金叉日行号均 ≤ t−1，天然 < t+1；eve 日本身不可能是金叉日，因 DIF_t ≤ DEA_t）；
- dif_lift_star := DEA_t − DIF[C(k−1)] ≥ 0.001（含义：若明日金叉，金叉处 DIF = DEA_t，抬升即此值）；
- 价格条件（前夜 provisional 口径，与确认口径差异见第 4 节）：
  - 变体1：min close over (C(k−1), t] < min close over (C(k−2), C(k−1)]（左开右闭按行号，严格新低）；
  - 变体2：provisional 锚 = (C(k−1), t] 内确认局部低点（左3右2，仅采纳确认日 s+2 ≤ t 的低点；沿用 scan_v2 的相邻保留低点间隔 ≥ 3 贪心过滤；无确认低点则回退为区间最低收盘价日，并列取最早）的收盘价 < anchor(k−1) 收盘价；
则记一个前夜候选，字段含 ts_code、eve_date=t、target_date=t+1（该股下一根实际 K 线）、c*、margin_ratio = c*/close[t] − 1、dif_star = DEA_t、所属变体、两金叉日期与 DIF、变体2 的双锚日期与收盘价。
同一股票连续多日候选允许（金叉迟迟不叉则每晚重算），各候选独立成单。

## 4. provisional 口径与确认口径（scan_v1/v2 冻结版）的差异披露

1. 变体1 确认口径比较 (C(k−1), C(k)] 与 (C(k−2), C(k−1)] 的区间最低收盘价；provisional 把当前区间右端从（未知的）C(k) 截到 eve 日 t。若 t 之后才出现更低收盘价，确认口径可能成立而 provisional 不成立，反之不成立（右端截断只可能让 min 偏大，即 provisional 通过 ⟹ 确认口径同窗口段价格条件亦已通过；但最终事件还要求 dif_lift（=确认后 DIF[C(k)]−DIF[C(k−1)] = dif_lift_star，一致）与 event_date 窗口）。
2. 变体2 确认口径当前锚窗口为 (C(k−1), C(k)+5个交易日]（含金叉后 5 日）；provisional 无金叉日可言，窗口为 (C(k−1), t]，不向右延伸。
3. 变体2 确认口径的低点确认依赖右侧 2 日；provisional 只采纳确认日 s+2 ≤ t 的低点（eve 当日晚可知）。
4. anchor(k−1) 按 scan_v2 原窗口 (C(k−2), C(k−1)+5] 口径计算，但做因果截断：窗口右端截到 min(C(k−1)+5, t)，低点确认同样只采纳 s+2 ≤ t 者。当 t ≥ C(k−1)+7 时与 scan_v2 完全一致；更早的 eve 使用截断版（前夜真实可知）。此差异逐条披露。
5. 事件日定义（确认口径 event_date = max(C(k), anchor(k)+2)）在预判场景无对应物：预判单在 target_date 即入场，不等 anchor+2。这是预判设计与确认基线的结构性差异，配对比较（第 8 节）按"同一金叉日"对齐。

## 5. 两档挂法（事件级逐笔模拟）

名义本金 100,000 元/笔（与 backtest 事件层口径一致），单候选独立现金池。
成本口径与引擎逐条一致：佣金双边万 2.5（单笔最低 5 元）、印花税卖出 0.05%（窗口全在 2023-08-28 后）、整手 100 股（int(预算/执行价/100)×100；不足一手置一手；现金不足逐手递减至 0 记 dropped_cash）、价格容差 1e-9、基准 000905.SH。

主边际参数 m = 1%（另报 m = 0%、m = 3% 敏感性行）：仅当 c* ≤ close[t]×(1−m) 才出手，否则放弃记 skipped_margin（逐配置披露数量，不产生交易不产生成本）。

### 设计1（前夜决策·目标日开盘买）

- target_date = t+1（该股下一根实际 K 线）开盘市价买入，执行价 = open×1.001（滑点照收）。
- 开盘涨停（open ≥ up_limit − 1e-9）拒买记 dropped_limitup，不递补；stk_limit 当日文件缺失视为无约束并计数。
- 确认：close[t+1] > c* → 金叉成立，按出场规则持有（第 6 节）；
- 失败：close[t+1] ≤ c* → t+2（日历次日）开盘认错卖出，执行价 = open×0.999，佣金印花税照收；触发日收盘 ≤ 跌停价 + 1e-9 则顺延至首个可卖日（deferred_days 计数）；无行情日跳过不计顺延；日历末端仍无法卖出记 incomplete 剔除并披露。

### 设计2（临界价限价单）

- t+1 在 c* 挂限价买单。成交规则（日线数据，顺序判定）：
  1. 开盘涨停锁死（open ≥ up_limit − 1e-9）→ 不成交，记 unfilled；
  2. open ≤ c* → 按 open 成交（优于限价，按更优价成交）；
  3. 否则 low ≤ c* → 按 c* 成交；
  4. 否则不成交，记 unfilled（无交易无成本，不递补）。
- 限价成交不加滑点（理由：限价单成交价格上限已被 c* 锁定，open 成交取实际 open 已是保守/真实更优价；对其再加 0.1% 滑点会系统性高估买入价，与限价单机制矛盾；滑点本质是市价单的价格不确定补偿，限价单无此不确定性）。佣金、印花税照收。
- 确认/失败判定同设计1；失败单 t+2 开盘卖出（口径同设计1，含跌停顺延与 incomplete 剔除）。

## 6. 确认单出场规则（与 backtest 事件层同口径，逐条对齐 README 增补 D 节）

- E1-H12：无屏障，持有满 12 个交易日（entry 日记第 1 日，held = 日历下标差 + 1，按 000905.SH 交易日历）当日收盘卖。
- A13：tp = 入场执行价×1.25，sl = 入场执行价×0.86；T+1 起逐日评估（买入日不评估）；tp 独中 → open 越屏障按 open 否则屏障价；sl 中（含同日双触发，保守取止损）→ open 破屏障按 open 否则屏障价；held ≥ 12 → 当日收盘卖。
- 跌停顺延：触发日 close ≤ down_limit + 1e-9 → 顺延至首个可卖日；无行情日不评估、持仓沿用。
- 卖出执行价 = raw×0.999；佣金 = max(5, 成交额×0.00025)；印花税 = 成交额×0.0005。
- 日历末端（2026-08-31）仍无法完整出场记 incomplete，剔除出统计并披露数量，不盯市（与 backtest 事件层同口径）。

## 7. 基线（对照行）与对拍

同脚本内重算基线：events_v1 / events_v2 全量事件，event_date 次日（000905.SH 日历）开盘买（×1.001），出场 E1-H12 / A13，逐笔口径完全复用 backtest/event_study.py 的 simulate_event（import 复用，不复制实现，保证逐位同源）。
与既有 backtest/event_study.parquet 对拍：
- 每 (signal, config ∈ {A13, E1-H12}) 总行数与 closed 笔数一致；
- closed 笔按 (ts_code, event_date) 配对，|Δret| < 1e-9 的比例 > 99%（预期 100%；不一致逐笔披露原因）。
不通过则在最终报告原样披露。

## 8. 指标与汇总表

基准超额：bench_ret = 000905.SH close(entry_date) → close(exit_date) 简单收益，excess = ret − bench_ret（确认单与失败单同口径）。
t 值：单样本 t = mean/(std/√n)，对合并净收益 ret（确认+失败合并）。
主汇总表（配置为行：变体 × 设计 × 出场，m = 1%）列：
总候选数、skipped_margin、unfilled（设计2）、dropped_limitup（设计1）、dropped_cash、成交笔数（确认/失败分列）、确认率（= 确认/(确认+失败)）、笔均净收益（确认单 / 失败单 / 合并三列）、合并胜率、合并笔均超额、t 值、incomplete 数、顺延事件数。
入场价改善（配对列）：同一确认事件的设计入场执行价 vs 基线入场执行价（基线 = 同 (ts_code, 金叉日) 确认事件的次日开盘价×1.001），改善 = (基线价 − 设计价)/基线价，报均值/中位，并披露配对笔数与未配对笔数（预判确认但确认口径未成事件者）。
m = 0% / 3% 敏感性行只报：合并笔均净收益、确认率（附成交笔数）。
基线行（design = baseline_next_open）同表列出 n、ret_mean、win_rate、excess_mean、t 值，供直接对照。
失败单计入规则：失败单（金叉未成立、认错卖出）计入合并净收益、合并胜率、合并超额与 t 值；skipped_margin / unfilled / dropped_* / incomplete 均不计入任何收益统计，仅数量披露。

## 9. 已知结构性口径差异（披露）

1. target_date 取该股下一根实际 K 线（扫描条件"t+1 在数据内"即此义）；基线入场日取 000905.SH 日历次日（无行情记 dropped_no_quote）。若个股在日历次日停牌，预判单顺延到其实际复牌日入场，基线则丢弃；两口径差异天然存在，配对改善列按各自实际入场价计算。
2. 预判确认的金叉未必成为确认口径事件（变体2 锚可能在 +5 窗口内右移导致价格条件反转，或 anchor+2 出窗），确认口径事件也未必有对应前夜候选（provisional 右端截断偏严）。两集合不对称，逐笔披露配对/未配对数。
3. trades_predictive.parquet 中失败单在三个 margin × 两个出場下重复成行（失败与出场规则无关，为汇总表按行直接可groupby而冗余存储），确认单按 margin × 出场各一行；解读时以 (variant, design, exit, margin, ts_code, eve_date) 为行键。
4. 设计2 的 low ≤ c* 成交判定使用日线 low，无法区分"盘中触及但未成交"的极端微观结构情形，属日线模拟固有近似，与既有事件层口径的近似同级。

## 10. 产物布局（本目录）

- README.md（本预登记）
- predictive_scan.py（不变量验证 + 前夜候选枚举）、predictive_scan.log
- candidates.parquet（全部前夜候选，含未过边际闸门的）
- run_predictive.py（两设计 × 两出场 × 三 margin 逐笔模拟 + 基线重算对拍 + 汇总）、run_predictive.log
- trades_predictive.parquet（逐笔：variant/design/exit/margin/确认与否/入场价/出场/净收益/超额/持有天数/status）
- summary_predictive.csv（主表 m=1% 全列 + 敏感性行 m=0%/3% 简列 + 基线行）

## 11. 自检项（执行后登记于最终报告）

1. 三条数学不变量全市场验证数量与失败数（须 0 失败）。
2. 基线对拍一致率（须 > 99%）。
3. 校准抽查：600283.SH 金叉 2026-07-29 前夜 2026-07-28 的 c*、dif_star = DEA_t，与实际 close/DIF 对照。
4. 披露数字：skipped_margin / unfilled / dropped_limitup / dropped_cash / incomplete / 顺延事件数。
5. 全程禁网络。
