# V1(最早背离设计)按 #31 口径重跑统计 —— 预登记(先于任何跑数落盘)

实验日期:2026-09-07。
依据:GitHub issue #34(本票口径,最高规格)、issue #31(统计链口径源头)、issue #2(V1 事件池产物与因果性审计)。
性质:不重跑 v6、不碰背离v6 底层信号;v6-1~5 与 v6 ALL 的数字直接引用 #31 已落盘结果。
本文件冻结口径,出数后不许增删改;阴性结果原样交付。

## 0. 显著披露(先于一切设计)

1. **V1 事件日即低点日本身 → bounce 恒 < 0 → V1 版 v6-1/2/3 结构性为空(纯逻辑推导,非数据)**。
   V1 背离条件含"当前低点收盘价 < 前低点收盘价"(`src/divergence_detector.py` L157),而 bounce = close_ev / anchor_close − 1,
   anchor_close = 前低点收盘价,故 V1 全池 bounce < 0 恒成立。
   v6-1/2/3 要求 0.02 < bounce ≤ 0.08,在 V1 池上必然 0 选中。
   本实验仍按本票口径套同款五组条件全出数(0 选中原样出数),v6-4/v6-5 无 bounce 条件可正常出数。
2. **V1 事件池直接复用 issue #2 产物** `v3_pipeline/reports/divergence_event_study/events.parquet`(1,033,193 事件,5,813 股,1992-02-14 ~ 2026-08-31)。
   该产物由逐日截断因果模拟生成(等价生产截断语义,无未来函数),issue #2 已用真实 V1 检测器对拍(val_mismatch=0)。
   本实验仍按本票要求独立抽样对拍(自检 b),逐行一致才放行。
3. **事件索引口径**:events.parquet 的 t_idx/prev_idx 基于事件研究的预处理序列
   (dropna(close) → drop_duplicates(trade_date) → sort_values(trade_date))。
   模拟 worker 采用完全相同的预处理,逐事件断言 预处理序列[t_idx].trade_date == 事件日,不一致计数须为 0。
4. **退市股处理**:事件后数据不足记 truncated,按"没买成"处理,结论偏乐观方向(同 #31 披露 3)。
5. **涨跌停判定**用 stock_data/stk_limit 逐日文件(2007-01-04 起),缺文件日视为无涨跌停约束,缺文件天数披露(同 #31 披露 4)。
6. **数据加载自写**(逐股 parquet,不复权);成本、滑点、整手调冻结引擎 strategy_engine 原语(同 #31 披露 2)。

## 1. 研究问题(issue #34 原文)

V1(最早背离设计,用户记忆中的 v2)的信号价值是否高于 v6-1~5 种子?
两层口径全出数,由用户看表拍板(不设硬线)。

## 2. V1 信号定义(issue #34 拍板,逐字)

`src/divergence_detector.py`(V1):收盘价滑窗低点(window=6,步长 3,锚定索引 0)+ 价格创新低且 MACD(DIF)比前低点高 ≥ 0.001。
与旧仓库 stock_qt 的 v2 检测器算法等价(issue #2 审计,仅差一个恒 0 死字段)。
工程注意(issue #2 已记录):输入列需 `volume`(parquet 原名 `vol`,改名即可);`_find_close_lows` 全历史跑有未来函数,
生产逐日截断语义无泄漏 —— 本实验采用逐日截断语义的事件池(issue #2 产物),不重扫全历史一次性检测。

### 2.1 事件池字段(events.parquet,逐日截断因果模拟产物)

每事件一行:event_id / ts_code / t_idx(事件日在预处理序列的行号)/ date(事件日)/ compare_rank(1 或 2,与第几个前低比较命中)/ prev_idx(前低点行号)。
同一 (ts_code, date) 至多一行(逐日截断语义下同日多次命中只记首个,compare_rank 小者优先)。

### 2.2 anchor_close 口径(本票钉死)

anchor_close = V1 背离事件的**前低点收盘价** = 预处理序列 close[prev_idx],等价于检测器输出的 close_previous 字段。
anchor 在事件日之前,因果干净。

## 3. 种子条件(逐字照抄 #31 §2.2,套在 V1 事件池上)

记号:j = 事件日在该股预处理序列的行号(= t_idx),close_ev = close[j]。

- `dd20 = close_ev / max(close[max(0, j-20) .. j]) - 1`(≤0,前 20 交易日窗口含事件日共 21 根;上市初期 j<20 时截断到 0)。
- `bounce = close_ev / anchor_close - 1`(V1 池恒 < 0,见披露 1)。

| 名称 | 别名 | 条件 |
|---|---|---|
| V1 版 v6-1 | S1 | dd20 ≤ −0.15 且 0.02 < bounce ≤ 0.08 |
| V1 版 v6-2 | S2 | dd20 ≤ −0.20 且 0.02 < bounce ≤ 0.08 |
| V1 版 v6-3 | S3 | dd20 ≤ −0.25 且 0.02 < bounce ≤ 0.08 |
| V1 版 v6-4 | S4 | dd20 ≤ −0.15(无 bounce 条件) |
| V1 版 v6-5 | S5 | dd20 ≤ −0.25(无 bounce 条件) |
| V1 全池 | ALL | 全池不筛选(对照行) |

## 4. 交易机制(逐字 #31 §2.3,直接复用 run_seeds.py 模拟原语)

- 事件日下一交易日(个股序列下一行)开盘买;事件日无下一行(含退市)→ truncated_no_next,计数不进收益统计。
- 开盘无报价 → dropped_no_quote;开盘价 ≥ 当日涨停价 − 1e-9 → dropped_limitup(买不进,计数不递补)。
- 每笔独立 10 万本金:px = open×1.001(0.1% 买入滑点);sh = int(100000/px/100)×100;不足一手取一手;
  while sh>0 且 sh×px+买佣 > 100000+1e-6:减一手;减到 0 → dropped_cash。
- 入场日记持有第 1 日;第 H 个交易日(H ∈ {10, 20, 25},按个股序列行号计)收盘卖。
- 收盘 ≤ 当日跌停价 + 1e-9 → 顺延至下一非跌停收盘日;数据耗尽 → truncated_exhausted,计数不进收益统计。
- 卖出执行价 xs = 收盘×0.999(0.1% 卖出滑点)。
- 净收益 = (sh×(xs−px) − 买佣 − 卖佣 − 印花税) / (sh×px + 买佣)。
- 成本常量(`v3_pipeline/scripts/strategy_engine.py`,冻结引擎):佣金双边万 2.5、单笔最低 5 元;
  印花税 0.1%(2023-08-27 含之前)/ 0.05%(2023-08-28 起);整手 100 股;滑点单边 0.1%。
- 涨跌停判定用 stock_data/stk_limit/YYYYMMDD.parquet(2007-01-04 起),缺文件日视为无涨跌停约束。
- 实现:直接调用 `experiments/divergence_seed_trial_history/run_seeds.py` 的 `_simulate_one` / `load_limits` / `cluster_t`,
  逐股 worker 本实验自写(事件字段不同),模拟原语零改动。

## 5. 统计指标(逐字 #31 §2.4 + 本票增补)

- n_closed、净笔均、净中位、胜率、cluster_t(按入场日聚类的 Liang-Zeger 稳健 t,G<2 记 NaN)、
  盈利年占比(有 ≥30 笔成交的年份中当年净笔均 > 0 的占比,按入场年)、
  日期集中度(top5 入场日净盈亏合计 / 全部净盈亏合计,总净盈亏 ≤ 0 时记 NaN)。
- 五线判活宣判(#31 同款五线:n≥300 / 净笔均>0 / cluster_t≥2 / 盈利年≥60% / 日期集中度≤50%)。
  本票不设硬线(#34:"由用户看表拍板"),五线只作参照出数,不做封档/放行决定。
- 收益分布表:呈现口径 = "覆盖率→门槛"方向("X%信号涨幅>" = 涨幅最高的 X% 信号都超过该值),
  档位 90/85/75/50/25/10/5%,另出均值/最好/最差;禁裸 P-分位列;收益带 %。
- 净收益 > +1% 的信号占比(closed 口径,扣完成本),分母 = n_closed。
- 信号时间分布(事件级,与 H 无关):
  汇总行(总数/年均/月均/零信号月占比/单月最多/单日最多;年均 = 总数/35;
  月均 = 总数/种子并集首末事件月跨度月数(含两端,全表共享同一跨度);零信号月占比 = (跨度 − 该行有信号月数)/跨度;
  该月口径已按 #31 §4.4 发表值逐格反推核对一致,v6 侧跨度 = 398(1993-07~2026-08));
  逐日信号数维度(有信号日数、零信号日占比(分母 = 市场日历 8000 个交易日,仅落入日历的有信号日计入)、
  "50%日信号数>"、"5%日>"、max);
  逐年信号数 + 逐年有信号日数 + 占当年交易日 %(市场日历 = stock_data/daily/000001.SH.parquet 上证指数日线,
  1993-10-08 ~ 2026-08-31,8000 个交易日,与 #31 §4.7 同源;日历起点前的事件计数披露、不入日历占比)。

## 6. 格子与并表(配置为行,全出数)

- V1 侧:6 配置 {V1 全池, V1 版 v6-1~5} × H {10, 20, 25} = 18 行,全出数。
- v6 侧:v6-1~5 与 v6 ALL 的数字直接引用 #31 已落盘结果
  (summary_seed.csv / trades_seed.parquet / report.md / addendum_distribution.md / addendum_temporal.md),不重跑 v6。
  凡从 #31 落盘 parquet 重算的引用数(分布、时间分布),须与 #31 已发表表格逐格核对一致(自检 d)。
- 并表:V1 全池、V1 版 v6-1~5 为行,与 v6 ALL、v6-1~5 并排;池价值(全池行)与池内筛选增益(种子行 − 全池行)分开表述。

## 7. 自检(全过才可出报告)

- **自检 a(模拟函数保真)**:用 #31 的 events_history_v1.parquet 以复用的模拟原语重跑 v6-1/H20 一格,
  逐笔 net_ret 与 #31 trades_seed.parquet 对拍(排序后 max abs diff = 0),净笔均须复现 +4.89%(与 summary_seed.csv 六位小数一致)。
- **自检 b(V1 事件因果对拍)**:固定种子抽 6 只股票,① 逐日截断因果模拟(des.simulate_events_idx)重算的事件
  与 events.parquet 该股子集 (t_idx, compare_rank, prev_idx) 逐行一致;② 真实 DivergenceDetector 逐日重算
  (全部事件日 + 抽样非事件日,截断输入),日级命中布尔与事件池逐日一致。
  另:全部模拟事件中 预处理序列[t_idx].trade_date == 事件日 的逐事件断言计数须 0 违例。
- **自检 c(计数/因果/守恒,仿 run_seeds.py)**:交易因果(entry_date > event_date、exit_date > entry_date);
  计数守恒(n_selected ≤ n_universe;ALL 行 n_selected == n_universe;n_selected = n_closed + n_dropped + n_truncated);
  事件守恒(trades 行数 = 3 × 事件池行数)。
- **自检 d(v6 引用数核对)**:从 #31 落盘 trades_seed.parquet 重算的 v6 分布/时间分布关键格
  与 #31 issue 已发表数字一致(如 v6-1/H20 分布七档、§4.4 汇总行、维度 1 有信号日数)。

## 8. 纪律

- 阴性结果原样交付;所有格子全出数,配置为行。
- 分布统计一律"覆盖率→门槛"方向;收益带 %;占比写清分母。
- 全程 progress.log 心跳,每大步一行带时间戳;确定性可复跑(产物排序落盘,跑两遍逐位一致)。
- 不分析背离v6 底层信号;不评议 V1 优劣,只出数。
- *.parquet 不入库(.gitignore 政策),本地落盘供复核;禁 git commit、禁评论 issue。

## 9. 交付物(本目录内)

README.md(本文件)、run_v1.py(唯一执行脚本)、progress.log、
trades_v1.parquet(每事件 × H 一行 + 五组选择布尔列)、summary_v1.csv(18 行)、
verdict.json(自检明细 + 参照五线)、report.md(全表 + 分布 + 时间分布 + 并表对比 + 自检,阴性原样)。
