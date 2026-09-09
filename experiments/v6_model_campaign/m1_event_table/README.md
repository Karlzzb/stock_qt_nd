# M1 事件表扩展与扫描器 —— 预登记军令状(先于跑数落盘,冻结勿改)

- 票:`Karlzzb/stock_qt_nd` issue #48(Part of #47)。
- 分支:`v6-portfolio`。
- 日期:2026-09-09。
- 口径源头:#31 §2.1(背离v6 金叉对金叉扫描逐条口径)、#42 §3.1(事件表缺口清单)与 §7(因果性验收断言 11 条)、`experiments/divergence_anchor_eval_2026/scan_v1.py`(扫描器口径源头)、`experiments/divergence_seed_trial_history/run_seeds.py`(全历史扫描 frozen 参照,只读)。
- 对账基准:`experiments/divergence_seed_trial_history/events_history_v1.parquet`(96,577 行 × 9 列,只读,不许改)。
- 本阶段只做事件口径,不模拟交易、不建标签、不建特征主表(后两者归 M2/M3)。

## 一、施工范围(钉死)

在不改变任何既有信号口径的前提下,把扫描器逐字复刻扩展,在事件行上补落 #42 §3.1 缺口列,重算全历史 96,577 事件表。
铁律:背离v6 信号定义一个字不许改;只加列不改口径;事件集合(哪些行算事件)必须与 frozen 基准逐行一致。

## 二、记号(与 #42 §一 逐字一致)

- 个股**不复权**日线按 `trade_date` 升序、`reset_index(drop=True)` 后的 0 基行号索引。
- `DIF, DEA, _ = talib.MACD(close)`,12/26/9 默认参数,按个股全历史一次性计算。
- 金叉三元组行号:`i2 = C(k-2)`、`i1 = C(k-1)`、`j = C(k)`(事件日行号)。
- 锚点行:`a = (i1, j]` 内最低收盘价所在行(并列取最早);前区间最低收盘价行:`p_low = (i2, i1]` 内最低收盘价所在行(并列取最早)。
- 区间语义:左开右闭,按行号切。

## 三、新增列清单(16 列;旧 9 列原样保留,逐字口径不变)

每列给出:列名 / 中文全称 / 精确公式 / 因果论证(最晚可知时点)/ 量纲。
全部 16 列的输入均为个股序列行号 `<= j` 的行与由 `close[0..t](t<=j)` 递推的 DIF/DEA 值,最晚可知时点 = 事件日(第 j 行)收盘(#42 引理 1+2+3)。

| # | 列名 | 中文全称 | 精确公式 | 因果论证 | 量纲 |
|---|---|---|---|---|---|
| 1 | cross_prev2_row | 远金叉行号 | `i2`(金叉三元组第 1 个金叉的 0 基行号) | i2 < i1 < j,金叉日本身早于事件日 | 计数(行号) |
| 2 | cross_prev2_date | 远金叉日期 | 第 i2 行交易日 | 同上 | 日期 |
| 3 | cross_prev2_dif | 远金叉 DIF 值 | `DIF[i2]` | DIF[i2] 只是 close[0..i2] 的函数(引理 2),i2 < j | 价格量纲(元) |
| 4 | cross_prev2_dea | 远金叉 DEA 值 | `DEA[i2]` | 同上 | 价格量纲(元) |
| 5 | cross_prev_row | 前金叉行号 | `i1`(金叉三元组第 2 个金叉的 0 基行号) | i1 < j | 计数(行号) |
| 6 | cross_prev_dea | 前金叉 DEA 值 | `DEA[i1]` | 引理 2,i1 < j | 价格量纲(元) |
| 7 | event_row | 事件日行号 | `j = C(k)` 的 0 基行号 | 事件日本身 | 计数(行号) |
| 8 | event_close | 事件日收盘价 | `close[j]` | 事件日收盘可知 | 价格(元) |
| 9 | event_vol | 事件日成交量 | `vol[j]`(个股日线原始值,未复权未归一) | 事件日收盘可知 | 成交量(手) |
| 10 | event_amount | 事件日成交额 | `amount[j]`(同上;文件缺 amount 列时记 NaN 并披露,实测 0 文件触发) | 事件日收盘可知 | 成交额(千元) |
| 11 | cross_dea | 事件日金叉 DEA 值 | `DEA[j]` | 引理 2,第 j 行收盘可知 | 价格量纲(元) |
| 12 | min_prev_close | 前区间最低收盘价 | `min(close[(i2, i1]])`,左开右闭 | 输入行号 <= i1 < j(引理 1) | 价格(元) |
| 13 | min_prev_row | 前区间最低收盘价行号 | `p_low = argmin close[(i2, i1]]`,并列取最早(np.argmin 首个最小值) | 同上 | 计数(行号) |
| 14 | min_prev_date | 前区间最低收盘价日期 | 第 p_low 行交易日 | 同上 | 日期 |
| 15 | anchor_row | 锚点行号 | `a = argmin close[(i1, j]]`,并列取最早 | 引理 3:a <= j | 计数(行号) |
| 16 | anchor_bars | 锚点距事件日交易日数 | `j - a` | 由 7 与 15 派生 | 交易日 |

与旧列的衔接(不新增、仅登记):DIF[i1] = 旧列 `cross_prev_dif`;DIF[j] = 旧列 `cross_dif`;区间 (i1, j] 最低收盘价 min_cur = 旧列 `anchor_close`(由构造恒等,不重复落列,见披露 1)。
#42 §3.1 缺口覆盖对照:C(k-2) 日期与 DIF/DEA 值 = 列 2/3/4;DEA[j]、DEA[i1] = 列 11/6;两区间最低收盘价 = 列 12 + 旧列 anchor_close;锚点行号距离 = 列 16;量能字段 = 列 9/10;急跌窗口内部路径由列 7(event_row)+ 原始日线在 M2 重建(见披露 2)。
行号三元组(i2/i1/j)与 p_low/a 落盘后,#42 六族 43 条特征的全部输入均可凭事件表 + 原始日线在 M2 无重扫重建。

## 四、验收断言清单(全过才算完)

- 验收 A(与 frozen 基准对账):`events_ext_v1.parquet` 行数 = 96,577;`(ts_code, event_date)` 双侧唯一;按该键对齐后共有 9 列逐位一致(浮点最大绝对差 = 0.0,非容差);行序(event_date, ts_code 升序)亦逐行一致。
- 验收 B(截断历史重算):随机抽 20 只股票 × 每只 3 个截断行(个股历史长度的 40%/60%/80% 处,去重,且截断后行数 >= 100;抽样种子 20260909,合格池 = 事件数 >= 5 且历史行数 >= 300 的股票),仅用行号 <= 截断行的数据重跑同一扫描函数,断言:截断重算产出的事件集合(事件日行号 <= 截断行者)与全量表中该股同行号范围的事件,全部 25 列逐位一致(浮点逐位相等,非容差)。
- 验收 C(#42 §7 的 11 条因果性断言):逐条实现为 `selfcheck_causality.py` 可机检代码并全过;M1 适配映射见第六节。
- 验收 D(缺失与 warm-up):新增 16 列逐列 NaN 占比落盘披露;MACD warm-up 段行为与旧表一致(行数对账 + 全部 DIF/DEA 列非 NaN 硬断言)。
- 验收 E(确定性):全量扫描连跑两次,两次产出全部 25 列逐位一致。

## 五、自检设计(抽样参数预登记)

| 自检 | 样本 | 种子 | 说明 |
|---|---|---|---|
| 截断重算(验收 B) | 20 股 × 3 截断行 | 20260909 | 见验收 B |
| 断言 2(区间语义自洽,独立代码路径重算) | 2,000 事件 | 20260909 | 独立于扫描器的重写实现逐字段核对 |
| 断言 3(MACD 无前视) | 300 事件(限 j >= 100) | 20260909 | 截尾重算 talib.MACD,容差 1e-8(预期实测 0) |
| 断言 7(量纲缩放不变性) | 50 股(事件数 >= 3)全部事件,lambda = 2.718 | 20260909 | close 同乘 lambda 重扫,比值类特征变化 < 1e-9 |
| 断言 4/10(dd20 与种子资格) | 全表 96,577 事件 | — | 从原始日线凭 event_row 独立重算,与 frozen `trades_seed.parquet`(v1)逐位对账 |

## 六、#42 §7 十一断言的 M1 适配映射(预登记)

#42 §7 断言针对「特征构建产物表」;M1 产物 = 扩展事件表(中间量层),43 条特征的构建归 M2。
适配原则:凡可在 M1 产物上完整执行者原样执行;凡指向 M2 产物者,M1 执行其可在中间量层执行的部分并对 M2 部分立移交声明。

1. 断言 1(行号上界):全表向量化断言 i2 < i1 < j、p_low ∈ (i2, i1]、a ∈ (i1, j]、anchor_bars == j - a;外加验收 B 截断重算作构造性证明(全表可用不晚于事件日的数据逐位复现)。
2. 断言 2(区间语义自洽):抽样 2,000 事件,用独立重写的实现从原始日线重算 i2/i1/j/a/p_low/min_prev/min_cur 与全部派生列,逐字段逐位核对(含并列取最早规则复现)。
3. 断言 3(MACD 无前视):抽样 300 事件(j >= 100),close[0..j] 截尾重算 talib.MACD,|DIF 差| 与 |DEA 差| < 1e-8。
4. 断言 4(dd20 窗口与符号):全表凭 event_row 从原始日线重算 dd20,断言 dd20 <= 1e-12、窗口 = [max(0, j-20), j] 长度 = min(21, j+1);并与 frozen trades_seed(v1)逐位对账。dd20 本体不落 M1 事件表(见披露 3)。
5. 断言 5(锚点因果与取值域):全表断言 a <= j、anchor_close > 0;a == j 蕴含 |event_close/anchor_close - 1| <= 1e-12(bounce 恒等式,由表内列可算)。
6. 断言 6(事件条件自洽):全表断言 dif_lift >= 0.001 且 min_prev_close > anchor_close(min_cur);抽样 2,000 事件从原始日线重算核对(与断言 2 同批)。
7. 断言 7(量纲缩放不变性):抽样 50 股,close 同乘 lambda = 2.718 重扫;断言事件集合(按 event_row 对齐)不变、行号/日期/量能列严格不变、DIF/DEA/价格列按 lambda 等比缩放(相对误差 < 1e-9),代表性无量纲比值(DIF[j]/close[j]、DEA[j]/close[j]、dif_lift/close[j]、(DIF[j]-DEA[j])/close[j]、anchor_close/min_prev_close - 1、(j-i1)/(i1-i2))变化 < 1e-9。
8. 断言 8(环境特征来源):机检 M1 产物列集 == 预登记 25 列(9 旧 + 16 新),不含任何快照覆盖特征(E2/E3/E4 归 M2 日频快照,移交声明)。
9. 断言 9(特征-标签隔离):机检无 `label_` 前缀列;M1 不产出任何标签表(目录内无标签产物);join 键 (ts_code, event_date) 约定移交 M2/M3。
10. 断言 10(成员资格重算一致):由断言 4 重算的 dd20 与表内 bounce 恒等式(event_close/anchor_close - 1)重算 v6-1~v6-5 布尔,断言计数 == frozen 值(5580/2419/1159/11888/3271)、嵌套链 v6-3⊂v6-2⊂v6-1⊂v6-4 与 v6-5⊂v6-4 零违反、与 frozen trades_seed(v1)sel_S1~S5 逐位一致。
11. 断言 11(缺失披露):16 新列逐列 NaN 占比落盘;断言 NaN 列 ⊆ 预登记允许集(仅 event_amount 因缺列守卫可能 NaN,实测披露);含守卫 NaN 分支的特征(B3/B4/B6/C4/E1)归 M2,移交声明。

## 七、披露项(原样,不粉饰)

1. min_cur(区间 (i1, j] 最低收盘价)不单独落列:由构造恒等于旧列 anchor_close,断言 6 以 min_prev_close > anchor_close 实现;零信息损失。
2. #42 §3.1「急跌窗口内部路径」不落盘(变长结构不宜入事件表):改落 event_row,窗口 W = [max(0, j-20), j] 内全部路径由 M2 凭 event_row 从原始日线重建;因果性不变(行号上界 j 由断言 1 保证);急跌窗口派生量(dd20 等)的因果性由断言 4 在全表上独立重算验证。
3. dd20/bounce 与种子布尔不落 M1 事件表:#42 标记二者「已存在」(种子特征,由 run_seeds.py 计算),归 M2 特征主表;M1 以全表独立重算 + 与 frozen trades_seed.parquet 逐位对账作为验收(断言 4/10)。
4. E2/E3/E4 按 #42 族E 立规由 v5 日频快照覆盖,不进 v6 专属层,M1 不落。
5. amount 缺列守卫:全市场仅 2 个指数文件(000001.SH/399001.SZ)缺 amount,且二者本就被 schema 过滤(缺 ts_code/vol)跳过,实测 0 只个股文件触发守卫。
6. 行号为个股序列 0 基行号(非市场日历对齐),跨股不可比;个股序列无停牌行。
7. 事件日量能(vol/amount)为原始口径,未复权、未归一;归一化特征(如 E1 量比)归 M2。
8. 扫描跳过口径与 frozen 参照逐字一致:缺 ts_code/vol 列跳过(schema)、行数 < 100 跳过(short)、DIF 全 NaN 跳过(nan)。
9. parquet 产物与 progress.log 按仓库 .gitignore 政策不入库;代码、README、report.md、自检 JSON 入库。

## 八、复现步骤

1. 数据就位:`stock_data/daily/*.parquet`(5,891 个,含 2 个指数文件)、frozen 基准 events_history_v1.parquet 与 trades_seed.parquet。
2. 依赖:python3 + pandas/numpy/pyarrow/talib。
3. 运行:`python3 experiments/v6_model_campaign/m1_event_table/run_event_table_ext.py`(扫描两次验证确定性 + 验收 A/B/D,预计约 10 分钟)。
4. 运行:`python3 experiments/v6_model_campaign/m1_event_table/selfcheck_causality.py`(11 断言 + 生成 report.md,预计约 10 分钟)。
5. 期望产物:events_ext_v1.parquet(96,577 行 × 25 列)、m1_scan_results.json、m1_causality_results.json、report.md、progress.log。

## 修订记录(预登记正文逐字未动,仅追加)

- 2026-09-09 首跑后:①断言 7 预登记实现「缩放后事件集合(按 event_row 对齐)不变」与冻结信号口径的绝对阈值 `dif_lift >= 0.001`(价格量纲,#31 §2.1)矛盾——close 同乘 lambda 后 dif_lift 同步放大 lambda 倍,缩放版会新增 unscaled dif_lift ∈ [0.001/lambda, 0.001) 的事件(首跑 50 股中 5 股各 +1 起)。
  按铁律「#42 缺口清单/断言与 scan_v1.py 实际可计算性矛盾时,按 scan_v1.py 因果可行为准并披露偏差,不许私改信号定义」,断言 7 改实现为:(a) 基线事件零缺失;(b) 每个新增事件 dif_lift 落在阈值带 [0.001/lambda, 0.001)(归因于绝对阈值效应);(c) 基线事件交集上行号/日期/量能严格不变、价格列等比(相对误差 < 1e-9)、无量纲比值变化 < 1e-9。
  信号定义未做任何改动;披露同步落 report.md 第 8 节第 10 条。
  ②自检实现修正两处(均为自检代码 bug,非数据/口径问题):断言 10 布尔逐位比较改为在合并表自身列上重算后比对(原实现受外连接行序影响);trades_seed 参照列显式改名 `_frozen` 后缀(原实现因无列名碰撞未加后缀导致 KeyError)。
