# M3 标签赛 —— 预登记军令状(先于跑数落盘,冻结勿改)

- 票:`Karlzzb/stock_qt_nd` issue #50(Part of #47)。
- 分支:`v6-portfolio`。
- 日期:2026-09-09。
- 施工图:`reports/wayfinder/43-v6-label-race.md`(候选清单 §3.2 与裁决规则 §3.3 逐字落地)。
- 机制源头:#31 §2.3(事件级模拟逐条口径)与 §2.4(cluster_t 口径);冻结成本原语 `v3_pipeline/scripts/strategy_engine.py`(只读)。
- 切分常量唯一权威落点:`v3_pipeline/src/feature_master.py`(import 使用,零改动)。
- 本阶段只建标签、跑标签赛、出裁决;不精选特征(归 M4)、不做策略层审判(归 M5)、测试段零触碰。

## 一、施工范围(钉死)

1. 事件级直接模拟构建标签表 `labels_v6.parquet`(与 #31 run_seeds 同机制,不沿用 v5 全序列毛利路径),覆盖全部 96,577 事件 × 4 档 H ∈ {10,20,25,60}。
2. 8 候选 × 36 组预登记网格单池正赛,训练只用 seg=='train' 且标签非 NaN 的行,val 指标只用 seg=='val' 的行;embargo/pre2001 不作建模型行;test 段零触碰(只在场断言,不计算任何 test 指标、不做 test 逐行统计)。
3. 裁决、自检、报告全链路落盘;任何系统性偏差停下来记录并向监工报告,不私改口径迁就。

## 二、输入(全部只读)

- 事件表:`experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet`(96,577 行 × 25 列,M1 复核锁定)。
- 特征主表:`experiments/v6_model_campaign/m2_feature_master/master_v6.parquet`(96,577 × 2,002;元数据列 = event_id/ts_code/date/event_row/seg,其余 1,997 特征列)。
- 原始日线:`stock_data/daily/{ts_code}.parquet`(统一 `pd.to_datetime(df['trade_date'].astype(str))` 后 `sort_values('trade_date').reset_index(drop=True)`,行号与事件表 event_row 对齐)。
- 涨跌停:`stock_data/stk_limit/YYYYMMDD.parquet`(2007-01-04 起);缺文件日视为无涨跌停约束;2007 年前受影响成交占比按本池逐 H 重算披露。
- 对账基准:`experiments/divergence_seed_trial_history/trades_seed.parquet`(frozen;只用 variant=='v1')。

## 三、候选清单(8 个,候选序 = H 升序、同 H 二分类先)

统一标签口径(每个候选共用,逐条对齐 #43 §3.2 与 #31 §2.3,逐字):
信号日 T(背离金叉事件日,变体1)的下一交易日开盘入场:买入执行价 px = open×1.001(0.1% 买入滑点)。
开盘无报价 → dropped_no_quote 剔除;开盘价 ≥ 当日涨停价 − 1e-9 → dropped_limitup 剔除(买不进,不递补)。
每笔独立 10 万本金整手折算:sh = int(100000/px/100)×100,不足一手取一手,while sh>0 且 sh×px+买佣 > 100000+1e-6 减一手,减到 0 → dropped_cash 剔除。
入场日记持有第 1 日;第 H 个交易日(按个股序列行号计)收盘出场:卖出执行价 xs = close×0.999(0.1% 卖出滑点)。
出场日收盘 ≤ 当日跌停价 + 1e-9 → 顺延至下一非跌停收盘日;数据耗尽(含退市)→ truncated_exhausted 剔除。
事件日无下一行(个股数据耗尽,含退市)→ truncated_no_next 剔除(#31 同机制,本赛并入剔除类别计数)。
净收益 = (sh×(xs−px) − 买佣 − 卖佣 − 印花税) / (sh×px + 买佣)。
成本常量复用冻结引擎 `strategy_engine.py`(只读):佣金双边万2.5、单笔最低5元;印花税 0.1%(2023-08-27 含之前)/ 0.05%(2023-08-28 起);整手 100 股;滑点单边 0.1%;价格比较容差 1e-9。
涨跌停判定用 `stock_data/stk_limit/YYYYMMDD.parquet`(2007-01-04 起),缺文件日视为无涨跌停约束(2007 年前受影响成交占比须披露,本赛按本池本 H 重算披露)。
剔除类别(dropped_no_quote / dropped_limitup / dropped_cash / truncated_no_next / truncated_exhausted / 键缺失)逐类计数按段落盘并披露占比(只对 train/val 出逐类计数与指标,test 段零触碰只在场断言)。

| # | 工程名 | 中文全称 | 精确定义 | 头型 |
|---|---|---|---|---|
| 1 | net_pos_10d | 次日开盘入场净收益大于零二分类标签(10 个交易日视野) | 1{ 净收益(H=10)> 0 } | 二分类 |
| 2 | net_ret_10d | 次日开盘入场净收益幅度回归标签(10 个交易日视野) | 净收益(H=10)原值 | 回归 |
| 3 | net_pos_20d | 次日开盘入场净收益大于零二分类标签(20 个交易日视野) | 1{ 净收益(H=20)> 0 } | 二分类 |
| 4 | net_ret_20d | 次日开盘入场净收益幅度回归标签(20 个交易日视野) | 净收益(H=20)原值 | 回归 |
| 5 | net_pos_25d | 次日开盘入场净收益大于零二分类标签(25 个交易日视野) | 1{ 净收益(H=25)> 0 } | 二分类 |
| 6 | net_ret_25d | 次日开盘入场净收益幅度回归标签(25 个交易日视野) | 净收益(H=25)原值 | 回归 |
| 7 | net_pos_60d | 次日开盘入场净收益大于零二分类标签(60 个交易日视野) | 1{ 净收益(H=60)> 0 } | 二分类 |
| 8 | net_ret_60d | 次日开盘入场净收益幅度回归标签(60 个交易日视野) | 净收益(H=60)原值 | 回归 |

候选序(破平用)固定为上表序:H 升序,同 H 二分类先于回归。
二分类标签在跑训驱动内由净收益派生:net_pos_{H}d = 1{ net_ret_{H}d > 0 },NaN 保留(不由 (ret > 0) 吞成 0)。
回归标签 = 净收益原值,NaN 保留。
剔除事件发生任意剔除类别时该 H 标签为 NaN;剔除时机在跑训驱动按段登记(只对 train/val 登记逐类计数,test 零逐行统计)。

## 四、标签表 schema 设计(labels_v6.parquet)

- 键:event_id(int64,自主表元数据列按 (ts_code, date) 一对一合并取得)、ts_code(str)、date(datetime64,事件日);另带 event_row(int64)与 seg(str)两列便于核验与分段,seg 落盘时以 `feature_master.segment_of` 独立重算并与主表 seg 逐行一致断言。
- entry_date(datetime64):入场日(下一交易日);入场侧剔除(dropped_no_quote / dropped_limitup / dropped_cash / truncated_no_next / 键缺失)时为 NaT;该列 H 不变(入场不依赖 H),供主指标 cluster_t 按入场日聚类使用。
- 每 H 三列(H ∈ {10,20,25,60}):
  - `net_ret_{H}d`(float64):净收益;非 closed 一律 NaN(NaN 保留原则:不由任何派生吞成 0/ False)。
  - `status_{H}d`(str):closed / dropped_no_quote / dropped_limitup / dropped_cash / truncated_no_next / truncated_exhausted / key_missing 七类之一;入场侧四类 H 不变,逐 H 重复落列以对账。
  - `exit_date_{H}d`(datetime64):出场日;非 closed 为 NaT。
- 合计 3 键列 + 2 核验列 + 1 入场日列 + 4×3 标签列 = 18 列。
- 行序:与事件表行序一致(event_date, ts_code 升序);(ts_code, date) 与 event_id 双侧唯一断言。

## 五、裁决规则(#43 §3.3 十条逐字落地)

1. 正赛池 = ALL 全池(变体1),无主备结构、无兜底升级条款;死锁条款:若训练段有效标签事件数 < 3000,本赛季不裁决、回用户拍板(v5 的 3000 阈值保留为纯样本量保险丝,不再触发任何池切换)。
2. 网格沿用 v5 预登记 36 组全因子(num_leaves{15,31,63} × min_data_in_leaf{50,100,200} × learning_rate{0.05,0.10} × feature_fraction{0.6,0.8},≤50 上限,config_id 按 `label_race.GRID` 原序 0~35);二分类头 objective 沿用 v5 口径(binary + average_precision),回归头 objective 改 regression、评估指标改回归对应(钉死为 rmse,与 l2 同 argmin、单调等价,早停最优轮数不变),其余参数与复现性四件套沿用 #25 口径(`train_eval_pipeline.DEFAULT_LGBM_PARAMS`:deterministic=true、force_col_wise=true、num_threads=8、seed/feature_fraction_seed/bagging_seed/data_random_seed 全固定 20260902;num_boost_round 上限 1000、早停 50 轮)。
3. 每配置管线沿用 v5:训练段五折时间序列折外(按事件日分块、折界不落日内)→ 全训练段终模(轮数 = 五折 best_iteration 均值取整,下限 1)→ train_oof/val 双段指标表,配置为行。
   校准层分头处理:二分类头沿用逻辑回归校准层 [p, p²](`SquaredLogitCalibrator`,拟合于折外非首块样本);回归头不配概率校准层,排序直接用折外/终模预测原值(裁决只看排序型指标,单调变换不影响结果,此条按预登记写明)。
4. 主裁决指标:验证段头部五名净笔均(日加权)——每个事件日按模型分数排序取 top-min(5, 当日事件数)(无信号日不计),对所取事件按该候选自身 H 口径的真实净收益求日截面均值,再按日等权平均;同指标附带 cluster_t(按入场日聚类的 Liang-Zeger 稳健 t,#31 §2.4 口径:x̄ = 净收益均值;S_c = 聚类 c 内 (x_i − x̄) 之和;var(x̄) = G/(G−1) × ΣS_c² / n²;G < 2 记 NaN;t = x̄/√var(x̄),var ≤ 0 记 NaN;cluster_t 在全部所取事件上合并计算,聚类 = 标签表 entry_date)。
   二分类头分数 = 校准后概率,回归头分数 = 预测幅度原值,两者在同一净收益标尺上可比,双头同场裁决成立。
   头部排序平局裁决(钉死):按 (分数降序, ts_code 升序, event_id 升序) 确定性裁决(与 v5 topk_precision 同口径)。
   train_oof 同指标同口径落盘(分数 = 折外口径,二分类头经校准层,首块无折外概率的行不入 train_oof 指标)。
   辅助指标(落盘、不作裁决输入):二分类头另出 v5 口径的头部五名精确率(日加权/事件加权)与平均精确率(仅供与 v5 历史产物口径对照参考;v5 实证结论不作任何输入);回归头另出 top-5 命中率(所取事件净收益>0 占比,事件加权为主、日加权附出)。
5. 每候选选配置:主指标(val 头部五名净笔均日加权)最高 → 平局 cluster_t 较高 → 再平局网格序(config_id)靠前。
6. 总裁决:八候选各取当选配置,主指标最高者当选;平局 cluster_t 较高 → 再平局候选序(§三 表序)靠前。
7. 双约束(保留 v5 的防硬选机制并加显著性闸):当选候选主指标 ≥ 八候选(各取当选配置)主指标的中位数;且当选候选 val cluster_t ≥ 2。任一不满足 → 本赛季无当选,阴性结论如实落盘,不降格另选。
8. 复现与纪律沿用 v5 全套:八候选当选配置折外全程复跑逐位一致断言(不可旁路,以跑赛落盘折外概率为对照基准重算比对);test 段零触碰只在场断言;段界与隔离带硬断言(`feature_master.assert_segment_integrity`);泄漏列排除断言(训练特征集 = 主表 1,997 特征列,对 `feature_master.EXCLUDE_PATTERNS` 零命中重扫);断点续跑(每配置完成即落盘,重跑跳过已完成)与心跳日志;独立证伪式复核通过后锁定,锁定后候选集与当选结果不再变更。
9. 与阶段门的衔接:标签赛当选只决定训练目标(H 与主头),不构成过门证据;模型是否过阶段门由第二轮裁决 Q4 主门在策略层实证(终审同型组合骨架上 top-K 净笔均 ≥ 最好手工挑选规则 +2pp 且 cluster_t ≥ 2),标签赛主指标是该主门的事件级同型预览。
10. 双头主用裁决:八候选同场赛跑即天然完成主头裁决——当选候选属哪个头,定版模型主用哪个头;副头概率/幅度可保留为参考输出,是否作为辅助特征或辅助排序入后续阶段,由特征精选与策略层阶段另行裁决,不在本赛预登记范围内。

## 六、验收断言清单(执行版,全过才算完)

1. **因果性**:抽样 ≥15 只股票(种子 20260909,合格池 = 该股事件数 ≥1 且截头 200 行后仍有事件)截掉序列头部 200 行重算标签,交集事件(原 event_row ≥ 200 者,行号平移 200)全部标签列逐位一致(容差 1e-9,NaN 视同相等)。
2. **对账断言**:variant=='v1' 且 H∈{10,20,25} 的成交事件,labels_v6 净收益与 trades_seed.net_ret 逐笔对账(容差 1e-9),status 剔除类别交叉核对(逐类计数一致、事件级一致);任何系统性偏差停下来记录并报告,不私改口径迁就。
3. **标签-特征隔离**:标签构建器不读主表任何特征列(只读元数据列 event_id/ts_code/date/event_row/seg 与事件表);跑训管线训练特征集列名无 label_ 混入(机检)。
4. **段界硬断言**:`feature_master.assert_segment_integrity` 在主表 (date, seg) 上执行;训练/验证行 seg 核验(train 模型行 seg 全 'train',val 指标行 seg 全 'val')。
5. **泄漏列断言**:训练特征集 = 主表 1,997 特征列(`feature_master.feature_columns` 口径),列名对 EXCLUDE_PATTERNS 零命中重扫(selfcheck 独立重扫)。
6. **确定性**:标签构建双跑 md5 逐位一致;八候选当选配置 OOF 复跑逐位一致断言(不可旁路)。
7. **剔除类别计数**:按段落盘(train/val 逐类计数与占比;test 只断言行数在场,零逐行统计);2007-01-04 前入场(全程无涨跌停约束)的成交占比按本池逐 H 重算披露;stk_limit 缺文件天数与受影响计数披露。
8. **行数与键守恒**:labels_v6 行数 == 96,577;(ts_code, date) 与 event_id 双侧唯一;与事件表/主表键对账一对一。
9. **裁决完整性**:metrics 表每候选恰 36 行(配置为行全出数);summary 恰 8 行;双约束核验落盘;无当选则阴性结论如实落盘不降格。
10. **test 零触碰**:跑赛与自检全程不计算任何 test 指标、不做 test 逐行统计;只在场断言(test 行数 > 0)。

## 七、机时台账预登记

- 基线:v5 合并池(train 22,963 行 × 2,060 列)19 候选 × 36 配置实测约 2~6 小时(workers=3 × num_threads=8)。
- v6 线性放大预估:训练段 50,165 行(≈2.18×)× 1,997 列(≈0.97×)× 288 配置(vs 684,≈0.42×)→ 总量 ≈ 0.9× v5,**预估 2~6 小时**;标签构建(96,577 事件 × 4 H,逐股并行)预估 < 5 分钟(参照 run_seeds 2 变体 × 3 H 全程 22 秒)。
- 冒烟纪律:全量开跑前先做 1 候选 × 1 配置(net_pos_10d × config_id=0)冒烟计时,实测单配置耗时与全量(8×36)预估写入本节修订记录后再开跑全量。

## 八、产物清单与路径(全部在 experiments/v6_model_campaign/m3_label_race/)

| 产物 | 入库 | 说明 |
|---|---|---|
| README.md | 是 | 本预登记(冻结,变更只追加修订记录) |
| build_labels_v6.py | 是 | 标签表构建驱动(事件级直接模拟) |
| run_label_race_v6.py | 是 | 跑赛驱动(8×36,断点续跑) |
| selfcheck_m3.py | 是 | 独立自检(不复用构建脚本函数) |
| labels_v6.parquet | 否(*.parquet 按 .gitignore) | 标签表(§四 schema) |
| metrics_v6_{candidate}.csv | 是 | 每候选指标表(配置为行,36 行) |
| oof_v6_{candidate}.parquet | 否 | 当选对照基准:36 配置折外分数(复现性断言用) |
| summary_v6.csv | 是 | 八候选当选配置汇总(裁决输入) |
| adjudication_v6.json | 是 | 总裁决结果(含双约束核验) |
| race_results_v6.json | 是 | 口径/断言/台账/剔除类别计数 |
| selfcheck_results.json | 是 | 自检结果(全绿 all_pass=true) |
| report.md | 是 | 主指标全表(8×36 配置为行)+ 辅助指标 + 裁决 |
| progress.log | 否(*.log) | 长任务心跳日志(驱动名标签+时间戳) |

## 九、执行纪律

- 长任务落 progress.log(各驱动追加,带驱动名标签与心跳);派活即报日志路径与预计时长。
- 本阶段对 v3_pipeline 零改动;发现 bug 停下来在 progress.log 记录并向监工报告,不私自改。
- 汇报口径:配置为行全出数;分布统计用「覆盖率→门槛」方向;特征/标签命名用中文全称;禁用任何泄漏产物作对照;v5 实证结论不作任何输入。
- commit 到 v6-portfolio,不加 co-author 行,不 push;不关 #50(监工独立证伪式复核后处置)。

## 修订记录(预登记正文逐字未动,仅追加)

- 2026-09-09 冒烟计时(先于全量开跑):net_pos_10d × config_id=0(num_leaves=15, min_data_in_leaf=50, learning_rate=0.05, feature_fraction=0.6)单配置实测 19.7 秒(含 worker 载表初始化);
  全量 8 候选 × 36 配置 = 288 任务,按 workers=3 线性预估 ≈ 288÷3×19.7s ≈ 32 分钟,加八候选当选配置 OOF 复现断言 ≈ 1 分钟与主进程开销,全量预估 35~60 分钟,优于 §七 预估区间(2~6 小时,按 v5 实测线性放大的保守值)。
  冒烟附带观测(不构成任何裁决输入):该配置 val 头部五名净笔均(日加权)= −0.005472,cluster_t = −2.736。
  死锁条款实测不触发:八候选训练段有效标签事件数 49,806~49,847,均 ≫ 3000。
- 2026-09-09 施工实现修正(构建驱动一处,非口径变更):`feature_master.segment_of` 返回 numpy 数组而非 Series,首跑 AttributeError 后以 `np.asarray` 适配,预登记口径与断言语义不变。
