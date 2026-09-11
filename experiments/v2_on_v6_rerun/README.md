# v2 路线复刻(v2 特征工程逻辑 + v2 模型架构 × v6 事件池)—— 预登记军令状(先于跑数落盘,冻结勿改)

- 票:`Karlzzb/stock_qt_nd` issue #54。
- 分支:`v6-portfolio`。
- 日期:2026-09-11。
- 指令来源:用户 2026-09-11 原话「按照我之前v2版本的逻辑做特征工程,模型架构和v2一样,去做」。
- 起因:用户对 M4(#51)/M5(#52)「终选 10 特征、模型路线判死」的裁决不服,主张 v2 时代 LightGBM+SHAP 有约 170 特征,170 vs 10 的鸿沟需要实证归因。
- 考古依据:v2 时代 commit `ba9d2ed`,解剖报告 = 本目录 `anatomy_report.md`(md5 67fe294bd1590ecf9b1f3b6237366863,与监工抽验互证:D1 跨股票污染、横截面口径、标签公式、LGB 参数、158 选择规则五项已由监工肉眼复核源码属实)。
- 总原则:v2 的**方法论**(特征公式体系、特征选择规则、stacking 模型架构、标签定义)原样复刻;v2 的**缺陷**按 §三「修正性偏离登记表」逐条处理,每条偏离先于跑数登记,不允许静默偏离,不允许看着结果调口径。

## 一、施工范围(钉死)

1. 特征层:在 v6 事件池(96,577 事件,M1 事件表只读)上重建 v2 FULL 610 列逻辑,族构成 = 基础 TA 族 + 进阶 TA 族(含 MACD 深度族)+ alpha 族 + 结构化族 + lag 族 + 市场大盘族 + 背离事件结构族(v6 语义改写)+ 横截面族(rankpct/z × 197 + cs_n)。
2. 横截面口径 = v2 原口径:同日**全市场**股票当日行,不是事件行内横截面。
   为此必须建全市场逐日基础特征面板(逐股全历史向量化计算,再按事件日聚合)。
3. 标签层:v2 公式逐项(§五),训练标签 = `future_return_15d > 0.01`。
4. 特征选择层:v2 规则程序化复刻(§六),不接受手抄 v2 的 158/105 清单(那是 v2 数据上的产物,且选择过程含手工迭代)。
5. 训练层:v2 stacking 架构逐项(§七)。
6. 审判层:M5 同款生死门 18 格(§八),条款机械执行。
7. 段界:训练 = seg=='train'(2001-01-01~2019-12-31,扣隔离带),审判 = seg=='val'(2020-01-01~2023-12-31),test 段零触碰(只出分数不出任何指标),embargo/pre2001 不作建模型行。

## 二、输入(全部只读)

- 事件表:`experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet`(96,577 行 × 25 列)。
- 段标:`experiments/v6_model_campaign/m2_feature_master/master_v6.parquet` 的 (event_id, seg) 两列(train 50,165 / val 24,405 / test 17,902 / embargo 3,063 / pre2001 1,042,断言核验)。
- 个股日线:`stock_data/daily/{code}.parquet`(OHLCV,与 M2 同源,复权口径 = 事件表扫描器口径即不复权,构建时对事件表 event_close 与日线 close 逐位对账断言)。
- 指数日线:`stock_data/daily/000001.SH.parquet`(含 volume)与 `stock_data/daily/399001.SZ.parquet`(在场断言,v2 sz_* 族依赖)。
- 生死门机器:`experiments/v6_long_hold_trial/run_long_hold.py`(md5 a8f1cca6d2bad167899454f896f6d026 断言)与 M5 驱动 `experiments/v6_model_campaign/m5_stage_gate/run_stage_gate_m5.py`(克隆使用,改动面见 §八)。

## 三、修正性偏离登记表(先于跑数,逐条;未登记者一律按 v2 原样)

| # | v2 缺陷(解剖 §3) | 本线处理 | 理由 |
|---|---|---|---|
| P1 | D1:基础/进阶 TA 在按 timestamp 排序的多股票拼帧上计算,全特征跨股票污染(含背离检测用 macd) | 全部按 symbol 分组、逐股全历史因果计算;公式一字不改 | 污染是 bug 不是方法论;用户要的是特征逻辑 |
| P2 | D2:用次日 low ≤ 信号收盘过滤训练行(前视样本选择) | 不过滤任何行;标签对全事件按 v2 公式计算 | 行筛选是选择偏差型前视;v6 引擎自带入场可行性机制 |
| P3 | D3:close_wavelet 小波去噪非因果(v2 注释自认泄漏) | 剔除 close_wavelet 及其 rankpct/z(共 3 列) | 因果版需逐日逐股全窗重跑 pywt,工程量不可行且 v2 自认有问题 |
| P4 | D4:close_d0.4 分数阶差分跨股票边界污染(同上自认) | 剔除 close_d0.4 及其 rankpct/z(共 3 列) | 同上 |
| P5 | D5:boxcox_atr 的 lambda 用全窗全股票一次性估计 | lambda 按**同日全市场横截面**估计(收盘可知,因果) | 保变换逻辑,修统计量口径 |
| P6 | D6:vol_divergence 分母为全帧全局标量 | 分母 = rolling(60).std() 的**滚动 100 个交易日内全市场均值**(只用 ≤ 当日数据) | 同上 |
| P7 | D7:macd_percentile 的 100 行窗口跨股票 | 按 symbol 取截尾 100 行(不含当日)percentileofscore(kind='rank');不足 100 行 → NaN(不删行,NaN 交由模型侧 fillna(0) 口径,见 §七) | 修分组,保公式 |
| P8 | D8:is_quick_divergence_x 恒 0 废列 | 剔除;is_quick_divergence_y 保留真值式 = `anchor_bars < 3` | 零方差废列无信息 |
| P9 | v2 复制列(macd_signal_cross≡macd_golden_cross、macd_signal_convergence≡macd_signal_distance、sh/sz_price_wave_abs≡amplitude、daily_return≡pct_change) | 保留(v2 原样,零成本) | 原样复刻 |
| P10 | compare_rank ∈ {1,2}(v2 对前 1/前 2 低点各出一行) | v6 事件恒为相邻金叉对 → compare_rank 恒 1(常量列保留);j=2 语义另立**增补列** `v2j2_macd_increase_pct = (cross_dif − cross_prev2_dif)/abs(cross_prev2_dif)`(cross_prev2_dif=0 → 0),命名隔离,明示非 v2 原列 | v6 事件定义不含 j=2 行;增补列是给模型的额外信息,不计入「v2 原样」口径 |
| P11 | divergence_amount = 当日 v2 检测器事件总数 | = 当日 v6 事件表行数(按 event_date 计数) | 语义对应物 |
| P12 | volume_signal 需前低点成交量 | 取 cross_prev_date 当日个股成交量;比较与编码规则按 v2(event_vol < prev → bullish;> 1.5×prev → bearish;否则 neutral;LabelEncoder 拟合只在 train 段) | 数据可得,公式不变 |
| P13 | v2 训练/测试 = 85/15 时间切分 | 不映射;训练 = train 段(扣带),OOF 在 train 内 5 折,分数段 = val(居 v2 test 的协议位置),test 零触碰 | 战役段界纪律(#30 US12)优先 |
| P14 | v2 标签不含手续费 | 原样不含;生死门引擎自带真实成本,与标签构造无关 | 原样复刻 |
| P15 | v2 特征快照只用最后 100 个交易日计算(Wilder 系未收敛) | 逐股**全历史**计算(因果,EMA 只回看) | #30 US6 既有口径;消除 warm-up 截断偏差 |
| P16 | v2 optimal_threshold 搜索(F0.5) | 不复刻(v2 中纯信息性,不影响打分;解剖 §5.6) | 死代码不迁 |

未列入本表的 v2 行为一律原样复刻;施工中发现的任何新缺陷,停下来记录并向监工报告,不私改。

## 四、特征层规格(族级;逐列公式 = anatomy_report.md §2,落盘词典为准)

1. 基础 TA 族(talib,逐股全历史):MACD(12,26,9) 三列、RSI(6/14/24)、MA(5/20/60)、BBANDS(20,2,2) 三列、volume_ma_20、OBV、ATR(14)、STOCH(5,3,3) 两列。
2. 进阶 TA 族:price_vs_ma{5,20,60}、ma_arrangement、bb_position、bb_squeeze、rsi_oversold/overbought_{6,14,24}、macd_signal_distance、volume_ma20、volume_ratio、volume_spike、volume_dryup、atr_ratio、hammer_pattern、downtrend、hammer_signal、doji_pattern、distance_to_support/resistance(20 窗)、stoch_oversold/overbought、macd_percentile(P7 口径)、obv_trend(6 点斜率)、volatility_{3,5,10,15,20,25,30}d、engulfing_pattern、signed_volume_strength、close_vs_high、volume_ma_ratio(shift(1))、rsi_momentum、rsi_turning_simple、rsi_turning、volume_trend_{5,10}、price_trend_5、price_volume_divergence、volume_consistency。
3. MACD 深度族:macd_hist_trend_5、macd_golden_cross、macd_death_cross、macd_signal_cross(P9 复制列)、macd_zero_cross_up/down、macd_zero_cross、macd_hist_amplitude、macd_hist_direction、macd_hist_acceleration、macd_signal_convergence(P9 复制列)、macd_signal_convergence_trend。
4. alpha 族:pct_change、clv、upper_shadow_ratio、body_strength、rank_return、rank_volume、signed_vol_strength、pv_corr_10、dist_to_high_60、vol_divergence(P6 口径)。
5. 结构化族:vol_gk、vol_gk_ratio、illiq、efficiency_ratio、intraday_pos、ret_overnight、ret_intraday、smart_money_diff、high_mean_20、low_mean_20、support_resistance_ratio、log_volume、boxcox_atr(P5 口径)、rsi_robust、macd_robust(逐股全历史 expanding 中位数/IQR,只用 ≤ 当日;IQR=0 退化为 x−median)。
   注:rsi_robust/macd_robust 的 v2 口径是「当前 100 天窗口全窗统计」,本线取 expanding 全历史(与 P15 同原则),登记为 P5~P7 同类修正。
6. lag 族:OHLC × lag{3,5,10,15,20,25,30}、volume × 同七档、daily_return + return_lag{1,2,3,5,10,20}、amplitude + amplitude_lag{1,3,5}、vol_gk/vol_gk_ratio_lag{1,3,5,10}、illiq_lag{1,3,5}、efficiency_ratio_lag{1,3,5}、intraday_pos_lag{1,2,3}、smart_money_diff/ret_overnight/ret_intraday_lag{1,3,5}、support_resistance_ratio_lag{1,3,5}。
7. 市场大盘族:sh_*/sz_* 各 7 列(price_change、amplitude、volume_ratio、price_change_abs、price_wave_abs、sentiment、volume_signal,指数序列截断 ≤ 当日,公式 = 解剖 §2.6)+ sh_sz_sync_direction、sh_sz_sync_strength、market_avg_change、market_avg_amplitude、market_sentiment、market_sync_score。
8. 背离事件结构族(v6 语义改写,公式 = 解剖 §6.3 逐条):close_current(=anchor_close)、close_previous(=min_prev_close)、macd_current(=cross_dif)、macd_previous(=cross_prev_dif)、price_decline_pct、macd_increase_pct、compare_rank(P10)、formation_period(=anchor_bars)、is_quick_divergence_y、divergence_strength、volume_signal(P12)、price_macd_ratio、divergence_magnitude、confirmation_score、divergence_amount(P11)、v2j2_macd_increase_pct(P10 增补)。
9. 横截面族:对第 1~7 族全部 float64/int64 列(v2 排除集原样:timestamp/symbol/label/OHLC/28 标签列/bool 列)按 event_date 的全市场当日行做 rank(pct=True) 与 (x−median)/(std(ddof=0)+1e-9);cs_n = 当日全市场行数。
   预计列数 ≈ 610 − 6(P3/P4)− 1(P8)+ 1(P10 增补)= 604 左右,以落盘词典为准,词典含中文全名列(命名全称纪律)。

## 五、标签规格(v2 公式逐项)

- 持有期 RETURN_PERIODS = [3,5,10,15,20,25,30]。
- 买入价 = 事件日收盘价(个股行号 j);窗口 = [j+2, j+1+p] 含端点(个股自身交易日行号)。
- 原版收益:窗口内任一交易日 high ≥ buy×1.15 → 收益 = +0.15,卖出日 = 首触日;否则收益 = close[j+1+p]/buy − 1。
- 止损版收益(信息列):自 j+1 起逐日,open ≤ buy×0.65 → 按 open 卖;low ≤ buy×0.65 → 按 buy×0.65 卖;high ≥ buy×1.15 → 按 buy×1.15 卖;时间优先;皆未触发 → 期末收盘。
- 窗口末端超出个股历史 → 该期标签 NaN(不删行)。
- 训练标签 = (future_return_15d > 0.01),NaN → 不入训练行(与 v2 的 data_clean 删 NaN 行对应,但不作 P2 的次日 low 过滤)。
- 28 个标签列(future_return/future_sell_date/stop_loss_return/stop_loss_sell_date × 7)全部隔离落盘,绝不进特征表(断言守护)。

## 六、特征选择规则(v2 程序化复刻,先于跑数)

1. 基座 LGBM(§七参数)在 train 段(扣带,标签非 NaN)以 FULL 全列训练,TimeSeriesSplit 5 折,逐折取 feature_importances_ top-250,入选折占比 ≥ 0.6 的特征 → `OPTIMIZED_V2ON6` 清单落盘(条数落盘,不作条数预期)。
2. 定版基座 = OPTIMIZED_V2ON6 上重跑同一 5 折管线(OOF + 折权重)。
3. STABLE_V2ON6:在 train OOF 的栈结构上(`pred_lgb, pred_lgb_squared` + FULL 全列,StandardScaler,LR 同 §七参数)跑 `collect_lr_coefs`(TimeSeriesSplit 12 折)与 `analyze_coef_stability`,取 selection_rate ≥ 0.6 者 → STABLE_V2ON6 清单落盘。
4. 终版元模型输入 = pred_lgb + pred_lgb_squared + STABLE_V2ON6。

## 七、训练管线(v2 架构逐项)

- LGB_PARAMS 逐字 = comm_fun.py L206-229(n_estimators=1000、learning_rate=0.02、num_leaves=63、max_depth=4、min_child_samples=58、min_split_gain=0.6、min_child_weight=0.03、reg_alpha=2、reg_lambda=2、bagging_fraction=0.875、bagging_freq=7、feature_fraction=0.85、scale_pos_weight=1.0、objective=binary、random_state=42、verbosity=-1;n_jobs 按本机核数落盘)。
- 动态覆盖:折内训练样本 <10,000 时 min_child_samples = max(5, int(len×0.01))(v2 原样)。
- 特征矩阵:float32、inf→NaN→fillna(0)(v2 prepare_features 原样);median imputer 在 train 上 fit(v2 原样,实为冗余保险)。
- TimeSeriesSplit 5 折,折内 fit(eval_set=验证折, eval_metric=[auc,binary_logloss], early_stopping 150 轮)。
- 折权重 = 折验证段 PR-AUC 归一化;val 段基座概率 = 5 折模型概率加权平均。
- 元模型:LR(penalty=l1, solver=saga, C=0.01, class_weight=balanced, max_iter=1000, fit_intercept=True, random_state=42),输入 StandardScaler(train 栈 fit)变换后的 [pred_lgb, pred_lgb_squared, STABLE_V2ON6]。
- 最终分数 = lr_meta 概率;`scores_v2on6.parquet` 列 = event_id/ts_code/date/seg/score 恰五列,覆盖主表全 96,577 行(train 行 = OOF;val/test/embargo/pre2001 行 = 折加权基座 + 元模型;标签 NaN 的 train 行 score=NaN)。
- 复现断言:全链第二进程重跑,清单/分数逐位一致,双 md5 落台账。

## 八、审判口径(M5 同款生死门,机械执行)

- 驱动 = 克隆 `run_stage_gate_m5.py` 为 `run_gate_v2on6.py`,改动面白名单:scores 输入路径、输出目录、格 ID 前缀(v2on6)、verdict/报告文件名、README 引用;**机制零改动**(底座 md5 断言、E1_A6、S1~S5、P7、cluster_t、退化格对账、标签对账、双跑 detcmp 全部继承)。
- 网格 = 18 格:挑选规则 {M=v2-stack 分数, S1 先到先得, S2 前20日跌幅最深, S3 种子强度, S4 随机(种子 42), S5 已弹最少+量比最冷} × K∈{3,5,10};骨架 P7(100万÷K)+ E1_A6(tp=+0.25, sl=−0.18)+ H=60;事件集 = val 段 24,405 起(与 M5 同集,逐键一致断言)。
- 主门(生死):任一 K 档 M 净笔均 ≥ max(S1~S5 同 K)+2pp 且 M cluster_t ≥ 2 → 过线。
- 过线 → 推翻 #52 的 MODEL_ROUTE_DEAD,模型路线复活,承认此前管线裁决有误;全不过 → v2 路线与 M 路线互证判死,「170 vs 10」归因于 v2 池污染与宽松选择规则,回用户拍板。
- 副门(信息性):precision@top10%(pooled,与 M5 同口径)。
- 日历延伸、truncated_window==0 硬断言、标签对账、退化格对账均按 M5 预登记 §四原样。

## 九、验收断言清单(执行版,全过才算完)

1. 因果性:抽样 12 股 × 3 日截断重算(仅用 ≤ 当日数据)全特征逐位一致(rtol=1e-9);横截面族同日性断言(rp/z 仅依赖当日截面)。
2. 泄漏:特征表无 future_/stop_loss_/label_ 列;事件表直透列最晚可知时点 = 事件日收盘(M1 断言继承);LabelEncoder/imputer/StandardScaler/lambda 拟合侧全部只用 train 段或同日截面。
3. 段界:`assert_segment_integrity` 在主表执行;训练行 seg 全 'train';选择/SHAP 不用 val 以外段;test 零触碰(分数表无标签列,台账无 test 指标)。
4. 行数与键:特征表 96,577 行守恒,event_id 唯一,(ts_code, event_date) 与事件表互证;event_close 对日线 close 逐位对账(复权口径断言)。
5. 背离结构族:price_decline_pct/macd_increase_pct/formation_period 等抽样 500 事件自写重算逐位一致;seed 布尔不在本线(归 M2 口径,不重算)。
6. 词典:每列中文全称非空。
7. 确定性:特征面板、标签表、分数表各双跑 md5 逐位一致。
8. 选择规则程序化:OPTIMIZED_V2ON6/STABLE_V2ON6 由代码产出,清单与重要性/系数台账同盘。
9. 门:18 格全出数(配置为行),trades/equity 落盘,verdict JSON 机械生成。

## 十、工程量与预算(解剖 §7 为据)

- 瓶颈 = 全市场逐日基础特征面板(约 5,188 股 × 全历史 × 约 200 列):v2 的 Python 循环(slope/cv/percentile/obv_trend)全部向量化重写;存储与 M2 的 v4daily parts 同量级(约 21 GiB)。
- 标签扫描向量化(首触 argmax 法)。
- 预算:特征面板 ≤ 3 小时,横截面聚合 ≤ 1 小时,标签 ≤ 20 分钟,训练+选择 ≤ 30 分钟,门 ≤ 5 分钟(M5 实测 11 秒级);全程落 progress.log 心跳。
- 冒烟纪律:全量前先 50 股 × 200 日冒烟,实测写入修订记录后再开全量。

## 十一、产物清单与路径(全部在 experiments/v2_on_v6_rerun/)

| 产物 | 入库 | 说明 |
|---|---|---|
| README.md | 是 | 本预登记(冻结,变更只追加修订记录) |
| anatomy_report.md | 是 | v2 解剖报告(md5 67fe294bd1590ecf9b1f3b6237366863) |
| build_panel_v2on6.py | 是 | 全市场逐日基础特征面板构建(含横截面聚合) |
| build_labels_v2on6.py | 是 | 标签构建(向量化) |
| build_master_v2on6.py | 是 | 事件×特征主表合成(事件行切片 + 背离结构族 + 词典) |
| train_stack_v2on6.py | 是 | 选择 + 训练 + 三段分数 |
| run_gate_v2on6.py | 是 | M5 驱动克隆(改动面白名单见 §八) |
| selfcheck_v2on6.py | 是 | 独立自检(不 import 上述驱动任何函数) |
| master_v2on6.parquet / labels_v2on6.parquet / scores_v2on6.parquet | 否(*.parquet) | 数据产物 |
| dictionary_v2on6.csv | 是 | 特征词典(含中文全称) |
| optimized_features_v2on6.json / stable_features_v2on6.json | 是 | 程序化选择清单 |
| selection_results_v2on6.json | 是 | 选择台账(重要性/系数/折明细) |
| summary_gate_v2on6.csv / verdict_gate_v2on6.json / report.md | 是 | 生死门产物(18 格全出数) |
| progress.log | 否(*.log) | 长任务心跳 |

## 十二、执行纪律

- 长任务落 progress.log(驱动名标签+心跳);派活即报日志路径与预计时长。
- 对 v3_pipeline 与 M1~M5 既有产物零改动;发现 bug 停下来记录并向监工报告,不私改。
- 汇报口径:配置为行全出数;分布统计用「覆盖率→门槛」方向;特征命名用中文全称;信号池价值与池内排序增益分开表述。
- commit 到 v6-portfolio,不加 co-author 行,不 push;不关 #54(监工独立证伪式复核后处置)。

## 修订记录(预登记正文逐字未动,仅追加)

### 2026-09-11 冒烟实测记录(build_panel_v2on6.py --smoke,50 股 × 最近 200 事件日)

- 实测宇宙:stock_data/daily/*.parquet 共 5,891 个文件,减 2 个指数文件 = **5,889 股**(§十 估计「约 5,188 股」为旧估计,以实测为准);事件表 96,577 行、(ts_code,event_date) 无重复、事件日 **5,696 个**(§一 估计「约 4,000+ 日」以实测为准)。
- 冒烟范围:字母序前 50 股 × 最近 200 个事件日(2025-11-06 ~ 2026-08-31);事件行切出 61 行 = 该 50 股在 200 日内的事件数,与事件表逐日核对全等。
- 实测机时(28 核,workers=26):Pass A 逐股特征 1s(50 股,全历史行 282,316,事件日行 7,517);因果性抽检 3s(12 股 × 3 日 = 36 单元全过,rtol=1e-9 equal_nan);市场族+P6 分母 <1s(sh 缺日 0、sz 缺日 0、非指数日 vol_long 行 1,098);重打包 <1s;横截面聚合 27s(200 日,1 块);全程 32s。
- P5 台账:boxcox_lambda_smoke.parquet 200 日全出数,lambda 范围 [-0.4053, -0.0074],无非有限值。
- 冒烟结论:全链机检(§九.1 因果性抽检、事件行计数、event_close/event_vol/event_amount 逐位对账、event_row 行号一致、cross_prev_date 在日线存在)全过,开全量。

### 2026-09-11 口径裁定:止损版标签窗口(§五 行文歧义,按军令状以 anatomy 为准)

- 歧义:§五 止损版行文「自 j+1 起逐日」与 §五 窗口定义「窗口 = [j+2, j+1+p] 含端点」字面冲突。
- 裁定依据:军令状规定公式唯一来源 = anatomy_report.md;anatomy §1.3(L67)明确「从买入日后第 1 天到第 p 天逐日检查(L565-592)」,买入日 = j+1,故首检日 = j+2;v2 源码 `_calculate_stop_loss_return` 循环 `days_after_buy in range(1, p+1)`、`current_idx = (j+1) + days_after_buy`,首检日同为 j+2。
- 裁定:止损版窗口 = [j+2, j+1+p],与原版收益窗口一致;「自 j+1 起」按「自买入日(j+1)之后起」解读。
- 影响面:仅止损版信息列(不进训练标签);训练标签 = future_return_15d > 0.01 用原版收益,窗口本无歧义。

### 2026-09-11 口径裁定:P6 分母与市场族在非指数事件日的取值(全量首跑停工记录)

- 停工事实:全量首跑在横截面聚合阶段 KeyError —— 8 个事件日(1992-12-25、1993-06-04、1993-07-20、1993-07-21、1993-07-23、1993-07-27、1993-09-06、1993-09-21,均 pre2001 段)不在上证指数日线日历内(个股当日有交易、指数文件缺日,早期数据缺口);深证成指另多缺 1995-02-07、1995-02-09 两天(共 10 天)。
- 市场族:§四.7/解剖 §2.6 已登记缺失日 = v2 默认行,原样执行(8/10 天默认行,计数落台账)。
- P6 分母裁定:预登记只定义「市场交易日历 = 上证指数交易日」上的分母序列,未定义非指数事件日的取值;裁定取 **≤ 当日的最近市场交易日分母**(pandas asof,只用 ≤ 当日数据,因果保持),8 天逐日名单落台账(panel_ledger 的 denom_asof_dates)。
- 处置:修复后以 --aggregate-only 复用 Pass A/重打包落盘产物重跑聚合;本裁定同步向监工报告。

### 2026-09-11 口径裁定(二轮,作废上轮 asof):P6 分母日历 = 全市场个股日线交易日并集

- 新事实:asof 修复后聚合再停 —— 1992-12-25 的 asof 无值可取;排查确认**指数日线文件仅近 8,000 行**(000001.SH 起于 1993-10-08、399001.SZ 起于 1993-09-30),1993-10 前的市场交易日指数日历物理缺失,上轮「asof 回退」裁定对该时段无解,作废。
- 二轮裁定:P6 分母的市场交易日历由「上证指数交易日」改为「全市场个股日线 trade_date 并集」。
  理由:P6 预登记语义为「滚动 100 个交易日内全市场均值(只用 ≤ 当日数据)」;个股日线并集是该语义在无指数日历时段的唯一因果可实现口径,1993-10-08 后与指数日历实测一致(两日历差异日全在指数起点前,落台账 n_days_pre_index_start);并集日历每一天由构造必有 ≥1 股 vol_long 有限,M_t 无 NaN,每个事件日必有分母。
- 影响面:仅 vol_divergence 一列(及其 rankpct/z 两列);1993-10-08 前的 8 个事件日由「无分母」变为正常取值,1993-10-08~1994 年初的滚动窗因并集日历多出的早期交易日有轻微数值变化,之后逐位不变。
- 冒烟修订记录中「市场交易日历 = 上证指数交易日」一句由本裁定取代。

### 2026-09-11 全量实测记录(build_panel_v2on6.py --full,run1)

- 实测机时(28 核,workers=26):Pass A 逐股特征 545s(5,889 股,全历史);因果性抽检 36 单元全过;市场族+P6 分母 15s(sh 缺日 8、sz 缺日 10、并集日历 8,610 日、sidecar 16,975,096 行、指数起点前 610 日、零覆盖日 0);重打包 501s(23 块,15,455,882 行);横截面聚合 1,061s(23 块);全程约 35 分钟。
- 产出:事件行 **96,577**(与事件表行数一致,§九.4);boxcox_lambda_run1.parquet **5,696 日全出数**,lambda 范围 [-1.1417, 0.6673],无非有限值;事件行分日 md5 台账落 panel_ledger_run1.json(供 §九.7 双跑逐位比对)。
- 标签线独立实测(build_labels_v2on6.py,双跑):每次约 4s(5,748 事件股);run1/run2 md5 逐位一致 = ce6a584fb65716ac777ed98669f00b46;窗口末端超史 NaN 计数 {3d:326, 5d:412, 10d:476, 15d:522, 20d:696, 25d:1229, 30d:1490}(事件日临近数据末端所致,台账落盘)。

### 2026-09-11 全链收官实测记录(主表/训练/门/自检,全断言机检通过)

- 主表(build_master_v2on6.py,双跑):每次 7~8s;(96,577, 609) = 5 键列 + 604 特征列;run1/run2 md5 逐位一致 = a9fe203bcfb2273dad9b8ccb8c822714;背离结构族 500 事件抽检全过;词典 609 列中文全称全非空;段计数与 §三 PREREG 逐位一致;`assert_segment_integrity` 通过。
- 面板双跑(§九.7):run2 全程 1,623s(Pass A 545s + 因果抽检 + 市场族 15s + 重打包 + 聚合 1,063s);eventrows 23 个分块文件 run1/run2 md5 逐位一致,事件行 96,577 双跑守恒。
- 训练(train_stack_v2on6.py,双跑):每次约 780s;§六.1 FULL 604 五折 OOF PR-AUC 0.4979;OPTIMIZED_V2ON6 = **250 条**(≥60% 折规则顶格);§六.2 定版基座 OOF PR-AUC 0.4933,折权重 [0.2284, 0.1434, 0.2536, 0.1755, 0.1992];§六.3 STABLE_V2ON6 = **64 条**(含 pred 列 0 条,12 折系数 selection_rate≥0.6);元模型输入 66 列(pred_lgb + pred_lgb_squared + 64);分数表 (96,577, 5),run1/run2 md5 逐位一致 = af2903e7a0c4181db457b96ad106a41c;train 段 score NaN 4 行 = train 标签 NaN 4 行(窗口末端超史),val 24,405 行全覆盖。
- 生死门(run_gate_v2on6.py --mode full,15s):底座 run_long_hold.py md5 校验通过;18 格全出数;pass2 确定性重算 + detcmp 双跑逐位对拍 0 不一致;#40 机制对账(退化全覆盖 + 网格交集)5 条手工规则全 PASS;标签同机制对账 18/18 PASS。
  - 主门三档全不过:**K=3** M 净笔均 +2.5773% vs 最好手工 S5 +1.9923%,差值 +0.5850pp(阈值 ≥+2pp),cluster_t 0.9856(阈值 ≥2);**K=5** M −0.2144% vs S2 +0.7649%,差值 −0.9793pp,cluster_t −0.1174;**K=10** M +0.6342% vs S2 +0.7687%,差值 −0.1345pp,cluster_t 0.4684。
  - 宣判:**V2_ROUTE_DEAD** —— 按 §八 原文,v2 路线与 M 路线互证判死,「170 vs 10」归因于 v2 池污染与宽松选择规则,回用户拍板。
  - 副门(信息性):precision@top10% = 0.438351 vs 池基线 0.459768,提升 **−2.14pp**(排序全集 24,247,前 2,425)。
- 独立自检(selfcheck_v2on6.py):13 项全 PASS(行数与键/泄漏扫描/段界/特征口径抽检 192 单元/背离结构族 500×16/标签 300×7×4 自写重算/词典/三件套双跑 md5/选择清单/分数表口径/门产物机械一致),7s。

### 2026-09-11 口径裁定(监工复核发现,三轮):macd_percentile 暖机窗 NaN 口径 = scipy 字面语义

- 发现(监工独立复核 B2 层):`_macd_percentile_p7` 对「窗内含 NaN 的 100 行窗」(事件行号 j∈[100,134),talib MACD 暖机 33 行 NaN 落入窗内)给出 (less+leq)/2/100 实值;P7 登记引用函数 `scipy.stats.percentileofscore(kind='rank')` 在本环境(scipy 1.17.1)对含 NaN 输入返回 NaN。
  工人代码注释「与 scipy count_nonzero 语义一致」经实测不成立(scipy 1.17.1 实测:含 NaN → NaN)。
- 裁定:P7 暖机语义以 scipy 字面为准 —— 窗内含任一 NaN → 整窗结果 NaN(随后经 v2 L897 面板 fillna(0) → 0.0,与原管线汇合);不足 100 行 → NaN(原实现已一致)。
- 影响面:j∈[100,134) 事件 1,114 起(train 596 / val 341 / test 93 / embargo 66 / pre2001 18,占全池 1.15%)的 macd_percentile 基列由 0~34 实值变为 0.0,及其 rankpct/z 两横截面衍生列在含暖机股的事件日微变。
  j<100 的 592 起事件修复前后落盘值不变(均为 NaN→fillna(0)→0.0)。
  FULL 阶段五折重要性台账证实 macd_percentile 三列分裂数全 0(LGBM 路径零影响);STABLE 64 条含 macd_percentile(LR 元模型路径理论上非零影响)。
- 处置:按「构造缺陷 → 修复重跑」纪律,修复后全链重跑(面板 p7fix/p7fix2 双跑 → 主表 → 训练双跑 → 终审复跑;终审 scores 输入路径与主表输入路径指向 p7fix 产物,均属 §八/§九 白名单与输入路径项)。
  修复前裁决(K=3 +0.5850pp/t 0.9856,K=5 −0.9793pp/t −0.1174,K=10 −0.1345pp/t 0.4684,副门 −2.14pp,宣判 V2_ROUTE_DEAD)已在 commit 5e9d7ff 全量归档,修复后数字连同前后对照一并落盘。

### 2026-09-11 口径裁定(监工复核发现,四轮):build_master 输入目录硬编码 run1 → 随 tag 走;三轮暖机行数勘误

- 发现(监工独立复核,p7fix 链后验):p7fix/p7fix2 面板逐股 part 已是修复后值(暖机窗 macd_percentile=0.0,实测 000001.SZ 2001-09-19),但 `build_master_v2on6.py` 输入目录硬编码 `eventrows_run1`/`event_aux_run1`(原 382-387 行),`--rerun-tag` 只改输出文件名 —— p7fix/p7fix2 主表静默吃 run1 修复前事件行(md5 与 run1 全等,scores md5 随之与修复前全等,终审复跑等于没跑)。
  该硬编码对 run2 无害(run1/run2 事件行按设计逐位相等,双跑对账由面板台账 md5 独立承担),仅在修复链暴露。
- 裁定:build_master 输入目录随 `--rerun-tag` 走(`eventrows_{tag}`/`event_aux_{tag}`,断言信息同步);标签输入 `labels_v2on6_run1` 不受 macd_percentile 修复影响(标签只由未来价格计算),维持不变。
- 勘误(三轮):talib MACD 线暖机实测 33 行 NaN(首个有效 index=33),故含 NaN 窗事件行号为 j∈[100,132],计 1,069 起(train 569 / val 331 / test 61 / embargo 46 / pre2001 62,占全池 1.11%);三轮文中「j∈[100,134)、1,114 起」的分段计数以此为准,口径结论不变。
- 处置:p7fix/p7fix2 面板产物(逐股 part、事件行 chunks、台账 md5)本身正确,保留;自主表起下游重跑(build_master p7fix/p7fix2 → train p7fix/p7fix2 → 终审复跑),修复前裁决归档不变(commit 5e9d7ff)。
  三轮修复链已产生的主表/分数/终审产物全部由本轮重跑覆盖,不作证据引用。
- 附记(同日终审复跑停工排查):build_master 修复后首次终审复跑 detcmp 报 2 格(M K3/K10)不一致 —— 排查确认为 `checkpoint_pass1.pkl`(修复前 13:14 留存,gitignored)被断点续跑机制当作 pass1 结果复用,与新分数 pass2 对拍自然不符;单格 emit 双跑逐位相等证明引擎确定性本身无恙。
  处置:删除陈旧 checkpoint 后全新终审复跑;此为运维面教训(checkpoint 未按分数 md5 命名),非引擎缺陷,机制代码不改。
