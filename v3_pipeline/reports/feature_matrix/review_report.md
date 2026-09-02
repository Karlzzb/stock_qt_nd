# 特征引擎独立复核报告

> 复核对象： `v3_pipeline/src/feature_engine.py`、`v3_pipeline/scripts/build_feature_matrix.py`、`v3_pipeline/tests/test_feature_engine.py`、`v3_pipeline/reports/feature_matrix/{main,backup}_pool_features.parquet`、`feature_dictionary.csv`。
> 复核依据： `v3_pipeline/reports/feature_harvest/feature_master_spec.md`（主表 v1.0）。
> 复核立场： 怀疑者，逐项实证后才认可。
> 复核日期： 2026-09-01。
> 复核过程未修改被复核代码（第 3 项红绿验证的临时篡改已恢复，恢复后 `git diff` 为空、测试全绿）。

## 总体判定： 有条件通过

唯一实质缺陷是 backup 池矩阵中 1 个 inf 单元格（引擎 corr 输出缺有限性护栏，系统性问题、当前仅 1  cell 泄漏进交付物）。
其余全部声称复验成立。
建议： 下游训练前给 corr 类特征加有限性清洗（或接受该 1 cell 由模型 NaN/inf 处理兜底），并对"指数文件事件行进矩阵"做一次明确取舍。

---

## 1. 重跑测试 —— 通过

- `pytest v3_pipeline/tests/` 实跑： **63 passed**（本机 6.0s），与声称的 63 个一致。
- 其中 `test_feature_engine.py` 21 个全过，与交付说明一致。
- "24 秒"无法复验（机器相关，本机 6s），不影响正确性结论。

## 2. 公式抽验 —— 通过

**种子随机抽样 10 特征（seed=7，覆盖 10 个不同家族），逐条对照字典公式与代码实现：**

| 列 | 家族 | 字典公式 | 代码核对 |
|---|---|---|---|
| RET10 | 反转 | CF/shift(CF,10)-1 | 一致（L515-516） |
| RET_ID | 隔夜/日内 | C/O-1 | 一致，原始价口径符合主表 1.2 |
| CPV10 | 量价相关 | Corr(C̃,V,10) | 一致，min_periods=10 |
| DIST_SUPPORT20 | 位置结构 | (C̃-Min(L̃,20))/C̃ | 一致 |
| CNTD20 | 技术指标 | (Count(R>0,20)-Count(R<0,20))/20 | 一致 |
| IMIN20 | 位置 | IdxMin(L̃,20)/20 | 一致（`_idx_extreme` ties 取最旧） |
| LOG_PRICE | 价格水平 | ln(C) 原始价 | 一致，失真标注符合主表 1.2 |
| CAL_DOW_2 | 日历 | 星期 one-hot (0=Mon) | 一致（(days+3)%7，1970-01-01 周四锚定正确） |
| MKT_SH_PRICE_CHANGE_ABS | 市场状态 | \|(C-O)/O\| | 一致 |
| DIV_ZERO_AXIS_DEPTH | 事件结构 | max(DIF,DEA)[T]/(C̃[T]×ATRp[T]) | 一致 |

**小样本独立手算 3 特征（300721.SZ @ 2019-06-28，event_id=4000，从原始 parquet 独立重算）：**

- RET20： 矩阵 -0.0091700237 vs 手算 -0.0091700238，差 6.8e-11（f32 舍入内）。
- ATRN： 矩阵 0.0364816636 vs 手算（talib.ATR(14) 复权价/C̃）0.0364816650，差 1.4e-09。
- CPV20： 矩阵 0.7753895521 vs 手算 0.7753895618，差 9.7e-09。
- 附带验证： 矩阵 `sig_idx` 与独立清洗（dropna/dedup/sort）后的行号一致。

## 3. 泄漏防线实测 —— 通过（附 1 条观察）

**① 黑名单 14 条硬卡点： 通过。**

- 14 条正则全覆盖注入（含大小写变体、`rank_` 前缀全家）均被 `assert_no_blacklisted` 拦截。
- 关键场景实证： 向注册表注入泄漏名 `stop_loss_return_9` 后走 `assemble_pool`，构建硬失败（"特征矩阵列命中黑名单"）。
- 观察（低危）： 经 `feat_rows` 直接注入未注册坏列时，`assemble_pool` 不报错而是**静默丢弃**（`feature_column_order` 只保留注册表列）。
- 结果安全（输出矩阵绝不可能含黑名单列），但检测是静默的，注入尝试不会留下痕迹。

**② 截断对拍红绿验证： 通过。**

- 临时将 RET5/10/20/60 改为 `scf.shift(-5)/scf.shift(w)-1`（未来数据），3 个 `test_truncation_recompute_stock_features` 全部变红；恢复后 63/63 全绿，`git diff` 为空。
- 证明截断对拍确实在断言数值差异，不是空转。

**③ 矩阵无标签列： 通过。**

- 两个交付 parquet 的全部 183 列实测通过黑名单 + 标签命名空间断言。
- 引擎源码不读 labels.parquet（仅 docstring 提及），事件输入只来自 events.parquet。

## 4. 矩阵体检 —— 存疑（1 个 inf cell）

| 项 | main | backup |
|---|---|---|
| 形状 | 8158×183 ✓ | 37012×183 ✓ |
| event_id 与 events.parquet 集合一致 | ✓ 无重复 | ✓ 无重复 |
| 事件日期范围 | 1993-04-26 ~ 2026-05-26，与 events 完全一致 | 1992-10-12 ~ 2026-08-31，与 events 完全一致 |
| 逐行 date 对拍 | 0 处不符 | 0 处不符 |
| 整行重复 | 0 | 0 |
| NaN 总占比 | 0.38% | 0.57% |
| **inf** | 0 | **1** |

- NaN 分布合理： Top 列为 BBW_PCTILE250（7.7%/10.4%）、VOL_DRYUP_EXTREME、POS_52W/DIST_52W_*/MA_STACK（250 日窗，新股历史不足）、T1_GAP（窗口无符合条件的交易日）等，均有明确机制解释；无列 NaN>50%。
- **问题： backup 池 HLV_DIV10 在 000556.SZ @ 2001-11-02 为 inf。**
- 根因已定位： 该股该窗口连续 10 根一字板（H==L），H/L 为常数序列，pandas `rolling.corr` 在零方差窗口数值上产出 inf 而非 NaN；引擎对 corr 类输出无有限性护栏。
- 实测该股全历史中此类 inf 共 31 个；交付矩阵仅漏入 1 cell，但 CPV10/CPV20/CPV_VWAP10/RVC20 共享同一暴露面。
- 主表口径下常数窗口相关应为 NaN，属实现缺陷（低危，1 cell；系统性护栏缺失）。

## 5. 口径一致性 —— 通过

**CF 复权链（主表 1.2）真实除权日验证： 通过。**

- 600036.SH @ 2025-07-11（除息日）： 原始价收益 -5.39%，真实收益（pct_chg/100）-1.30%。
- 引擎 RET1 = -0.01298 ✓（未把除权失真当收益）。
- RET20（i+10，窗口跨除权日）： 引擎 0.0119 = CF 链手算 0.0119；若用原始价链则为 -0.0301。
- 证明跨日价格比较确实走 CF 复权链。

**事件锚点抽 5 行手验： 通过。**

- 随机 5 事件（600379.SH/300096.SZ/002216.SZ/603533.SH/000502.SZ），i1/i2 由 events.parquet 的 prev_low_date/low_date 独立定位。
- DIV_SPAN_BARS、DAYS_SINCE_L2、DIV_PRICE_NEWLOW_DEPTH、REBOUND_FROM_L2、RSI_DIV 全部与手算一致（f32 舍入内）。
- 锚点映射方向正确： i1=prev_low_date（前低）、i2=low_date（次低），与主表记号一致。

## 6. 声称核对

| 声称 | 复核结果 |
|---|---|
| 63 个测试通过 | ✓ 实跑 63/63 |
| 24 秒 | 未能复验（本机 6s，机器相关，非正确性问题） |
| 锚点缺失 0 | ✓ 独立重算两池锚点（low_date/prev_low_date 逐一对该股清洗后交易日集合），main 0 / backup 0 |
| REGIME 一致率 100% / 99.98% | ✓ 与 events.parquet 的 regime 列逐事件对拍： main 100.0000%、backup 99.9757%（9/37012 不符，均为 ±10% 阈值边界的浮点/口径微差，方向可解释） |
| 矩阵形状 8158×183 / 37012×183 | ✓ |
| 字典 179 特征 | ✓（179 行；183 = 4 meta + 179 特征） |

## 问题清单（按严重度）

1. **低（建议修复）： corr 输出无有限性护栏** —— backup 矩阵 1 个 inf cell（HLV_DIV10, 000556.SZ 2001-11-02）；同一暴露面覆盖全部 rolling-corr 特征；引擎及装配层均无 inf 断言（本复核的 inf 检查是事后手工做的，不是管线自带关卡）。
2. **低（观察）： `assemble_pool` 对未注册列静默丢弃而非报错** —— 黑名单硬卡点仅对注册表来源列有效（已实证该路径会硬失败）；直接注入未注册泄漏列会被无声丢弃，无审计痕迹。
3. **观察： 指数文件事件行进矩阵**（main 4 行 / backup 26 行，000001.SH/399001.SZ 上检测出的"背离事件"）—— 继承自 divergence_lab 事件集，指数行的制度类特征口径无意义（如对指数按 10% 涨停计 LIMITCNT）；下游训练/标签侧需明确取舍。
4. **观察： NEW_LISTING 用"股内行号<250"代替主表 list_date 口径** —— 数据起始于上市日时等价，注册表已注明，可接受。

## 未能复验的声称

- "24 秒"（运行时长，机器相关；本机全量测试 6s）。
- 矩阵构建过程日志本身（未重跑全量构建，所有矩阵级结论均为对交付 parquet 的直接实证）。
