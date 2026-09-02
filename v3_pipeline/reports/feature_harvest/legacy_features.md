# 历史特征总账（Legacy Feature Harvest）

**日期**：2026-09-01
**范围**：仓库历史上全部版本的特征工程代码——`src/` 旧生产代码（v1/v2 特征管线、背离检测器、网格/狙击策略脚本）、`v3_pipeline/`（特征缓存、筛选、背离实验室、架构赛）、`v3_pipeline/results/` 与 `experiments/` 里的历史重要性产物。
**计数口径**：以"列名/列名模式"为 1 个特征条目；滞后与横截面变体单独注明展开列数。
**判定标准**：安全 = 只用信号日 T 及之前数据且逐股隔离；可疑 = 口径/来源需复核；实锤泄漏 = 使用 T 之后数据（注明证据）。

---

## 0. 总量概览

| 类别 | 条目数 | 展开列数（约） | 说明 |
|---|---|---|---|
| 安全 | 114 | ~630（含 ~400 横截面变体） | 可直接进入新特征引擎候选池 |
| 可疑 | 6 | ~28 | 需复核口径或来源后才能用 |
| 实锤泄漏 | 12 | ~50（含 5 个回测执行列；另有 4 个标签族单列于 1.4） | 永久封杀，见第 1 章反面清单 |

历史特征展开总量峰值约 714 列（v3 Stage 2 基线，见 `v3_pipeline/results/feature_importance_3d.json`），其中安全部分与 `comm_fun.py` `FULL_FEATURE_COLS`（626 列）+ v2.3.0 微观结构 7 列基本吻合。

---

## 1. 实锤泄漏反面清单（永久封杀，新引擎必须内置防线）

这一章是本总账最重要的部分。
以下特征/机制已被证明使用未来数据，任何新特征引擎必须以**列名黑名单 + 单元测试**双重手段杜绝其复活。

### 1.1 v3 特征级标签泄漏（V4 里程碑实锤作废的根源）

| 特征 | 计算口径 | 出处 | 泄漏原因 | 处置 |
|---|---|---|---|---|
| `stop_loss_return_{3,5,10,15,20,25,30}d`（7 列） | 信号日收盘价买入后，逐日检查未来 open/low 是否触及止损（0.65×）、high 是否触及止盈（1.15×），返回模拟卖出收益 | `src/feature_pipeline_v2.py:415-438` `_calculate_stop_loss_return`（v1 同 `src/feature_pipeline.py:569-617`） | 用未来 N 天 OHLC 模拟交易结果。与 label `future_return_3d` 完全相等行占 99.87%、相关 0.9991；泄漏模型 gain 占比 73.3%+17.5%（V4_MILESTONE_REPORT.md §3 证据链） | 废弃。V4 已物理删除全部 21 列（`build_v4_clean_cache.py:84-91`），config exclude 永久含 `^stop_loss_return_`（`v4_0_0_clean.yaml`） |
| `stop_loss_sell_date_{*}d`（7 列） | 上述模拟卖出的触发日期 | 同上 | 未来日期 | 废弃（已物理删除） |
| `future_sell_date_{*}d`（7 列） | 止盈/到期卖出日期 | `src/feature_pipeline_v2.py:392-413` `_calculate_original_return` | 未来日期 | 废弃（已物理删除） |
| `future_return_{*}d`（止盈截断版，7 列） | 未来窗口内 high 触及 +15% 即按 +15% 截断（`EXPECTED_PROFIT=1.15`），否则期末收盘收益 | `src/feature_pipeline_v2.py:392-413`；参数 `src/comm_fun.py:188` | 本身是 label；v3 的教训是 (a) 它混入特征即泄漏（train_ranking 的 `system_patterns` 挡住了它但没挡 stop_loss），(b) 截断口径使 label 失真（"删赢家"偏差的源头） | 作特征=泄漏，永久排除；作 label 用 V4 重建的纯 close-to-close 版（`build_v4_clean_cache.py:54-58`） |

漏检教训：当时的 `check_v3_leakage.py` 只按列名关键词（future/next/forward/target/label）扫描，`stop_loss_return` 不含这些词所以漏网。
新引擎的泄漏防线不能靠关键词，要靠"特征构建函数是否访问 t 之后行"的结构性校验。

### 1.2 v1 管线泄漏（`src/feature_pipeline.py`，已官方废弃，文件头 14-25 行有废弃警告）

| 特征/机制 | 计算口径 | 出处 | 泄漏原因 | 处置 |
|---|---|---|---|---|
| `close_wavelet` | 对 `df['close'].values` 整面板（跨股串联）做 pywt 小波去噪，`mode='per'` 周期边界 | `src/feature_pipeline.py:1201-1284`，调用 `:1384` | 全面板混合跨股数据 + 小波窗内双向依赖；文件头注释自承"前向泄露源" | 废弃。v2 平替 `close_smooth_10`（EMA10，安全） |
| `close_d0.4` | 全面板分数阶差分（FFD, d=0.4） | `src/feature_pipeline.py:1299-1318`，调用 `:1387` | 跨股串联窗口 + 长记忆权重跨越股票边界 | 废弃。如需分数阶差分，必须逐股实现后再评估 |
| `boxcox_atr`（v1 版） | `stats.boxcox` 全样本拟合 lambda 后变换整列 | `src/feature_pipeline.py:1373-1377` | 全局统计量（lambda）含全窗信息，跨股+不可增量 | 废弃。v2 平替 `log1p(atr)`（`feature_pipeline_v2.py:825`，同名 `boxcox_atr`，安全）。注意 v2 缓存里该列实为 log1p 口径 |
| `rsi_robust` / `macd_robust`（v1 版） | 全窗 median/IQR 稳健 z-score（非滚动） | `src/feature_pipeline.py:1186-1199`，调用 `:1380-1381` | 分母用整个窗口的全期统计，口径不可增量、训练/实盘不一致 | 废弃 v1 版。v2 平替 rolling(100) 版（`feature_pipeline_v2.py:787-792,827-828`，安全） |
| v1 未逐股隔离的结构族（`vol_gk`、`vol_gk_ratio`、`illiq`、`efficiency_ratio`、`intraday_pos`、`ret_overnight`、`ret_intraday`、`smart_money_diff`、`high_mean_20`、`low_mean_20`、`support_resistance_ratio`，11 列） | 与 v2 同名特征公式相同，但 rolling/shift/pct_change 均未 groupby('symbol') | `src/feature_pipeline.py:1320-1366` | 面板按 (symbol,timestamp) 排序后，每只股票头部窗口吃到前一只股票的尾部数据（跨股污染） | 废弃 v1 版。v2 同名特征已全部逐股隔离，安全可用 |
| `vol_divergence`（v1 版） | 短期波动 / `volatility_long.mean()`（全面板均值） | `src/feature_pipeline.py:910-919` | 分母是全市场全窗均值，跨股污染 | 废弃 v1 版。v2 版为组内 `rolling(60).std().rolling(60).mean()`（`feature_pipeline_v2.py:616-619`，安全） |
| 次日最低价样本过滤（机制，非列） | 背离信号过滤时偷看 T+1 的 low 是否 ≤ 信号日收盘价，判"能否成交" | `src/feature_pipeline.py:720-750`（`is_valid_row` 内 `:746-747`） | 样本选择阶段使用 T+1 数据 → 训练集被未来信息筛选 | 废弃。v2 已改为只用 T 日可得信息（`feature_pipeline_v2.py:538-555`），可成交性改由 label NaN 处理 |

### 1.3 回测执行列（非特征，但必须防止误入特征矩阵）

| 列 | 出处 | 说明 |
|---|---|---|
| `next_open` / `next_high` / `next_low` / `next_close` / `entry_date` | `src/grid_trading_simulation_v{6,7,8,10,12,13}.py`（如 v13:92-96） | 网格回测的 T+1 执行价，`groupby('code').shift(-1)` 显式未来数据。只服务于回测撮合，若任何特征表出现 `next_*`/`entry_date` 列即报警 |

### 1.4 标签族（合法的未来数据，但永不可作特征）

以下全部为 label，新引擎应将其隔离在独立命名空间（现有约定：`future_return_*`/`rank_future_return_*`/`open_exec_return_*`/`rank_open_exec_return_*`/`mfr_*`/`ret_h*`/`hit_N*_k*`/`mfe_*`/`mae_*`/`dyn_*`），特征选择一律 exclude。

| 标签 | 口径 | 出处 |
|---|---|---|
| `future_return_{h}d`（V4 干净版） | close[t+1+h]/close[t]-1，纯收盘 | `v3_pipeline/scripts/build_v4_clean_cache.py:54-58` |
| `rank_future_return_{h}d` | 按 timestamp 横截面百分位排名 [0,1] | `v3_pipeline/src/ranking_labels.py:13-146` |
| `cur_return` / `open_exec_return` / `max_forward_return`（V5 三族） | T 收盘→T+1+h 收盘 / T+1 开盘→T+1+h 收盘 / T+1 开盘→窗口最大 high | `v3_pipeline/src/label_candidates.py:18-43` |
| `ret_h{h}` / `hit_N{N}_k{k}` / `mfe_h{w}` / `mae_h{w}` / `tmfe/tmae` / `dyn_c{c}` | 背离事件标签：固定窗收益、狙击命中（T+1 开盘+k·ATR 目标）、MFE/MAE 及到达时间、formation 比例动态窗 | `v3_pipeline/scripts/divergence_lab.py:319-384`、`v3_pipeline/scripts/distribution_audit.py:43-84` |

---

## 2. 安全特征总账（v2 管线，`src/feature_pipeline_v2.py` —— 当前唯一可信特征来源）

v2 管线版本号 2.3.0（`feature_pipeline_v2.py:23`）。
全部特征满足：逐股 `groupby('symbol')` 隔离、rolling/shift 只看历史、横截面变换只按 timestamp 分组。
数据字段需求：个股日频 OHLCV（open/high/low/close/volume）+ 指数日频 OHLCV（000001.SH、399001.SZ）。
注意工程细节：v2 按 `FEATURE_NEED_MAX_DAYS=100`（`comm_fun.py:203`）先截断 100 天再算特征，导致 100 窗特征（`macd_percentile`、`rsi_robust`、`macd_robust`）贴着窗口边缘运行；新引擎应改为"全历史计算、末端切片"。

### 2.1 基础 TA-Lib 指标（17 列）——`calc_talib`，`feature_pipeline_v2.py:645-660`

| 特征 | 公式 | 所需字段 | 复用建议 |
|---|---|---|---|
| `macd` / `macd_signal` / `macd_hist` | talib.MACD(close) 默认 12/26/9 | close | 直接可用 |
| `rsi_6` / `rsi_14` / `rsi_24` | talib.RSI(close, 6/14/24) | close | 直接可用（`rsi_6` 为 v3 gain #15） |
| `ma_5` / `ma_20` / `ma_60` | talib.MA(close, 5/20/60) | close | 直接可用 |
| `bb_upper` / `bb_middle` / `bb_lower` | talib.BBANDS(close) 默认 20,2,2 | close | 直接可用 |
| `volume_ma_20` | talib.MA(volume, 20) | volume | 直接可用 |
| `obv` | talib.OBV(close, volume) | close, volume | 直接可用 |
| `atr` | talib.ATR(high, low, close) 默认 14 | OHLC | 直接可用 |
| `slowk` / `slowd` | talib.STOCH(high, low, close) 默认 9,3,3 | OHLC | 直接可用（`slowk` v3 gain #37） |

### 2.2 MACD 滚动族（12 列）——`macd_features_rolling`，`feature_pipeline_v2.py:92-135`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `macd_hist_trend_5` | macd_hist 6 窗线性斜率 | 直接可用 |
| `macd_golden_cross` / `macd_death_cross` | DIF 上/下穿 DEA（shift(1) 判交叉） | 直接可用（golden_cross_rankpct 为 v3 gain #19） |
| `macd_signal_cross` | = golden_cross（别名） | 冗余，可合并 |
| `macd_zero_cross_up` / `macd_zero_cross_down` / `macd_zero_cross` | DIF 上/下/任一穿零轴 | 直接可用（zero_cross_rankpct 在 walk-forward 稳定 12 特征内） |
| `macd_hist_amplitude` | macd_hist 20 窗极差 | 直接可用（v3 gain #24/#34） |
| `macd_hist_direction` | sign(macd_hist) | 直接可用 |
| `macd_hist_acceleration` | macd_hist 二阶差分 | 直接可用 |
| `macd_signal_convergence` / `macd_signal_convergence_trend` | DIF-DEA 及其 5 窗斜率 | 直接可用 |

### 2.3 量价族（5 列）——`calculate_volume_features_vectorized`，`feature_pipeline_v2.py:138-172`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `volume_trend_5` / `volume_trend_10` | volume 6/11 窗线性斜率（向量化） | 直接可用（volume_trend_10_z 为 v3 gain #17） |
| `price_trend_5` | close 6 窗线性斜率 | 直接可用（price_trend_5_rankpct 为 v3 gain #3） |
| `price_volume_divergence` | (price_trend_5 × volume_trend_5) < 0 标记（注意：与 18.5 节同名特征会被后者覆盖，见 2.7） | 直接可用，需解决重名 |
| `volume_consistency` | volume 6 窗变异系数 CV | 直接可用 |

### 2.4 RSI 衍生族（3 列）——`optimized_rsi_features`，`feature_pipeline_v2.py:175-195`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `rsi_momentum` | rsi_14 一阶差分 | 直接可用 |
| `rsi_turning_simple` | rsi_14 差分变号标记 | 直接可用 |
| `rsi_turning` | rsi_14 三点局部顶/底标记 | 直接可用 |

### 2.5 波动率与趋势位置（11 列）

| 特征 | 公式 | 出处 | 复用建议 |
|---|---|---|---|
| `volatility_{3,5,10,15,20,25,30}d`（7 列） | log 收益 rolling(w).std × √252 年化 | `:210-222` | 直接可用（volatility_10d v3 gain #36；volatility_5d/15d 在 walk-forward 稳定 12 特征内） |
| `obv_trend` | OBV 6 窗线性趋势 | `:225-246` | 直接可用（v3 gain #27） |
| `macd_percentile` | macd 在自身过去 100 值中的百分位 | `:249-268` | 直接可用；注意口径包袱：窗口不足为 NaN、下游把 ==50 置 NaN（`:622-623`），新引擎建议重定义为 trailing 百分位 |
| `distance_to_support` / `distance_to_resistance` | (close − rolling20 min low)/close；(rolling20 max high − close)/close | `:271-279` | **优先复用**：v3 gain #1/#2/#10/#13；V4 干净模型 top 特征（6.3% gain）；walk-forward 稳定名单在列 |

### 2.6 K 线形态（3+3 列）——`:282-323, 936-940`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `hammer_pattern` | 下影 ≥2×实体 且 上影 ≤0.5×实体 且 下影 ≥0.6×全长 | 直接可用 |
| `downtrend` / `hammer_signal` | ma5<ma10；hammer ∧ downtrend | 直接可用 |
| `doji_pattern` | 实体 < 0.1×全长 | 直接可用 |
| `engulfing_pattern` | 前阴今阳/前阳今阴且实体吞没（向量化） | 直接可用 |

### 2.7 Alpha 族（10 列）——`_generate_alpha_features`，`feature_pipeline_v2.py:584-624`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `pct_change` | close 日收益 | 直接可用（v3 gain #5） |
| `clv` | (close−low)/(high−low)，收盘位置 | 直接可用 |
| `upper_shadow_ratio` | (high−max(open,close))/(high−low)；一字板置 0 | 直接可用（其 rankpct 为 v3 gain #31） |
| `body_strength` | (close−open)/(high−low) | 直接可用 |
| `rank_return` / `rank_volume` | pct_change / volume 的当日横截面百分位 | 直接可用（rank_return v3 gain #37 附近）；volume 若存在 `hs_volume_ratio` 列会被替换为指数量比——该 fallback 是历史包袱，新引擎应固定用个股 volume |
| `signed_vol_strength` | volume × sign(close−open) | 直接可用 |
| `pv_corr_10` | close 与 volume 10 窗滚动相关；跨股边界用 cumcount<10 置 NaN 屏蔽 | 直接可用（v3 gain #12/#21）；新引擎建议改为标准 groupby-rolling 写法 |
| `dist_to_high_60` | close / rolling60 max(close) | 直接可用（STABLE_FEATURES 在列） |
| `vol_divergence` | close 5 窗 std / 其 60 窗 std 的 60 窗均值（组内） | 直接可用（v2 已修跨股污染） |

### 2.8 进阶技术族（26 列）——`_calculate_advance_technical_features`，`feature_pipeline_v2.py:898-956`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `price_vs_ma5` / `price_vs_ma20` / `price_vs_ma60` | close/ma−1 | **优先**（price_vs_ma60 为 walk-forward 稳定名单首位、v3 gain #35） |
| `ma_arrangement` | ma5>ma20>ma60 多头 +1 / 空头 −1 / 否则 0 | 直接可用（其 rankpct 在稳定 12 特征内） |
| `bb_position` / `bb_squeeze` | (close−bb_lower)/(带宽)；带宽/close<0.05 挤压标记 | **优先**（bb_squeeze_rankpct 为 walk-forward mean_delta 第 2 且在稳定 12 特征内） |
| `rsi_oversold_{6,14,24}` / `rsi_overbought_{6,14,24}`（6 列） | RSI<30 / >70 | 直接可用 |
| `macd_signal_distance` / `macd_above_signal` | DIF−DEA；>0 标记 | 直接可用 |
| `volume_ma20` / `volume_ratio` / `volume_spike` / `volume_dryup` | 量 20 均；量比；>2 放量；<0.5 缩量 | 直接可用 |
| `atr_ratio` | atr/close | **优先**（walk-forward 稳定 12 特征内） |
| `stoch_oversold` / `stoch_overbought` | slowk<20 / slowd>80 | 直接可用 |
| `signed_volume_strength` | volume_ratio × (close−open)/open | 直接可用 |
| `close_vs_high` | (high−close)/(high−low) | **优先**（v3 gain #11，STABLE_FEATURES 在列） |
| `volume_ma_ratio` | 量 MA5/MA10（均 shift(1)） | 直接可用 |

### 2.9 流动性微观结构族（7 列，v2.3.0 新增，Issue #12）——`feature_pipeline_v2.py:958-1033`

代码内逐条注释"无泄露"。

| 特征 | 公式 | 出处 | 复用建议 |
|---|---|---|---|
| `amihud_illiq_intraday` | |close/open−1| / (volume×close) | `:964-970` | 直接可用 |
| `hl_spread` | (high−low)/((high+low)/2) | `:975-979` | 直接可用 |
| `effective_spread` | (high−low)/close | `:984-989` | 直接可用 |
| `alpha12` | sign(Δvolume) × (−Δclose)，WorldQuant Alpha#12 | `:994-997` | 直接可用 |
| `price_volume_divergence` | 收盘创 20 日新高 ∧ 量低于 20 日均量 | `:1002-1008` | 直接可用；与 2.3 节同名特征互相覆盖，新引擎必须改名其一 |
| `volume_momentum` | (量 MA5 − MA20)/MA20 | `:1013-1023` | **优先**（v3 gain #22/#28/#33） |
| `price_impact` | |close−open| / √volume | `:1028-1033` | 直接可用 |

### 2.10 结构族（16 列）——`generate_structure_features`，`feature_pipeline_v2.py:794-833`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `vol_gk` / `vol_gk_ratio` | Garman-Klass 波动率 √(0.5·ln²(H/L)−(2ln2−1)·ln²(C/O))；/20 窗均值 | 直接可用 |
| `illiq` | log(|ret|/(close×volume)+1)，Amihud 非流动性 | **优先**（v3 gain #6/#14） |
| `efficiency_ratio` | |close−close.shift10| / Σ|Δclose|(10)，Kaufman 效率系数 | 直接可用 |
| `intraday_pos` | (close−low)/(high−low)；一字板按涨跌置 1/0 | 直接可用 |
| `ret_overnight` / `ret_intraday` / `smart_money_diff` | open/prev close−1；close/open−1；两者差 | **优先**（ret_overnight 为 v3 gain #4） |
| `high_mean_20` / `low_mean_20` / `support_resistance_ratio` | 高/低 20 均及其比（RSRS 思路） | **优先**（support_resistance_ratio 及 lag_5 在 walk-forward 稳定 12 特征内） |
| `log_volume` | log1p(volume) | 直接可用（log_volume_z v3 gain #16） |
| `boxcox_atr` | log1p(atr.clip(lower=eps))（v2 安全平替，沿用旧名） | 直接可用；建议新引擎改名 `log_atr` 以与 v1 泄漏版划清界限 |
| `rsi_robust` / `macd_robust` | rolling(100) 稳健 z-score（median/IQR） | 直接可用（v2 已修为滚动口径） |
| `close_smooth_10` | close 的 EMA10 | 直接可用（v1 `close_wavelet` 的安全平替） |

### 2.11 滞后族（72 列）——`generate_lag_features`，`feature_pipeline_v2.py:835-867`

| 特征组 | 展开 | 复用建议 |
|---|---|---|
| `{close,open,high,low,volume}_lag_{3,5,10,15,20,25,30}` | 35 列，原始量 lag | 直接可用；注意绝对价格 lag 跨股票不可比，模型层面主要靠其 rankpct/z 变体 |
| `return_lag_{1,2,3,5,10,20}`（基于 `daily_return`） | 6 列 | 直接可用 |
| `amplitude_lag_{1,3,5}`（基于 `amplitude`=(H−L)/L） | 3 列 + `amplitude` 本身 | 直接可用 |
| `vol_gk_lag_{1,3,5,10}` / `vol_gk_ratio_lag_{1,3,5,10}` | 8 列 | 直接可用 |
| `{illiq,efficiency_ratio,smart_money_diff,ret_overnight,ret_intraday,support_resistance_ratio}_lag_{1,3,5}` | 18 列 | 直接可用（efficiency_ratio_lag_5 为 v3 gain #7/#9） |
| `intraday_pos_lag_{1,2,3}` | 3 列 | 直接可用 |
| `daily_return` / `amplitude` | 2 列 | 直接可用 |

### 2.12 横截面变换机制（cs_n + 每个数值特征 ×2 变体）——`_calculate_cross_features`，`feature_pipeline_v2.py:869-896`

| 机制 | 口径 | 复用建议 |
|---|---|---|
| `cs_n` | 当日样本数 | 直接可用（v3 gain #23） |
| `{feat}_rankpct` | 按 timestamp 分组的百分位排名 | 机制安全，全面复用；v3 gain 榜首 `distance_to_support_rankpct` 即此族 |
| `{feat}_z` | 按 timestamp 分组的 (x−median)/std | 机制安全，全面复用 |

横截面变换只依赖当日全市场快照，无时间方向泄漏，是 ranking 任务下性价比最高的特征增强手段，新引擎应保留为通用后处理。

### 2.13 大盘/市场族（20 列）——`_calculate_market_features`，`feature_pipeline_v2.py:673-765`

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `sh_price_change` / `sz_price_change` | 指数 (close−open)/open | 直接可用 |
| `sh_amplitude` / `sz_amplitude` | 指数 (high−low)/low | 直接可用 |
| `sh_volume_ratio` / `sz_volume_ratio` | 指数量 / 20 日均量 | 直接可用 |
| `sh_price_change_abs` / `sz_price_change_abs`、`sh_price_wave_abs` / `sz_price_wave_abs` | 上述绝对值 | 直接可用 |
| `sh_sentiment` / `sz_sentiment` | 涨>1% 且振幅<2% → 2；跌>1% 且振幅>3% → 0；否则 1 | 直接可用 |
| `sh_volume_signal` / `sz_volume_signal` | 量比 >1.2 | 直接可用 |
| `sh_sz_sync_direction` / `sh_sz_sync_strength` | 两指数涨跌同号；|涨跌幅差| | 直接可用 |
| `market_sentiment` / `market_avg_change` / `market_avg_amplitude` / `market_sync_score` | 两指数均值版情绪/涨幅/振幅/同步度 | **优先**（market_avg_amplitude、market_sync_score 在 walk-forward 稳定 12 特征内） |

数据需求：000001.SH 与 399001.SZ 指数日线，仅 ≤T 数据（`:724` `index_hist = df_index[df_index.index <= timestamp]`），安全。

---

## 3. 背离事件特征总账（信号侧，全部安全）

### 3.1 v1/v2 检测器事件字段（`src/divergence_detector.py`、`src/divergence_detector_v2.py`）

两个检测器的事件字段完全一致；区别只在低点锚定算法（见 4.2 可疑项）。
输入数据在调用前已按 `index <= target_date` 截断（`feature_pipeline_v2.py:1169-1171`），因此全部字段因果。

| 特征 | 公式 | 出处 | 复用建议 |
|---|---|---|---|
| `compare_rank` | 与前 2 个低点中的第 j 个形成背离（j=1,2） | `divergence_detector.py:175` | 直接可用 |
| `close_current` / `close_previous` | 当前/前低收盘价 | `:176-177` | 直接可用（绝对价，靠横截面变体起作用） |
| `macd_current` / `macd_previous` | 当前/前低 DIF | `:178-179` | 直接可用 |
| `price_decline_pct` | (close_cur−close_prev)/close_prev | `:161` | 直接可用 |
| `macd_increase_pct` | (macd_cur−macd_prev)/|macd_prev| | `:162` | 直接可用 |
| `formation_period` | 两低点间隔 K 线数 | `:182` | 直接可用 |
| `is_quick_divergence` | formation_period ≤ 3（pipeline 内重算） | `feature_pipeline_v2.py:507` | 直接可用 |
| `divergence_strength` | 0.4×min(1,|跌幅|/0.1) + 0.6×min(1,macd升幅/0.1) | `divergence_detector.py:70-75` | 直接可用 |
| `price_macd_ratio` | |macd_increase_pct / price_decline_pct| | `:97-99` | 直接可用 |
| `divergence_magnitude` | (|price_decline_pct|+|macd_increase_pct|)/2 | `:101-104` | 直接可用 |
| `confirmation_score` | 跌幅<−2% 且 macd 升幅>1% → 1 | `:106-110` | 直接可用 |
| `divergence_amount` | 当日全市场背离点数（每行填当日总数） | `feature_pipeline_v2.py:528` | **优先**（v3 gain #8；walk-forward mean_delta 第 1、4 年全过 null95） |

### 3.2 divergence_lab 新一代事件检测（v3 后期，严格因果设计）

| 机制/特征 | 口径 | 出处 | 复用建议 |
|---|---|---|---|
| `fractal_lows` | 因果分形低点：i 为 p[i−k..i+k] 严格最小，**信号日 = i+k**（右窗走完日） | `divergence_lab.py:122-134` | 直接可用，是"锚点不漂移"的正确实现 |
| `zigzag_lows` | 因果 ZigZag 流式状态机：自最低点反弹 ≥pct 当日确认 | `divergence_lab.py:137-162`（numba） | 直接可用 |
| `apply_min_sep` | 低点最小间隔过滤 | `divergence_lab.py:165-175` | 直接可用 |
| 背离事件判定（`_pair_ok` + `detect_divergence_events`） | 价格新低（>min_decline）+ 指标抬高（>min_change）+ 可选 below_zero/volume_confirm 过滤；multi=3 时要求连续三低点两两背离 | `divergence_lab.py:179-229` | 直接可用；参数族在 `v3_pipeline/configs/divergence_lab/*.json`（fractal/zigzag × min_decline/multi/below_zero/volume_confirm 全扫描） |
| `build_market_regime` | 全样本等权日收益累积指数，滚动 window 收益 >±th 判 up/down/sideways | `divergence_lab.py:388-408` | 直接可用（等权指数本身只用 ≤T 数据） |

### 3.3 architecture_race 事件级特征（17 列，狙击模型特征集）

`v3_pipeline/scripts/architecture_race.py:63-69, 88-156, 189-190`。
报告（`v3_pipeline/reports/architecture_race.md:7`）明示"全部严格因果（仅用信号日 T 及之前）"。
与 3.1 重叠的（compare_rank、formation≈formation_period、price_decline≈price_decline_pct）不重复计数。

| 特征 | 公式 | 复用建议 |
|---|---|---|
| `confirm_lag` | sig_idx − low_idx（低点确认耗时） | 直接可用 |
| `dif_lift_atr` | (dif_low − dif_prev_low)/ATR14(sig) | 直接可用（ATR 归一比 v1 的 pct 口径更稳） |
| `price_drop_atr` | (close_prev − close_low)/ATR14(sig) | 直接可用 |
| `dif_sig` / `dif_low` | 信号日/低点日 DIF | 直接可用 |
| `atr_pct` | ATR14/close（信号日） | 直接可用 |
| `log_close` | log(close_sig) | 直接可用 |
| `above_ma200` / `ma200_ratio` | close>MA200 标记；close/MA200−1 | 直接可用 |
| `regime_code` | build_market_regime 输出映射 {unknown:−1, sideways:0, up:1, down:2} | 直接可用 |
| `vol_shrink` | vol_low/vol_prev_low（低点缩量） | 直接可用 |
| `vol_ratio` | vol_sig / mean(vol[sig−20:sig]) | 直接可用 |
| `ret_5` / `ret_10` / `ret_20` | close_sig/close_{sig−h}−1 | 直接可用 |

---

## 4. 可疑特征（复核后才可用）

| 特征/机制 | 出处 | 疑点 | 处置建议 |
|---|---|---|---|
| `hs_*` / `gq_*` / `hc_*` 市场列（各 7 列：price_change/amplitude/volume_ratio/price_change_abs/price_wave_abs/sentiment/volume_signal）+ `hs_gq/hs_hc/gq_hc_sync_{direction,strength}` 6 列 | 现存代码仅存于 `_get_default_combined_features` 兜底（`feature_pipeline_v2.py:749-765`）；但 v3 筛选产物证明 cache 中真实存在这些列（`feature_importance_3d.json` 的 collinearity_removed 列表） | 来源为仓库首个提交之前的三指数旧口径，hs/gq/hc 对应哪三个指数已不可考；与现行 sh/sz 双指数口径并存会造成语义混乱 | 不直接复用旧列；如需三指数口径，按 2.13 节模式用确定指数（如 000300.SH/399006.SZ/000905.SH）重算并显式命名 |
| `volume_signal`（背离事件字段，字符串 'bullish'/'bearish'/'neutral'） | `divergence_detector.py:77-87` | 非数值，v3 训练被 `train_ranking.py:82` 的 `.*_signal$` 规则整体排除，等于从未生效；且注意它会误伤 `sh_volume_signal` 等同后缀数值特征 | 改为 one-hot 或数值编码（缩量 1/放量 −1/平 0）后可用；新引擎的排除规则应精确到列名而非后缀 |
| v1 滑窗低点锚定（`_find_close_lows`，6 窗 step=3 取窗内最小） | `divergence_detector.py:194-214` | 窗口对齐依赖序列起点，同一历史低点随回放起点不同而漂移；右边缘低点依赖截断位置 | 废弃该锚定，统一用 v2 检测器的 left=3/right=2 稳定锚（`divergence_detector_v2.py:194-242`）或 divergence_lab 的 fractal/zigzag |
| `macd_percentile` 的 NaN/缺省口径 | `feature_pipeline_v2.py:249-268` + `:622-623` | 窗口不足 NaN、历史代码把 ==50 置 NaN，下游 fillna(0) 与 NaN 混用 | 复用时重定义口径（trailing 百分位 + 明确 min_periods） |
| `rank_volume` / `signed_vol_strength` 的 `hs_volume_ratio` fallback | `feature_pipeline_v2.py:603-607` | 若上游 merge 顺序变化引入指数量比列，特征语义会被静默替换 | 新引擎固定用个股 volume，删除 fallback |
| `pv_corr_10` 的跨股边界屏蔽写法 | `feature_pipeline_v2.py:609-611` | 现行写法（整面板 rolling + cumcount 屏蔽 + fillna(0)）正确但脆弱；前 10 行填 0 是脏值 | 复用时改写为标准 `groupby('symbol').rolling(10).corr()`，min_periods=10，保留 NaN 交由模型处理 |

---

## 5. 历史重要性证据与复用优先名单

### 5.1 三份证据源及其可信度

| 证据 | 文件 | 口径 | 可信度备注 |
|---|---|---|---|
| v3 Stage 2 gain importance（454 特征） | `v3_pipeline/results/feature_importance_3d.json` | LightGBM gain，screen_features.py:61-63 已排除 stop_loss/future_return/future_sell 列 | 特征侧无 stop_loss 泄漏，但训练 label 仍是 +15% 截断版、universe 有"删赢家"偏差 → 排序仅作弱证据 |
| walk-forward 多年稳定性筛选（552→12） | `experiments/screening_summary_2021-12-31_2022-01-01_2025-07-31.json` + `screening_importance_*.csv` | 按年 2022-2025 分年算 mean_delta、要求 ≥3 年过 null95 | v2 时代产物，label 同为截断版；但"跨 4 年稳定"这一维度独立于 label 口径，参考价值较高 |
| V4 干净模型健康度 | `V4_MILESTONE_REPORT.md` §5 | v4_0_0_clean，624 特征，无泄漏 | 唯一全干净证据："top 特征 distance_to_support 仅占 6.3% gain，无单列主导" |

### 5.2 复用优先 Top20（全部判定安全）

合并规则：walk-forward 稳定 12 特征全入选（跨 4 年验证），其余按 v3 gain 排名补足，两源重叠者优先级最高。

| # | 特征 | 证据 | 出处 |
|---|---|---|---|
| 1 | `distance_to_support` (+`_rankpct`) | v3 gain #1(rankpct)/#2；V4 干净模型 top(6.3%)；STABLE 名单 | feature_pipeline_v2.py:271-279 |
| 2 | `divergence_amount` | walk-forward mean_delta 第 1（4/4 年过 null95）；v3 gain #8 | feature_pipeline_v2.py:528 |
| 3 | `bb_squeeze` (+`_rankpct`) | walk-forward mean_delta 第 2 + 稳定 12； | feature_pipeline_v2.py:909 |
| 4 | `price_vs_ma60` | walk-forward 稳定 12；v3 gain #35 | feature_pipeline_v2.py:902 |
| 5 | `atr_ratio` | walk-forward 稳定 12 | feature_pipeline_v2.py:933 |
| 6 | `support_resistance_ratio` (+`_lag_5`) | walk-forward 稳定 12（含 lag_5） | feature_pipeline_v2.py:819-821 |
| 7 | `volatility_5d` / `volatility_15d` | walk-forward 稳定 12 | feature_pipeline_v2.py:210-222 |
| 8 | `market_avg_amplitude` | walk-forward 稳定 12 | feature_pipeline_v2.py:704 |
| 9 | `market_sync_score` | walk-forward 稳定 12 | feature_pipeline_v2.py:707 |
| 10 | `ma_arrangement` (+`_rankpct`) | walk-forward 稳定 12 | feature_pipeline_v2.py:903 |
| 11 | `macd_zero_cross` (+`_rankpct`) | walk-forward 稳定 12 | feature_pipeline_v2.py:118-120 |
| 12 | `price_trend_5` (+`_rankpct`) | v3 gain #3 | feature_pipeline_v2.py:169 |
| 13 | `ret_overnight` | v3 gain #4 | feature_pipeline_v2.py:815 |
| 14 | `pct_change` | v3 gain #5 | feature_pipeline_v2.py:592 |
| 15 | `illiq` (+`_rankpct`) | v3 gain #6/#14 | feature_pipeline_v2.py:803 |
| 16 | `efficiency_ratio_lag_5` (+`_rankpct`) | v3 gain #7/#9 | feature_pipeline_v2.py:808,858-861 |
| 17 | `close_vs_high` | v3 gain #11；STABLE 名单 | feature_pipeline_v2.py:950 |
| 18 | `pv_corr_10` | v3 gain #12；STABLE 名单 | feature_pipeline_v2.py:609-611 |
| 19 | `distance_to_resistance` | v3 gain #13 | feature_pipeline_v2.py:271-279 |
| 20 | `volume_momentum` (+`_rankpct`) | v3 gain #22/#28/#33 | feature_pipeline_v2.py:1013-1023 |

候选替补（同属安全 + 有证据）：`rsi_6`（v3 #15）、`log_volume_z`（#16）、`volume_trend_10_z`（#17）、`macd_golden_cross_rankpct`（#19）、`obv_trend`（#27）、`upper_shadow_ratio_rankpct`（#31）、`volatility_10d`（#36）、`slowk`（#37）、`rank_return`、事件级 `dif_lift_atr`/`price_drop_atr`/`vol_shrink`（狙击模型特征）。

重要警示：以上全部重要性证据都产自"截断 label"时代。
v3 Stage 1 那个 ICIR 44.2 的基线后来被证明是泄漏撑起来的，v3 gain 排序整体高估了与止盈模式相关的特征。
Top20 应作为新引擎的**首批候选**而非结论，必须在 V4 干净 label 上重新做特征筛选。

---

## 6. 新特征引擎复用工程建议

1. **泄漏防线**：内置列名黑名单（`^stop_loss_`、`^future_`、`^next_`、`^rank_`、`^label_`、`^mfr_`、`^open_exec_return_`、`ret_h\d+`、`hit_N`、`mfe_|mae_|dyn_`、`entry_date`），并加"构建函数不得访问 t 之后行"的结构测试；不要依赖关键词扫描（stop_loss 漏检的教训）。
2. **特征计算顺序**：全历史逐股计算 → 末端按日期切片，不要沿用 v2 的"先截 100 天再算"（100 窗特征贴边运行）。
3. **命名清理**：`boxcox_atr`（实为 log1p）建议改名 `log_atr`；两处 `price_volume_divergence` 重名必须区分；`macd_signal_cross` 与 `macd_golden_cross` 冗余合并。
4. **横截面后处理通用化**：`_rankpct`/`_z`/`cs_n` 机制保留为对所有数值特征的自动增强（2.12）。
5. **背离信号统一**：低点锚定只用 v2 检测器（left3/right2）或 divergence_lab fractal/zigzag，废弃 v1 滑窗锚；事件特征以 3.1+3.3 两节的并集为准。
6. **市场特征**：用 sh/sz 双指数现行口径；hs/gq/hc 旧列不进新引擎。
7. **重建路径**：缓存与模型产物已在仓库瘦身时删除，特征需从 `stock_data/daily/*.parquet` 重算；v2 管线的指纹缓存机制（`compute_cache_fingerprint`，`feature_pipeline_v2.py:26-67`）可直接沿用。
