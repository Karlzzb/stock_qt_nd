# v2 时代特征工程 + 模型解剖报告（复刻规格书）

源 commit：`ba9d2ed`（提交信息自述：「future_return_15d LGB:158个特征 LR: 107个特征」）。
解剖对象：`feature_pipeline.py`（2025 行）、`stock_model_Tflod_v2.py`（790 行）、`predictor_model.py` / `predictor_model_v2.py`、`comm_fun.py`（711 行）、`data_process.py`（227 行）、`divergence_detector.py`（216 行）、`features.md`、`feature_readme.md`。
行号引用格式：`feature_pipeline.py:L919` 指该文件第 919 行。

---

## 0. 管线地图（一图速览）

离线建库（每日一进程）：`load_price_data`（feature_pipeline.py:L1665）加载 `{symbol}_price_data.csv` →
`feature_generator(target_date)`（L1813）按日截断 → `FeaturePipeline.enrich`（L604）算特征+检测背离+打标签 →
落盘 `realistic_features_{YYYYMMDD}.csv`（L683）。
汇总：`data_process.main`（data_process.py:L169）合并所有日频 csv → 60/20/20 切成 `train_set.csv / test_set.csv / validation_set.csv`（L220-222）。
训练：`stock_model_Tflod_v2.main`（L348）读 `t*.csv`（只命中 train_set + test_set，validation_set 不参与训练，L64）→ 85/15 时间切分 → 5 折 TimeSeriesSplit LGBM → OOF + LR 元模型。
预测：`predictor_model_v2.PriceChangePredictor`（L18）复用同一条 stacking 推理链。

---

## 1. 数据装配流

### 1.1 t*.csv 每行是什么

每一行 = 一个「个股 × 事件日」的底背离事件点，不是全交易日。
事件由 `DivergenceDetector._detect_divergence_by_close_historical`（divergence_detector.py:L119）定义：价格创新低且 MACD 抬升（L157-160）。
同一股票同一天可能检出多个背离点（对前 1/前 2 个低点各一条），在 `data_clean`（data_process.py:L114-139）按 `['timestamp','symbol']` 去重、按 `confirmation_score` 排序 keep='last'（L118）。
训练脚本 glob `DATASET_DIR/t*.csv`（stock_model_Tflod_v2.py:L64），因此实际吃进的是 `train_set.csv`（60%）+ `test_set.csv`（20%）；`validation_set.csv`（20%）只用于 `predictor_model_v2.main` 的事后评估（predictor_model_v2.py:L189-204）。

### 1.2 日频特征 csv 的生成（enrich 全流程）

入口 `FeaturePipeline.enrich(all_2_target_day_df, df_sh, df_sz)`（feature_pipeline.py:L604-688），逐步如下：

1. 按 timestamp 排序，取**最后 100 个不重复交易日**的子集 `feature_used_df`（L609-616，`FEATURE_NEED_MAX_DAYS=100`，comm_fun.py:L203）。
2. 对 `feature_used_df` 依次计算：基础 TA（L617）→ 进阶 TA（L618）→ alpha 特征（L619）→ 结构化特征（L620）→ lag 特征（L621）。
   注意 L617-618 在「按 timestamp 排序的多股票拼接帧」上直接跑 talib/rolling，存在跨股票污染，详见第 3 节。
3. 对全量帧再算一遍基础 TA（L622），供背离检测取 macd 列用（同样受污染，见第 3 节）。
4. 取特征帧中最大日期为 target_date，抽出当日全市场行 `target_df`（L625-626）。
5. 计算大盘特征并按 timestamp 左连接进 `target_df`（L629-639，公式见 2.6）。
6. 对 `target_df` 逐行取 symbol，调 `divergence_detector.detect_daily_divergence`（L645-652）检出当日背离点表。
7. 对当日全市场行（此时已含市场特征、尚未含背离列）做横截面 `_calculate_cross_features`（L659），并写入恒 0 的 `is_quick_divergence`（L660，详见 2.8 的 _x/_y 来源）。
8. 当日全市场特征帧与背离点表按 `['symbol','timestamp']` **inner join**（L663-667），只保留当日有背离的股票行。
9. `_filter_valid_rows_apply`（L670 → L691-721）剔除「次日最低价 > 信号日收盘价」的行（次日无法按信号价买入）。
10. 逐行 `_calculate_and_assign`（L677-680 → L723-747）计算 7 个持有期的 4 组标签列。
11. `divergence_amount = len(target_divergence_df)`（L682），即当日全市场背离事件总数，同日所有行同值。
12. `save_to_csv` 落盘（L683 → L749-771，`index=False` 除非索引有名）。

### 1.3 标签公式（_calculate_future_return 族）

持有期 `RETURN_PERIODS=[3,5,10,15,20,25,30]`（comm_fun.py:L200）。
每行追加 7×4=28 列：`future_return_{p}d / future_sell_date_{p}d / stop_loss_return_{p}d / stop_loss_sell_date_{p}d`（L734-746）。

买入假设（`_calculate_future_return`，L425-512）：

- 信号日收盘价 = 目标买入价 `signal_price`（L445）。
- 若次日最低价 > 信号价，视为买不进，全部标签置 NaN（L457-466）。
- 成交则 `buy_price = signal_price`，买入日记为 `signal_idx+1`（L469、L520）。

原版收益（`_calculate_original_return`，L514-551）：

- 观察窗 = 买入日次日（`signal_idx+2`）至买入后第 p 天（`signal_idx+1+p`），含端点（L520-527）。
- 目标价 = `buy_price × EXPECTED_PROFIT`（L532），`EXPECTED_PROFIT=1.15`（comm_fun.py:L188），即止盈 +15%（注释里写「5%」是过时注释，以代码为准）。
- 若窗口内任一交易日 `high ≥ 目标价`：以首次触及日为卖出日，`future_return = 目标价/buy_price − 1 = +0.15`（L535-541）。
- 否则以窗口末日收盘价卖出：`future_return = (close[end_idx] − buy_price)/buy_price`（L543-547）。

止损版收益（`_calculate_stop_loss_return`，L553-602）：

- 从买入日后第 1 天到第 p 天逐日检查，时间优先（L565-592）。
- 开盘止损：当日 `open ≤ buy_price × 0.65` → 按开盘价卖出（L577-580）。
- 盘中止损：当日 `low ≤ buy_price × 0.65` → 按 `buy_price × 0.65` 卖出，即 −35%（L583-586）。
- 止盈：当日 `high ≥ buy_price × 1.15` → 按 `buy_price × 1.15` 卖出，即 +15%（L589-592）。
- 都未触发：期末收盘价卖出（L594-598）。
- `EXPECTED_LOSS=0.65`（comm_fun.py:L189）注释写「3%止损」，代码实为 −35% 止损，复刻以代码为准。

手续费：标签公式中**不含任何手续费**。
`comm_fun.calculate_total_fees`（L100-142，佣金万 2.5 最低 5 元 + 印花税 0.05% + 过户费 0.001%）存在但在标签链中无任何调用。

训练标签（stock_model_Tflod_v2.py:L77）：
`label = (future_return_15d > 0.01).astype(int)`。
`LABEL_COL = future_return_15d`（`SETTLEMENT_DAYS=15`，comm_fun.py:L186-187）。
`get_return_threshold`（comm_fun.py:L175-178）的分位数实现已被注释，**恒返回常数 0.01**，不存在全量数据阈值泄漏。
注意标签用的是「原版收益」（15% 止盈或期末收盘版），止损版收益仅作为信息列落盘，不参与标签。

### 1.4 训练侧装配（stock_model_Tflod_v2.load_and_preprocess_data）

合并 t*.csv → 按 timestamp 排序（L64-73）→ 造 label（L77）。
`prepare_features`（L767-788）：取 `FULL_FEATURE_COLS`（610 列）转 float32 → inf→NaN → **fillna(0)**。
`data_process.prepare_real_daily_features` 还生成过 day_of_year/week_of_year/quarter 等 8 个时间列（data_process.py:L148-155），但它们不在 FULL_FEATURE_COLS 中，模型未使用。
`label_encoding`（comm_fun.py:L145-173）把字符串列 `volume_signal`（bullish/bearish/neutral）与布尔列 `hammer_signal` 做 LabelEncoder。

---

## 2. 特征全目录

FULL_FEATURE_COLS 共 **610 列**（comm_fun.py:L232-361，经 AST 复核 610 无重复）：
216 个基础列 + 197 个 `_rankpct` + 197 个 `_z`。
无 `_rankpct/_z` 变体的 19 列 = 背离结构族 16 列 + `cs_n` + `downtrend` + `hammer_signal`（后两者为 bool dtype，被 L1459-1460 的 dtype 过滤排除）。

### 2.1 基础 TA 族（talib，`_calculate_basic_technical_features`，L908-949）

| 特征 | 公式 / talib 调用 | 行号 |
|---|---|---|
| macd, macd_signal, macd_hist | `talib.MACD(close)` 默认 fast=12, slow=26, signal=9；hist=macd−signal | L919 |
| rsi_6 / rsi_14 / rsi_24 | `talib.RSI(close, timeperiod=6/14/24)`（Wilder 平滑） | L922-924 |
| ma_5 / ma_20 / ma_60 | `talib.MA(close, 5/20/60)`，SMA | L927-929 |
| bb_upper / bb_middle / bb_lower | `talib.BBANDS(close)` 默认 period=20, nbdevup=2, nbdevdn=2, SMA 中轨 | L932 |
| volume_ma_20 | `talib.MA(volume, 20)` | L935 |
| obv | `talib.OBV(close, volume)` | L936 |
| atr | `talib.ATR(high, low, close)` 默认 period=14 | L939 |
| slowk, slowd | `talib.STOCH(high, low, close)` 默认 fastk=5, slowk=3(SMA), slowd=3(SMA) | L942 |

### 2.2 进阶 TA 族（`_calculate_advance_technical_features`，L1520-1632）

| 特征 | 公式 | 行号 |
|---|---|---|
| price_vs_ma5/20/60 | `close/ma_n − 1` | L1525-1527 |
| ma_arrangement | ma5>ma20>ma60 → 1；ma5<ma20<ma60 → −1；否则 0 | L1528-1536 |
| bb_position | `(close − bb_lower)/(bb_upper − bb_lower)`，带宽为 0 时 0.5 | L1539-1545 |
| bb_squeeze | `(bb_upper−bb_lower)/close < 0.05` → 1 | L1547-1551 |
| rsi_oversold_6/14/24 | rsi_n < 30 | L1555-1557 |
| rsi_overbought_6/14/24 | rsi_n > 70 | L1559-1561 |
| macd_signal_distance | `macd − macd_signal` | L1564 |
| macd_golden_cross | 先在 L1565 赋为 `macd_signal_distance>0`，**随后在 L1630 被 macd_features_rolling 覆盖为「当日发生金叉事件」**（定义见 2.2.4） | L1565、L59-62 |
| volume_ma20 | `volume.rolling(20, min_periods=1).mean()`（注意与 volume_ma_20 是两列） | L1568 |
| volume_ratio | `volume/volume_ma20`，分母 0 → 1 | L1569-1573 |
| volume_spike | `volume_ratio > 2.0` | L1575 |
| volume_dryup | `volume_ratio < 0.5` | L1577 |
| atr_ratio | `atr/close`，close≤0 → 0 | L1580-1584 |
| hammer_pattern | 下影 ≥2×实体 且 上影 ≤0.5×实体 且 下影 ≥0.6×全日振幅 | L1587 → L336-356 |
| downtrend | `close.rolling(5).mean() < close.rolling(10).mean()`（bool，无 rp/z） | L1588 |
| hammer_signal | hammer_pattern & downtrend（bool，无 rp/z） | L1589 |
| doji_pattern | 实体 < 全日振幅 × 0.1 | L1590 → L358-372 |
| distance_to_support | `(close − low.rolling(20,min_periods=1).min())/close`，前 20 行强制 0 | L1593 → L301-334 |
| distance_to_resistance | `(high.rolling(20,min_periods=1).max() − close)/close`，前 20 行强制 0 | 同上 |
| stoch_oversold | `slowk < 20` | L1597 |
| stoch_overbought | `slowd > 80` | L1599 |
| macd_percentile | 当前 macd 在此前 100 行（不含当日）中的 `percentileofscore(kind='rank')`；不足处默认 50，后在 L898-901 被置 NaN 并在 data_clean 中删行 | L1602 → L277-299 |
| obv_trend | OBV 在 6 点窗口（含当日）上的最小二乘斜率（公式法） | L1605 → L238-275 |
| volatility_{3,5,10,15,20,25,30}d | `log(close/close.shift(1)).rolling(n,min_periods=n).std() × √252` | L1608 → L212-236 |
| engulfing_pattern | 前后两根 K 线颜色相反且今实体包住昨实体（看涨/看跌合并） | L1611 → L374-408 |
| signed_volume_strength | `volume_ratio × (close−open)/open` | L1614-1615 |
| close_vs_high | `(high − close)/(high − low)` | L1616-1617 |
| volume_ma_ratio | `MA5(volume).shift(1) / MA10(volume).shift(1)`，分母 0 或 NaN → 1 | L1620 → L186-210 |
| rsi_momentum | `rsi_14.diff()` | L1623 → L157 |
| rsi_turning_simple | `rsi_diff1.shift(1) × rsi_diff1 < 0` | L165 |
| rsi_turning | 顶部转折（昨>前 且 今<昨）或底部转折（昨<前 且 今>昨） | L169-174 |
| volume_trend_5 / volume_trend_10 | volume 在 6/11 点窗口上的 linregress 斜率 | L1626 → L126-127 |
| price_trend_5 | close 在 6 点窗口上的斜率 | L130 |
| price_volume_divergence | `price_trend_5 × volume_trend_5 < 0` | L133-135 |
| volume_consistency | volume 6 点窗口变异系数 std/mean | L138-147 |

MACD 深度族（`macd_features_rolling`，L32-104，均基于 inf/NaN 清零后的 macd 三列）：

- `macd_hist_trend_5`：macd_hist 在 6 点窗口（min_periods=2）上的斜率（L53-55）。
- `macd_golden_cross`：昨 macd<昨 signal 且 今 macd>今 signal（L59-62，最终生效版）。
- `macd_death_cross`：反向（L64-67）。
- `macd_signal_cross` = macd_golden_cross（L69，完全重复列）。
- `macd_zero_cross_up/down`：macd 上/下穿越 0 轴（L72-78）；`macd_zero_cross` = 两者或（L80）。
- `macd_hist_amplitude`：macd_hist 20 点窗口极差（L83-85）。
- `macd_hist_direction`：`sign(macd_hist)`（L89）。
- `macd_hist_acceleration`：`macd_hist.diff().diff()`（L92）。
- `macd_signal_convergence`：`macd − macd_signal`（L95，与 macd_signal_distance 完全重复）。
- `macd_signal_convergence_trend`：该差值 5 点窗口（min_periods=2）斜率（L96-98）。

### 2.3 Alpha 族（`_generate_alpha_features`，L776-903）

先按 `['symbol','timestamp']` 重排（L795），此后 groupby 计算。

| 特征 | 公式 | 行号 |
|---|---|---|
| pct_change | `groupby(symbol).close.pct_change()` | L805 |
| clv | `(close − low)/(high − low + 1e-9)` | L814 |
| upper_shadow_ratio | `(high − max(open,close))/(high − low + 1e-9)` | L819-820 |
| body_strength | `(close − open)/(high − low + 1e-9)` | L823 |
| rank_return | 当日全市场 `pct_change` 的 `rank(pct=True)`（groupby timestamp） | L834 |
| rank_volume | 当日全市场 volume 的 rank(pct)（若有 hs_volume_ratio 则用之，实际无此列 → 用 volume） | L838-839 |
| signed_vol_strength | `volume × sign(close − open)` | L851-852 |
| pv_corr_10 | `groupby(symbol) close.rolling(10).corr(volume)` | L861-865 |
| dist_to_high_60 | `close / (groupby(symbol) close.rolling(60).max() + 1e-9)` | L874-875 |
| vol_divergence | `close.rolling(5).std() / (volatility_long_mean + 1e-9)`，其中 volatility_long_mean 是 rolling(60).std() 在**整个特征帧（全部股票×全部 100 天）上的全局均值标量** | L881-890 |
| 收尾 | 全帧 fillna(0)；macd_percentile==50 的置 NaN | L897-901 |

### 2.4 结构化族（`generate_structure_features`，L1291-1360）

| 特征 | 公式 | 行号 |
|---|---|---|
| vol_gk | `sqrt(0.5·ln(high/low)² − (2ln2−1)·ln(close/open)²)`（Garman-Klass） | L1304-1306 |
| vol_gk_ratio | `vol_gk / vol_gk.rolling(20).mean()` | L1308 |
| illiq | `ln(|pct_change(close)| / (close×volume + eps) + 1)`（Amihud 对数版，用 close×volume 代金额） | L1313 |
| efficiency_ratio | `|close − close.shift(10)| / (|close.diff()|.rolling(10).sum() + eps)`（Kaufman ER，period=10） | L1317-1320 |
| intraday_pos | `(close − low)/(high − low + eps)`（与 clv 公式同形，eps 不同） | L1325 |
| ret_overnight | `open/close.shift(1) − 1` | L1330 |
| ret_intraday | `close/open − 1` | L1331 |
| smart_money_diff | `ret_intraday − ret_overnight` | L1332 |
| high_mean_20 / low_mean_20 | `high.rolling(20).mean()` / `low.rolling(20).mean()` | L1335-1336 |
| support_resistance_ratio | `high_mean_20 / low_mean_20` | L1337 |
| log_volume | `log1p(volume)` | L1341 |
| boxcox_atr | `scipy.stats.boxcox(atr)`，atr≤0 或 NaN 处填 eps；**lambda 在整个特征帧（全股票×100 天）上一次性估计** | L1346-1348 |
| rsi_robust / macd_robust | `groupby(symbol).transform(robust_zscore)`；robust_zscore = `(x − median)/(IQR/1.34896)`，IQR=0 时退化为 `x − median`；median/IQR 为**该 symbol 当前 100 天窗口全窗统计** | L1351-1352 → L1157-1170 |
| close_wavelet | 对 `close.values` 做 db4、level=1、mode='per'、软阈值小波去噪（阈值 = MAD/0.6745 × √(2ln n)），**直接作用在拼帧数组上** | L1355 → L1172-1255 |
| close_d0.4 | 固定窗分数阶差分 `frac_diff_ffd(close, d=0.4, thres=1e-5, lim=10000)`，权重 `w_k = −w_{k−1}/k·(d−k+1)`，输出从第 width 行起 | L1358 → L1257-1289 |

### 2.5 Lag 族（`generate_lag_features`，L1362-1440）

全部 `groupby('symbol').shift(lag)`。
基础列 × lag 值矩阵：

- OHLC：`close/open/high/low × lag∈{3,5,10,15,20,25,30}`（LAG_PERIODS，comm_fun.py:L202），28 列（L1386-1391）。
- 成交量：`volume × lag∈{3,5,10,15,20,25,30}`，7 列（L1394-1396）。
- 收益率：`daily_return = groupby(symbol).close.pct_change()`（L1399，与 pct_change 重复）→ `return_lag_{1,2,3,5,10,20}`（L1400-1402）。
- 振幅：`amplitude = (high−low)/low`（L1405）→ `amplitude_lag_{1,3,5}`（L1406-1407）。
- GK：`vol_gk_lag_{1,3,5,10}`、`vol_gk_ratio_lag_{1,3,5,10}`（L1414-1416）。
- 非流动性：`illiq_lag_{1,3,5}`（L1419-1420）。
- 效率系数：`efficiency_ratio_lag_{1,3,5}`（L1423-1424）。
- 日内强度：`intraday_pos_lag_{1,2,3}`（L1427-1428）。
- 隔夜/日内：`smart_money_diff_lag_{1,3,5}`、`ret_overnight_lag_{1,3,5}`、`ret_intraday_lag_{1,3,5}`（L1431-1434）。
- 支撑阻力比：`support_resistance_ratio_lag_{1,3,5}`（L1437-1438）。

### 2.6 市场大盘族（`_calculate_market_features` / `_calculate_single_index_features`，L954-1151）

指数代码：上证 `000001.SH`、深证 `399001.SZ`（L25-26），取自与个股相同的 STOCK_DATA_DIR。
截断：`index_hist = df_index[df_index.index <= timestamp]`（L1056），只用信号日及之前。

单指数（prefix ∈ {sh, sz}，L1048-1117）：

| 特征 | 公式 | 行号 |
|---|---|---|
| {p}_price_change | `(close − open)/open`（当日指数） | L1065 |
| {p}_amplitude | `(high − low)/low` | L1066 |
| {p}_volume_ratio | `volume / volume.rolling(20).mean()`（rolling 含当日；历史不足 20 行 → 1） | L1069-1073 |
| {p}_price_change_abs | `abs(price_change)` | L1076 |
| {p}_price_wave_abs | = amplitude（复制列） | L1077 |
| {p}_sentiment | 涨>1% 且 振幅<2% → 2（强势）；跌>1% 且 振幅>3% → 0（弱势）；否则 1 | L1080-1085 |
| {p}_volume_signal | `volume_ratio > 1.2` → 1 | L1088 |

组合（L988-1028）：

- `sh_sz_sync_direction`：`sh_price_change × sz_price_change > 0` → 1（L991-992）。
- `sh_sz_sync_strength`：`abs(sh_price_change − sz_price_change)`（L993-994）。
- `market_avg_change`：两指数 price_change 均值（L1010、L1020）。
- `market_avg_amplitude`：两指数 amplitude 均值（L1011、L1021）。
- `market_sentiment`：avg_change>0.01 且 avg_amplitude<0.02 → 2；avg_change<−0.01 且 avg_amplitude>0.03 → 0；否则 1（L1014-1019）。
- `market_sync_score`：`1 − min(1, (max(price_changes) − min(price_changes))/0.02)`；仅一个指数时 0.5（L1024-1028）。

异常兜底默认值见 L1092-1101 与 L1119-1151（后者仍残留 hs/gq/hc 港股前缀，属历史遗留死代码）。

### 2.7 横截面族（`_calculate_cross_features`，L1445-1466）

- 横截面范围 = **同一 timestamp 的全市场当日行**（groupby('timestamp')），作用对象是 enrich 第 7 步的当日全市场特征帧（含市场特征、不含背离列与标签列），不是「仅背离事件行」，也不是历史多日。
- 排除集：`{timestamp, symbol, label, open, close, high, low}` + 28 个标签列 + 非 float64/int64 dtype 列（L1457-1460）。
- `cs_n` = 当日帧行数（当日有数据股票数）（L1462）。
- 对每个保留列 f：`{f}_rankpct = groupby(timestamp)[f].rank(pct=True)`（L1464）。
- `{f}_z = (x − median)/(std(ddof=0) + 1e-9)`，按当日横截面（L1465）。
- 实际入选 rp/z 的基础列共 197 个（经 AST 复核），背离族 16 列在 merge 之后才加入，故无 rp/z 变体。

### 2.8 背离事件结构族（`divergence_detector.py` + enrich merge）

低点定义（`_find_close_lows`，L194-214）：window_size=6、step=3 的滑动窗口内收盘价最小值的索引序列（全历史扫描，但调用方已把数据截断到 ≤ 当前日）。
背离判定（`_detect_divergence_by_close_historical`，L119-188）：对第 i 个低点，与前 1、前 2 个低点（lookback_lows=2，L132）逐一比较；间隔 <5 根 K 线跳过（L146）；条件 = `close_current < close_previous` 且 `macd_current > macd_previous + 0.001`（min_macd_change，L133、L157-158）。

| 特征 | 公式 | 行号 |
|---|---|---|
| close_current | 当前低点收盘价 | L176 |
| close_previous | 被比较前低点收盘价 | L177 |
| macd_current / macd_previous | 两个低点处的 macd（DIF）值 | L178-179 |
| price_decline_pct | `(close_current − close_previous)/close_previous`（负值） | L161 |
| macd_increase_pct | `(macd_current − macd_previous)/abs(macd_previous)`，prev=0 → 0 | L162 |
| compare_rank | j∈{1,2}，「与前第 j 个低点比」 | L175 |
| formation_period | `current_idx − prev_idx`（K 线根数） | L182 |
| is_quick_divergence(_y) | `formation_period < 3` → 1 | L183 |
| divergence_strength | `0.4·min(1, |price_decline_pct|/0.1) + 0.6·min(1, macd_increase_pct/0.1)` | L165 → L70-75 |
| volume_signal | 当前低点量 < 前低点量 → 'bullish'；> 前低点量×1.5 → 'bearish'；否则 'neutral'（入模前 LabelEncoder） | L168 → L77-87 |
| price_macd_ratio | `abs(macd_increase_pct / max(|price_decline_pct|, 1e-6))` | L98-99 |
| divergence_magnitude | `(|price_decline_pct| + |macd_increase_pct|)/2` | L102-104 |
| confirmation_score | `price_decline_pct < −0.02 且 macd_increase_pct > 0.01` → 1 | L107-110 |
| divergence_amount | 当日全市场背离事件总数（enrich L682；data_process.py:L179 按文件行数重写一遍） | 见左 |
| is_quick_divergence_x | enrich L660 在全市场帧上写的恒 0 列（`.get('formation_period',10)` 取到标量默认 10 → 10<=3 为 False）；与 detector 的 is_quick_divergence 在 merge 时重名 → 自动改名 _x（恒 0）/_y（真值） | L660、L183 |

---

## 3. 因果性审计

### 3.1 确定缺陷/泄漏点（按严重度排序）

**D1. 基础 TA 与进阶 TA 在「按 timestamp 排序的多股票拼接帧」上计算，全部跨股票污染（系统性缺陷）。**
enrich L609 排序键只有 timestamp，`convert_dict_to_dataframe_from_index` 排序为 `['timestamp','symbol']`（L1805）。
L617-618 直接对该帧跑 `talib.MACD/RSI/MA/BBANDS/OBV/ATR/STOCH`（L919-942）及 rolling 系进阶特征（L1568、L1588、L1602、L1605、L1608、L1620、L1626、L1630）。
相邻行是同一交易日的不同股票，因此每只股票事件日的 MACD/RSI/均线/布林/OBV/ATR/STOCH 及其全部衍生列（含 197 对 rp/z 的底数）都是跨股票混合序列的产物。
alpha 族 L795 之后才按 symbol 重排，但只影响 L619 之后的新列，不修复已有列。
L622 对全量帧的同法计算使**背离检测所用的 macd 值同样被污染**，即背离事件结构族（macd_current/macd_previous/macd_increase_pct/divergence_strength 等）的语义建立在坏 MACD 上。
复刻时必须按 symbol 分组计算（这会让复刻版与 v2 实跑值不可比对，属于「修正性偏离」，需显式决策）。

**D2. 次日数据参与训练样本筛选（选择偏差型前视）。**
`_filter_valid_rows_apply` 用 `次日 low ≤ 信号日 close` 过滤训练行（feature_pipeline.py:L717-718），`_calculate_future_return` L457-466 同逻辑置 NaN 后被 data_clean 删除。
效果是训练集只含「次日能以信号价买到」的事件；实盘当日收盘后无法知道次日最低价。
标签侧的买入可行性假设合理，但行筛选使样本分布相对全事件池有偏。

**D3. close_wavelet 小波去噪：非因果变换。**
`wavelet_denoising` 对整个 100 天×全股票拼帧数组做 wavedec/waverec（L1355、L1215-1241），每个点的去噪值依赖窗口内前后样本（且跨股票边界），代码注释自认「可能有不一致性问题，导致预测时候的数据泄露（暂时不用）」（L1354），但该列仍在 FULL（comm_fun.py:L249）与 OPTIMIZED（L386）中，实际被使用。
由于每日快照只用 ≤ 当日数据生成，目标日行是各 symbol 块最后一行，泄漏形态主要是 periodization 边界效应与跨股票污染，不是直接的未来数据。

**D4. close_d0.4 分数阶差分：跨股票边界污染。**
`frac_diff_ffd` 对拼帧数组做卷积（L1358、L1284-1287），每个 symbol 块前 width 行的窗口包含上一只股票的尾部数据；代码注释同样自认风险（L1357）但仍入 FULL 与 OPTIMIZED（L392）。
d=0.4、thres=1e-5 下 width 可达数百，超过 100 天窗口时长时输出为空序列（目标日行可能取不到值）。

**D5. boxcox_atr：lambda 用全窗全股票数据估计。**
`stats.boxcox(atr_positive)`（L1348）的 lambda 由整个特征帧一次性拟合，属于全局统计量；不同日期快照的 lambda 不同，训练/预测口径不一致（轻度泄漏/不一致）。

**D6. vol_divergence：分母是全局标量。**
`volatility_long.mean()`（L887）是 rolling(60).std() 在全帧（全股票×100 天）上的均值，单只股票的特征值混入全市场信息（同日横截面意义上尚可接受，但它还混入窗口内其他日期的数据）。

**D7. macd_percentile：100 行窗口跨股票。**
`calculate_macd_percentile_vectorized`（L277-299）在拼帧上取 `iloc[i-100:i]` 作为「历史」，窗口内混入其他股票的 macd（在 D1 已污染的基础上再错一层）。

**D8. is_quick_divergence_x 恒 0。**
零方差特征（来源见 2.8），不是泄漏但占特征位。

### 3.2 审计后确认「无泄漏」的设计点

- 标签列绝不进特征：`_calculate_cross_features` 在标签赋值之前执行（enrich L659 vs L677），且排除集显式剔除 28 个标签列（L1448-1456）；FULL_FEATURE_COLS 不含任何 future/stop_loss 列。
- 横截面 rp/z 只用同日数据（L1464-1465），对收盘后预测是合法的。
- 标签阈值恒 0.01，未用全量分位数（comm_fun.py:L177-178）。
- 大盘特征显式截断 `index <= timestamp`（L1056）。
- volume_ma_ratio 用 shift(1) 排除当日（L197-198）。
- 每日快照独立生成：特征只用 ≤ 当日的数据（feature_generator L1823-1831），训练集的每个历史日都不含当日之后的数据。
- 中位数 imputer 只在训练 85% 上 fit（stock_model_Tflod_v2.py:L365-367）；meta 层 StandardScaler 只在训练栈上 fit（L280-283）。
- TimeSeriesSplit 折内训练集恒早于验证集（L106）。

### 3.3 可疑点（需复刻时显式决策）

- `divergence_amount`、`cs_n` 是当日全市场计数，收盘后可得，合法但属强市场状态代理。
- 只取最后 100 个交易日算特征（L612-616）：talib Wilder 系指标（RSI/ATR/MACD signal）在 100 天上未完全收敛，与全历史计算存在系统性偏差；`dist_to_high_60`、`ma_60`、`volatility_*` 不受影响，但 `macd_percentile` 需要 101+ 行，恰在边界。
- `optimal_threshold`（F-beta 搜索）在全训练 OOF 上求得（L169），属折间信息混用；但最终预测实际不用它（见 5.6）。
- `FeatureScaler(method='standard')`（L370）实例化后从未 fit，被原样 joblib.dump（L421）——死代码，无缩放发生。

---

## 4. 特征精选逻辑

### 4.1 OPTIMIZED_FEATURE_COLS（158 条，comm_fun.py:L364-398）

- 158 条全部 ∈ FULL（AST 复核 158/158），是 LGBM 层的实际输入（stock_model_Tflod_v2.py:L381-382）。
- 代码内可复核的选择机制 = `analyze_feature_importance`（stock_model_Tflod_v2.py:L630-708）：对 5 个 fold 的 `feature_importances_` 分别取 top-250（`top_n=250`，L630），统计每个特征进入 top-250 的 fold 占比，保留占比 ≥60%（`threshold=0.6`，L630、L684）的特征，输出 `lgb_stable_features.csv`（L701）。
- commit 考古：LGB 侧在可见历史中始终 158 条（8af42e4「特征精简：LGB:158 LR:184」→ 965024a「LR:198」为序列起点，之前仅有 a960a2a「开始 测试」），158 的精确产生轮次不在当前浅历史中；可复原的规则即「≥60% 折叠进 top-250 + importance_mean 排序截取」，手工固化为常量。
- 结论：复刻时建议直接把 158 条清单当作规格照搬，而非重跑选择流程（重跑会得到不同集合）。

### 4.2 STABLE_FEATURES（105 条，comm_fun.py:L401-425）

- 实际条数 **105**（AST 复核，任务书所称 107 = 105 + pred_lgb + pred_lgb_squared，与 ba9d2ed 提交信息「LR: 107个特征」吻合）。
- 角色：LR 元模型输入中的「稳态原始特征」组（stock_model_Tflod_v2.py:L273-276，注释「C 类：少量稳态特征」）。
- 选择路径（commit 序列 + 代码佐证）：LR 输入从 198 → 184 → 112 → 107 逐轮「特征精简」（commits 965024a、8af42e4、6add2b6、ba9d2ed），工具链 = `collect_lr_coefs`（12 折 TimeSeriesSplit 重训 LR 收集系数，L711-729）+ `analyze_coef_stability`（按 selection_rate 非零出现率、sign_consistency 符号一致性、strength_stability 排序，L731-755）+ `analyze_meta_feature_importance`（L562-605）+ `analyze_lr_shap_importance`（L534-559）。
- 105 条与 OPTIMIZED 158 条有交集但互不包含，两者均全部 ∈ FULL。

---

## 5. 训练管线全口径（stock_model_Tflod_v2.py）

### 5.1 数据切分与预处理

- 输入：`t*.csv` 合并、按 timestamp 排序（L64-73），标签 `future_return_15d > 0.01`（L77）。
- 特征矩阵：FULL 610 列、float32、inf→NaN→fillna(0)（L769-778）。
- 切分：`cut = int(len × (1 − 0.15))`，前 85% 训练、后 15% 测试，按时间序（L358-362，`TEST_RATIO=0.15`，comm_fun.py:L182）。
- 缺失值：`SimpleImputer(strategy='median')` 在训练集 fit、测试集 transform（L365-367）；因 prepare_features 已 fillna(0)，实际无可填补值，属冗余保险。
- 无特征缩放（FeatureScaler 未 fit，L370-372 注释掉）。

### 5.2 LGBM 基座（train_lgb_models，L93-188）

- 输入列 = OPTIMIZED 158（L381-382）。
- `TimeSeriesSplit(n_splits=5)`（L95，`N_SPLITS=5`）。
- 参数 = `Config.LGB_PARAMS`（comm_fun.py:L206-229）：n_estimators=1000、learning_rate=0.02、num_leaves=63、max_depth=4、min_child_samples=58、min_split_gain=0.6、min_child_weight=0.03、reg_alpha=2、reg_lambda=2、bagging_fraction=0.875、bagging_freq=7、feature_fraction=0.85、scale_pos_weight=1.0、objective=binary、metric=[auc, binary_logloss]、n_jobs=4、random_state=42、verbosity=-1。
- 动态覆盖：折内训练样本 <10000 时 `min_child_samples = max(5, int(len×0.01))`（L130-132）。
- 样本权重：`sample_weights` 与 `compute_sample_weight('balanced')` 仅计算打印，fit 时 sample_weight 被注释（L116-126、L138），实际未加权。
- 每折：`fit(eval_set=[(X_val,y_val)], eval_metric=['auc','binary_logloss'], early_stopping(150), log_evaluation(100))`（L136-145）。
- OOF = 各折验证段 predict_proba（L147）。
- 折得分 = `average_precision_score(y_val, oof[val_idx])`（PR-AUC，L152）。

### 5.3 折权重与测试集基座预测

- `weights = fold_scores / sum(fold_scores)`（L396-398），即按 PR-AUC 归一化。
- 测试集 LGB 概率 = 5 个模型概率的加权平均（L401-405）。

### 5.4 元模型（meta_features_process + train_meta_model_reduce，L242-307）

- 栈特征（`USE_LGBM_LEAF=False`、comm_fun.py:L194，故无 leaf 列；`USE_PCA=False`，L195）：
  - A 类：`pred_lgb`（OOF 或加权测试概率）、`pred_lgb_squared`（L249-251）。
  - C 类：STABLE_FEATURES 105 条原值（L274-276）。
  - 合计 107 列，与 ba9d2ed 提交信息吻合。
- `StandardScaler`：训练时 fit_transform（L280-281），测试时 transform（L283、L408）；scaler 落盘 `stack_scaler.pkl`（L425）。
- LR：`LogisticRegression(penalty='l1', solver='saga', C=0.01, class_weight='balanced', max_iter=1000, fit_intercept=True, random_state=42)`（L294-304）。
- **最终打分公式：`final_proba = lr_meta.predict_proba(StandardScaler.transform([pred_lgb, pred_lgb², STABLE_105]))[:, 1]`**（L410；预测侧同构，predictor_model_v2.py:L51-86）。

### 5.5 评估与过拟合检查

- `enhanced_evaluate`（L310-346）：以 `PROBA_THRESHOLD=0.3`（comm_fun.py:L191）出 precision/recall/f1，另报 PR-AUC、阈值 0.3/0.5/0.7 的 F1、top-1%/5%/100 的 precision。
- 过拟合度 = (训练 OOF PR-AUC − 测试 PR-AUC)/测试 PR-AUC ×100%（L480-489）。

### 5.6 阈值搜索的真实作用

- `find_optimal_threshold_beta(beta=0.5)`（L28-46）在全训练 OOF 上搜 F0.5 最优阈值（L169），`find_optimal_threshold`（F1，L48-60）只用于折内打印（L157）。
- 搜索结果落盘 `optimal_threshold.pkl`（L426），但预测侧 `self.optimal_threshold = model_config.PROBA_THRESHOLD` 恒 0.3（predictor_model_v2.py:L37），`enhanced_evaluate` 也用 0.3（L315）。
- 结论：阈值搜索在 v2 中**纯信息性**，不影响任何下游打分或决策；复刻可省略。

---

## 6. 复刻依赖清单与新事件表改写

### 6.1 新事件表已有列（任务书给定）

ts_code / event_date / anchor_close / min_prev_close / dif_lift / cross_dif / cross_prev_dif / cross_prev2_dif / cross_dea / cross_prev_dea / cross_prev2_dea / anchor_bars / event_close / event_vol / event_amount + 个股全历史日线 OHLCV + 上证指数日线。

### 6.2 缺失数据（逐条）

1. **深证成指 399001.SZ 日线 OHLCV（含成交量）**：sz_* 七列 + sh_sz_sync_* + market_* 全部依赖，无替代。
2. **上证指数需含 volume**：sh_volume_ratio / sh_volume_signal 需要指数成交量（L1069-1073、L1088）。
3. **个股日线需含 open/high/low/close/volume 五列且复权口径明确**：v2 原始 csv 仅 OHLCV 五列（L1707-1710），illiq 用 close×volume 代金额（L1313），event_amount 在 v2 特征中无对应物；复权方式（前/后/不复权）在 v2 代码中不可考，需与新库对齐。
4. **前一金叉日的成交量**：volume_signal 特征需要「前低点量」（divergence_detector.py:L79-87），新表只有 event_vol；需用 anchor_bars 回推前金叉日期后从日线取量（前提：anchor_bars 语义=两金叉间距，需确认）。
5. **全市场日线池（横截面底座）**：rp/z、rank_return/rank_volume、cs_n、divergence_amount 的 v2 语义都是「当日全市场所有股票」，不是「当日事件股票」；若只在 96,577 事件行内做横截面，口径不同，必须显式决策（建议保 v2 语义：对全市场逐日算特征再抽事件行）。
6. **talib MACD(12,26,9) 的 DIF 口径**：新表的 cross_dif 须与 talib 默认参数一致，否则 macd_increase_pct 等不可比。
7. 不需要的：波段高低点检测结果（新表 anchor_close/min_prev_close 已直接给出两区间最低点，替代 v2 的滑动窗口低点检测）。

### 6.3 背离事件结构族的改写公式（金叉对金叉语义）

v2 语义 → 新语义映射：`close_current → anchor_close`；`close_previous → min_prev_close`；`macd_current → cross_dif`；`macd_previous → cross_prev_dif`（compare_rank=1）或 `cross_prev2_dif`（compare_rank=2）。

逐条改写：

| 新特征 | 公式 |
|---|---|
| close_current | = anchor_close |
| close_previous | = min_prev_close |
| macd_current | = cross_dif |
| macd_previous | = cross_prev_dif（j=1 行）/ cross_prev2_dif（j=2 行，可选） |
| price_decline_pct | `(anchor_close − min_prev_close)/min_prev_close` |
| macd_increase_pct | `dif_lift / abs(cross_prev_dif)`；等价于 `(cross_dif − cross_prev_dif)/abs(cross_prev_dif)`；cross_prev_dif=0 → 0 |
| compare_rank | 恒 1（新表仅一对金叉）；若生成 j=2 变体则取 2 |
| formation_period | = anchor_bars |
| is_quick_divergence_y | `anchor_bars < 3` → 1（v2 用 <3，注意 enrich L660 的 ≤3 恒 0 列不要复刻） |
| divergence_strength | `0.4·min(1, |price_decline_pct|/0.1) + 0.6·min(1, macd_increase_pct/0.1)` |
| price_macd_ratio | `abs(macd_increase_pct / max(|price_decline_pct|, 1e-6))` |
| divergence_magnitude | `(|price_decline_pct| + |macd_increase_pct|)/2` |
| confirmation_score | `price_decline_pct < −0.02 且 macd_increase_pct > 0.01` → 1 |
| volume_signal | `event_vol < prev_event_vol` → bullish；`event_vol > 1.5×prev_event_vol` → bearish；否则 neutral（prev_event_vol 需按 6.2-4 补齐），再 LabelEncoder |
| divergence_amount | 当日事件表行数（按 event_date 计数） |
| is_quick_divergence_x | 建议剔除（v2 恒 0 废列） |

DEA 三列（cross_dea 等）在 v2 无对应特征，可作为新增信息列，但注意偏离「原样复刻」口径。

### 6.4 标签复刻注意

买入可行性过滤（次日 low ≤ event_close）需要事件日次日的日线 low——新库有全历史日线，可实现，但是否保留该前视筛选需显式决策（见 D2）。
止盈/止损阈值按代码实值：止盈 ×1.15、止损 ×0.65，观察窗 = 事件日后第 2 根 K 线起 p 根。
标签 = `future_return_15d > 0.01`。

---

## 7. 复刻工程量估算

### 7.1 特征总列数预估

- 原样复刻：610（FULL）= 216 基础 + 197×2 横截面。
- 训练实际用：158（LGB）+ 107（LR 栈）去重后约 250 列需要真正计算。
- 若修正 D1（按 symbol 分组算 TA）并剔除废列（macd_signal_cross、macd_signal_convergence、is_quick_divergence_x、price_wave_abs 复制列等约 5 列），基础列约 211，总列数约 600。

### 7.2 计算瓶颈

1. **全市场逐日 TA 计算**（保 rp/z 口径所需）：约 4000+ 股票 × 全历史日线跑 talib + 60 余列 rolling/斜率/相关；rolling_slope/rolling_cv/macd_percentile/obv_trend 在 v2 中是 Python 循环实现（L114-145、L262-272、L291-297），复刻应改写为 numpy stride/卷积向量化，否则单日全市场分钟级、全历史不可行。
2. **横截面 rp/z**：197 列 × 交易日数（约 4000+ 日）的 groupby-transform，pandas 可做但内存峰值高，建议按日落盘中间件。
3. **标签计算**：96,577 事件 × 7 持有期 × 逐日止损扫描（L565-592 是 Python 循环），需向量化（首触 high≥1.15 / low≤0.65 / open≤0.65 的 argmax 法）。
4. **小波/FFD**：pywt 对全市场每股票 100 天窗口逐日快照代价极高；建议决策剔除 close_wavelet/close_d0.4（D3/D4 已证其有问题且 OPTIMIZED 中权重有限）。
5. **训练**：96,577×610 的 5 折 LGBM（158 列）很轻；瓶颈全在特征侧。

### 7.3 需要新写的代码模块清单

1. `daily_ta.py`：按 symbol 分组的 talib + rolling 特征（2.1/2.2 族，向量化重写 slope/cv/percentile/obv_trend）。
2. `alpha_features.py`：2.3 族（含 groupby rolling corr、dist_to_high_60）。
3. `structure_features.py`：2.4 族（剔除或隔离 wavelet/FFD/boxcox 全局量）。
4. `lag_features.py`：2.5 族（纯 shift，最简）。
5. `market_features.py`：2.6 族（需补深证成指数据）。
6. `cross_section.py`：2.7 族（按 event_date 的全市场 rankpct/z + cs_n）。
7. `divergence_structure.py`：6.3 改写公式（直接吃事件表列，无检测器）。
8. `label_maker.py`：1.3 标签（向量化止盈止损扫描 + 买入可行性过滤开关）。
9. `dataset_build.py`：按日快照拼接 + 去重（timestamp,symbol keep=confirmation_score 最大）+ label_encoding。
10. `train_stack.py`：第 5 节全链（85/15、median imputer、5 折 LGBM、PR-AUC 折权重、LR 元模型、0.3 阈值评估）；FeatureScaler 死代码与阈值搜索可不复刻。
11. 配置层：LGB_PARAMS / FULL(或 158+105 清单) / RETURN_PERIODS / 常量（1.15/0.65/0.01/0.3）照搬 comm_fun.py。
