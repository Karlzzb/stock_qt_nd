# 股票技术分析特征文档

## 📊 特征分类总览

### 1. 基础背离特征
### 2. MACD相关特征  
### 3. RSI动量特征
### 4. 均线系统特征
### 5. 布林带特征
### 6. 成交量特征
### 7. 波动率特征
### 8. K线形态特征
### 9. 大盘市场特征
### 10. 时间序列特征
### 11. 随机指标特征
### 12. 目标变量（标签）

---

## 1. 基础背离特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `price_decline_pct` | 价格下跌幅度 | `(当前低点价格 - 前低点价格) / 前低点价格` | float | 负值表示价格下跌 |
| `macd_increase_pct` | MACD指标上升幅度 | `(当前MACD - 前MACD) / abs(前MACD)` | float | 正值表示MACD上升 |
| `divergence_strength` | 背离强度评分 | `价格强度×0.4 + MACD强度×0.6` | float(0-1) | 综合背离强度 |
| `formation_period` | 背离形成周期 | `当前低点索引 - 前低点索引` | int | 形成时间长度 |
| `confirmation_score` | 背离确认度 | 价格跌>2%且MACD升>1%时为1 | int(0/1) | 强背离信号 |
| `divergence_magnitude` | 背离幅度 | `(abs(价格变化) + abs(MACD变化)) / 2` | float | 综合变化幅度 |
| `price_macd_ratio` | 价格-MACD变化比率 | `abs(MACD增幅 / 价格降幅)` | float | 变化相对比率 |

## 2. MACD相关特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `macd` | MACD值 | `talib.MACD(close)` | float | 基础MACD值 |
| `macd_signal` | MACD信号线 | MACD的9日EMA | float | 信号线 |
| `macd_hist` | MACD柱状图 | `MACD - 信号线` | float | 柱状图高度 |
| `macd_signal_distance` | 信号线距离 | `macd - macd_signal` | float | 金叉死叉距离 |
| `macd_golden_cross` | 金叉信号 | `macd_signal_distance > 0` | int(0/1) | 1表示金叉 |
| `macd_percentile` | MACD历史分位 | 最近100天MACD百分位 | float(0-100) | 历史位置 |
| `macd_hist_trend` | MACD柱状图趋势 | 5日MACD柱状图线性趋势 | float | 斜率值 |
| `macd_signal_cross` | 信号线交叉 | 前一日死叉当前金叉 | int(0/1) | 交叉信号 |
| `macd_zero_cross` | 零轴穿越 | MACD穿越零轴 | int(0/1) | 零轴信号 |
| `macd_hist_amplitude` | MACD柱状图振幅 | 20日MACD柱状图极差 | float | 波动幅度 |

## 3. RSI动量特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `rsi_6` | 6日RSI | `talib.RSI(close, 6)` | float(0-100) | 短期RSI |
| `rsi_14` | 14日RSI | `talib.RSI(close, 14)` | float(0-100) | 中期RSI |
| `rsi_24` | 24日RSI | `talib.RSI(close, 24)` | float(0-100) | 长期RSI |
| `rsi_oversold` | RSI超卖 | `rsi_14 < 30` | int(0/1) | 超卖信号 |
| `rsi_overbought` | RSI超买 | `rsi_14 > 70` | int(0/1) | 超买信号 |
| `rsi_oversold_6` | 6日RSI超卖 | `rsi_6 < 30` | int(0/1) | 短期超卖 |
| `rsi_oversold_14` | 14日RSI超卖 | `rsi_14 < 30` | int(0/1) | 中期超卖 |
| `rsi_oversold_24` | 24日RSI超卖 | `rsi_24 < 30` | int(0/1) | 长期超卖 |
| `rsi_overbought_6` | 6日RSI超买 | `rsi_6 > 70` | int(0/1) | 短期超买 |
| `rsi_overbought_14` | 14日RSI超买 | `rsi_14 > 70` | int(0/1) | 中期超买 |
| `rsi_overbought_24` | 24日RSI超买 | `rsi_24 > 70` | int(0/1) | 长期超买 |
| `rsi_momentum` | RSI动量 | `当前RSI - 前一期RSI` | float | 动量变化 |
| `rsi_turning` | RSI转折点 | RSI方向发生改变 | int(0/1) | 转折信号 |

## 4. 均线系统特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `ma_5` | 5日均线 | `talib.MA(close, 5)` | float | 短期均线 |
| `ma_20` | 20日均线 | `talib.MA(close, 20)` | float | 中期均线 |
| `ma_60` | 60日均线 | `talib.MA(close, 60)` | float | 长期均线 |
| `price_vs_ma5` | 价格vs5日线 | `(close / ma_5) - 1` | float | 相对位置 |
| `price_vs_ma20` | 价格vs20日线 | `(close / ma_20) - 1` | float | 关键位置 |
| `price_vs_ma60` | 价格vs60日线 | `(close / ma_60) - 1` | float | 长期趋势 |
| `ma_arrangement` | 均线排列 | 5>20>60为1, 5<20<60为-1 | int(-1,0,1) | 多头/空头排列 |

## 5. 布林带特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `bb_upper` | 布林上轨 | `talib.BBANDS上轨` | float | 压力线 |
| `bb_middle` | 布林中轨 | `talib.BBANDS中轨` | float | 均线 |
| `bb_lower` | 布林下轨 | `talib.BBANDS下轨` | float | 支撑线 |
| `bb_position` | 布林带位置 | `(close - bb_lower) / (bb_upper - bb_lower)` | float(0-1) | 相对位置 |
| `bb_squeeze` | 布林带收缩 | `带宽/close < 0.05` | int(0/1) | 收缩信号 |
| `price_bb_position` | 价格布林位置 | 同`bb_position` | float(0-1) | 别名特征 |

## 6. 成交量特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `volume` | 成交量 | 当日成交量 | float | 原始成交量 |
| `volume_ma20` | 20日均量 | `talib.MA(volume, 20)` | float | 均量线 |
| `volume_ratio` | 成交量比率 | `volume / volume_ma20` | float | 相对均量 |
| `volume_spike` | 成交量突增 | `volume_ratio > 2.0` | int(0/1) | 放量信号 |
| `volume_dryup` | 成交量萎缩 | `volume_ratio < 0.5` | int(0/1) | 缩量信号 |
| `volume_signal` | 成交量信号 | 比较前后低点成交量 | enum | bullish/bearish/neutral |
| `obv` | 能量潮 | `talib.OBV(close, volume)` | float | 累积量价指标 |
| `obv_trend` | OBV趋势 | 5日OBV线性趋势 | float | 量能趋势 |
| `volume_trend_5` | 5日成交量趋势 | 5日成交量线性趋势 | float | 短期量能趋势 |
| `volume_trend_10` | 10日成交量趋势 | 10日成交量线性趋势 | float | 中期量能趋势 |
| `price_volume_divergence` | 价量背离 | 价格与成交量趋势相反 | int(0/1) | 背离信号 |
| `volume_consistency` | 成交量一致性 | 5日成交量变异系数 | float | 量能稳定性 |
| `volume_ma_ratio` | 成交量MA比率 | `5日均量 / 10日均量` | float | 量能短期强度 |

## 7. 波动率特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `atr` | 平均真实波幅 | `talib.ATR(high, low, close)` | float | 绝对波幅 |
| `atr_ratio` | ATR比率 | `atr / close` | float | 相对波幅 |
| `volatility_20d` | 20日波动率 | 20日收益率标准差×√252 | float | 历史波动率 |
| `volatility_20` | 波动率20 | 同`volatility_20d` | float | 别名特征 |

## 8. K线形态特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `hammer_pattern` | 锤子线 | 下影线≥2×实体，上影线短 | int(0/1) | 反转信号 |
| `doji_pattern` | 十字星 | 实体<总范围的10% | int(0/1) | 犹豫信号 |
| `engulfing_pattern` | 吞没形态 | 阳线完全包裹前一根阴线 | int(0/1) | 强反转信号 |

## 9. 大盘市场特征

### 上证指数特征
| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `sh_price_change` | 上证涨跌幅 | `(收盘-开盘)/开盘` | float | 日内涨跌 |
| `sh_amplitude` | 上证振幅 | `(最高-最低)/最低` | float | 波动幅度 |
| `sh_volume_ratio` | 上证量比 | `成交量/20日均量` | float | 相对成交量 |
| `sh_price_change_abs` | 上证绝对涨跌 | `abs(涨跌幅)` | float | 波动强度 |
| `sh_price_wave_abs` | 上证绝对波动 | 同`sh_amplitude` | float | 波动幅度 |

### 深证成指特征
| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `sz_price_change` | 深证涨跌幅 | `(收盘-开盘)/开盘` | float | 日内涨跌 |
| `sz_amplitude` | 深证振幅 | `(最高-最低)/最低` | float | 波动幅度 |
| `sz_volume_ratio` | 深证量比 | `成交量/20日均量` | float | 相对成交量 |
| `sz_price_change_abs` | 深证绝对涨跌 | `abs(涨跌幅)` | float | 波动强度 |
| `sz_price_wave_abs` | 深证绝对波动 | 同`sz_amplitude` | float | 波动幅度 |

### 市场协同特征
| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `market_sync_direction` | 市场同步方向 | 两市涨跌同向为1 | int(0/1) | 方向一致性 |
| `market_sync_strength` | 市场同步强度 | `abs(上证涨跌幅-深证涨跌幅)` | float | 差异程度 |
| `market_sync_score` | 市场同步评分 | `1 - min(1, 差异/0.02)` | float(0-1) | 同步性评分 |
| `market_avg_change` | 市场平均变化 | `(abs(上证涨跌幅)+abs(深证涨跌幅))/2` | float | 平均波动 |
| `market_avg_amplitude` | 市场平均振幅 | `(上证振幅+深证振幅)/2` | float | 平均振幅 |
| `market_sentiment` | 市场情绪 | 2:强势, 1:中性, 0:弱势 | int(0,1,2) | 情绪评分 |
| `volume_sync` | 成交量协同 | 两市量比均>1.2为1 | int(0/1) | 量能协同 |
| `avg_price_change` | 平均价格变化 | `(上证涨跌幅+深证涨跌幅)/2` | float | 市场平均表现 |
| `avg_price_wave` | 平均价格波动 | `(上证振幅+深证振幅)/2` | float | 市场平均波动 |

## 10. 时间序列特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `time_since_last_div` | 距上次背离天数 | `当前时间 - 上次背离时间` | int | 背离频率 |
| `position_in_trend` | 趋势位置 | 基于背离点密度的趋势判断 | int(-1,0,1) | 趋势阶段 |

## 11. 随机指标特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `stoch_k` | 随机指标K值 | `talib.STOCH的K值` | float(0-100) | 快速线 |
| `stoch_d` | 随机指标D值 | `talib.STOCH的D值` | float(0-100) | 慢速线 |
| `stoch_oversold` | 随机指标超卖 | `stoch_k < 20` | int(0/1) | 超卖信号 |
| `stoch_overbought` | 随机指标超买 | `stoch_k > 80` | int(0/1) | 超买信号 |

## 12. 其他技术特征

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `is_quick_divergence` | 快速背离 | `formation_period <= 3` | int(0/1) | 短期背离 |
| `distance_to_support` | 距支撑距离 | `(close - 近期低点) / close` | float | 支撑位置 |
| `distance_to_resistance` | 距阻力距离 | `(近期高点 - close) / close` | float | 阻力位置 |

## 13. 目标变量（标签）

| 特征名 | 意义 | 算法 | 数据类型 | 备注 |
|--------|------|------|----------|------|
| `future_return_3d` | 3日收益率 | `(3日后收盘价-当前收盘价)/当前收盘价` | float | 短期收益 |
| `future_return_5d` | 5日收益率 | `(5日后收盘价-当前收盘价)/当前收盘价` | float | 中期收益 |
| `future_return_10d` | 10日收益率 | `(10日后收盘价-当前收盘价)/当前收盘价` | float | 长期收益 |

---

## 🔧 技术实现说明

### 数据安全措施
- ✅ **无未来函数**：所有特征仅使用历史数据计算
- ✅ **时间序列保护**：严格按时间顺序处理数据
- ⚠️ **唯一注意事项**：`macd_percentile`特征在数据不足100天时使用默认值50

### 特征工程特点
1. **多时间维度**：包含6/14/24日等多周期指标
2. **多市场维度**：个股技术指标 + 大盘环境特征
3. **量价结合**：价格指标与成交量特征协同分析
4. **形态识别**：包含多种K线反转形态检测

### 适用场景
- 机器学习模型训练
- 量化策略开发
- 技术分析研究
- 交易信号验证

---