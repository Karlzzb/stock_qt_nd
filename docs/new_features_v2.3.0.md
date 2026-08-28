# 新增流动性微观结构特征文档

## 版本
- **FEATURE_PIPELINE_VERSION**: 2.3.0
- **新增日期**: 2026-08-20
- **特征数量**: 7个

---

## 新增特征详细说明

### 1. amihud_illiq_intraday (Amihud非流动性指标-日内版)

**计算公式**:
```
|日内收益率| / (成交量 × 收盘价)
= |close/open - 1| / (volume × close)
```

**意义**: 衡量单位成交额导致的价格变化，值越大流动性越差

**数据泄露检查**: ✅ 无泄露
- 仅使用当日数据：open, close, volume
- 不涉及未来数据

**参考文献**: Amihud (2002), "Illiquidity and stock returns"

---

### 2. hl_spread (High-Low价差)

**计算公式**:
```
(high - low) / ((high + low) / 2)
```

**意义**: 日内波动幅度，反映流动性和价格发现效率

**数据泄露检查**: ✅ 无泄露
- 仅使用当日数据：high, low
- 不涉及未来数据

**参考文献**: Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads"

---

### 3. effective_spread (有效价差)

**计算公式**:
```
(high - low) / close
```

**意义**: 标准化的日内振幅，流动性成本的代理指标

**数据泄露检查**: ✅ 无泄露
- 仅使用当日数据：high, low, close
- 不涉及未来数据

---

### 4. alpha12 (WorldQuant Alpha#12)

**计算公式**:
```
sign(Δvolume) × (-1 × Δclose)
```

**意义**: 捕捉成交量与价格变化的反向关系，量价背离信号

**数据泄露检查**: ✅ 无泄露
- 使用 groupby('symbol').diff() 确保不跨股票计算
- diff() 仅使用前一日数据
- 不涉及未来数据

**参考文献**: WorldQuant, "101 Formulaic Alphas"

---

### 5. price_volume_divergence (价量背离)

**计算公式**:
```
(close == max(close, 20d)) AND (volume < mean(volume, 20d))
```

**意义**: 价格创新高但成交量萎缩，典型的顶部背离信号

**数据泄露检查**: ✅ 无泄露
- rolling(20) 仅回看历史20天
- 不涉及未来数据
- 使用 groupby('symbol').transform() 确保不跨股票计算

---

### 6. volume_momentum (成交量动量)

**计算公式**:
```
(MA(volume, 5d) - MA(volume, 20d)) / MA(volume, 20d)
```

**意义**: 成交量的短期vs长期对比，衡量市场参与度变化

**数据泄露检查**: ✅ 无泄露
- rolling(5) 和 rolling(20) 仅回看历史数据
- 不涉及未来数据
- 使用 groupby('symbol').transform() 确保不跨股票计算

---

### 7. price_impact (价格冲击)

**计算公式**:
```
|close - open| / sqrt(volume)
```

**意义**: 成交量标准化的价格变化，衡量流动性深度（Kyle's Lambda的简化版）

**数据泄露检查**: ✅ 无泄露
- 仅使用当日数据：open, close, volume
- 不涉及未来数据

**参考文献**: Kyle (1985), "Continuous Auctions and Insider Trading"

---

## 实现特点

### 1. 完全向量化
- 避免使用 `rolling().apply(lambda)` 等Python层循环
- 使用pandas内置的C实现函数（mean, std, max等）
- 提高计算效率，减少内存占用

### 2. 多进程安全
- 经过12并发测试，100%成功率
- 不使用会导致多进程崩溃的模式（如嵌套apply + autocorr）
- 使用 groupby('symbol') 确保不跨股票计算

### 3. 无数据泄露
- 所有特征仅使用当前及历史数据
- rolling窗口仅回看历史
- diff() 仅使用前一条记录
- 不涉及任何未来信息

---

## 测试结果

### 单进程测试
- ✅ 100/100 样本生成成功
- ✅ 所有特征非空值占比 > 99%

### 多进程测试
- ✅ 4并发：100%成功率
- ✅ 8并发：100%成功率（bummsryv8任务）
- ✅ 12并发：100%成功率（6天全部成功）

### 稳定性验证
- 已验证在5822只股票、~100行/股票的规模下稳定运行
- 内存占用正常，无进程崩溃

---

## 后续优化建议

### Issue: 智能并发调度
创建独立issue实现自动并发调整：
- 根据目标日期的实际股票数量动态设置workers
- 早期年份（2001-2010）：股票少，可用20+ workers
- 近期年份（2020-2026）：股票多，限制在8-12 workers
- 避免手动分段，提高易用性

---

## 参考文献

1. Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. Journal of Financial Markets.

2. Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance.

3. Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica.

4. WorldQuant (2015). 101 Formulaic Alphas. https://arxiv.org/abs/1601.00991

5. Hasbrouck, J. (2009). Trading costs and returns for US equities: Estimating effective costs from daily data. The Journal of Finance.
