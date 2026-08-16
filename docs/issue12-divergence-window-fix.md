# Issue #12 - 背离检测v2的关键修复

## 🎯 问题发现

在尝试全量特征重算时，发现**所有日期都失败**，错误信息："无有效背离数据或生成失败"。

经过深入调查，发现v2背离检测器存在根本性设计缺陷：
- v2检测到了背离（如43个低点），但无法生成任何特征文件
- v1系统有很多背离信号，v2几乎检测不到
- 2005-2010年测试期间，所有日期都返回0个背离

## 🔍 根本原因

### v2背离检测器的设计

v2使用严格的波谷检测算法：

```python
def _find_close_lows(self, df, left_window=3, right_window=2):
    # 要求：
    # 1. 比左侧3天都低（下跌确认）
    # 2. 比右侧2天都低（反弹确认）
```

**问题**：
1. 需要右侧2天窗口来确认低点是否真的是低点
2. 对于目标日期T，最后可能的低点是T-2天
3. 背离点的timestamp是低点日期
4. 特征管线要求：`timestamp == target_date`
5. 结果：几乎所有日期都无法匹配到背离信号

### 示例

2010-06-01查询时：
- 检测到43个低点，最后一个是2010-05-17（距离目标日期15天）
- 这些低点可能形成背离信号
- 但merge时要求`timestamp == 2010-06-01`
- 由于右侧窗口限制，6月1日不可能有timestamp=6月1日的低点
- 结果：返回空，"无背离数据"

## ✅ 解决方案

### 核心思路

**背离信号在形成后的一段时间内持续有效**（符合实际交易逻辑）

### 实现

1. **检测所有历史背离**（不限定当天）
   ```python
   all_symbol_divergence = self.divergence_detector._detect_divergence_by_close_historical(
       data_with_indicators
   )
   ```

2. **保留最近10天内的背离信号**
   ```python
   DIVERGENCE_LOOKBACK_DAYS = 10  # 背离信号有效期
   
   recent_divergence = all_symbol_divergence[
       (all_symbol_divergence['timestamp'] >= target_date - 10天) &
       (all_symbol_divergence['timestamp'] <= target_date)
   ]
   ```

3. **按symbol合并，而不是精确匹配日期**
   ```python
   target_divergence_df = enriched_points.merge(
       all_divergence_points,
       on='symbol',  # 只按symbol匹配
       how='inner'
   )
   ```

4. **保留背离点的原始timestamp**
   - 背离的timestamp = 低点日期
   - 新增divergence_date字段 = 检测日期（target_date）

## 📊 验证结果

### 修复前
```
2010-06月：✅ 0天成功 | ❌ 30天失败
错误：所有日期都是"无有效背离数据"
```

### 修复后
```
2010-06月：✅ 18天成功 | ❌ 0天失败
生成文件：11个特征文件（部分日期可能合并）
单文件数据量：1032行背离信号
```

## 🎓 关键洞察

1. **v1 vs v2的差异**
   - v1：滑动窗口，宽松检测，最后一个窗口可以包含目标日期
   - v2：严格波谷，需要右侧确认，无法检测近期低点

2. **为什么v1"有效"**
   - v1的窗口检测允许目标日期本身成为低点
   - 但v1有"锚点漂移"问题（PRD提到的bug）

3. **完美方案的权衡**
   - v2的严格检测避免了假信号，但引入了时滞
   - 解决方法：允许背离信号在确认后N天内有效
   - 这符合实际交易逻辑：一个背离信号不会立即失效

## 🔄 后续影响

### 对特征数据的影响

每只股票在某个日期可能有多个背离信号：
- 10天前形成的背离（仍然有效）
- 5天前形成的背离
- ...

这会导致：
1. **特征文件更大**：同一个日期可能有更多行（每个有效背离一行）
2. **需要去重或聚合**：训练时可能需要选择"最近的"或"最强的"背离
3. **更符合实际**：实际交易中，多个背离信号叠加往往意味着更强的信号

### 建议的后续优化

```python
# 选项1：只保留最近的背离
recent_divergence = recent_divergence.sort_values('timestamp', ascending=False)
recent_divergence = recent_divergence.groupby('symbol').first().reset_index()

# 选项2：只保留最强的背离
recent_divergence = recent_divergence.sort_values('divergence_strength', ascending=False)
recent_divergence = recent_divergence.groupby('symbol').first().reset_index()

# 选项3：聚合多个背离的特征
recent_divergence = recent_divergence.groupby('symbol').agg({
    'divergence_strength': 'max',
    'formation_period': 'mean',
    # ...
})
```

## 📝 修改文件

- `src/feature_pipeline_v2.py`：背离检测和合并逻辑

## 🎉 状态

✅ **已修复并验证**

现在可以继续issue #12的全量特征重算工作。
