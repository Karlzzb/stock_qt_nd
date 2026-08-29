# V3 Pipeline Final Audit Report

**Date**: 2026-08-28  
**Status**: ❌ **REJECTED - Label Leakage Detected**  
**Issue**: #27

---

## Executive Summary

V3 ranking pipeline显示极端高收益（年化929万% - 1422万%），但经过全面审查，确认这是**标签泄漏**导致的虚假结果。模型学会了识别"会被数据截断的股票"，而不是真正预测收益。

**核心问题**：数据截断（+15%上限）→ 截断股票rank≈1.0 → 模型学习截断特征 → 虚假高收益

---

## 1. 数据泄漏审查

### 1.1 时间分割 ✓

```
Train:  964,855 rows (2001-01-02 to 2021-12-31)
Val:    337,185 rows (2022-01-01 to 2025-07-31)
Test:   100,227 rows (2025-08-01 to 2026-08-14)
```

- ✓ 无overlap
- ✓ 严格时间顺序

### 1.2 特征排除 ✓

```python
exclusion_patterns = [
    r".*_date$",
    r".*_signal$",
    r"^prev_",
    r"^rank_future_return_",
    r"^future_return_",
]
```

- ✓ 所有future_return列被正确排除
- ✓ 所有rank_future_return列被正确排除
- ✓ 631个特征，无future信息
- ⚠️ `future_sell_date_*`未被`.*_date$`匹配（但实际未用作特征）

### 1.3 Rank Label计算 ✓

- ✓ Cross-sectional ranking（仅用同期数据）
- ✓ actual_return与cache一致
- ✓ 无跨期泄漏

### 1.4 数据截断分析 ❌

**验证集（2022-2025）**：
```
3d:   0.00% capped (0/301,022)
5d:   0.00% capped (0/300,947)
10d:  0.00% capped (0/300,845)
...
```

**但**在pred_3d.parquet中：
```
- 总样本：250,712行
- 截断样本（return=0.15）：9,644行（2.86%）
- 截断股票的平均rank：0.957
- 非截断股票的平均rank：0.485
```

**测试集（2025-2026）**：
```
- 总样本：78,865行
- 截断样本：3,181行（3.46%）
- Top1选中截断股票：230/248期（92.7%）
```

---

## 2. 标签泄漏证据

### 2.1 截断股票Rank分布异常

|指标|截断股票|非截断股票|
|---|---|---|
|平均rank|**0.957**|0.485|
|中位数|0.978|0.484|
|最小值|0.500|0.000|
|标准差|0.059|0.282|

**结论**：截断股票获得极高rank（接近1.0），而非截断股票正常分布。

### 2.2 模型预测与截断状态高度相关

**测试集分析**：
- 预测与is_capped的相关性：**0.55**
- 随机基准Top1截断率：3.51%
- 实际Top1截断率：**92.7%**
- **提升倍数：26.4x**

**结论**：模型学会了识别会被截断的股票。

### 2.3 样本期展示

**2022-01-04**（72只股票）：
- 3只截断股票（return=0.15）全部获得rank=0.986
- 这3只股票包揽Top 3排名
- 第4名是非截断股票，return=0.0416，rank=0.958

**泄漏机制**：
1. 真实收益：20%、50%、100%
2. 被截断到：15%、15%、15%
3. Cross-sectional rank：全部打成平手，rank≈1.0
4. 模型学习：这些特征 → rank=1.0
5. 预测时：识别截断特征 → 预测high rank

---

## 3. 策略回测结果

### 3.1 验证集（2022-2025，865期）

| 策略 | 年化收益 | Sharpe | MaxDD | 胜率 |
|------|----------|--------|-------|------|
| Long-Top1 | 6,140,914% | 237,562 | -9.4% | 99.4% |
| L/S Top1vsBot1 | 11,811,074% | 227,355 | 0.0% | 99.2% |
| L/S Top5vsBot5 | 1,065,076% | 26,900 | 0.0% | 98.3% |

- 平均3天期收益：14.06%（接近15%上限）
- Top1截断率：87.2%

### 3.2 测试集（2025-2026，248期）

| 策略 | 年化收益 | Sharpe | MaxDD | 胜率 |
|------|----------|--------|-------|------|
| Long-Top1 | 9,289,552% | 637,636 | 0.0% | 100.0% |
| L/S Top1vsBot1 | 14,216,801% | 405,476 | 0.0% | 100.0% |
| L/S Top5vsBot5 | 2,044,170% | 59,228 | 0.0% | 100.0% |

- 平均3天期收益：14.60%（接近15%上限）
- Top1截断率：**92.7%**

**验证集vs测试集**：表现一致，证明泄漏在训练时已发生，不是过拟合。

---

## 4. 根本原因分析

### 4.1 泄漏路径

```
真实世界收益（未知）
    ↓
数据截断：15%上限
    ↓
future_return_3d = 0.15（多只股票打成平手）
    ↓
Cross-sectional ranking
    ↓
rank_future_return_3d ≈ 1.0（截断股票）
    ↓
LightGBM训练
    ↓
模型学习：feature pattern → rank=1.0
    ↓
预测：识别截断特征 → high prediction
    ↓
回测：选中截断股票 → return=0.15 → 高收益
```

### 4.2 为什么不是真实预测能力？

**如果模型真的有预测能力**：
- Top1截断率应该 = 整体截断率 ≈ 3.5%
- 实际Top1截断率 = **92.7%**
- 提升26.4x = **检测截断，不是预测收益**

**如果是真实高收益**：
- 应该有部分期间Top1 > 15%（但数据截断了）
- 应该有部分期间Top1 < 15%但仍是最高（很少）
- 实际：92.7%的期间Top1 = 15%（异常）

---

## 5. Code Review 1 - 训练流程

### 检查项

✓ `v3_pipeline/src/ranking_labels.py`: Cross-sectional ranking正确  
✓ `v3_pipeline/scripts/build_feature_cache.py`: 从V2 cache加载，无新泄漏  
✓ `v3_pipeline/scripts/train_ranking.py`: 特征排除正确  
✓ `check_v3_leakage.py`: 7项检查全通过  

### 发现的问题

❌ `.*_date$`正则无法匹配`future_sell_date_3d`（后缀是数字）  
→ 但实际未用作特征（被missing rate过滤）

❌ **数据截断是V2遗留问题**，V3继承了  
→ 需要追溯V2的feature_cache_all.parquet生成逻辑

---

## 6. Code Review 2 - 回测流程

### 检查项

✓ `test_set_strategy_evaluation.py`: 
- 时间分割正确
- Return格式处理正确（加法格式）
- Long/Short计算正确
- 复利计算正确

✓ `detailed_strategy_analysis.py`: 与测试集脚本一致

### 发现的问题

无代码错误。回测逻辑正确，但输入数据（predictions）有泄漏。

---

## 7. 结论

### 7.1 V3是否可信？

**❌ 不可信**

- 模型学习的是"截断检测"，不是"收益预测"
- 极端高收益来自标签泄漏
- 测试集同样受影响（泄漏在数据生成阶段）

### 7.2 是否无泄漏？

**❌ 有泄漏**

泄漏类型：**间接标签泄漏**
- 非传统意义的"未来数据泄漏"（时间分割正确）
- 而是"标签信息泄漏"（截断状态编码在rank中）

### 7.3 是否有漂移？

**不适用**

由于泄漏问题，无法评估真实的模型漂移。

---

## 8. 修复方案

### 8.1 短期方案（验证假设）

1. **移除截断数据**：
   - 过滤掉所有future_return_3d == 0.15的样本
   - 重新计算rank labels
   - 重新训练
   - 观察：如果收益骤降，证明泄漏假设正确

2. **添加随机噪声**：
   - 对截断值添加小噪声：0.15 → 0.15 + ε (ε ~ U[-0.001, 0.001])
   - 打破平手，rank分散
   - 重新训练

### 8.2 长期方案（根治）

1. **追溯V2数据源**：
   - 检查feature_cache_all.parquet生成代码
   - 找到截断逻辑（EXPECTED_PROFIT=1.15）
   - 评估是否可以获取未截断数据

2. **重新生成feature cache**：
   - 从原始数据重新计算future_return
   - 不做任何截断
   - 重建整个V3 pipeline

3. **改用classification**：
   - 不用regression/ranking
   - 分类任务：Top 10% vs 其他
   - 避免截断影响label

---

## 9. 附件

### 生成的文件

1. `check_v3_leakage.py` - 7项泄漏检查
2. `generate_test_predictions.py` - 测试集预测生成
3. `test_set_strategy_evaluation.py` - 测试集策略评估
4. `diagnose_label_leakage.py` - 标签泄漏诊断
5. `test_set_results_20260828_203231.json` - 完整测试集结果
6. `test_predictions.log` - 预测生成日志
7. `test_strategy_results_corrected.log` - 回测日志

### 关键数据文件

1. `v3_pipeline/models/v3_0_1_label_selection/test_pred_*.parquet` - 测试集预测（7个horizon）
2. `v3_pipeline/feature_cache_v3.parquet` - V3 feature cache

---

## 10. 最终裁决

**V3 Ranking Pipeline: REJECTED**

理由：
1. ❌ 标签泄漏（截断 → rank bias）
2. ❌ 虚假高收益（检测截断，非预测收益）
3. ❌ 不可实盘（真实世界无15%上限）

**下一步行动**：
1. 追溯V2数据截断源头
2. 评估获取未截断数据的可行性
3. 如果无法获取，考虑：
   - 使用classification任务
   - 或接受截断，但明确说明局限性
4. 暂停V3相关工作，直到数据问题解决

---

**Audit completed by**: Claude (Kiro)  
**Audit duration**: Full session (data inspection + leakage检测 + code review × 2)  
**Confidence**: **High** (多重证据链交叉验证)
