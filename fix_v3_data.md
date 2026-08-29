# V3 失败根因与修复方案

## 根因确认

**数据质量问题：feature_cache_v3.parquet 的收益率被止盈逻辑截断到 +15%**

### 证据
```
future_return_30d: 444,701 个样本（31.7%）= 0.15
future_return_20d: 339,762 个样本（26.6%）= 0.15
future_return_15d: 276,954 个样本（21.6%）= 0.15
```

### 源头
`src/feature_pipeline_v2.py:401-406`
```python
target_price = buy_price * model_config.EXPECTED_PROFIT  # 1.15
if target_hit_mask.any():
    future_return = (target_price - buy_price) / buy_price  # 固定 0.15
```

`src/comm_fun.py:188`
```python
EXPECTED_PROFIT = 1.15  # 特征中的止盈
```

### 影响链条
1. **训练时**：模型学习截断后的收益（max = 0.15）
2. **评估时**：
   - Stage 1-2 用 Rank IC/ICIR（排名指标）→ 虚高（截断让排名更容易正确）
   - Stage 3-4 用真实价格回测 → 失败（分布不一致）
3. **结果**：IC = 0.98（虚假成功）vs 年化 -71%（真实失败）

---

## 修复方案

### 方案 A：生成无截断的 feature cache（推荐）

**步骤：**

1. **修改 V2 特征管线**：添加"原始收益"列（不截断）
   - 保留现有 `future_return_*d`（止盈版，V2 需要）
   - 新增 `raw_future_return_*d`（无截断，V3 专用）

2. **重新生成 feature_cache_all.parquet**
   - 时间成本：~2-3 小时（140 万行 × 7 horizons）

3. **重建 V3 缓存**：
   ```bash
   python v3_pipeline/scripts/build_feature_cache.py --use-raw-returns
   ```

4. **重做 Stage 1**：用未截断数据重新训练 7 个 horizon
   - 预期：IC 会下降（0.98 → 0.3-0.5），但更真实
   - 用**回测收益**（不是 IC）选最优 horizon

**优点**：
- 彻底解决分布不一致问题
- V2 和 V3 都能用（分别用各自的收益列）
- 一劳永逸

**缺点**：
- 需要重跑特征管线（2-3 小时）
- Stage 1-4 全部重做

---

### 方案 B：直接用回测评估 Stage 1（快速验证）

**跳过重新生成数据，直接改评估逻辑：**

1. **修改 Stage 1 评估**：
   - 不用 IC/ICIR 选 horizon
   - 直接对 7 个 horizon 各跑一次简化回测（validation set 2022-2025）
   - 按年化收益选最优

2. **用现有模型**：
   - 已有 `v3_0_1_label_selection/model_*d.txt`（7 个）
   - 直接评估回测表现

3. **如果还是负收益**：
   - 说明即使用截断数据，模型也没有真实边缘
   - 需要回到特征工程（V3 本身的问题，不只是数据问题）

**优点**：
- 快速（1-2 小时）
- 不需要重跑特征管线
- 能验证"数据截断是不是唯一问题"

**缺点**：
- 治标不治本（数据还是截断的）
- 如果要继续迭代 V3，最终还是要方案 A

---

## 推荐执行顺序

**先 B 后 A**：

1. **立即执行方案 B**（1-2 小时）
   - 用现有 7 个模型跑回测
   - 如果某个 horizon 回测正收益 → 说明数据截断是主因
   - 如果全部负收益 → 说明还有更深层问题（特征不行）

2. **如果 B 验证通过**，再执行方案 A（2-3 小时）
   - 生成无截断数据
   - 重做 Stage 1-4

3. **如果 B 验证失败**（全负收益）
   - 放弃 V3 当前特征集
   - 回到 V2 修复或重新设计特征

---

## 下一步行动

我现在执行**方案 B**：用现有 7 个模型直接跑回测验证。

预期用时：1 小时
