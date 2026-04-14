# V9/V10 策略升级设计文档

**日期:** 2026-04-13
**版本:** 3.0
**状态:** 待实施（V9）/ 待实施（V10）

---

## 1. 概述

V9 和 V10 是两条独立的策略增强路径，均基于 V8 成功经验进行开发：

- **V8**: 基础策略，简单直接的止盈止损逻辑 + 概率加权仓位分配
- **V9**: 基于 V8 + 动态风控增强（涨幅过滤、RQ动态止损、大盘仓位）
- **V10**: 基于 V8 + 止盈方式增强（移动止盈、分批止盈、跟踪止损）

**设计原则：** 不改变原有 V8 代码，分别在 V9 和 V10 分支进行独立增强。

**约束:**
- 仅修改策略层，不涉及模型训练和特征变化
- V9 和 V10 **禁止 import V8**，所需部分直接复制实现，确保版本间完全隔离

---

## 0. V8 核心逻辑清单（V9/V10 必须保持一致）

以下逻辑为 V8 验证有效的核心实现，V9 和 V10 必须**原样复制**，不得修改：

### 0.1 每日流程顺序
```
卖出/管理持仓 -> 买入新仓 -> 结算当日资产
```

### 0.2 _manage_positions 核心逻辑

| 逻辑 | 实现要求 |
|------|---------|
| T+1规则 | `days_held < 1` 不能卖 |
| 硬止损触发 | `curr_o <= stop_price or curr_l <= stop_price` |
| 硬止损卖出价 | `sell_price = curr_o if curr_o <= stop_price else stop_price`，再 `sell_price = min(sell_price, curr_h)` |
| 止盈触发 | `curr_h >= target_price`（只要High碰到目标价就算成交）|
| 止盈卖出价 | `sell_price = max(curr_o, target_price)`，再 `sell_price = min(sell_price, curr_h)` |
| 时间退出 | `days_held >= max_hold_days`，按 `curr_c` 结算 |

### 0.3 _open_new_positions 核心逻辑

| 逻辑 | 实现要求 |
|------|---------|
| 选股过滤 | `y_pred_proba >= min_probability` + `close <= AFFORDABLE_PRICE` |
| 资金冻结机制 | 挂单阶段不扣 self.cash，只冻结资金 |
| 成交条件 | `next_low <= signal_price` 才可能成交 |
| 跌停处理 | 跌停也买入（实盘挂单无法控制） |
| 成交价 | `actual_price = signal_price`（不是 next_open） |
| 资金结算 | 只扣除实际成交金额 `actual_cost`，冻结多余返还 |

### 0.4 概率加权仓位分配（ALLOCATION_STRATEGY）

| 方案 | 逻辑 |
|------|------|
| z_score | `weight_factor = 1.0 + (0.2 * z_score)`，限制在 [0.8, 1.4] |
| tiered | proba>=0.72 → 1.3，0.60~0.72 → 1.0，<0.60 → 0.8 |

### 0.5 _record_daily_asset
```
total = self.cash + sum(price * shares for each position)
```

### 0.6 实盘模拟一致性检查点
- 成交价固定用 signal_price，不依赖 next_open
- 跌停时按规则仍尝试买入（不 continue 跳过）
- 冻结资金机制确保资金使用效率与实盘一致

---

## 2. V9 设计（基于 V8 + 动态风控）

### 2.1 V8 基础参数（保持原 V8 逻辑，不修改）

| 参数 | 类型 | 候选值 | 说明 |
|------|------|--------|------|
| base_ratio | float | [1.0, 0.86] | 开仓比例 |
| target_profit | float | [0.25, 0.3, 0.35] | 止盈阈值 |
| hard_stop_loss | float | [-0.12, -0.1, -0.14] | 止损阈值 |
| max_hold_days | int | [13, 15, 18, 20] | 最大持仓天数 |
| min_probability | float | [0.5, 0.55] | 预测概率阈值 |

### 2.2 V9 新增参数

| 参数 | 类型 | 候选值 | 说明 |
|------|------|--------|------|
| recent_rise_n | int | [10, 15, 20] | 统计近N天涨幅 |
| recent_rise_pct | float | [0.2, 0.25, 0.3] | 超过此值则排除 |
| rq_window | int | [5, 10, 15] | RQ统计窗口 |
| rq_shrink_threshold | float | [0.5, 0.6] | RQ收缩触发阈值 |
| rq_recover_threshold | float | [0.7, 0.8, 0.9] | RQ恢复阈值 |

### 2.3 V9 参数组合数

- V8基础: 2 × 3 × 3 × 4 × 2 = 144 组合
- V9新增: 3 × 3 × 3 × 2 × 3 = 162 组合
- **V9总计: 144 × 162 = 23,328 组合**

### 2.4 V9 功能详细设计

#### 2.4.1 涨幅过滤（Recent Rise Filter）

**位置:** `_open_new_positions` 方法

**逻辑:**
```python
def _calc_recent_rise(self, code, today, df):
    """计算股票近N天涨幅"""
    stock_data = df[df['code'] == code].sort_values('date')
    cutoff_date = today - pd.Timedelta(days=self.recent_rise_n)
    recent = stock_data[stock_data['date'] >= cutoff_date]
    if len(recent) < 2:
        return 0
    return (recent['close'].iloc[-1] / recent['close'].iloc[0]) - 1
```

**过滤条件:** `recent_rise > recent_rise_pct` 的股票被排除

#### 2.4.2 动态止盈止损（RQ-Based Dynamic Thresholds）

**位置:** `_manage_positions` 方法

**RQ定义:** 近期收益降低率（Recent Profit Reduction Rate）

**RQ计算公式:**
```
RQ = 最近rq_window次交易收益的移动平均值
其中单次交易收益 = (卖出价 - 买入价) / 买入价
```

**RQ物理含义:**
- RQ = 1.0: 收益处于正常水平
- RQ = 0.5: 收益下降50%，市场变差，触发止损止盈收缩
- RQ = 0.2: 收益严重下滑，大幅收缩阈值避险
- RQ回升至0.8以上: 市场恢复，恢复正常阈值

**状态机:**
- `rq < rq_shrink_threshold`: 进入收缩状态，`target_profit *= shrink_ratio`, `hard_stop_loss *= shrink_ratio`
- `rq > rq_recover_threshold`: 恢复正常状态，阈值还原
- 中间状态：保持当前状态

**收缩比例:** 固定 0.8（同时收缩止盈和止损）

#### 2.4.3 大盘动态仓位（Market-Based Allocation）

**位置:** `_open_new_positions` 方法

**大盘指标:** `market_avg_change`

**分档规则:**

| market_avg_change | 仓位系数 |
|-----------------|---------|
| > 1.0% | 1.2 |
| 0.5% ~ 1.0% | 1.0 |
| -0.5% ~ 0.5% | 0.8 |
| -1.0% ~ -0.5% | 0.6 |
| < -1.0% | 0.6 |

**计算:** `actual_base_ratio = base_ratio * market_multiplier`

---

## 3. V10 设计（基于 V8 + 止盈方式增强）

### 3.1 设计背景

V8 的成功因素：简单直接的止盈止损逻辑 + 概率加权仓位分配

V10 改进思路：**在 V8 成功基础上，只做简单有效的增强，不做减法**

V10 与 V9 的关键区别：
- V9: 限制开仓 + 动态风控 → 负向效果
- V10: 增强止盈方式 + 优化资金利用 → 正向增强

### 3.2 V8 最佳参数（不修改）

```python
base_ratio = 1.0  # V8最佳
target_profit = 0.25  # V8最佳
hard_stop_loss = -0.12  # V8最佳
max_hold_days = 18  # V8最佳
min_probability = 0.5  # V8最佳
max_positions = 3  # V8最佳
```

### 3.3 V10 新增参数

| 参数 | 类型 | 候选值 | 说明 |
|------|------|--------|------|
| trailing_offset | float | [0.05, 0.08, 0.10] | 触发移动止盈的涨幅 |
| trailing_pct | float | [0.15, 0.20, 0.25] | 触发后回撤百分比 |
| use_partial_take_profit | bool | [True, False] | 是否启用分批止盈 |
| partial_profit_1 | float | [0.10, 0.12, 0.15] | 第一批止盈目标 |
| partial_profit_2 | float | [0.20, 0.25, 0.30] | 第二批止盈目标 |
| partial_ratio_1 | float | [0.4, 0.5, 0.6] | 第一批卖出比例 |
| use_trailing_stop | bool | [True, False] | 是否启用跟踪止损 |
| trailing_stop_pct | float | [0.15, 0.20, 0.25] | 跟踪止损回撤比例 |

### 3.4 V10 参数组合数

- V8固定: 1 组合
- V10新增: 3 × 3 × 2 × 3 × 3 × 3 × 2 × 3 = **2916 组合**

### 3.5 V10 功能详细设计

#### 3.5.1 移动止盈 (Trailing Take Profit) — 核心增强

**问题:** V8用固定止盈，涨多了会卖飞

**方案:**
```
持有期间:
  如果当前价 > 成本价 * (1 + trailing_offset):
    移动止盈线 = max(当前价 * (1 - trailing_pct), 原止盈线)
```

**效果:** 涨得越多，锁定越多利润，但不会卖飞

#### 3.5.2 分批止盈 (Partial Take Profit)

**问题:** 一次性全卖，错过大涨

**方案:**
```
目标价1: 成本价 * (1 + partial_profit_1)
  → 卖出 partial_ratio_1 比例

目标价2: 成本价 * (1 + partial_profit_2)
  → 卖出剩余全部

如果没到目标价2:
  用移动止盈或原止损/时间退出
```

**效果:** 锁定部分利润，剩下的博更大涨幅

#### 3.5.3 跟踪止损 (Trailing Stop Loss) — 可选增强

**问题:** 固定止损在大涨后会被打掉

**方案:**
```
持仓后最高价 = max(持仓期间所有收盘价)

如果 当前价 < 最高价 * (1 - trailing_stop_pct):
    触发跟踪止损
```

**效果:** 让利润奔跑，在回撤到一定程度时保护

### 3.6 V10 策略伪代码

```python
class SmartSniperStrategyV10(SmartSniperStrategy):  # 直接继承V8

    def _manage_positions(self, daily_data, today):
        for code, pos in self.positions.items():
            # 1. 固定止损 (保持V8)
            if curr_o <= stop_price or curr_l <= stop_price:
                self._execute_trade(..., 'INTRADAY_STOP_LOSS', ...)
                continue

            # 2. 分批止盈 (V10新增)
            if self.use_partial_take_profit:
                # 检查第一批止盈
                if curr_h >= partial_target_1 and not pos.get('partial_1_done'):
                    # 卖出partial_ratio_1比例
                    self._execute_partial_trade(..., partial_ratio_1, 'PARTIAL_TAKE_PROFIT_1')
                    pos['partial_1_done'] = True

                # 检查第二批止盈
                if curr_h >= partial_target_2:
                    # 卖出剩余全部
                    self._execute_partial_trade(..., 1.0, 'PARTIAL_TAKE_PROFIT_2')
                    continue

            # 3. 移动止盈 (V10新增)
            if curr_price > pos['avg_cost'] * (1 + trailing_offset):
                new_trailing_stop = curr_price * (1 - trailing_pct)
                pos['trailing_stop'] = max(pos.get('trailing_stop', 0), new_trailing_stop)

            # 4. 跟踪止损 (V10新增)
            if self.use_trailing_stop:
                highest_price = pos.get('highest_price', pos['avg_cost'])
                if curr_c < highest_price * (1 - trailing_stop_pct):
                    self._execute_trade(..., 'TRAILING_STOP', ...)
                    continue

            # 5. 固定止盈 (V8原有)
            if curr_h >= target_price:
                self._execute_trade(..., 'TAKE_PROFIT', ...)
                continue

            # 6. 时间退出 (V8原有)
            if days_held >= max_hold_days:
                self._execute_trade(..., 'TIME_EXIT', ...)
                continue
```

### 3.7 V10 优先级

1. **必须实现:** 移动止盈 (trailing take profit)
2. **可选实现:** 分批止盈 (partial take profit)
3. **可选实现:** 跟踪止损 (trailing stop loss)

建议先实现移动止盈，跑完网格搜索确认有效后再加其他功能。

### 3.8 V10 预期结果

对比V8 (635%收益率，-37%最大回撤):

| 场景 | 预期收益 | 预期回撤 |
|------|---------|---------|
| V8基准 | 635% | -37% |
| V10 + 移动止盈 | > 635% | < 37% |
| V10 + 移动止盈 + 分批止盈 | >> 635% | < 40% |

**目标:** 收益率提升10-20%，回撤不显著增加

---

## 4. 文件结构

**重要:** V9/V10 文件均为独立实现，禁止 import V8。所需 V8 逻辑直接复制到各自文件中。

```
src/
├── strategies/
│   ├── smart_sniper_strategy.py           # V8 原版（不修改）
│   ├── smart_sniper_strategy_v9.py        # V9 独立实现（复制V8基础 + 新增逻辑）
│   └── smart_sniper_strategy_v10.py       # V10 独立实现（复制V8基础 + 新增逻辑）
├── grid_trading_simulation_v8.py          # V8 单进程（不修改）
├── grid_trading_simulation_v8_mp.py       # V8 多进程批量测试（不修改）
├── grid_trading_simulation_v9.py          # V9 单进程（独立实现）
├── grid_trading_simulation_v9_mp.py       # V9 多进程批量测试（独立实现）
├── grid_trading_simulation_v10.py         # V10 单进程（独立实现）
└── grid_trading_simulation_v10_mp.py      # V10 多进程批量测试（独立实现）
```

---

## 5. 测试数据

- **训练数据:** `test_set.csv`, `validation_set.csv`（与V8一致）
- **回测指标:** 回报率、最大回撤、胜率、夏普比率

---

## 6. 实现优先级

### V9 实现顺序:
1. V9 基础框架 + 涨幅过滤
2. V9 动态止盈止损
3. V9 大盘动态仓位
4. V9 参数网格搜索脚本

### V10 实现顺序:
1. V10 基础框架（继承V8）
2. 移动止盈 (trailing take profit)
3. 分批止盈 (partial take profit)
4. 跟踪止损 (trailing stop loss)
5. V10 参数网格搜索脚本
