# V9/V10 策略实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 V9 和 V10 两套独立策略，每套包含策略类 + 单进程模拟 + 多进程网格搜索

**Architecture:**
- V9/V10 均**禁止 import V8**，所需 V8 逻辑直接复制到各自文件中
- 每套策略完全自包含，独立运行，互不影响
- 核心逻辑（每日流程、资金管理、成交机制）必须与 V8 保持完全一致

**Tech Stack:** Python, pandas, numpy, concurrent.futures, tqdm

---

## Chunk 1: V9 策略类实现

### Task 1: 创建 smart_sniper_strategy_v9.py

**文件:**
- 创建: `src/strategies/smart_sniper_strategy_v9.py`

**实现步骤:**

- [ ] **Step 1: 复制 V8 基础代码**

从 `src/strategies/smart_sniper_strategy.py` 复制完整代码到新文件，作为 V9 的基础。

- [ ] **Step 2: 添加 V9 新增参数**

在 `__init__` 方法中添加 V9 特有参数:

```python
# V9 特有参数 - 涨幅过滤
self.recent_rise_n = 15       # 统计近N天涨幅
self.recent_rise_pct = 0.25   # 超过此值则排除

# V9 特有参数 - RQ动态止损
self.rq_window = 10            # RQ统计窗口
self.rq_shrink_threshold = 0.5 # RQ收缩触发阈值
self.rq_recover_threshold = 0.8 # RQ恢复阈值
self.rq_shrink_ratio = 0.8     # 收缩比例
```

- [ ] **Step 3: 添加 RQ 统计相关属性**

在 `__init__` 中初始化:

```python
self.recent_trades_profits = []  # 最近交易收益记录
self.current_rq = 1.0           # 当前RQ值
self.is_shrunk = False          # 是否处于收缩状态
```

- [ ] **Step 4: 实现 _calc_recent_rise 方法**

在 `_open_new_positions` 方法**之前**添加:

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

- [ ] **Step 5: 修改 _open_new_positions - 添加涨幅过滤**

在选股过滤部分，添加:

```python
# 2. 涨幅过滤 (V9新增)
recent_rise = self._calc_recent_rise(code, today, df)
if recent_rise > self.recent_rise_pct:
    continue  # 跳过近期涨幅过大的股票
```

在 `candidates = daily_data[...]` 之后，遍历 `top_candidates` 时添加上述过滤。

- [ ] **Step 6: 修改 _open_new_positions - 添加大盘仓位**

在计算 `base_budget_per_slot` 之后，应用仓位系数:

```python
# 大盘仓位 (V9新增)
market_avg_change = row.get('market_avg_change', 0)
if market_avg_change > 0.01:
    market_multiplier = 1.2
elif market_avg_change > 0.005:
    market_multiplier = 1.0
elif market_avg_change >= -0.005:
    market_multiplier = 0.8
else:
    market_multiplier = 0.6

actual_base_ratio = self.base_ratio * market_multiplier
weighted_budget = base_budget_per_slot * weight_factor * (actual_base_ratio / self.base_ratio)
```

- [ ] **Step 7: 修改 _manage_positions - 添加 RQ 动态阈值**

在 `_manage_positions` 方法开头添加 RQ 状态机:

```python
# RQ 动态阈值 (V9新增)
# 根据当前 RQ 状态调整阈值
if self.is_shrunk:
    if self.current_rq > self.rq_recover_threshold:
        self.is_shrunk = False  # 恢复正常
    else:
        effective_target_profit = self.target_profit * self.rq_shrink_ratio
        effective_stop_loss = self.hard_stop_loss * self.rq_shrink_ratio
else:
    if self.current_rq < self.rq_shrink_threshold:
        self.is_shrunk = True  # 进入收缩状态
    effective_target_profit = self.target_profit
    effective_stop_loss = self.hard_stop_loss
```

将 `_manage_positions` 中的 `self.target_profit` 和 `self.hard_stop_loss` 替换为 `effective_target_profit` 和 `effective_stop_loss`。

- [ ] **Step 8: 修改 _execute_trade - 记录交易收益用于 RQ 计算**

在 `_execute_trade` 方法中，卖出交易完成后记录收益:

```python
def _execute_trade(self, today, code, action, price, shares, pos, days_held):
    revenue = price * abs(shares)
    profit = revenue - (pos['avg_cost'] * abs(shares))
    profit_pct = (price - pos['avg_cost']) / pos['avg_cost']
    self.cash += revenue

    # RQ 计算: 只记录已结束的卖出交易 (V9新增)
    if action in ['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT']:
        self.recent_trades_profits.append(profit_pct)
        # 保持滑动窗口
        if len(self.recent_trades_profits) > self.rq_window:
            self.recent_trades_profits.pop(0)
        # 更新 RQ 值
        if len(self.recent_trades_profits) > 0:
            self.current_rq = np.mean(self.recent_trades_profits)

    self._log(today, code, action, price, shares, profit, profit_pct, days_held, pos['proba'])
```

- [ ] **Step 9: 提交代码**

```bash
git add src/strategies/smart_sniper_strategy_v9.py
git commit -m "feat: add SmartSniperStrategyV9 with dynamic risk control

- 涨幅过滤 (recent_rise_n, recent_rise_pct)
- RQ动态止盈止损 (rq_window, rq_shrink_threshold, rq_recover_threshold)
- 大盘动态仓位 (market_avg_change based)
- 禁止import V8，完整复制V8基础逻辑"
```

---

## Chunk 2: V9 模拟脚本实现

### Task 2: 创建 grid_trading_simulation_v9.py

**文件:**
- 创建: `src/grid_trading_simulation_v9.py`

- [ ] **Step 1: 复制 V8 模拟脚本基础代码**

从 `src/grid_trading_simulation_v8.py` 复制完整代码。

- [ ] **Step 2: 修改 import**

将 `from grid_trading_simulation_v8 import SmartSniperStrategy, data_process, logger`
改为 `from strategies.smart_sniper_strategy_v9 import SmartSniperStrategy`
添加 `from src.comm_fun import data_process, logger`（需要从 comm_fun 复制 data_process 或直接内联）

注：由于禁止 import V8，需要将 `grid_trading_simulation_v8.py` 中的 `data_process` 函数也复制过来，或者在该文件中内联实现。

**简化处理**: 直接在该文件中内联 `data_process` 函数（从 `grid_trading_simulation_v8.py` 复制）。

- [ ] **Step 3: 修改参数设置部分**

将 V8 参数替换为 V9 参数空间（设计文档 2.1 + 2.2 节）:

```python
# V8 基础参数
strategy.base_ratio = param_dict['base_ratio']
strategy.target_profit = param_dict['target_profit']
strategy.hard_stop_loss = param_dict['hard_stop_loss']
strategy.max_hold_days = param_dict['max_hold_days']
strategy.min_probability = param_dict['min_probability']

# V9 新增参数
strategy.recent_rise_n = param_dict['recent_rise_n']
strategy.recent_rise_pct = param_dict['recent_rise_pct']
strategy.rq_window = param_dict['rq_window']
strategy.rq_shrink_threshold = param_dict['rq_shrink_threshold']
strategy.rq_recover_threshold = param_dict['rq_recover_threshold']
```

- [ ] **Step 4: 定义 V9 参数网格**

```python
param_grid = {
    # V8 基础参数
    'base_ratio': [1.0, 0.86],
    'target_profit': [0.25, 0.3, 0.35],
    'hard_stop_loss': [-0.12, -0.1, -0.14],
    'max_hold_days': [13, 15, 18, 20],
    'min_probability': [0.5, 0.55],
    # V9 新增参数
    'recent_rise_n': [10, 15, 20],
    'recent_rise_pct': [0.2, 0.25, 0.3],
    'rq_window': [5, 10, 15],
    'rq_shrink_threshold': [0.5, 0.6],
    'rq_recover_threshold': [0.7, 0.8, 0.9],
}
```

- [ ] **Step 5: 提交代码**

```bash
git add src/grid_trading_simulation_v9.py
git commit -m "feat: add grid_trading_simulation_v9 for V9 backtesting"
```

---

### Task 3: 创建 grid_trading_simulation_v9_mp.py

**文件:**
- 创建: `src/grid_trading_simulation_v9_mp.py`

- [ ] **Step 1: 复制 V8_MP 基础代码**

从 `src/grid_trading_simulation_v8_mp.py` 复制完整代码。

- [ ] **Step 2: 修改 import**

将 `from grid_trading_simulation_v8 import SmartSniperStrategy, data_process, logger`
改为 `from strategies.smart_sniper_strategy_v9 import SmartSniperStrategy`
添加 `from src.comm_fun import data_process, logger`

- [ ] **Step 3: 修改参数设置**

同 Task 2 Step 3，添加 V9 参数设置。

- [ ] **Step 4: 更新输出文件名**

将 `output_file = 'parameter_optimization_results_concurrent_v8.csv'`
改为 `output_file = 'parameter_optimization_results_concurrent_v9.csv'`

- [ ] **Step 5: 提交代码**

```bash
git add src/grid_trading_simulation_v9_mp.py
git commit -m "feat: add grid_trading_simulation_v9_mp for V9 concurrent optimization"
```

---

## Chunk 3: V10 策略类实现

### Task 4: 创建 smart_sniper_strategy_v10.py

**文件:**
- 创建: `src/strategies/smart_sniper_strategy_v10.py`

- [ ] **Step 1: 复制 V8 基础代码**

从 `src/strategies/smart_sniper_strategy.py` 复制完整代码到新文件。

- [ ] **Step 2: 添加 V10 新增参数**

在 `__init__` 方法中添加:

```python
# V10 特有参数 - 移动止盈
self.trailing_offset = 0.08   # 触发移动止盈的涨幅
self.trailing_pct = 0.20      # 触发后回撤百分比

# V10 特有参数 - 分批止盈
self.use_partial_take_profit = False
self.partial_profit_1 = 0.12   # 第一批止盈目标
self.partial_profit_2 = 0.25   # 第二批止盈目标
self.partial_ratio_1 = 0.5      # 第一批卖出比例

# V10 特有参数 - 跟踪止损
self.use_trailing_stop = False
self.trailing_stop_pct = 0.20  # 跟踪止损回撤比例
```

- [ ] **Step 3: 修改 _manage_positions - 添加 V10 增强逻辑**

在 V8 原有止盈逻辑**之后**添加:

```python
# ===== V10 增强逻辑 =====

# 1. 分批止盈 (Partial Take Profit) - 在固定止盈之前检查
if self.use_partial_take_profit:
    partial_target_1 = pos['avg_cost'] * (1 + self.partial_profit_1)
    partial_target_2 = pos['avg_cost'] * (1 + self.partial_profit_2)

    # 检查第一批止盈
    if curr_h >= partial_target_1 and not pos.get('partial_1_done'):
        # 卖出 partial_ratio_1 比例
        shares_to_sell = int(pos['shares'] * self.partial_ratio_1)
        if shares_to_sell > 0:
            self._execute_partial_trade(today, code, 'PARTIAL_TAKE_PROFIT_1',
                                         curr_o, shares_to_sell, pos, days_held)
        pos['partial_1_done'] = True

    # 检查第二批止盈
    if curr_h >= partial_target_2:
        # 卖出剩余全部
        remaining_shares = pos['shares'] - int(pos['shares'] * self.partial_ratio_1)
        if remaining_shares > 0:
            self._execute_partial_trade(today, code, 'PARTIAL_TAKE_PROFIT_2',
                                         curr_o, remaining_shares, pos, days_held)
        codes_to_remove.append(code)
        continue

# 2. 移动止盈 (Trailing Take Profit)
if curr_price > pos['avg_cost'] * (1 + self.trailing_offset):
    new_trailing_stop = curr_price * (1 - self.trailing_pct)
    pos['trailing_stop'] = max(pos.get('trailing_stop', 0), new_trailing_stop)

# 3. 检查移动止盈是否触发
if 'trailing_stop' in pos:
    if curr_c <= pos['trailing_stop']:
        self._execute_trade(today, code, 'TRAILING_TAKE_PROFIT', curr_c,
                           -pos['shares'], pos, days_held)
        codes_to_remove.append(code)
        continue

# 4. 跟踪止损 (Trailing Stop Loss)
if self.use_trailing_stop:
    highest_price = pos.get('highest_price', pos['avg_cost'])
    highest_price = max(highest_price, curr_h)  # 更新最高价
    pos['highest_price'] = highest_price

    if curr_c < highest_price * (1 - self.trailing_stop_pct):
        self._execute_trade(today, code, 'TRAILING_STOP_LOSS', curr_c,
                           -pos['shares'], pos, days_held)
        codes_to_remove.append(code)
        continue
```

**重要**: 需要在持仓初始化时添加 `highest_price` 字段（见 Step 4）。

- [ ] **Step 4: 在开仓时初始化 V10 特有字段**

在 `_open_new_positions` 的成交部分，为新持仓添加:

```python
self.positions[code] = {
    'avg_cost': actual_price,
    'shares': planned_shares,
    'last_buy_price': actual_price,
    'entry_date': order.get('entry_date'),
    'proba': order['proba'],
    # V10 特有字段
    'highest_price': actual_price,  # 用于跟踪止损
    'trailing_stop': 0,              # 用于移动止盈
}
```

- [ ] **Step 5: 实现 _execute_partial_trade 方法**

添加分批交易执行方法:

```python
def _execute_partial_trade(self, today, code, action, price, shares, pos, days_held):
    """执行部分卖出交易"""
    revenue = price * abs(shares)
    profit = revenue - (pos['avg_cost'] * abs(shares))
    profit_pct = (price - pos['avg_cost']) / pos['avg_cost']
    self.cash += revenue

    # 更新持仓数量
    pos['shares'] -= abs(shares)

    self._log(today, code, action, price, shares, profit, profit_pct, days_held, pos['proba'])
```

- [ ] **Step 6: 重要 - 调整检查顺序**

V10 的 `_manage_positions` 执行顺序应为:
1. 固定止损 (V8原有)
2. 分批止盈 (V10新增) - 第一批
3. 分批止盈 (V10新增) - 第二批（触发后移除）
4. 移动止盈 (V10新增)
5. 跟踪止损 (V10新增)
6. 固定止盈 (V8原有)
7. 时间退出 (V8原有)

**注意**: 移动止盈和跟踪止损需要更新 `highest_price`，需要在检查止盈之前更新。

- [ ] **Step 7: 提交代码**

```bash
git add src/strategies/smart_sniper_strategy_v10.py
git commit -m "feat: add SmartSniperStrategyV10 with trailing take profit

- 移动止盈 (trailing_offset, trailing_pct)
- 分批止盈 (use_partial_take_profit, partial_profit_1/2, partial_ratio_1)
- 跟踪止损 (use_trailing_stop, trailing_stop_pct)
- 禁止import V8，完整复制V8基础逻辑"
```

---

## Chunk 4: V10 模拟脚本实现

### Task 5: 创建 grid_trading_simulation_v10.py

**文件:**
- 创建: `src/grid_trading_simulation_v10.py`

- [ ] **Step 1: 复制 V8 模拟脚本基础代码**

从 `src/grid_trading_simulation_v8.py` 复制完整代码。

- [ ] **Step 2: 修改 import**

将策略类改为 V10，添加内联 data_process（如 Task 2 所述）。

- [ ] **Step 3: 修改参数设置**

添加 V10 参数设置:

```python
# V8 最佳固定参数 (不复用搜索)
# base_ratio = 1.0, target_profit = 0.25, hard_stop_loss = -0.12,
# max_hold_days = 18, min_probability = 0.5, max_positions = 3

# V10 新增参数
strategy.trailing_offset = param_dict['trailing_offset']
strategy.trailing_pct = param_dict['trailing_pct']
strategy.use_partial_take_profit = param_dict['use_partial_take_profit']
strategy.partial_profit_1 = param_dict['partial_profit_1']
strategy.partial_profit_2 = param_dict['partial_profit_2']
strategy.partial_ratio_1 = param_dict['partial_ratio_1']
strategy.use_trailing_stop = param_dict['use_trailing_stop']
strategy.trailing_stop_pct = param_dict['trailing_stop_pct']
```

- [ ] **Step 4: 定义 V10 参数网格**

```python
param_grid = {
    # V10 新增参数 (V8参数固定为最佳值)
    'trailing_offset': [0.05, 0.08, 0.10],
    'trailing_pct': [0.15, 0.20, 0.25],
    'use_partial_take_profit': [True, False],
    'partial_profit_1': [0.10, 0.12, 0.15],
    'partial_profit_2': [0.20, 0.25, 0.30],
    'partial_ratio_1': [0.4, 0.5, 0.6],
    'use_trailing_stop': [True, False],
    'trailing_stop_pct': [0.15, 0.20, 0.25],
}
```

- [ ] **Step 5: 提交代码**

```bash
git add src/grid_trading_simulation_v10.py
git commit -m "feat: add grid_trading_simulation_v10 for V10 backtesting"
```

---

### Task 6: 创建 grid_trading_simulation_v10_mp.py

**文件:**
- 创建: `src/grid_trading_simulation_v10_mp.py`

- [ ] **Step 1: 复制 V8_MP 基础代码**

从 `src/grid_trading_simulation_v8_mp.py` 复制完整代码。

- [ ] **Step 2: 修改 import**

将策略类改为 V10，添加内联 data_process。

- [ ] **Step 3: 修改参数设置**

同 Task 5 Step 3。

- [ ] **Step 4: 更新输出文件名**

改为 `parameter_optimization_results_concurrent_v10.csv`

- [ ] **Step 5: 提交代码**

```bash
git add src/grid_trading_simulation_v10_mp.py
git commit -m "feat: add grid_trading_simulation_v10_mp for V10 concurrent optimization"
```

---

## Chunk 5: 验证与测试

### Task 7: 验证 V9 实现

- [ ] **Step 1: 运行单次 V9 回测**

```bash
cd src
python grid_trading_simulation_v9.py
```

验证点:
- 无 import 错误
- 策略正常初始化
- 交易日志正常生成

### Task 8: 验证 V10 实现

- [ ] **Step 1: 运行单次 V10 回测**

```bash
cd src
python grid_trading_simulation_v10.py
```

验证点:
- 无 import 错误
- 移动止盈/分批止盈逻辑正常执行
- 交易日志正常生成

### Task 9: 提交全部更改

- [ ] **Step 1: 检查 git 状态**

```bash
git status
```

应显示新增的 6 个文件。

- [ ] **Step 2: 提交**

```bash
git add -A
git commit -m "feat: complete V9 and V10 strategy implementation

V9: 涨幅过滤 + RQ动态止盈止损 + 大盘动态仓位
V10: 移动止盈 + 分批止盈 + 跟踪止损

禁止import V8，各版本完全独立实现"
```

---

## 关键检查点

1. **V8 核心逻辑一致性**: 验证以下逻辑在 V9/V10 中与 V8 完全一致:
   - 每日流程顺序: 卖出 → 买入 → 结算
   - T+1 规则: `days_held < 1` 不能卖
   - 硬止损触发条件: `curr_o <= stop_price or curr_l <= stop_price`
   - 止盈触发条件: `curr_h >= target_price`
   - 成交价: `actual_price = signal_price`
   - 跌停处理: 不跳过，照常买入

2. **版本隔离**: 确认 V9/V10 文件中无 `import smart_sniper_strategy` 或 `import grid_trading_simulation_v8`

3. **参数组合数**:
   - V9: 23,328 组合
   - V10: 2,916 组合
