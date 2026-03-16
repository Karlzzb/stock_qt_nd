# WEB页面操作功能更新设计

**Date:** 2026-03-16

## 1. 概述

更新股票AI量化交易系统的WEB页面，增加四个主要功能：
1. 自定义数据路径配置（可配置的股票数据、股票列表文件路径）
2. 手动交易功能（手动买入和卖出操作）
3. st_stock_filter 脚本执行功能
4. stock_nd 脚本执行功能

## 2. 数据路径配置

### 2.1 配置文件

创建 `web_interface/paths_config.json`，存储默认路径：

```json
{
  "stock_data_dir": "E:/stock_data",
  "st_stocks_list": "data/st_stocks_list.csv",
  "new_stocks_list": "data/new_stocks_list.csv"
}
```

路径相对于项目根目录（`BASE_DIR`）。如果使用绝对路径，则直接使用。

### 2.2 Flask 配置加载

在 `app.py` 启动时加载配置：

```python
import json
from pathlib import Path

PATHS_CONFIG_FILE = Path(__file__).parent / 'paths_config.json'

def load_paths_config():
    """加载路径配置"""
    if PATHS_CONFIG_FILE.exists():
        with open(PATHS_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "stock_data_dir": "stock_data",
        "st_stocks_list": "data/st_stocks_list.csv",
        "new_stocks_list": "data/new_stocks_list.csv"
    }

def save_paths_config(config):
    """保存路径配置"""
    with open(PATHS_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
```

### 2.3 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/config/paths` | 获取当前配置的路径 |
| PUT | `/api/config/paths` | 更新路径配置（保存到 paths_config.json） |

### 2.4 执行时传递路径

修改 `app.py` 中的 `run_executor` 函数，将路径作为命令行参数传递给 `grid_trading_realworld_v8.py`：

```bash
python grid_trading_realworld_v8.py --stock-data-dir <path> --st-stocks-list <path> --new-stocks-list <path>
```

**注意：** 需要在 `grid_trading_realworld_v8.py` 中添加 argparse 参数支持，并修改实际使用路径的代码位置。

### 2.4.1 修改 grid_trading_realworld_v8.py

**修改位置1：** 脚本开头添加 argparse（支持命令行参数覆盖配置文件）

```python
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--stock-data-dir', type=str, default=None)
parser.add_argument('--st-stocks-list', type=str, default=None)
parser.add_argument('--new-stocks-list', type=str, default=None)
parser.add_argument('--predict-date', type=str, required=True)
parser.add_argument('--feature-date', type=str, required=True)
parser.add_argument('--update-data', action='store_true')
args, unknown = parser.parse_known_args()
```

**修改位置2：** 第356-357行，将：
```python
st_df = pd.read_csv(DATASET_DIR / 'st_stocks_list.csv')
new_df = pd.read_csv(DATASET_DIR / 'new_stocks_list.csv')
```

改为（支持命令行参数覆盖）：
```python
st_list_path = args.st_stocks_list if args.st_stocks_list else DATASET_DIR / 'st_stocks_list.csv'
new_list_path = args.new_stocks_list if args.new_stocks_list else DATASET_DIR / 'new_stocks_list.csv'
st_df = pd.read_csv(st_list_path)
new_df = pd.read_csv(new_list_path)
```

### 2.4.2 修改 feature_pipeline.py

`get_cached_dataset` 方法使用 `STOCK_DATA_DIR`。需要支持环境变量覆盖：

```python
import os

# 支持环境变量覆盖
DEFAULT_STOCK_DATA_DIR = Path(__file__).parent.parent / 'stock_data'
STOCK_DATA_DIR = Path(os.environ.get('STOCK_DATA_DIR', DEFAULT_STOCK_DATA_DIR))
```

然后在 `app.py` 执行脚本前设置环境变量：
```python
# 执行前设置环境变量
env = os.environ.copy()
env['STOCK_DATA_DIR'] = paths_config.get('stock_data_dir', '')

process = subprocess.Popen(cmd, ..., env=env)
```

### 2.4.3 路径传递流程

```
用户配置路径 → paths_config.json → app.py 读取配置
                                            ↓
                    执行时作为 CLI 参数传递 + 环境变量设置
                                            ↓
              grid_trading_realworld_v8.py ← argparse 接收
              feature_pipeline.py ← 环境变量 STOCK_DATA_DIR
```

## 3. 手动交易功能

### 3.1 可卖出股票列表

**GET** `/api/trading/sellable-stocks`

返回当前持仓中未卖出的股票列表（从 `portfolio_positions.csv` 读取 `is_sold = 'NO'` 的记录）。

响应示例：
```json
{
  "success": true,
  "data": [
    {
      "code": "600200.SH",
      "name": "某股票",
      "shares": 1000,
      "avg_cost": 10.5,
      "entry_date": "2026-03-01"
    }
  ]
}
```

### 3.2 可买入股票列表

**GET** `/api/trading/buyable-stocks?days=3`

读取过去N天（默认3天）的 `trade_suggestions_YYYYMMDD.csv` 文件。

**文件路径：** `real_trading_data/investment_data/trade_suggestions_YYYYMMDD.csv`

响应示例：
```json
{
  "success": true,
  "data": [
    {
      "code": "600200.SH",
      "name": "某股票",
      "score": 0.85,
      "predict_date": "2026-03-13"
    }
  ]
}
```

### 3.3 手动卖出

**POST** `/api/trading/sell`

请求体：
```json
{
  "stock_code": "600200.SH",
  "quantity": 500,
  "price": 12.0
}
```

处理逻辑：
1. 验证 quantity ≤ 当前持仓数量
2. **部分卖出处理**：
   - 如果 quantity = 当前持仓数量：标记 `is_sold = 'YES'`
   - 如果 quantity < 当前持仓数量：更新 `shares = shares - quantity`，保留原持仓记录
3. 填入 `actual_sell_price` 和 `actual_sell_date`
4. 更新 `portfolio_cash.csv`：现金增加 `quantity * price`
5. 更新 `portfolio_assets.csv`：重新计算资产总额

**CSV 文件路径：**
- `portfolio_positions.csv` → `real_trading_data/investment_data/`
- `portfolio_cash.csv` → `real_trading_data/investment_data/`
- `portfolio_assets.csv` → `real_trading_data/investment_data/`

**CSV 文件结构：**

`portfolio_positions.csv` 字段：
| 字段 | 说明 |
|------|------|
| code | 股票代码 |
| avg_cost | 平均成本 |
| shares | 持股数量 |
| entry_date | 建仓日期 |
| stop_loss_price | 止损价 |
| take_profit_price | 止盈价 |
| should_sell_date | 应卖出日期 |
| actual_sell_price | 实际卖出价 |
| actual_sell_date | 实际卖出日期 |
| is_sold | 是否已卖出 (YES/NO) |

`portfolio_cash.csv` 字段：
| 字段 | 说明 |
|------|------|
| cash | 当前现金 |
| last_run_date | 最后运行日期 |
| update_time | 更新时间 |

`portfolio_assets.csv` 字段：
| 字段 | 说明 |
|------|------|
| predict_date | 预测日期 |
| total | 总资产 |
| cash | 现金 |
| positions_value | 持仓市值 |

响应：
```json
{
  "success": true,
  "message": "卖出成功",
  "data": {
    "cash": 15000,
    "sold_quantity": 500
  }
}
```

### 3.4 手动买入

**POST** `/api/trading/buy`

请求体：
```json
{
  "stock_code": "600200.SH",
  "price": 11.0,
  "quantity": 1000
}
```

处理逻辑：
1. 验证现金充足：`price * quantity ≤ current_cash`
2. 更新 `portfolio_positions.csv`：添加新持仓记录，`avg_cost = price`
3. 更新 `portfolio_cash.csv`：现金减少 `quantity * price`
4. 更新 `portfolio_assets.csv`：重新计算资产总额

响应：
```json
{
  "success": true,
  "message": "买入成功",
  "data": {
    "cash": 4000,
    "new_position": {
      "code": "600200.SH",
      "shares": 1000,
      "avg_cost": 11.0
    }
  }
}
```

## 4. 前端页面更新

### 4.1 路径配置区域

在WEB页面添加"数据路径配置"区块：
- 显示当前配置的路径
- 提供输入框修改路径
- 保存按钮调用 PUT `/api/config/paths`

### 4.2 手动交易区域

添加"手动交易"区块，包含两个标签页：

**卖出标签页：**
- 下拉框选择持仓股票（调用 GET `/api/trading/sellable-stocks`）
- 输入框填写卖出数量和价格
- 提交按钮调用 POST `/api/trading/sell`

**买入标签页：**
- 下拉框选择建议买入股票（调用 GET `/api/trading/buyable-stocks`）
- 输入框填写买入价格和数量（价格可手动修改）
- 提交按钮调用 POST `/api/trading/buy`

## 5. 数据流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│  WEB UI     │────▶│ Flask API   │────▶│ PortfolioState      │
│             │     │ (app.py)    │     │ (grid_trading_...py)│
└─────────────┘     └─────────────┘     └─────────────────────┘
                                                 │
                    ┌────────────────────────────┤
                    ▼                            ▼
           ┌─────────────────┐         ┌─────────────────┐
           │ CSV数据文件     │         │ trade_suggestions│
           │ - positions.csv│         │ _YYYYMMDD.csv   │
           │ - cash.csv      │         └─────────────────┘
           │ - assets.csv    │
           └─────────────────┘
```

## 6. 辅助脚本执行功能

新增两个辅助脚本的执行功能，用于数据准备。

### 6.1 全局执行状态

添加全局状态变量来管理辅助脚本的执行：

```python
# 扩展 execution_status
execution_status = {
    "running": False,           # 主交易脚本运行中
    "st_filter_running": False,  # st_stock_filter 运行中
    "stock_nd_running": False,   # stock_nd 运行中
    "start_time": None,
    "log_file": None,
    "progress": "",
    "process": None
}
```

**状态锁定规则：**
- `st_filter_running` 和 `stock_nd_running` 互不影响（可同时运行）
- 当任一辅助脚本运行时，`running = True`（主交易脚本不可执行）
- 主交易脚本运行时，`st_filter_running` 和 `stock_nd_running` 保持不变

### 6.2 st_stock_filter 脚本执行

**脚本路径：** `src/st_stock_filter.py`

**API 端点：**

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/execute/st-filter` | 启动 st_stock_filter 脚本 |
| POST | `/api/execute/st-filter/stop` | 停止 st_stock_filter 脚本 |

**请求体（可选）：**
```json
{
  "date": "2026-03-16"
}
```

**响应：**
```json
{
  "success": true,
  "message": "st_stock_filter 开始执行",
  "log_file": "execution_20260316_143022.log"
}
```

### 6.3 stock_nd 脚本执行

**脚本路径：** `src/stock_nd.py`

**API 端点：**

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/execute/stock-nd` | 启动 stock_nd 脚本 |
| POST | `/api/execute/stock-nd/stop` | 停止 stock_nd 脚本 |

**请求体（可选）：**
```json
{
  "date": "2026-03-16"
}
```

**响应：**
```json
{
  "success": true,
  "message": "stock_nd 开始执行",
  "log_file": "execution_stocknd_20260316.log"
}
```

### 6.4 执行状态查询

扩展 `/api/status` 返回所有执行状态：

```json
{
  "running": false,
  "st_filter_running": true,
  "stock_nd_running": false,
  "st_filter_progress": "正在下载ST股票数据...",
  "stock_nd_progress": "",
  "st_filter_log": "execution_stfilter_20260316.log",
  "stock_nd_log": ""
}
```

### 6.5 前端页面更新

在WEB页面添加"数据准备"区块：

- **st_stock_filter 按钮：**
  - 显示"执行ST股票过滤"
  - 执行中显示"停止"按钮
  - 状态：空闲/运行中

- **stock_nd 按钮：**
  - 显示"下载股票数据"
  - 执行中显示"停止"按钮
  - 状态：空闲/运行中

**锁定逻辑：**
- 辅助脚本运行期间，主交易脚本按钮禁用
- 主交易脚本运行时，辅助脚本按钮仍可用（但不建议同时运行）

## 7. 错误处理

| 场景 | HTTP状态码 | 错误信息 |
|------|------------|----------|
| 卖出数量超过持仓 | 400 | "卖出数量超过当前持仓" |
| 现金不足买入 | 400 | "现金不足" |
| 股票不在持仓中 | 404 | "股票不在持仓中" |
| 路径配置无效 | 400 | "路径不存在或无效" |
| 辅助脚本运行中 | 400 | "辅助脚本运行中，无法执行主交易" |

### 7.1 事务处理

由于买卖操作涉及多个 CSV 文件的更新，为保证数据一致性，采用以下策略：

1. **先读取所有需要的数据**
2. **验证通过后再执行更新**
3. **按顺序更新各文件**，失败时返回错误

简化处理：手动交易操作频率较低，暂不实现复杂的回滚机制。操作失败时返回明确的错误信息，由用户决定是否重试。

## 7. 验证清单

- [ ] 路径配置可读取和更新
- [ ] 执行时正确传递路径参数
- [ ] 可卖出股票列表正确显示当前持仓
- [ ] 可买入股票列表正确显示过去3天的建议股票
- [ ] 手动卖出正确更新所有相关CSV文件
- [ ] 手动买入正确更新所有相关CSV文件
- [ ] st_stock_filter 脚本可执行、可停止
- [ ] stock_nd 脚本可执行、可停止
- [ ] 两脚本可同时运行
- [ ] 辅助脚本运行时，主交易脚本被锁定
- [ ] 错误情况正确处理并返回友好提示
- [ ] 前端页面交互流畅
