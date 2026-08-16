# 特征计算增量运行指南

## 问题
原始的 `run_feature_pipeline.py` 运行全量数据（2010-2024）需要很长时间，如果中途出错：
- 不知道哪些日期失败了
- 无法从断点继续
- 重跑代价很大

## 解决方案：增量脚本

新的 `run_feature_pipeline_incremental.py` 提供：
1. **分批运行** - 按月/季度/年份分批处理
2. **进度跟踪** - 自动记录已完成和失败的日期
3. **断点续跑** - 出错后从上次位置继续
4. **失败重试** - 单独重试失败的日期

## 推荐用法

### 方案 A：按月份逐个跑（最推荐）

```bash
# 逐月处理，每个月完成后有状态保存
for year in {2010..2024}; do
  for month in {1..12}; do
    echo "处理 $year 年 $month 月..."
    uv run python scripts/run_feature_pipeline_incremental.py \
      --year $year --month $month \
      --workers 24 --batch-size 20
    
    # 每月完成后检查状态
    uv run python scripts/run_feature_pipeline_incremental.py --status
  done
done
```

### 方案 B：按季度跑

```bash
# 2010 Q1
uv run python scripts/run_feature_pipeline_incremental.py \
  --start-date 2010-01-01 --end-date 2010-03-31 \
  --workers 24 --batch-size 20

# 2010 Q2
uv run python scripts/run_feature_pipeline_incremental.py \
  --start-date 2010-04-01 --end-date 2010-06-30 \
  --workers 24 --batch-size 20

# ... 依此类推
```

### 方案 C：跑整年（适合测试）

```bash
# 只跑 2010 年
uv run python scripts/run_feature_pipeline_incremental.py \
  --year 2010 --workers 24 --batch-size 20
```

## 出错后的处理

### 1. 查看进度状态

```bash
uv run python scripts/run_feature_pipeline_incremental.py --status
```

输出示例：
```
============================================================
特征计算进度状态
============================================================
✅ 已完成: 1250 天
❌ 失败:   15 天
📅 最后更新: 2026-08-16T20:15:30

失败的日期（前10个）:
  1. 2010-05-12: TypeError: Can only merge Series or DataFrame...
  2. 2010-08-23: 未找到任何有效股票数据切片
  ...
============================================================
```

### 2. 重试失败的日期

```bash
# 重试所有之前失败的日期
uv run python scripts/run_feature_pipeline_incremental.py \
  --retry-failed --workers 24
```

### 3. 续跑未完成的工作

```bash
# 从某个大范围继续，会自动跳过已完成的日期
uv run python scripts/run_feature_pipeline_incremental.py \
  --start-date 2010-01-01 --end-date 2024-12-31 \
  --resume --workers 24 --batch-size 50
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--year YYYY` | 处理指定年份 | - |
| `--month M` | 配合 --year 使用，处理指定月份 | - |
| `--start-date` | 起始日期 YYYY-MM-DD | - |
| `--end-date` | 截止日期 YYYY-MM-DD | - |
| `--resume` | 跳过已完成的日期 | false |
| `--retry-failed` | 重试失败的日期 | false |
| `--status` | 显示进度状态 | false |
| `--workers N` | 并发进程数 | CPU核数-1 |
| `--batch-size N` | 每批处理天数（越小越频繁保存进度） | 50 |

## 状态文件

进度保存在 `.feature_pipeline_state.json`（项目根目录）：
```json
{
  "completed": ["2010-01-04", "2010-01-05", ...],
  "failed": {
    "2010-05-12": {
      "error": "TypeError: ...",
      "timestamp": "2026-08-16T19:49:06"
    }
  },
  "last_update": "2026-08-16T20:15:30"
}
```

## 性能优化建议

1. **batch-size 调优**：
   - 小值（10-20）：更频繁保存进度，出错损失小，但开销稍大
   - 大值（50-100）：性能更好，但出错后损失更多进度
   - 推荐：首次跑用 20，稳定后可以用 50

2. **workers 调优**：
   - 设为 CPU 物理核数 - 1（留一个核心给系统）
   - 16核机器推荐：12-14
   - 24核机器推荐：20-22

3. **按月跑的好处**：
   - 每月大约 20 交易日，batch-size=20 一批就完成
   - 出错影响范围小
   - 可以随时中断，下次继续下一个月

## 快速开始（推荐流程）

```bash
# 1. 先跑一个月测试（验证配置和性能）
uv run python scripts/run_feature_pipeline_incremental.py \
  --year 2010 --month 1 --workers 24 --batch-size 20

# 2. 检查状态
uv run python scripts/run_feature_pipeline_incremental.py --status

# 3. 如果成功，用循环跑全部
for year in {2010..2024}; do
  for month in {1..12}; do
    echo "=== 处理 $year-$(printf "%02d" $month) ==="
    uv run python scripts/run_feature_pipeline_incremental.py \
      --year $year --month $month --workers 24 --batch-size 20
  done
done

# 4. 最后检查是否有失败的，重试一次
uv run python scripts/run_feature_pipeline_incremental.py --status
uv run python scripts/run_feature_pipeline_incremental.py --retry-failed
```

## 与原脚本的对比

| 特性 | 原脚本 | 增量脚本 |
|------|--------|----------|
| 进度跟踪 | ❌ | ✅ 持久化到文件 |
| 断点续跑 | ❌ | ✅ --resume |
| 失败重试 | ❌ | ✅ --retry-failed |
| 分批运行 | ❌ 只能全跑 | ✅ 按月/季/年 |
| 错误可见性 | ⚠️ 日志流逝 | ✅ 记录到状态文件 |
| 适合场景 | 一次性全量跑 | 增量、分批、测试 |
