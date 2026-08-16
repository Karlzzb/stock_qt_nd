# Issue #12 修复总结

## 问题描述

执行 `uv run python scripts/run_feature_pipeline.py --start-date 2010-01-01 --workers 24` 时出错：

```
TypeError: Can only merge Series or DataFrame objects, a <class 'dict'> was passed
```

错误发生在 `src/feature_pipeline_v2.py:468`

## 根本原因

`_get_default_combined_features` 方法返回 `dict`，但调用方期望 `DataFrame`。

当 `_calculate_market_features` 的 `except:` 块（line 721）捕获异常时，它返回 `_get_default_combined_features(timestamp)` 的结果（一个 dict），导致后续的 merge 操作失败。

## 修复方案

修改 `_get_default_combined_features` 方法，使其返回带 timestamp 索引的 DataFrame，与成功路径（lines 723-726）保持一致。

**修改位置**: `src/feature_pipeline_v2.py:756-767`

```python
# 之前：返回 dict
return default_features

# 现在：返回 DataFrame
default_df = pd.DataFrame([default_features])
default_df['timestamp'] = timestamp
default_df.set_index('timestamp', inplace=True)
return default_df
```

## 测试验证

1. ✅ 新增回归测试：`tests/test_issue12_market_features_dataframe.py`
2. ✅ 全部 208 个测试通过
3. ✅ 原始错误不再复现

## 额外改进：增量运行脚本

针对用户关注的"运行时间长、出错后重跑代价大"问题，新增增量运行脚本：

**新文件**: `scripts/run_feature_pipeline_incremental.py`

### 核心功能

1. **分批运行** - 按月/季度/年份分批处理
2. **进度跟踪** - 自动记录已完成和失败的日期到 `.feature_pipeline_state.json`
3. **断点续跑** - 使用 `--resume` 跳过已完成的日期
4. **失败重试** - 使用 `--retry-failed` 单独重试失败的日期
5. **更好的可见性** - `--status` 查看进度统计

### 推荐用法

```bash
# 按月跑（最推荐）
for year in {2010..2024}; do
  for month in {1..12}; do
    uv run python scripts/run_feature_pipeline_incremental.py \
      --year $year --month $month --workers 24 --batch-size 20
  done
done

# 查看进度
uv run python scripts/run_feature_pipeline_incremental.py --status

# 重试失败的日期
uv run python scripts/run_feature_pipeline_incremental.py --retry-failed
```

详细文档见：`docs/feature-pipeline-incremental-guide.md`

## 文件清单

### 修改的文件
- `src/feature_pipeline_v2.py` - 修复 `_get_default_combined_features` 返回类型

### 新增的文件
- `tests/test_issue12_market_features_dataframe.py` - 回归测试
- `scripts/run_feature_pipeline_incremental.py` - 增量运行脚本
- `docs/feature-pipeline-incremental-guide.md` - 使用指南

## 下一步行动

1. **立即可用**：原始错误已修复，可以直接运行原脚本
2. **推荐方式**：使用新的增量脚本按月分批运行，更安全可控
3. **验证**：先跑一个月测试，确认无误后批量运行

```bash
# 快速验证（跑 2010 年 1 月）
uv run python scripts/run_feature_pipeline_incremental.py \
  --year 2010 --month 1 --workers 24 --batch-size 20
```
