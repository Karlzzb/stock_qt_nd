#!/bin/bash
# 全量特征重算 - 智能并发调度
# 根据历史数据量自动调整并发数

set -e

LOG_FILE="full_rebuild_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================================" | tee -a $LOG_FILE
echo "全量特征重算开始: $(date)" | tee -a $LOG_FILE
echo "版本: FEATURE_PIPELINE_VERSION 2.3.0" | tee -a $LOG_FILE
echo "新增: 7个流动性微观结构特征" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# 阶段1: 2001-2005 (股票数量少，~2000只，20并发)
echo "" | tee -a $LOG_FILE
echo "=== 阶段 1: 2001-2005 (20并发) ===" | tee -a $LOG_FILE
echo "数据特点: 股票数量少(~2000只)" | tee -a $LOG_FILE
echo "预计耗时: ~1小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2001-12-31 \
  --end-date 2005-12-31 \
  --workers 20 2>&1 | tee -a $LOG_FILE

# 阶段2: 2006-2010 (股票数量中等，~2500只，16并发)
echo "" | tee -a $LOG_FILE
echo "=== 阶段 2: 2006-2010 (16并发) ===" | tee -a $LOG_FILE
echo "数据特点: 股票数量中等(~2500只)" | tee -a $LOG_FILE
echo "预计耗时: ~1.5小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2006-01-01 \
  --end-date 2010-12-31 \
  --workers 16 2>&1 | tee -a $LOG_FILE

# 阶段3: 2011-2015 (股票数量增加，~3500只，14并发)
echo "" | tee -a $LOG_FILE
echo "=== 阶段 3: 2011-2015 (14并发) ===" | tee -a $LOG_FILE
echo "数据特点: 股票数量增加(~3500只)" | tee -a $LOG_FILE
echo "预计耗时: ~2小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2011-01-01 \
  --end-date 2015-12-31 \
  --workers 14 2>&1 | tee -a $LOG_FILE

# 阶段4: 2016-2019 (股票数量较多，~4500只，12并发)
echo "" | tee -a $LOG_FILE
echo "=== 阶段 4: 2016-2019 (12并发) ===" | tee -a $LOG_FILE
echo "数据特点: 股票数量较多(~4500只)" | tee -a $LOG_FILE
echo "预计耗时: ~2.5小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2016-01-01 \
  --end-date 2019-12-31 \
  --workers 12 2>&1 | tee -a $LOG_FILE

# 阶段5: 2020-2023 (股票数量最多，~5500只，10并发)
echo "" | tee -a $LOG_FILE
echo "=== 阶段 5: 2020-2023 (10并发) ===" | tee -a $LOG_FILE
echo "数据特点: 股票数量最多(~5500只)" | tee -a $LOG_FILE
echo "预计耗时: ~3小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --workers 10 2>&1 | tee -a $LOG_FILE

# 阶段6: 2024-2026 (最新数据，~5800只，8并发，最保守)
echo "" | tee -a $LOG_FILE
echo "=== 阶段 6: 2024-2026 (8并发) ===" | tee -a $LOG_FILE
echo "数据特点: 最新数据，股票数量~5800只" | tee -a $LOG_FILE
echo "预计耗时: ~2小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2026-08-14 \
  --workers 8 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "全量特征重算完成: $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# 统计结果
echo "" | tee -a $LOG_FILE
echo "生成的特征文件统计:" | tee -a $LOG_FILE
total_files=$(ls real_feature_data_daily/realistic_features_*.csv 2>/dev/null | wc -l)
echo "  总文件数: $total_files" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "按年份统计:" | tee -a $LOG_FILE
for year in {2001..2026}; do
    count=$(ls real_feature_data_daily/realistic_features_${year}*.csv 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo "  $year: $count 文件" | tee -a $LOG_FILE
    fi
done

echo "" | tee -a $LOG_FILE
echo "磁盘占用:" | tee -a $LOG_FILE
du -sh real_feature_data_daily/ | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "日志文件: $LOG_FILE"
echo "======================================================================" | tee -a $LOG_FILE
