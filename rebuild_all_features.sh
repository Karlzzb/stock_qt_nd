#!/bin/bash
# 全量特征重算 - 根据数据量动态调整并发数

set -e  # 遇到错误立即退出

LOG_FILE="full_rebuild_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================================" | tee -a $LOG_FILE
echo "全量特征重算开始: $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# 2001-2010: 数据量小，24并发
echo "" | tee -a $LOG_FILE
echo "=== 阶段 1: 2001-2010 (24并发) ===" | tee -a $LOG_FILE
echo "预计耗时: ~1-2小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2001-12-31 \
  --end-date 2010-12-31 \
  --workers 24 2>&1 | tee -a $LOG_FILE

# 2011-2015: 数据量中等，16并发
echo "" | tee -a $LOG_FILE
echo "=== 阶段 2: 2011-2015 (16并发) ===" | tee -a $LOG_FILE
echo "预计耗时: ~2-3小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2011-01-01 \
  --end-date 2015-12-31 \
  --workers 16 2>&1 | tee -a $LOG_FILE

# 2016-2020: 数据量较大，12并发
echo "" | tee -a $LOG_FILE
echo "=== 阶段 3: 2016-2020 (12并发) ===" | tee -a $LOG_FILE
echo "预计耗时: ~3-4小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2016-01-01 \
  --end-date 2020-12-31 \
  --workers 12 2>&1 | tee -a $LOG_FILE

# 2021-2026: 数据量最大，8并发
echo "" | tee -a $LOG_FILE
echo "=== 阶段 4: 2021-2026 (8并发) ===" | tee -a $LOG_FILE
echo "预计耗时: ~4-5小时" | tee -a $LOG_FILE
uv run python scripts/run_feature_pipeline.py \
  --start-date 2021-01-01 \
  --end-date 2026-08-14 \
  --workers 8 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "全量特征重算完成: $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# 统计结果
echo "" | tee -a $LOG_FILE
echo "生成的特征文件统计:" | tee -a $LOG_FILE
ls real_feature_data_daily/realistic_features_*.csv | wc -l | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "日志已保存到: $LOG_FILE"
