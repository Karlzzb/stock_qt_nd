#!/bin/bash
# 重新生成2022-07至2022-09特征（包含新增的流动性微观结构特征）

echo "=== 删除旧特征文件 ==="
rm -f real_feature_data_daily/realistic_features_202207*.csv
rm -f real_feature_data_daily/realistic_features_202207*.csv.fp
rm -f real_feature_data_daily/realistic_features_202208*.csv
rm -f real_feature_data_daily/realistic_features_202208*.csv.fp
rm -f real_feature_data_daily/realistic_features_202209*.csv
rm -f real_feature_data_daily/realistic_features_202209*.csv.fp

echo ""
echo "=== 重新生成 2022-07 到 2022-09 ==="
uv run python scripts/run_feature_pipeline.py --start-date 2022-07-01 --end-date 2022-09-30 --workers 24

echo ""
echo "=== 验证新特征效果 ==="
uv run python validate_market_features.py
