#!/bin/bash
# 完整的重训流程：验证特征 -> 重训 label grid -> 生成对比报告
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "步骤 1/4: 快速验证市场特征质量"
echo "========================================"
python3 scripts/analyze_feature_missing_rate.py --sample 100

echo ""
echo "========================================"
echo "步骤 2/4: 生成完整训练缓存"
echo "========================================"
# 使用训练脚本自动生成 feature_cache.parquet
MALLOC_ARENA_MAX=4 uv run python scripts/run_walk_forward_train.py \
  --single-split \
  --train-end 2021-12-31 \
  --score-start 2022-01-01 \
  --score-end 2025-07-31 \
  --dry-run

echo ""
echo "========================================"
echo "步骤 3/4: 重跑 label grid（7个label）"
echo "========================================"
MALLOC_ARENA_MAX=4 uv run python scripts/run_label_grid.py \
  --horizons 3,5,10,15,20,25,30

echo ""
echo "========================================"
echo "步骤 4/4: 生成对比报告"
echo "========================================"
python3 << 'PYEOF'
import json
from pathlib import Path
import pandas as pd

# 加载新的 rank metrics
new_metrics_path = Path('experiments/rank_metrics_2021-12-31_2022-01-01_2025-07-31.json')
if not new_metrics_path.exists():
    print("❌ 未找到新的 rank metrics")
    exit(1)

with open(new_metrics_path) as f:
    new_metrics = json.load(f)

# 准备对比表格
results = []
for label, data in new_metrics.items():
    icir = data['daily_rank_icir']
    top5_excess = data['daily_top5_excess_ret']
    results.append({
        'label': label,
        'IC': data['daily_rank_ic'],
        'ICIR': icir,
        'top5_excess': top5_excess,
        'tradable': '✅' if icir > 1.5 and top5_excess > 0.02 else '❌',
    })

df = pd.DataFrame(results).sort_values('ICIR', ascending=False)

print("\n" + "="*60)
print("恢复市场特征后的 Rank Metrics（验证集 2022-2025/07）")
print("="*60)
print(df.to_string(index=False))
print("\n可交易标准: ICIR > 1.5 AND top5_excess > 2%")

# 检查是否有可交易label
tradable_count = df['tradable'].str.contains('✅').sum()
if tradable_count > 0:
    print(f"\n🎉 有 {tradable_count} 个 label 达到可交易阈值！")
else:
    print("\n⚠️  暂无 label 达到可交易阈值")

# 保存报告
report_path = Path('reports/market_features_recovery_comparison.md')
with open(report_path, 'w') as f:
    f.write("# 市场特征恢复后的性能对比\n\n")
    f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
    f.write("## 问题背景\n\n")
    f.write("之前的训练缺少 13 个市场特征（sh_/sz_ 前缀）和 1 个背离特征（divergence_magnitude），\n")
    f.write("原因是数据加载 bug 导致指数数据未正确合并。\n\n")
    f.write("## 修复内容\n\n")
    f.write("1. run_feature_pipeline.py: 指数文件列名 volume 不是 vol\n")
    f.write("2. feature_pipeline_v2.py load_price_data: 自动加载指数数据\n")
    f.write("3. _calculate_single_index_features: 时间戳类型转换\n")
    f.write("4. calculate_macd_percentile_vectorized: 窗口不足返回 NaN\n\n")
    f.write("## 恢复后的 Rank Metrics\n\n")
    f.write("验证集: 2022-01-01 至 2025-07-31\n\n")
    f.write("```\n")
    f.write(df.to_string(index=False))
    f.write("\n```\n\n")
    f.write("可交易标准: ICIR > 1.5 AND top5_excess > 2%\n\n")
    if tradable_count > 0:
        f.write(f"**结果: ✅ 有 {tradable_count} 个 label 达到可交易阈值**\n")
    else:
        f.write("**结果: ⚠️ 暂无 label 达到可交易阈值**\n")

print(f"\n报告已保存: {report_path}")

PYEOF

echo ""
echo "========================================"
echo "✅ 完整流程执行完毕"
echo "========================================"
