#!/usr/bin/env python3
"""
分析训练段特征缺失率
用于验证市场特征修复后的数据质量
"""
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
import sys

def analyze_missing_rates(start_date='20010101', end_date='20211231', sample_size=None):
    """
    分析指定时间段的特征缺失率

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        sample_size: 抽样文件数，None=全部
    """
    feature_dir = Path('real_feature_data_daily')

    # 获取时间段内的文件
    all_files = sorted([
        f for f in feature_dir.glob('realistic_features_*.csv')
        if start_date <= f.stem.split('_')[-1] <= end_date
    ])

    if not all_files:
        print(f"❌ 未找到 {start_date}-{end_date} 范围内的特征文件")
        return None

    print(f"时间段: {start_date} - {end_date}")
    print(f"总文件数: {len(all_files)}")

    # 抽样
    if sample_size and sample_size < len(all_files):
        # 均匀抽样
        step = len(all_files) // sample_size
        files = all_files[::step][:sample_size]
        print(f"抽样: {len(files)} 个文件")
    else:
        files = all_files
        print(f"使用全部 {len(files)} 个文件")

    # 加载数据
    print("\n加载数据...")
    all_data = []
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  已加载 {i+1}/{len(files)}")
        df = pd.read_csv(f)
        all_data.append(df)

    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"总样本数: {len(df_combined):,} (股票-天)")

    # 计算缺失率
    print("\n计算缺失率...")
    missing_rates = {}
    for col in df_combined.columns:
        if col == 'ts_code':
            continue
        missing_rate = df_combined[col].isna().sum() / len(df_combined) * 100
        missing_rates[col] = missing_rate

    # 分类统计
    market_features = {k: v for k, v in missing_rates.items() if k.startswith(('sh_', 'sz_'))}
    other_features = {k: v for k, v in missing_rates.items() if not k.startswith(('sh_', 'sz_'))}

    print("\n" + "="*60)
    print("市场特征 (sh_/sz_)")
    print("="*60)
    print(f"总数: {len(market_features)}")

    if market_features:
        high_missing = {k: v for k, v in market_features.items() if v > 30}
        if high_missing:
            print(f"\n⚠️  缺失率 >30% 的特征 ({len(high_missing)} 个):")
            for k, v in sorted(high_missing.items(), key=lambda x: -x[1])[:10]:
                print(f"  {k}: {v:.2f}%")
        else:
            print("\n✅ 所有市场特征缺失率 ≤30%")

        avg_missing = sum(market_features.values()) / len(market_features)
        max_missing = max(market_features.values())
        print(f"\n平均缺失率: {avg_missing:.2f}%")
        print(f"最高缺失率: {max_missing:.2f}%")

    print("\n" + "="*60)
    print("其他关键特征")
    print("="*60)

    # 重点检查之前有问题的特征
    key_features = [
        'macd_percentile',
        'macd_percentile_rankpct',
        'divergence_magnitude',
    ]

    for feat in key_features:
        if feat in missing_rates:
            rate = missing_rates[feat]
            status = "✅" if rate < 30 else "⚠️"
            print(f"{status} {feat}: {rate:.2f}%")
        else:
            print(f"❌ {feat}: 不存在")

    # 统计高缺失率特征
    print("\n" + "="*60)
    print("高缺失率特征汇总 (>30%)")
    print("="*60)
    high_missing_all = {k: v for k, v in missing_rates.items() if v > 30}
    print(f"总数: {len(high_missing_all)} / {len(missing_rates)}")

    if high_missing_all:
        print("\nTop 20:")
        for k, v in sorted(high_missing_all.items(), key=lambda x: -x[1])[:20]:
            print(f"  {k}: {v:.2f}%")

    # 保存结果
    result = {
        'time_range': f"{start_date}_{end_date}",
        'total_samples': len(df_combined),
        'total_features': len(missing_rates),
        'market_features': {
            'count': len(market_features),
            'avg_missing': float(sum(market_features.values()) / len(market_features)) if market_features else 0,
            'max_missing': float(max(market_features.values())) if market_features else 0,
            'high_missing_count': len([v for v in market_features.values() if v > 30]),
        },
        'high_missing_features': {k: float(v) for k, v in sorted(high_missing_all.items(), key=lambda x: -x[1])[:50]},
    }

    output_path = Path(f'reports/feature_missing_rate_{start_date}_{end_date}.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n结果已保存: {output_path}")

    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='分析特征缺失率')
    parser.add_argument('--start', default='20010101', help='开始日期 YYYYMMDD')
    parser.add_argument('--end', default='20211231', help='结束日期 YYYYMMDD')
    parser.add_argument('--sample', type=int, default=None, help='抽样文件数')

    args = parser.parse_args()
    analyze_missing_rates(args.start, args.end, args.sample)
