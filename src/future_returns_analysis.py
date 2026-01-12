import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings

from comm_fun import model_config, get_return_threshold
from config.settings import DATASET_DIR, RESULT_DIR

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")


def load_data(file_pattern):
    """加载数据集"""
    if isinstance(file_pattern, list):
        file_list = file_pattern
    else:
        file_list = glob.glob(file_pattern)

    all_data = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            all_data.append(df)
            print(f"✅ 成功加载: {file_path} ({len(df)} 行)")
        except Exception as e:
            print(f"❌ 加载文件 {file_path} 时出错: {e}")

    return pd.concat(all_data, ignore_index=True)


def analyze_predictability(series, name):
    """分析预测难度"""
    print(f"\n{name} 预测性分析:")
    print(f"方差: {series.var():.6f}")

    # 处理空值或单一值情况
    if len(series.dropna()) < 2:
        print("数据不足，跳过自相关计算")
        return None

    autocorr = series.autocorr()
    print(f"自相关性(滞后1): {autocorr:.4f}" if not pd.isna(autocorr) else "自相关性: NaN")

    # 避免除以0
    if abs(series.mean()) > 1e-10:
        predictability = series.var() / abs(series.mean())
        print(f"预测难度指标: {predictability:.4f} (越小越易预测)")
        return predictability
    else:
        print("均值接近0，跳过预测难度计算")
        return None


def analyze_feature_correlation(data, numeric_cols, return_cols):
    """分析特征与收益之间的相关性"""
    corr_matrix = data[numeric_cols].corrwith(data[return_cols])

    print("\n特征与收益相关性分析:")
    print(corr_matrix.sort_values(ascending=False))

    # 可视化
    plt.figure(figsize=(12, 8))
    corr_matrix.abs().sort_values().plot(kind='barh')
    plt.title('特征与收益的绝对相关性')
    plt.tight_layout()
    plt.savefig(str(DATASET_DIR / 'feature_correlation.png'), dpi=150)
    plt.close()


def visualize_return_distribution(data, return_cols, bins=50):
    """可视化收益分布"""
    n_cols = min(3, len(return_cols))
    n_rows = (len(return_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(return_cols):
        ax = axes[idx]
        data[col].hist(bins=bins, ax=ax, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'{col} 分布\n均值: {data[col].mean():.4f}')
        ax.set_xlabel('收益')
        ax.set_ylabel('频数')

    for idx in range(len(return_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(str(DATASET_DIR / 'return_distributions.png'), dpi=150)
    plt.close()


def select_best_return_label(df, return_cols):
    """选择能够带来最大收益的标签"""
    # 计算每个样本的最佳持有期
    best_returns = df[return_cols].max(axis=1)
    best_periods = df[return_cols].idxmax(axis=1)

    print("\n最佳持有期分布:")
    print(best_periods.value_counts())

    # 分析各期限的相对表现
    period_performance = {}
    for col in return_cols:
        # 计算该期限作为最佳选择的频率
        freq = (best_periods == col).mean()
        avg_return_when_best = df.loc[best_periods == col, col].mean()

        period_performance[col] = {
            'best_frequency': freq,
            'avg_return_when_best': avg_return_when_best,
            'overall_avg_return': df[col].mean()
        }

    return period_performance


def main():
    """主分析流程"""
    # 1. 数据加载
    file_pattern = [
        os.path.join(DATASET_DIR, "train_set.csv"),
        os.path.join(DATASET_DIR, "validation_set.csv"),
        os.path.join(DATASET_DIR, "test_set.csv")
    ]

    data = load_data(file_pattern)
    print(f"数据行:{len(data)}")
    print("数据列类型:")
    print(data.dtypes.value_counts())

    # 2. 数据类型检查
    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    print(f"\n非数值列: {list(non_numeric_cols)}")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    print(f"\n数值列: {list(numeric_cols)}")

    # 3. 收益列定义
    return_cols = []
    for period in model_config.RETURN_PERIODS:
        return_cols.append(f'future_return_{period}d')
        return_cols.append(f'stop_loss_return_{period}d')

    # 4. 基础统计分析
    print("基础统计分析:")
    print(data[return_cols].describe())
    data[return_cols].describe().to_csv( RESULT_DIR / 'future_return_describe.csv')

    for col in return_cols:
        print(f"\n{col} 额外指标:")
        print(f"正收益比例: {(data[col] > get_return_threshold(data)).mean():.2%}")
        print(f"波动率(年化): {data[col].std() * np.sqrt(252):.2%}")
        print(f"夏普比率(假设无风险利率为0): {data[col].mean() / data[col].std():.4f}")

    # 5. 预测难度分析
    predictability_scores = {}
    for col in return_cols:
        score = analyze_predictability(data[col], col)
        predictability_scores[col] = score

    # 6. 最佳持有期分析
    performance = select_best_return_label(data, return_cols)
    for period, metrics in performance.items():
        print(f"\n{period}:")
        print(f"  作为最佳选择的频率: {metrics['best_frequency']:.2%}")
        print(f"  作为最佳时的平均收益: {metrics['avg_return_when_best']:.4f}")
        print(f"  整体平均收益: {metrics['overall_avg_return']:.4f}")

    # 7. 可选分析（已定义但未调用）
    # analyze_feature_correlation(data, numeric_cols, return_cols)
    visualize_return_distribution(data, return_cols)


if __name__ == "__main__":
    main()