import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config.settings import RESULT_DIR

import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np


def comprehensive_correlation_analysis(df, col1, col2, output_dir):
    """
    专业相关性分析完整流程 - 所有图表单独保存
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 1. 数据检查
    print("=== 数据质量检查 ===")
    print(f"样本量: {len(df)}")
    print(f"缺失值: {col1}={df[col1].isnull().sum()}, {col2}={df[col2].isnull().sum()}")

    # 2. 描述性统计
    print("\n=== 描述性统计 ===")
    desc_stats = df[[col1, col2]].describe()
    print(desc_stats)

    # 保存描述性统计到CSV
    desc_stats.to_csv(os.path.join(output_dir, 'descriptive_statistics.csv'))
    print(f"描述性统计已保存到: {os.path.join(output_dir, 'descriptive_statistics.csv')}")

    # 3. 计算各种相关系数
    print("\n=== 相关系数分析 ===")
    methods = ['pearson', 'spearman', 'kendall']
    corr_results = {}
    for method in methods:
        corr = df[col1].corr(df[col2], method=method)
        corr_results[method] = corr
        results[f'{method}_corr'] = corr
        print(f"{method.title()}: {corr:.4f}")

    # 保存相关系数结果

    corr_df = pd.DataFrame.from_dict(corr_results, orient='index', columns=['correlation'])
    corr_df.to_csv(os.path.join(output_dir, 'correlation_coefficients.csv'))
    print(f"相关系数已保存到: {os.path.join(output_dir, 'correlation_coefficients.csv')}")

    # 4. 显著性检验
    r, p = pearsonr(df[col1].dropna(), df[col2].dropna())
    results['pearson_r'] = r
    results['p_value'] = p
    print(f"\nPearson检验: r = {r:.4f}, p值: {p:.6f}")
    print(f"显著性: {'显著' if p < 0.05 else '不显著'} (α=0.05)")

    # 5. 准备数据用于回归分析
    clean_df = df[[col1, col2]].dropna()
    if len(clean_df) < 2:
        print("警告: 清洗后数据不足，无法进行回归分析")
        return results

    X = clean_df[[col1]]
    y = clean_df[col2]

    # 6. 创建并保存各个图表

    # 图1: 基础散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(clean_df[col1], clean_df[col2], alpha=0.6,
                color='steelblue', edgecolor='w', s=80, label='数据点')

    # 添加标题和标签
    plt.xlabel(col1, fontsize=14, fontweight='bold')
    plt.ylabel(col2, fontsize=14, fontweight='bold')
    plt.title(f'{col1} vs {col2} 散点图', fontsize=16, fontweight='bold')

    # 添加相关系数标注
    stats_text = f'Pearson r = {r:.3f}\nn = {len(clean_df)}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    scatter_path = os.path.join(output_dir, f'scatter_{col1}_vs_{col2}.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'scatter_{col1}_vs_{col2}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"散点图已保存: {scatter_path}")

    # 图2: 带回归线和置信区间的散点图
    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制回归图（带置信区间）
    sns.regplot(x=col1, y=col2, data=clean_df,
                scatter_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'w', 'color': 'steelblue'},
                line_kws={'color': 'red', 'linewidth': 3, 'alpha': 0.8, 'label': '回归线'},
                ci=95)  # 95%置信区间

    # 拟合线性模型
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # 添加R²值
    r_squared = model.score(X, y)

    plt.xlabel(col1, fontsize=14, fontweight='bold')
    plt.ylabel(col2, fontsize=14, fontweight='bold')
    plt.title(f'{col1} vs {col2} 回归分析', fontsize=16, fontweight='bold')

    # 添加统计信息
    stats_text = f'r = {r:.3f}\nR² = {r_squared:.3f}\np = {p:.4f}\nn = {len(clean_df)}'
    if p < 0.001:
        stats_text = f'r = {r:.3f}\nR² = {r_squared:.3f}\np < 0.001\nn = {len(clean_df)}'

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    regression_path = os.path.join(output_dir, f'regression_{col1}_vs_{col2}.png')
    plt.savefig(regression_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'regression_{col1}_vs_{col2}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"回归图已保存: {regression_path}")

    # 图3: 残差图
    plt.figure(figsize=(10, 8))

    residuals = y - y_pred

    # 绘制残差散点图
    plt.scatter(y_pred, residuals, alpha=0.6,
                color='darkorange', edgecolor='w', s=60, label='残差')

    # 添加零线
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='零线')

    # 添加loess平滑线（可选）
    try:

        lowess_res = lowess(residuals, y_pred, frac=0.3)
        plt.plot(lowess_res[:, 0], lowess_res[:, 1],
                 color='blue', linewidth=3, alpha=0.7, label='趋势线')
    except ImportError:
        # 如果没有statsmodels，使用简单的移动平均
        from scipy.ndimage import uniform_filter1d
        if len(y_pred) > 10:
            window_size = max(3, len(y_pred) // 20)
            sorted_idx = np.argsort(y_pred)
            y_pred_sorted = y_pred.iloc[sorted_idx]
            residuals_sorted = residuals.iloc[sorted_idx]

            # 使用移动平均平滑
            residuals_smoothed = uniform_filter1d(residuals_sorted, size=window_size)
            plt.plot(y_pred_sorted, residuals_smoothed,
                     color='blue', linewidth=3, alpha=0.7, label='趋势线')

    plt.xlabel('预测值', fontsize=14, fontweight='bold')
    plt.ylabel('残差', fontsize=14, fontweight='bold')
    plt.title(f'{col1} vs {col2} 残差图', fontsize=16, fontweight='bold')

    # 添加残差统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    stats_text = f'均值 = {mean_residual:.3f}\n标准差 = {std_residual:.3f}\nn = {len(residuals)}'

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    residual_path = os.path.join(output_dir, f'residuals_{col1}_vs_{col2}.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'residuals_{col1}_vs_{col2}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"残差图已保存: {residual_path}")

    # 图4: 变量分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 第一个变量的分布
    axes[0].hist(clean_df[col1], bins=30, color='steelblue',
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(x=clean_df[col1].mean(), color='red',
                    linestyle='--', linewidth=2, label=f'均值 = {clean_df[col1].mean():.2f}')
    axes[0].set_xlabel(col1, fontsize=12, fontweight='bold')
    axes[0].set_ylabel('频数', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{col1} 分布', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 第二个变量的分布
    axes[1].hist(clean_df[col2], bins=30, color='darkorange',
                 edgecolor='black', alpha=0.7)
    axes[1].axvline(x=clean_df[col2].mean(), color='red',
                    linestyle='--', linewidth=2, label=f'均值 = {clean_df[col2].mean():.2f}')
    axes[1].set_xlabel(col2, fontsize=12, fontweight='bold')
    axes[1].set_ylabel('频数', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{col2} 分布', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    distribution_path = os.path.join(output_dir, f'distributions_{col1}_vs_{col2}.png')
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'distributions_{col1}_vs_{col2}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"分布图已保存: {distribution_path}")

    # 图5: 综合图表（原来的四合一图）
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 左上：散点图
    axes[0, 0].scatter(clean_df[col1], clean_df[col2], alpha=0.6)
    axes[0, 0].set_xlabel(col1, fontsize=11)
    axes[0, 0].set_ylabel(col2, fontsize=11)
    axes[0, 0].set_title(f'散点图 (r={r:.3f})', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 右上：回归图
    sns.regplot(x=col1, y=col2, data=clean_df, ax=axes[0, 1], ci=95,
                scatter_kws={'alpha': 0.6, 's': 30},
                line_kws={'color': 'red', 'linewidth': 2})
    axes[0, 1].set_xlabel(col1, fontsize=11)
    axes[0, 1].set_ylabel(col2, fontsize=11)
    axes[0, 1].set_title(f'回归图 (R²={r_squared:.3f})', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 左下：残差图
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值', fontsize=11)
    axes[1, 0].set_ylabel('残差', fontsize=11)
    axes[1, 0].set_title('残差图', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 右下：分布图
    clean_df[[col1, col2]].hist(ax=axes[1, 1], bins=20, alpha=0.7)
    axes[1, 1].set_title('变量分布', fontsize=12, fontweight='bold')

    plt.suptitle(f'{col1} 与 {col2} 相关性分析综合图', fontsize=16, fontweight='bold')
    plt.tight_layout()

    comprehensive_path = os.path.join(output_dir, f'comprehensive_{col1}_vs_{col2}.png')
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'comprehensive_{col1}_vs_{col2}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"综合图已保存: {comprehensive_path}")

    # 图6: 相关系数热力图（可选）
    plt.figure(figsize=(8, 6))

    corr_matrix = clean_df[[col1, col2]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 14, "weight": "bold"})

    plt.title(f'{col1} 与 {col2} 相关系数热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, f'heatmap_{col1}_vs_{col2}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'heatmap_{col1}_vs_{col2}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {heatmap_path}")

    # 7. 生成分析报告
    generate_analysis_report(results, clean_df, col1, col2, model, output_dir)

    print(f"\n=== 分析完成 ===")
    print(f"所有图表和结果已保存到: {output_dir}")
    print(f"共生成6个图表和2个数据文件")

    return results


def generate_analysis_report(results, df, col1, col2, model, output_dir):
    """生成详细的分析报告"""
    report_path = os.path.join(output_dir, f'correlation_analysis_report_{col1}_vs_{col2}.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"    相关性分析报告: {col1} 与 {col2}\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 数据概览\n")
        f.write("-" * 40 + "\n")
        f.write(f"分析变量: {col1} 与 {col2}\n")
        f.write(f"有效样本量: {len(df)}\n")
        f.write(f"数据收集时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("2. 描述性统计\n")
        f.write("-" * 40 + "\n")
        f.write(f"{col1}:\n")
        f.write(f"  均值: {df[col1].mean():.4f}\n")
        f.write(f"  标准差: {df[col1].std():.4f}\n")
        f.write(f"  最小值: {df[col1].min():.4f}\n")
        f.write(f"  最大值: {df[col1].max():.4f}\n\n")

        f.write(f"{col2}:\n")
        f.write(f"  均值: {df[col2].mean():.4f}\n")
        f.write(f"  标准差: {df[col2].std():.4f}\n")
        f.write(f"  最小值: {df[col2].min():.4f}\n")
        f.write(f"  最大值: {df[col2].max():.4f}\n\n")

        f.write("3. 相关性分析结果\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pearson相关系数: {results.get('pearson_corr', 'N/A'):.4f}\n")
        f.write(f"Spearman相关系数: {results.get('spearman_corr', 'N/A'):.4f}\n")
        f.write(f"Kendall相关系数: {results.get('kendall_corr', 'N/A'):.4f}\n")
        f.write(f"Pearson r值: {results.get('pearson_r', 'N/A'):.4f}\n")
        f.write(f"P值: {results.get('p_value', 'N/A'):.6f}\n")

        p_value = results.get('p_value', 1)
        significance = "显著" if p_value < 0.05 else "不显著"
        f.write(f"统计显著性 (α=0.05): {significance}\n\n")

        f.write("4. 回归分析结果\n")
        f.write("-" * 40 + "\n")
        f.write(f"回归方程: {col2} = {model.intercept_:.4f} + {model.coef_[0]:.4f} × {col1}\n")
        f.write(f"决定系数 (R²): {model.score(df[[col1]], df[col2]):.4f}\n")
        f.write(f"回归系数: {model.coef_[0]:.4f}\n")
        f.write(f"截距: {model.intercept_:.4f}\n\n")

        f.write("5. 结论与建议\n")
        f.write("-" * 40 + "\n")

        corr_strength = abs(results.get('pearson_r', 0))
        if corr_strength > 0.7:
            f.write(f"• {col1} 与 {col2} 呈现强相关关系 (|r| = {corr_strength:.3f})\n")
        elif corr_strength > 0.3:
            f.write(f"• {col1} 与 {col2} 呈现中等程度相关 (|r| = {corr_strength:.3f})\n")
        else:
            f.write(f"• {col1} 与 {col2} 相关性较弱 (|r| = {corr_strength:.3f})\n")

        if p_value < 0.05:
            f.write("• 统计上显著相关，结果具有统计学意义\n")
        else:
            f.write("• 统计上不显著，可能由于样本量不足或真实关系较弱\n")

        f.write("• 注意：相关性不代表因果关系\n")

    print(f"分析报告已保存: {report_path}")

def correlation_analyzer(version = "v5", param_suffix = "参数1"):
    # 1. 读取数据
    file_path = str(RESULT_DIR / f'simple_run_log_{version}/simple_run_grid_{version}_trade_log_{param_suffix}.csv')
    df = pd.read_csv(file_path)
    output_dir = str(RESULT_DIR / f"correlation_analysis_{version}_{param_suffix}")
    os.makedirs(output_dir, exist_ok=True)


    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[df['action'] != 'OPEN_BUY']

    # 使用示例
    results = comprehensive_correlation_analysis(df, 'profit_pct', 'proba', output_dir)
    print(results)

if __name__ == "__main__":
    correlation_analyzer()