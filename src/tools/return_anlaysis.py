import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config.settings import RESULT_DIR

def return_analyzer(version = "v5", param_suffix = "参数1"):
    # 1. 读取数据
    file_path = str(RESULT_DIR / f'simple_run_log_{version}/simple_run_grid_{version}_asset_log_{param_suffix}.csv')
    df = pd.read_csv(file_path)
    output_dir = str(RESULT_DIR / f"portfolio_analysis_charts_{version}_{param_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 1. 计算年度收益率
    # 重采样取每年最后一个数据的total值，计算百分比变化
    yearly_resampled = df['total'].resample('YE').sum()
    yearly_returns = yearly_resampled.pct_change()
    # 第一年因为没有前值通常是NaN，或者如果第一行是初始资金，根据数据逻辑处理
    # 这里简单起见，如果第一个点是NaN则去除，或者视为0起点
    yearly_returns = yearly_returns.dropna()

    # 绘图：年度
    plt.figure(figsize=(12, 6))
    colors_y = ['#d62728' if x >= 0 else '#2ca02c' for x in yearly_returns] # 红涨绿跌
    bars = plt.bar(yearly_returns.index.strftime('%Y'), yearly_returns * 100, color=colors_y)

    plt.title('年度收益率统计 (Yearly Returns)', fontsize=16, pad=15)
    plt.ylabel('收益率 (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.8)

    # 添加详细数字标签
    for bar in bars:
        height = bar.get_height()
        xy_pos = (bar.get_x() + bar.get_width() / 2, height)
        text_pos = height + 0.5 if height >= 0 else height - 1.5
        plt.text(xy_pos[0], text_pos, f'{height:.2f}%', ha='center', va='bottom' if height>=0 else 'top', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    yearly_chart_path = os.path.join(output_dir, 'yearly_returns.png')
    plt.savefig(yearly_chart_path, dpi=300, bbox_inches='tight')
    print(f"年度收益率图表已保存: {yearly_chart_path}")
    plt.close()


    # 2. 计算季度收益率
    quarterly_resampled = df['total'].resample('QE').last()
    quarterly_returns = quarterly_resampled.pct_change().dropna()

    # 绘图：季度 (超宽图表)
    plt.figure(figsize=(20, 8))
    colors_q = ['#d62728' if x >= 0 else '#2ca02c' for x in quarterly_returns]

    # 生成X轴标签 (e.g., 2010-Q1)
    q_labels = [f"{d.year}-Q{d.quarter}" for d in quarterly_returns.index]
    x_pos = range(len(quarterly_returns))

    bars_q = plt.bar(x_pos, quarterly_returns * 100, color=colors_q)

    plt.title('季度收益率统计 (Quarterly Returns)', fontsize=18, pad=20)
    plt.ylabel('收益率 (%)', fontsize=14)
    plt.xticks(x_pos, q_labels, rotation=90, fontsize=10) # 旋转X轴标签
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlim(-1, len(x_pos))

    # 添加详细数字标签 (垂直旋转以防重叠)
    for bar in bars_q:
        height = bar.get_height()
        # 调整标签位置，正数在上方，负数在下方
        offset = 1 if height >= 0 else -3
        plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
                 f'{height:.1f}%', # 季度保留1位小数以节省空间，或者2位
                 ha='center', va='bottom', rotation=90, fontsize=9)

    plt.tight_layout()
    # 保存图表
    quarterly_chart_path = os.path.join(output_dir, 'quarterly_returns.png')
    plt.savefig(quarterly_chart_path, dpi=300, bbox_inches='tight')
    print(f"季度收益率图表已保存: {quarterly_chart_path}")
    plt.close()

    # 3. 计算月度收益率并转换为透视表 (Pivot Table) 用于热力图
    monthly_resampled = df['total'].resample('ME').last()
    monthly_returns = monthly_resampled.pct_change().dropna()

    # 构建 Year x Month 的 DataFrame
    m_df = monthly_returns.to_frame(name='ret')
    m_df['year'] = m_df.index.year
    m_df['month'] = m_df.index.month
    pivot_df = m_df.pivot(index='year', columns='month', values='ret') * 100 # 转换为百分比

    # 绘图：热力图
    plt.figure(figsize=(14, 10))
    # cmap='RdYlGn_r': Green(Low) -> Red(High) 符合中国股市颜色习惯
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='RdYlGn_r', center=0,
                linewidths=0.5, linecolor='lightgray', cbar_kws={'label': '收益率 (%)'})

    plt.title('月度收益率全览 (Monthly Returns Heatmap)', fontsize=18, pad=20)
    plt.ylabel('年份', fontsize=14)
    plt.xlabel('月份', fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # 保存图表
    monthly_chart_path = os.path.join(output_dir, 'monthly_returns.png')
    plt.savefig(monthly_chart_path, dpi=300, bbox_inches='tight')
    print(f"月度收益率图表已保存: {monthly_chart_path}")
    plt.close()