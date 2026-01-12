import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import regex as re

# 设置绘图风格 (支持中文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config.settings import RESULT_DIR

def trades_analyzer(version = "v5", param_suffix = "参数1"):
    # 1. 读取数据
    file_path = str(RESULT_DIR / f'simple_run_log/simple_run_grid_{version}_trade_log_{param_suffix}.csv')
    df = pd.read_csv(file_path)
    output_dir = str(RESULT_DIR / f"portfolio_analysis_charts_{version}_{param_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # 使用正则表达式提取日期部分并转换
    def extract_and_convert_date(date_str):
        """使用正则表达式提取 YYYY-MM-DD 部分并转换为 datetime"""
        if pd.isna(date_str):
            return pd.NaT

        # 正则表达式匹配日期部分 (YYYY-MM-DD)
        # 匹配格式: 2025-12-1, 2025-1-01, 2025-1-1 等
        match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', str(date_str))

        if match:
            date_part = match.group(1)
            # 将日期部分转换为标准格式
            year, month, day = date_part.split('-')
            # 补零
            month = month.zfill(2)
            day = day.zfill(2)
            standard_date = f"{year}-{month}-{day}"
            return pd.to_datetime(standard_date, format='%Y-%m-%d')
        else:
            # 如果没有匹配到，返回 NaT (Not a Time)
            return pd.NaT

    # 转换日期格式
    df = df.loc[df['action'] != 'OPEN_BUY']
    df['date'] = df['date'].apply(extract_and_convert_date)

    # 2. 数据聚合统计

    # 按年统计 (Yearly)
    # 使用 resample 必须先设置 index 或指定 on 参数，这里我们构建一个辅助 Series
    df_counts = df.set_index('date')
    count_yearly = df_counts.resample('YE').size()

    # 按季度统计 (Quarterly)
    count_quarterly = df_counts.resample('QE').size()

    # 按月统计 (Monthly)
    count_monthly = df_counts.resample('ME').size()


    # 3. 绘图辅助函数 (柱状图)
    def plot_count_bar(data, title, xlabel, figsize=(12, 6), rotation=0, is_quarter=False):
        plt.figure(figsize=figsize)

        # 使用蓝色渐变，数量越多颜色越深
        norm = plt.Normalize(data.min(), data.max())
        colors = plt.cm.Blues(norm(data.values))

        # 绘制
        if is_quarter:
            # 季度图使用自定义标签
            labels = [f"{d.year}-Q{d.quarter}" for d in data.index]
            bars = plt.bar(range(len(data)), data.values, color=colors)
            plt.xticks(range(len(data)), labels, rotation=90, fontsize=9)
        else:
            # 年度图
            bars = plt.bar(data.index.strftime('%Y'), data.values, color=colors)
            plt.xticks(rotation=rotation)

        plt.title(title, fontsize=16, pad=20)
        plt.ylabel('交易次数 (笔)', fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        # 保存图表
        yearly_chart_path = os.path.join(output_dir, f'{xlabel}_交易统计.png')
        plt.savefig(yearly_chart_path, dpi=300, bbox_inches='tight')
        plt.close()


    # 4. 绘图辅助函数 (月度热力图)
    def plot_count_heatmap(monthly_data):
        # 转换数据结构
        m_df = monthly_data.to_frame(name='count')
        m_df['year'] = m_df.index.year
        m_df['month'] = m_df.index.month

        pivot_df = m_df.pivot(index='year', columns='month', values='count')
        # 填充NaN为0 (有些月份可能没有交易)
        pivot_df = pivot_df.fillna(0)

        plt.figure(figsize=(14, 10))

        # 使用 Blues 色谱：白色为0，深蓝为高频
        sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap='Blues',
                    linewidths=0.5, linecolor='lightgray', cbar_kws={'label': '交易次数'})

        plt.title('月度交易频率热力表 (单位: 笔数)', fontsize=18, pad=20)
        plt.ylabel('年份', fontsize=14)
        plt.xlabel('月份', fontsize=14)
        plt.yticks(rotation=0)
        plt.tight_layout()
        # 保存图表
        yearly_chart_path = os.path.join(output_dir, f'月度交易.png')
        plt.savefig(yearly_chart_path, dpi=300, bbox_inches='tight')
        plt.close()


    # --- 生成图表 ---

    # 图1: 年度交易次数
    plot_count_bar(count_yearly, '年度交易次数统计 (Yearly Trade Count)', '年份')

    # 图2: 季度交易次数 (拉宽显示)
    plot_count_bar(count_quarterly, '季度交易次数统计 (Quarterly Trade Count)', '季度', figsize=(20, 8), is_quarter=True)

    # 图3: 月度交易次数热力图
    plot_count_heatmap(count_monthly)