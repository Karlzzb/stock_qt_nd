import pandas as pd
import matplotlib.pyplot as plt
import os
import regex as re

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config.settings import RESULT_DIR

def profit_analyzer(version = "v5", param_suffix = "参数1"):
    # 1. 读取数据
    file_path = str(RESULT_DIR / f'simple_run_log_v12/simple_run_grid_{version}_trade_log_{param_suffix}.csv')
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

    # 过滤&转换日期格式
    df = df.loc[df['action'] != 'OPEN_BUY']
    df['date'] = df['date'].apply(extract_and_convert_date)

    # 2. 计算收益率并分类
    df['roi'] = df['profit_pct']


    def classify_trade(roi):
        if roi >= 0.25:
            return '3. >= 25% (暴利)'
        elif roi < 0.005:
            return '1. < 0.5% (止损/微利)'
        else:
            return '2. 0.5%~25% (常规)'


    df['category'] = df['roi'].apply(classify_trade)


    # 3. 数据聚合函数
    def get_stacked_data(resample_rule):
        # 先设置索引
        temp_df = df.set_index('date')
        # 使用 crosstab 或 groupby size + unstack
        # 这里为了保证时间连续性，建议用 groupby TimeGrouper (resample logic)
        # 但直接用 crosstab 比较简单，只是需要重新索引时间轴以防漏掉某些月份

        # 更好的方法：添加辅助列 Period
        if resample_rule == 'Y':
            temp_df['period'] = temp_df.index.strftime('%Y')
        elif resample_rule == 'Q':
            # 自定义排序键
            temp_df['period'] = temp_df.index.to_period('Q').astype(str)
        elif resample_rule == 'M':
            temp_df['period'] = temp_df.index.to_period('M').astype(str)

        pivot = pd.crosstab(temp_df['period'], temp_df['category'])

        # 确保列的顺序: 暴利在最上(红色), 常规中间(橙色), 亏损在下(绿色)
        # 现在的列名包含数字前缀 1. 2. 3. 方便排序
        cols = ['1. < 0.5% (止损/微利)', '2. 0.5%~25% (常规)', '3. >= 25% (暴利)']
        # 重新索引列，防止某时期没有某类交易报错
        pivot = pivot.reindex(columns=cols, fill_value=0)

        return pivot


    # 4. 通用绘图函数
    def plot_stacked_bar(data, title, xlabel, figsize=(12, 6), rotation=0):
        ax = data.plot(kind='bar', stacked=True, figsize=figsize,
                       color=['#2ca02c', '#ff7f0e', '#d62728'],  # 绿(亏), 橙(中), 红(盈)
                       width=0.8)

        plt.title(title, fontsize=16, pad=20)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('交易笔数', fontsize=12)
        plt.xticks(rotation=rotation)
        plt.legend(title='收益率区间', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # 添加总数标签 (可选，太挤就不加了，或者只加在总高处)
        for p in ax.patches:
            height = p.get_height()
            # 只显示数量比较多的标签，避免拥挤
            if height > 0 and height > data.max().max() * 0.05:
                ax.annotate(str(int(height)),
                            (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                            ha='center', va='center', color='white', fontsize=9, fontweight='bold')

        plt.tight_layout()
        # 保存图表
        yearly_chart_path = os.path.join(output_dir, f'{xlabel}_returns_stacked.png')
        plt.savefig(yearly_chart_path, dpi=300, bbox_inches='tight')
        plt.close()


    # --- 生成图表 ---

    # 1. 按年
    data_yearly = get_stacked_data('Y')
    plot_stacked_bar(data_yearly, '年度交易收益分布 (按年)', '年份')

    # 2. 按季度 (需要处理一下排序，虽然字符串排序YYYY-Qx通常是对的)
    data_quarterly = get_stacked_data('Q')
    plot_stacked_bar(data_quarterly, '季度交易收益分布 (按季度)', '季度', figsize=(20, 8), rotation=90)

    # 3. 按月 (数据太多，柱状图会很密，我们换一种方式或者拉超宽)
    data_monthly = get_stacked_data('M')
    # 考虑到月份太多，我们生成图表但不显示内部数字，仅展示色块分布
    plt.figure(figsize=(20, 8))
    # 为了月份排序正确，重新生成带DatetimeIndex的数据
    m_temp = df.set_index('date').resample('ME')['category'].value_counts().unstack().fillna(0)
    # 重新排序列
    cols = ['1. < 0.5% (止损/微利)', '2. 0.5%~25% (常规)', '3. >= 25% (暴利)']
    m_temp = m_temp.reindex(columns=cols, fill_value=0)

    # 绘制月度图
    ax = m_temp.plot(kind='bar', stacked=True, figsize=(24, 8),
                     color=['#2ca02c', '#ff7f0e', '#d62728'], width=1.0)  # width=1.0 也就是无缝
    plt.title('月度交易收益分布 (按月)', fontsize=18, pad=20)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('交易笔数', fontsize=12)
    # X轴标签处理：每隔6个月显示一个标签，防止重叠
    n = len(m_temp)
    ticks = range(0, n, 6)  # 每半年一个刻度
    labels = [m_temp.index[i].strftime('%Y-%m') for i in ticks]
    plt.xticks(ticks, labels, rotation=90, fontsize=10)
    plt.legend(title='收益率区间', loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    monthly_chart_path = os.path.join(output_dir, f'month_returns_stacked.png')
    plt.savefig(monthly_chart_path, dpi=300, bbox_inches='tight')
    plt.close()