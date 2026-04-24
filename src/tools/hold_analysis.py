import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import regex as re

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config.settings import RESULT_DIR

def hold_analyzer(version = "v5", param_suffix = "参数1"):
    # 1. 读取数据
    file_path = str(RESULT_DIR / f'simple_run_log_v12/simple_run_grid_{version}_trade_log_{param_suffix}.csv')
    df = pd.read_csv(file_path)
    output_dir = str(RESULT_DIR / f"portfolio_analysis_charts_{version}_{param_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 数据预处理
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

    # 计算持股天数
    df['holding_days'] = df['days_held']

    # 计算收益率用于分类
    df['roi'] = df['profit_pct']

    # 分类：盈利单 vs 亏损单
    df['result_type'] = df['roi'].apply(lambda x: '盈利 (Win)' if x > 0 else '亏损 (Loss)')

    # 3. 统计基础数据
    avg_days = df['holding_days'].mean()
    median_days = df['holding_days'].median()
    avg_win_days = df[df['roi'] > 0]['holding_days'].mean()
    avg_loss_days = df[df['roi'] <= 0]['holding_days'].mean()

    # --- 绘图 ---
    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # 图1: 持股天数分布直方图 (Histogram)
    ax1 = fig.add_subplot(grid[0, :]) # 占满第一行
    # 使用堆叠直方图，看不同天数下盈亏的分布
    sns.histplot(data=df, x='holding_days', hue='result_type', multiple='stack',
                 bins=50, palette={'盈利 (Win)': '#d62728', '亏损 (Loss)': '#2ca02c'}, ax=ax1)

    ax1.set_title(f'持股周期分布 (平均: {int(avg_days)}天, 中位数: {int(median_days)}天)', fontsize=16)
    ax1.set_xlabel('持股天数', fontsize=12)
    ax1.set_ylabel('交易笔数', fontsize=12)
    ax1.axvline(avg_days, color='blue', linestyle='--', label=f'平均持仓 ({int(avg_days)}天)')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # 图2: 盈亏单持仓时间对比 (Box Plot)
    ax2 = fig.add_subplot(grid[1, 0])
    sns.boxplot(data=df, x='result_type', y='holding_days', palette={'盈利 (Win)': '#ffcccc', '亏损 (Loss)': '#ccffcc'}, ax=ax2)
    ax2.set_title('盈利单 vs 亏损单 持仓时间对比', fontsize=14)
    ax2.set_xlabel('交易结果', fontsize=12)
    ax2.set_ylabel('持股天数', fontsize=12)
    # 标注平均值
    ax2.text(0, avg_win_days, f'平均: {int(avg_win_days)}天', ha='center', va='bottom', fontweight='bold', color='red')
    ax2.text(1, avg_loss_days, f'平均: {int(avg_loss_days)}天', ha='center', va='bottom', fontweight='bold', color='green')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # 图3: 收益率 vs 持仓时间 散点图 (Scatter Plot)
    ax3 = fig.add_subplot(grid[1, 1])
    # 区分颜色绘制
    sns.scatterplot(data=df, x='holding_days', y=df['roi']*100, hue='result_type',
                    palette={'盈利 (Win)': '#d62728', '亏损 (Loss)': '#2ca02c'}, alpha=0.6, ax=ax3)

    ax3.set_title('资金效率分析: 持仓时间 vs 收益率', fontsize=14)
    ax3.set_xlabel('持股天数', fontsize=12)
    ax3.set_ylabel('收益率 (%)', fontsize=12)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 添加象限说明
    xmax = df['holding_days'].max()
    ymax = df['roi'].max() * 100
    ymin = df['roi'].min() * 100

    ax3.text(xmax*0.8, ymax*0.9, '耐心资本\n(长持盈利)', color='red', ha='center', bbox=dict(facecolor='white', alpha=0.7))
    ax3.text(xmax*0.1, ymin*0.9, '快速止损\n(短持亏损)', color='green', ha='center', bbox=dict(facecolor='white', alpha=0.7))

    chart_path = os.path.join(output_dir, f'holding_days_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- 输出文本统计 ---
    print(f"=== 持股周期统计分析 ===")
    print(f"1. 总体平均持股: {avg_days:.2f} 天")
    print(f"2. 盈利单平均持股: {avg_win_days:.2f} 天")
    print(f"3. 亏损单平均持股: {avg_loss_days:.2f} 天")
    print(f"4. 时间盈亏比 (赢家时间/输家时间): {avg_win_days/avg_loss_days:.2f}")
    if avg_win_days > avg_loss_days:
        print("   ✅ 结论: 符合'截断亏损，让利润奔跑'原则。盈利单拿得比亏损单久。")
    else:
        print("   ⚠️ 结论: 亏损单持有时间过长，可能存在'死扛'现象，建议检查止损逻辑。")