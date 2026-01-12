import pandas as pd
import os
import glob
from comm_fun  import model_config, label_encoding, EPS
import matplotlib.pyplot as plt
import seaborn as sns  # 用于统计可视化
import numpy as np
# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from config.settings import  DAILY_FEATURE_DIR, DATASET_DIR

def print_nan_report(data, key_columns):
    """打印NaN详细报告"""
    print("=" * 50)
    print("NaN 检测报告")
    print("=" * 50)

    # 创建布尔掩码
    nan_mask = data[key_columns].isna()

    # 统计信息
    total_rows = len(data)
    nan_rows_count = nan_mask.any(axis=1).sum()

    print(f"数据总行数: {total_rows}")
    print(f"包含NaN的行数: {nan_rows_count}")
    print(f"将被删除的行数: {nan_rows_count}")
    print(f"保留的行数: {total_rows - nan_rows_count}")
    print()

    # 按列统计NaN
    nan_cols = []
    print("各列NaN数量统计:")
    for col in key_columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            nan_cols.append(col)
            loss_pct = nan_count / len(data) * 100
            if loss_pct > 15:
                print(f"  {col}: {nan_count}个NaN ({nan_count / len(data) * 100:.1f}%)")
    return nan_cols


def analyze_feature_correlation(X, selected_features):
    """
    计算并保存特征相关性
    :param X:
    :return:
    """
    # 用于计算各个列之间的皮尔逊相关系数
    corr_matrix = pd.DataFrame(X, columns=selected_features).corr()

    # 创建输出目录
    output_dir = DATASET_DIR / "feature_correlations"
    os.makedirs(output_dir, exist_ok=True)

    # ========== 改进部分：调整图形参数 ==========
    # 根据特征数量动态调整图形大小
    n_features = len(selected_features)
    fig_width = max(20, n_features * 0.8)  # 每个特征0.8英寸
    fig_height = max(15, n_features * 0.7)  # 每个特征0.7英寸

    # 创建更大的图形
    plt.figure(figsize=(fig_width, fig_height), dpi=300)  # 增加DPI提高分辨率

    # 设置字体大小
    font_scale = max(1.0, 14 / n_features)  # 根据特征数量调整字体缩放
    sns.set(font_scale=font_scale)

    # 绘制热力图 - 添加更多参数控制
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt=".2f",
        square=True,  # 保持单元格为正方形
        linewidths=0.5,  # 单元格间的线宽
        cbar_kws={"shrink": 0.8, "label": "相关系数"},  # 颜色条设置
        annot_kws={"size": 10 if n_features <= 30 else 8},  # 注释字体大小
        mask=None  # 如果用mask=mask则只显示下三角
    )

    # 调整布局
    plt.title('特征相关性热力图', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签
    plt.yticks(rotation=0)

    # 自动调整布局防止标签被截断
    plt.tight_layout()

    # 保存高清图片
    plt.savefig(
        output_dir / "feature_correlation.png",
        dpi=300,  # 高DPI
        bbox_inches='tight',  # 紧凑边界
        facecolor='white',  # 背景色
        edgecolor='none'
    )

    # 同时保存PDF格式（矢量图，无限清晰）
    plt.savefig(
        output_dir / "feature_correlation.pdf",
        format='pdf',
        bbox_inches='tight'
    )

    plt.close()
    print(f"特征相关性图已保存至: {str(output_dir)}/feature_correlation.png")
    print(f"矢量图已保存至: {str(output_dir)}/feature_correlation.pdf")

def data_clean(raw_data):
    # 1. 去重
    data = raw_data.copy()
    before_count = len(data)
    data = data.sort_values(['timestamp', 'symbol', 'confirmation_score']).drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
    after_count = len(data)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # print(f"numeric_cols: {[col for col in numeric_cols]}")
    print(f"已删除 {before_count - after_count} 重复行，剩余 {after_count} 行数据")
    key_columns = ['timestamp', model_config.LABEL_COL] + model_config.OPTIMIZED_FEATURE_COLS


    # 2.  缺失值 统计与处理
    # 用合理的值填充NaN（只填充features里的数值cols, 不填充label）
    nan_cols = print_nan_report(data, key_columns)
    # feature_numeric_cols = list(set(numeric_cols) & set(nan_cols) & set(model_config.OPTIMIZED_FEATURE_COLS))
    # daily_data[feature_numeric_cols] = daily_data[feature_numeric_cols].fillna(0)
    # daily_data[feature_numeric_cols] = daily_data[feature_numeric_cols].fillna(EPS)
    # print(f"有空的列{[col for col in key_columns if col not in nan_cols]}")

    # 关键列数据无效剔除， 只删除关键列为 NaN 的行
    before_count = len(data)
    data = data.dropna(subset=key_columns)
    after_count = len(data)
    print(f"已删除 {before_count - after_count} 个包含关键列 NaN 的行，剩余 {after_count} 行数据")
    return data

def prepare_real_daily_features(daily_data):
    # 1. 索引创建
    daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])
    daily_data = daily_data.sort_values('timestamp').reset_index(drop=True)

    # 2. 更精细的时间特征
    # print("  --> 创建精细时间特征...")
    daily_data['day_of_year'] = daily_data['timestamp'].dt.dayofyear
    daily_data['week_of_year'] = daily_data['timestamp'].dt.isocalendar().week.astype(int)
    daily_data['quarter'] = daily_data['timestamp'].dt.quarter
    daily_data['day_of_month'] = daily_data['timestamp'].dt.day
    daily_data['is_month_end'] = daily_data['timestamp'].dt.is_month_end.astype(int)
    daily_data['is_month_start'] = daily_data['timestamp'].dt.is_month_start.astype(int)
    daily_data['is_quarter_end'] = daily_data['timestamp'].dt.is_quarter_end.astype(int)
    daily_data['is_quarter_start'] = daily_data['timestamp'].dt.is_quarter_start.astype(int)

    # 3. 数据清洗
    daily_data = data_clean(daily_data)

    # 4. 标签列处理
    daily_data, _ = label_encoding(daily_data)

    # 5。分析特征关系
    # analyze_feature_correlation(daily_data[model_config.OPTIMIZED_FEATURE_COLS],model_config.OPTIMIZED_FEATURE_COLS)

    # 6. NOTE: 对ST和不能交易的进行过滤
    full_len = len(daily_data)
    st_df = pd.read_csv(DATASET_DIR / 'st_stocks_list.csv')
    data = daily_data[daily_data['symbol'].str.match(r'^[60]')]
    data = data[~data['symbol'].isin(st_df['ts_code'])]
    print(f"原始数据量: {full_len} 过滤后数据量: {len(data)} 过滤后%: {(len(data)/full_len)* 100:.2f}%")

    daily_data = daily_data.sort_values('timestamp').reset_index(drop=True)
    return daily_data

def main():
    # 1. 读取数据
    # 构建文件匹配模式
    pattern = os.path.join(str(DAILY_FEATURE_DIR / "realistic_features_*.csv"))
    # 查找所有匹配的文件
    file_list = glob.glob(pattern)
    all_data = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            df['divergence_amount']  = len(df)
            all_data.append(df)
            print(f"✅ 成功加载: {file_path} ({len(df)} 行)")
        except Exception as e:
            print(f"❌ 加载文件 {file_path} 时出错: {e}")
    data = pd.concat(all_data, ignore_index=True)

    data = prepare_real_daily_features(data)

    # 提取日期
    date_counts = data['timestamp'].value_counts().sort_index()

    # 计算每个日期的累计行数
    cumulative_counts = date_counts.cumsum()
    total_rows = cumulative_counts.iloc[-1]

    # 确定分割点
    train_target = int(total_rows * 0.6)
    val_target = int(total_rows * 0.8)

    # 找到最接近分割点的日期
    train_end_date = date_counts.index[
        (cumulative_counts - train_target).abs().argmin()
    ]
    val_end_date = date_counts.index[
        (cumulative_counts - val_target).abs().argmin()
    ]

    # 划分数据
    train_data = data[data['timestamp'] <= train_end_date]
    test_data = data[(data['timestamp'] > train_end_date) & (data['timestamp'] <= val_end_date)]
    val_data = data[data['timestamp'] > val_end_date]

    # 检查比例
    print(f"原始总行数: {total_rows}")
    print(f"训练集: {len(train_data)} ({len(train_data) / total_rows:.1%})")
    print(f"测试集: {len(test_data)} ({len(test_data) / total_rows:.1%})")
    print(f"验证集: {len(val_data)} ({len(val_data) / total_rows:.1%})")

    # 保存到不同文件
    os.makedirs(DATASET_DIR, exist_ok=True)
    train_data.to_csv(DATASET_DIR / 'train_set.csv', index=False)
    test_data.to_csv(DATASET_DIR / 'test_set.csv', index=False)
    val_data.to_csv(DATASET_DIR / 'validation_set.csv', index=False)

if __name__ == "__main__":
    main()


