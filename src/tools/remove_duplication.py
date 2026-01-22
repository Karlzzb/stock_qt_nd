import pandas as pd
from config.settings import RESULT_DIR

def remove_duplicates_csv(input_file, output_file, subset=None, keep='first'):
    """
    去除CSV文件中的重复行

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    subset: 指定哪些列判断重复，默认全部列
    keep: 'first'保留第一个，'last'保留最后一个，False删除所有重复
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 打印去重前信息
    print(f"去重前行数: {len(df)}")

    # 去重
    df_deduped = df.drop_duplicates(subset=subset, keep=keep)

    # 打印去重后信息
    print(f"去重后行数: {len(df_deduped)}")
    print(f"删除了 {len(df) - len(df_deduped)} 行重复数据")

    # 保存到新文件
    df_deduped.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")

    return df_deduped

# 使用示例
# 1. 基于所有列去重
# remove_duplicates_csv('input.csv', 'output.csv')

# 2. 基于特定列去重（如'email'列）
remove_duplicates_csv(RESULT_DIR / 'parameter_optimization_results_concurrent_v8_15d.csv', RESULT_DIR / 'parameter_optimization_results_concurrent_v8_15d_new.csv', subset=['return_rate'])

# 3. 保留最后一个重复项
# remove_duplicates_csv('input.csv', 'output.csv', keep='last')