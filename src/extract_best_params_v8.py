import pandas as pd
import argparse


def extract_parameters_from_csv(input_file, sort_column, top_k, output_file=None, ascending=False):
    """
    从CSV文件提取参数并格式化为JSON格式

    参数:
    input_file: 输入CSV文件路径
    sort_column: 用于排序的列名
    top_k: 要提取的参数数量
    output_file: 输出文件路径（可选）
    ascending: 排序方向，False为降序（默认），True为升序
    """

    # 读取CSV文件
    df = pd.read_csv(input_file)  # 使用制表符分隔

    # 检查排序列是否存在
    if sort_column not in df.columns:
        print(f"错误：排序列 '{sort_column}' 不存在！")
        print(f"可用的列有: {', '.join(df.columns)}")
        return None

    # 按指定列排序
    df_sorted = df.sort_values(by=sort_column, ascending=ascending)

    # 取前top_k行
    top_params = df_sorted.head(top_k)

     # 生成格式化输出
    output_lines = ["{"]

    for i, (index, row) in enumerate(top_params.iterrows(), 1):
        # 提取参数
        base_ratio = row['base_ratio']
        target_profit = row['target_profit']
        hard_stop_loss = row['hard_stop_loss']
        max_hold_days = int(row['max_hold_days'])
        max_positions = int(row['max_positions'])
        min_probability = row['min_probability']

        if row['win_rate'] < 0.49 or row['total_trades'] < 90 or row['max_drawdown'] < -0.30:
            continue

        # 添加注释行
        comment = f' "参数{i}":   {{  # 回报率:{row["return_rate"]:.4f} | 最大回撤:{row["max_drawdown"]:.4f} | 胜率:{row["win_rate"]:.4f} | 夏普:{row["sharpe_ratio"]:.4f} | 交易数:{int(row["total_trades"])}'
        output_lines.append(comment)

        # 添加参数字典
        param_dict = f"""        
        'base_ratio': {base_ratio},
        'target_profit': {target_profit},
        'hard_stop_loss': {hard_stop_loss},
        'max_hold_days': {max_hold_days},
        'max_positions': {max_positions},
        'min_probability': {min_probability},
    }},"""
        output_lines.append(param_dict)

    # 移除最后一个逗号并添加闭合大括号
    if output_lines[-1].endswith(','):
        output_lines[-1] = output_lines[-1][:-1]
    output_lines.append("}")

    # 组合输出字符串
    output_str = '\n'.join(output_lines)

    # 输出结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"结果已保存到: {output_file}")
    else:
        print("格式化参数:")
        print(output_str)

    # 返回结果的字典形式以便进一步处理
    result_dict = {}
    for i, (index, row) in enumerate(top_params.iterrows(), 1):
        result_dict[f"参数{i}"] = {
            'base_ratio': row['base_ratio'],
            'target_profit': row['target_profit'],
            'hard_stop_loss': row['hard_stop_loss'],
            'max_hold_days': int(row['max_hold_days']),
            'max_positions': int(row['max_positions']),
            'min_probability': row['min_probability']
        }

    return output_str, result_dict


def main():
    parser = argparse.ArgumentParser(description='从CSV文件提取参数并格式化为JSON格式')
    parser.add_argument('input', help='输入CSV文件路径')
    parser.add_argument('--sort-by', required=True, help='用于排序的列名')
    parser.add_argument('--top-k', type=int, default=10, help='要提取的参数数量（默认：10）')
    parser.add_argument('--output', help='输出文件路径（可选）')
    parser.add_argument('--ascending', action='store_true', help='升序排序（默认是降序）')
    parser.add_argument('--show-columns', action='store_true', help='显示CSV文件的所有列名')

    args = parser.parse_args()

    #如果只需要显示列名
    if args.show_columns:
        df = pd.read_csv(args.input, sep='\t')
        print("CSV文件列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        return

    # 提取参数
    extract_parameters_from_csv(args.input, args.sort_by, args.top_k, args.output, args.ascending)

from config.settings import RESULT_DIR

if __name__ == "__main__":
    extract_parameters_from_csv(str(RESULT_DIR / "parameter_optimization_results_concurrent_v8.csv"),
                                'return_rate',50, str(RESULT_DIR / "best_params_v8.json")

         )