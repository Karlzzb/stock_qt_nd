import tinyshare as ts
import pandas as pd
from datetime import datetime
from config.settings import STOCK_DATA_DIR, DATASET_DIR

# 1. 设置你的Tushare Token
# 替换 '你的Tushare Token' 为你在官网获取的真实字符串
token = "eSB3V16uqfii3t2QZa6RXqV4Xf5vVeaai5JXr2lv51mUVH0B0IXYx3tW56c04451"  # 去tushare.pro注册获取
ts.set_token(token)

# 2. 初始化Pro接口
pro = ts.pro_api()

# 3. 获取所有上市股票的基本信息
# 这里使用 `stock_basic` 接口，它包含了股票的上市状态和名称等信息。
stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_status,list_date')

# 4. 筛选出名称中包含 'ST' 或 '*ST' 的股票
# 在A股中，ST股票的名称中会明确包含“ST”或“*ST”字样。
st_stocks = stock_basic[stock_basic['name'].str.contains('ST')]
print(f"共找到 {len(st_stocks)} 只ST/*ST股票：")
print(st_stocks[['ts_code', 'symbol', 'name']].to_string(index=False))

# 5. 计算发行天数
print("正在计算发行天数...")
current_date = pd.Timestamp(datetime.now().date())
# 将上市日期转换为datetime格式
stock_basic['list_date'] = pd.to_datetime(stock_basic['list_date'], format='%Y%m%d')
# 计算发行天数（自然日）
stock_basic['list_days'] = (current_date - stock_basic['list_date']).dt.days

# 6. 筛选发行天数不足100天的股票
new_stocks = stock_basic[stock_basic['list_days'] < 100]
print(f"找到 {len(new_stocks)} 只发行天数不足100天的新股")

# 7. 保存到CSV文件
new_stocks.to_csv(DATASET_DIR / 'new_stocks_list.csv', index=False, encoding='utf-8-sig')
print(f"\n股票列表已保存到 'model_dataset/st_stocks_list.csv'")
st_stocks.to_csv(DATASET_DIR / 'st_stocks_list.csv', index=False, encoding='utf-8-sig')
print("\n股票列表已保存到 'model_dataset/st_stocks_list.csv'")