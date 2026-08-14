import pandas as pd
from datetime import datetime
from config.settings import DATASET_DIR
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tinyshare_auth import get_pro_api


def main():
    pro = get_pro_api()

    # 获取所有上市股票的基本信息
    stock_basic = pro.stock_basic(
        exchange='', list_status='L',
        fields='ts_code,symbol,name,area,industry,market,list_status,list_date'
    )

    # 筛选出名称中包含 'ST' 或 '*ST' 的股票
    st_stocks = stock_basic[stock_basic['name'].str.contains('ST')]
    logger.info(f”共找到 {len(st_stocks)} 只ST/*ST股票：”)
    logger.info(st_stocks[['ts_code', 'symbol', 'name']].to_string(index=False))

    # 计算发行天数
    logger.info(“正在计算发行天数...”)
    current_date = pd.Timestamp(datetime.now().date())
    stock_basic['list_date'] = pd.to_datetime(stock_basic['list_date'], format='%Y%m%d')
    stock_basic['list_days'] = (current_date - stock_basic['list_date']).dt.days

    # 筛选发行天数不足100天的股票
    new_stocks = stock_basic[stock_basic['list_days'] < 100]
    logger.info(f”找到 {len(new_stocks)} 只发行天数不足100天的新股”)

    # 保存到CSV文件
    new_stocks.to_csv(DATASET_DIR / 'new_stocks_list.csv', index=False, encoding='utf-8-sig')
    logger.info(f”\n股票列表已保存到 '{DATASET_DIR}/new_stocks_list.csv'”)
    st_stocks.to_csv(DATASET_DIR / 'st_stocks_list.csv', index=False, encoding='utf-8-sig')
    logger.info(f”\n股票列表已保存到 '{DATASET_DIR}/st_stocks_list.csv'”)


if __name__ == '__main__':
    main()