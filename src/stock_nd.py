from datetime import datetime, timedelta
import pandas as pd
import tinyshare as ts
import os
import pickle
import time
from config.settings import STOCK_ND_CSV_DIR, STOCK_ND_PKL_DIR
# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_tushare(token):
    """初始化tushare"""
    ts.set_token(token)
    return ts.pro_api()

def get_stock_data_tushare(pro, symbol, start_date='20100101', end_date=None):
    """
    使用tushare获取股票数据
    需要注册获取token: https://tushare.pro/
    """
    try:
        # 获取日线数据
        if end_date is None:
            df = pro.daily(ts_code=symbol, start_date=start_date)
        else:
            df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').set_index('trade_date')
        return df[['open', 'high', 'low', 'close', 'vol']].rename(columns={'vol': 'volume'})
    except Exception as e:
        logger.info(f"Tushare获取{symbol}失败: {e}")
        return None

def get_all_stocks_tushare(pro):
    """
    获取所有A股股票列表
    """
    try:
        # 获取基础信息数据
        stock_basic = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        logger.info(f"获取到 {len(stock_basic)} 只A股股票")
        return stock_basic
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return None


def fetch_stock_price_data(promision, symbol, start_date=None, end_date=None):
    """
    获取股票价格数据，优先从缓存加载

    Args:
        symbol: 股票代码
        start_date: 开始日期
        fetch_only: 是否强制从API更新

    Returns:
        price_data: 价格数据 DataFrame 或 None（如果获取失败）
    """
    price_data = None

    # 强制更新，从API获取
    logger.info(f"从API获取数据: {symbol}")
    retry = 0
    try:
        while retry < 800:
            # price_data = get_stock_data_akshare(symbol, start_date=start_date)
            price_data = get_stock_data_tushare(promision, symbol, start_date=start_date, end_date=end_date)

            # 保存到缓存
            if price_data is not None and len(price_data) > 0:
                save_price_data(price_data, symbol)
                save_price_data_csv(price_data, symbol)
                return price_data
                # 【关键】在每次请求后等待一段时间
                # 使用随机间隔，模拟人类行为，避免规律性的请求
            retry += 1
            sleep_time = min(retry*15,120)
            # sleep_time = random.uniform(retry, 10*retry)  # 随机等待2到4秒
            logger.info(f"retry:第{retry}次, 等待 {sleep_time:.2f} 秒...")
            time.sleep(sleep_time)
    except Exception as e:
        logger.error(f"获取 {symbol} 数据时出错: {e}")

    return price_data


def fetch_index_price_data(promision, symbol, start_date=None, end_date=None):
    """
    获取股票价格数据，优先从缓存加载

    Args:
        symbol: 股票代码
        start_date: 开始日期
        fetch_only: 是否强制从API更新

    Returns:
        price_data: 价格数据 DataFrame 或 None（如果获取失败）
    """
    price_data = None

    # 强制更新，从API获取
    logger.info(f"从API获取数据: {symbol}")
    retry = 0
    try:
        while retry < 4:
            """
            使用tushare获取股票数据
            需要注册获取token: https://tushare.pro/
            """
            # 获取日线数据
            if end_date is None:
                df = promision.index_daily(ts_code=symbol, start_date=start_date)
            else:
                df = promision.index_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').set_index('trade_date')
            price_data = df[['open', 'high', 'low', 'close', 'vol']].rename(columns={'vol': 'volume'})

            # 保存到缓存
            if price_data is not None and len(price_data) > 0:
                save_price_data(price_data, symbol)
                save_price_data_csv(price_data, symbol)
                return price_data
                # 【关键】在每次请求后等待一段时间
                # 使用随机间隔，模拟人类行为，避免规律性的请求
            retry += 1
            sleep_time = retry*15
            # sleep_time = random.uniform(retry, 10*retry)  # 随机等待2到4秒
            logger.warning(f"retry:第{retry}次, 等待 {sleep_time:.2f} 秒...")
            time.sleep(sleep_time)
    except Exception as e:
        logger.error(f"获取 {symbol} 数据时出错: {e}")

    return price_data

def save_price_data(price_data, symbol, data_dir=str(STOCK_DATA_PLK_DIR)):
    """
    保存股票价格数据到文件
    """
    os.makedirs(data_dir, exist_ok=True)

    filename = f"{data_dir}/{symbol}_price_data.pkl"

    # 保存数据
    with open(filename, 'wb') as f:
        pickle.dump(price_data, f)

    # logger.info(f"价格数据已保存: {filename}")
    return filename


def save_price_data_csv(price_data, symbol, data_dir=str(STOCK_DATA_DIR)):
    """
    保存为CSV文件（可读性更好）
    """
    os.makedirs(data_dir, exist_ok=True)

    filename = f"{data_dir}/{symbol}_price_data.csv"
    price_data.to_csv(filename)

    logger.info(f"CSV数据已保存: {filename}")
    return filename

def fetch_symbols(promision, sample_ratio=1.0, cache_dir='../cache', cache_hours=240):
    """
    带本地文件缓存的股票代码获取函数

    Parameters:
    - sample_ratio: 采样比例
    - cache_dir: 缓存目录
    - cache_hours: 缓存有效期（小时）
    """
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)

    # 生成缓存文件名（基于采样比例）
    cache_file = os.path.join(cache_dir, f'symbols_cache_{sample_ratio}.pkl')

    # 检查缓存是否存在且未过期
    if os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(hours=cache_hours):
            logger.info(f"从缓存加载股票数据，采样率: {sample_ratio}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['sampled_stocks']['ts_code'].tolist()

    # 如果没有缓存或已过期，重新获取数据
    logger.info("正在获取股票数据...")
    all_stocks = get_all_stocks_tushare(promision)

    # 随机采样一部分股票
    if sample_ratio < 1.0:
        sample_size = int(len(all_stocks) * sample_ratio)
        sampled_stocks = all_stocks.sample(n=sample_size, random_state=42)
        logger.info(f"随机采样 {sample_size} 只股票进行处理")
    else:
        sampled_stocks = all_stocks
        logger.info(f"处理全部 {len(sampled_stocks)} 只股票")

    # 保存到缓存文件
    cache_data = {
        'sampled_stocks': sampled_stocks,
        'timestamp': datetime.now(),
        'sample_ratio': sample_ratio
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    logger.info(f"股票数据已缓存到: {cache_file}")

    return sampled_stocks['ts_code'].tolist()

# 使用示例
# data = get_stock_data_akshare('000001', start_date='20200101')
# logger.info(f"{data}")
def main():
    from tqdm import tqdm
    index_symbols = ["000001.SH","399001.SZ"]
    token = "3Q4RY56w8deQac5uQkcba5wzoaUf8XBdiLvBti22gv5jTstJ4d0ywZKU247ade48"
    pro = init_tushare(token)
    start_date = "20090101"
    sample_ratio=1.0
    symbols = fetch_symbols(pro,sample_ratio)
    success_count = 0
    fail_count = 0

    # 指数信息获取
    for index_symbol in index_symbols:
        fetch_index_price_data(pro, index_symbol, start_date)

    pbar = tqdm(symbols, desc="下载股票数据")
    for symbol in pbar:
        pbar.set_postfix(成功=success_count, 失败=fail_count, 当前股票=symbol)
        try:
            price_data = fetch_stock_price_data(pro, symbol, start_date)
            if price_data is not None:
                success_count += 1
            else:
                fail_count += 1
                logger.info(f"\n下载 {symbol} 失败")
        except Exception as e:
            fail_count += 1
            logger.info(f"\n下载 {symbol} 失败: {e}")

    logger.info(f"\n完成！成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    main()