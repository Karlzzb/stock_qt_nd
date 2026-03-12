# settings.py 或 config.py
from pathlib import Path
import os


# 🆕 Docker支持：获取基础路径（支持本地和Docker环境）
def get_base_path():
    """获取项目基础路径，支持本地和Docker

    r环境"""
    # 检查是否在Docker环境中（通过环境变量或路径判断）
    if os.getenv('DOCKER_ENV') == '1' or os.path.exists('/app'):
        base_path = Path('/app')
    else:
        # 本地环境：使用脚本所在目录
        base_path = Path(__file__).resolve().parent
    return base_path


# 全局常量
PROJECT_ROOT = get_base_path()
BASE_DIR = PROJECT_ROOT  # 别名，和Django保持相似

# 关键路径
DATASET_DIR = PROJECT_ROOT.parent / 'data'
LOG_DIR = PROJECT_ROOT.parent / 'logs'
CONFIG_DIR = PROJECT_ROOT.parent / 'config'
MODEL_DIR = PROJECT_ROOT.parent / 'models'
RESULT_DIR = PROJECT_ROOT.parent / 'output'
REAL_TRADING_DIR = PROJECT_ROOT.parent / 'real_trading_data'
REAL_TRADING_DIR_SIMULATION = PROJECT_ROOT.parent / 'real_trading_data_simulation' #tmp_simulation_realworld_v8.py用来模拟
STOCK_DATA_PLK_DIR =  get_base_path().parent / 'stock_data'
STOCK_DATA_DIR =  get_base_path().parent / 'stock_data/csv'
DAILY_FEATURE_DIR = get_base_path().parent  / 'real_feature_data_daily'

# 🆕 股票数据下载存储路径配置（可配置为外部绝对路径）
# stock_nd.py 下载的股票数据存储目录
STOCK_ND_DATA_DIR = Path(r'E:\stock_data\stock_nd')  # 修改为你想要的绝对路径
STOCK_ND_PKL_DIR = STOCK_ND_DATA_DIR / 'pkl'
STOCK_ND_CSV_DIR = STOCK_ND_DATA_DIR / 'csv'

# st_stock_filter.py 下载的过滤数据存储目录
ST_FILTER_DATA_DIR = Path(r'E:\stock_data\st_filter')  # 修改为你想要的绝对路径

# 检查并创建必要目录
for dir_path in [DATASET_DIR, LOG_DIR, CONFIG_DIR, MODEL_DIR, RESULT_DIR, REAL_TRADING_DIR, STOCK_DATA_DIR, DAILY_FEATURE_DIR,
                 STOCK_ND_DATA_DIR, STOCK_ND_PKL_DIR, STOCK_ND_CSV_DIR, ST_FILTER_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42