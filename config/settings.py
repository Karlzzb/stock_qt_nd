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
STOCK_DATA_DIR =  get_base_path().parent / 'stock_data/csv'
DAILY_FEATURE_DIR = get_base_path().parent  / 'real_feature_data_daily'

# 检查并创建必要目录
for dir_path in [DATASET_DIR, LOG_DIR, CONFIG_DIR, MODEL_DIR, RESULT_DIR, REAL_TRADING_DIR, STOCK_DATA_DIR, DAILY_FEATURE_DIR]:
    dir_path.mkdir(exist_ok=True)

RANDOM_STATE = 42