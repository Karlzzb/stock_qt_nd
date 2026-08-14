# settings.py
from pathlib import Path
import os


def get_base_path():
    """获取项目基础路径，支持本地和 Docker 环境。"""
    if os.getenv('DOCKER_ENV') == '1' or os.path.exists('/app'):
        return Path('/app')
    return Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = get_base_path()
BASE_DIR = PROJECT_ROOT  # 别名

# ---------------------------------------------------------------------------
# 目录配置（全部使用 Linux/POSIX 路径，支持环境变量覆盖）
# ---------------------------------------------------------------------------

def _resolve(env_var: str, default: Path) -> Path:
    """从环境变量读取路径，未设置则使用默认值。"""
    val = os.environ.get(env_var, "").strip()
    return Path(val) if val else default

# 数据根目录：Parquet 格式的行情数据、特征数据均存于此
DATA_ROOT = _resolve("STOCK_DATA_ROOT", PROJECT_ROOT.parent / "stock_data")

# 日线行情 Parquet 目录（每股一文件：{ts_code}.parquet）
DAILY_PARQUET_DIR  = _resolve("DAILY_PARQUET_DIR",  DATA_ROOT / "daily")

# 时点股票池元数据（universe_*.parquet）
UNIVERSE_DIR       = _resolve("UNIVERSE_DIR",       DATA_ROOT / "universe")

# 特征数据目录
DAILY_FEATURE_DIR  = _resolve("DAILY_FEATURE_DIR",  PROJECT_ROOT.parent / "real_feature_data_daily")

# 原 stock_nd.py 下载目录（兼容旧流程，后续逐步迁移到 DAILY_PARQUET_DIR）
STOCK_ND_DATA_DIR  = _resolve("STOCK_ND_DATA_DIR",  DATA_ROOT)
STOCK_ND_PKL_DIR   = STOCK_ND_DATA_DIR
STOCK_ND_CSV_DIR   = STOCK_ND_DATA_DIR / "csv"

# ST 过滤数据目录（原 Windows 路径已清除，改为项目内路径）
ST_FILTER_DATA_DIR = _resolve("ST_FILTER_DATA_DIR", DATA_ROOT / "st_filter")

# 其他目录
DATASET_DIR        = _resolve("DATASET_DIR",        PROJECT_ROOT.parent / "data")
LOG_DIR            = _resolve("LOG_DIR",             PROJECT_ROOT.parent / "logs")
CONFIG_DIR         = PROJECT_ROOT.parent / "config"
MODEL_DIR          = _resolve("MODEL_DIR",           PROJECT_ROOT.parent / "models")
RESULT_DIR         = _resolve("RESULT_DIR",          PROJECT_ROOT.parent / "output")
REAL_TRADING_DIR   = PROJECT_ROOT.parent / "real_trading_data"

# ---------------------------------------------------------------------------
# 启动时创建必要目录
# ---------------------------------------------------------------------------
_ENSURE_DIRS = [
    DAILY_PARQUET_DIR,
    UNIVERSE_DIR,
    DAILY_FEATURE_DIR,
    STOCK_ND_DATA_DIR,
    STOCK_ND_PKL_DIR,
    STOCK_ND_CSV_DIR,
    ST_FILTER_DATA_DIR,
    DATASET_DIR,
    LOG_DIR,
    CONFIG_DIR,
    MODEL_DIR,
    RESULT_DIR,
    REAL_TRADING_DIR,
]

for _dir in _ENSURE_DIRS:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 其他常量
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
