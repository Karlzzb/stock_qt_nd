"""
配置文件示例
复制此文件为 config.py 并修改相关配置
"""

# 服务器配置
HOST = '0.0.0.0'  # 0.0.0.0 允许外部访问，127.0.0.1 仅本地访问
PORT = 5000       # 端口号
DEBUG = True      # 调试模式，生产环境设为 False

# 文件路径配置（相对于项目根目录）
CASH_FILE = "real_trading_data/investment_data/portfolio_cash.csv"
POSITION_FILE = "real_trading_data/investment_data/portfolio_positions.csv"
EXECUTOR_SCRIPT = "src/daily_trading_executor.py"
REPORTS_DIR = "real_trading_data/investment_reports"

# 日志配置
LOG_DIR = "web_interface/logs"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# 安全配置（可选）
# 取消注释以启用基础认证
# ENABLE_AUTH = True
# AUTH_USERNAME = "admin"
# AUTH_PASSWORD = "your-secure-password"

# IP白名单（可选）
# 取消注释以启用IP限制
# ENABLE_IP_WHITELIST = True
# ALLOWED_IPS = ['127.0.0.1', '192.168.1.100']

# 跨域配置
CORS_ORIGINS = "*"  # 允许所有来源，生产环境建议指定具体域名

# 执行超时时间（秒）
EXECUTION_TIMEOUT = 3600  # 1小时

# 状态轮询间隔（秒）
STATUS_POLL_INTERVAL = 2

# 日志轮询间隔（秒）
LOG_POLL_INTERVAL = 2
