"""
tinyshare 认证助手。

所有使用 tinyshare（及 tushare 兼容接口）的模块都应通过此函数初始化，
而不是在各处硬编码 token。

token 读取方式（按优先级）：
1. 系统环境变量 TINYSHARE_TOKEN（已设置则直接使用）
2. 项目根目录 .env 文件中的 TINYSHARE_TOKEN

.env 文件示例（不要加引号）：
    TINYSHARE_TOKEN=<你的 token>
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import ttshare as ts

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_pro_api():
    """
    从 .env 或环境变量读取 TINYSHARE_TOKEN，初始化并返回 tinyshare pro API 实例。

    load_dotenv 在此处调用（而非模块顶层），以便测试通过 monkeypatch
    控制环境变量时行为可预期。override=False 保证系统环境变量优先。

    Raises:
        EnvironmentError: token 未在 .env 或环境变量中找到，或为空时抛出。
    """
    # override=False：已有环境变量时不覆盖；.env 不存在时静默跳过
    load_dotenv(dotenv_path=_ENV_PATH, override=False)

    token = (
        os.environ.get("TTSHARE_TOKEN", "").strip()
        or os.environ.get("TINYSHARE_TOKEN", "").strip()
    )
    if not token:
        raise EnvironmentError(
            "TINYSHARE_TOKEN 未找到。\n"
            f"请在项目根目录的 .env 文件（{_ENV_PATH}）中添加：\n"
            "    TINYSHARE_TOKEN=<你的 token>\n"
            "注意：不要加引号，不要有多余空格。"
        )
    ts.set_token(token)
    return ts.pro_api()
