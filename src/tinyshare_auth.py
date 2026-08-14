"""
tinyshare 认证助手。

所有使用 tinyshare（及 tushare 兼容接口）的模块都应通过此函数初始化，
而不是在各处硬编码 token。

环境变量设置方式：
    export TINYSHARE_TOKEN="<你的 token>"
或在项目根目录创建 .env 文件（.env 已写入 .gitignore）：
    TINYSHARE_TOKEN=<你的 token>
"""

import os

import tinyshare as ts


def get_pro_api():
    """
    从环境变量 TINYSHARE_TOKEN 读取 token，初始化并返回 tinyshare pro API 实例。

    Raises:
        EnvironmentError: 环境变量未设置或为空时抛出，并给出明确的设置指引。
    """
    token = os.environ.get("TINYSHARE_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "环境变量 TINYSHARE_TOKEN 未设置或为空。\n"
            "请在终端执行：\n"
            "    export TINYSHARE_TOKEN='<你的 token>'\n"
            "或在项目根目录的 .env 文件中添加：\n"
            "    TINYSHARE_TOKEN=<你的 token>"
        )
    ts.set_token(token)
    return ts.pro_api()
