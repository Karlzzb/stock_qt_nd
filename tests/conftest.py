"""
pytest 全局配置与 fixtures。

后续测试可在此声明共享 fixtures，例如：
- 固定输入的价格数据 DataFrame（用于特征计算确定性回归测试，见 issue #7）
- tinyshare mock（避免测试调用真实 API）
"""
import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=False)
def no_tinyshare_token(monkeypatch, tmp_path):
    """
    清除 TINYSHARE_TOKEN，用于测试「缺失 token 时应报错」的场景。

    同时将 tinyshare_auth._ENV_PATH 重定向到一个不存在的文件，
    防止 load_dotenv 从真实 .env 文件恢复 token。
    """
    monkeypatch.delenv("TINYSHARE_TOKEN", raising=False)

    # patch tinyshare_auth._ENV_PATH，使 load_dotenv 找不到 .env
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import tinyshare_auth
    monkeypatch.setattr(tinyshare_auth, "_ENV_PATH", tmp_path / "nonexistent.env")
