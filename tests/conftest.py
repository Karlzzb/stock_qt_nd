"""
pytest 全局配置与 fixtures。

后续测试可在此声明共享 fixtures，例如：
- 固定输入的价格数据 DataFrame（用于特征计算确定性回归测试，见 issue #7）
- tinyshare mock（避免测试调用真实 API）
"""
import os
import pytest


@pytest.fixture(autouse=False)
def no_tinyshare_token(monkeypatch):
    """
    清除 TINYSHARE_TOKEN，用于测试「缺失 token 时应报错」的场景。
    标注 autouse=False 使其仅在明确使用的测试中生效。
    """
    monkeypatch.delenv("TINYSHARE_TOKEN", raising=False)
