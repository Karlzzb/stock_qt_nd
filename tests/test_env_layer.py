"""
环境层验收测试（issue #2）。

这组测试不依赖网络或真实 token，验证：
1. tinyshare_auth.get_pro_api() 在缺少 TINYSHARE_TOKEN 时抛出 EnvironmentError，
   且错误消息包含足够的指引信息。
2. requirements.txt 被标注为废弃（已由 pyproject.toml 取代）。
3. pyproject.toml 存在且包含必要的 uv 配置字段。
"""

import os
import sys
from pathlib import Path
import importlib

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"


# ---------------------------------------------------------------------------
# tinyshare_auth
# ---------------------------------------------------------------------------

class TestTinyshareAuth:
    def test_raises_when_token_missing(self, no_tinyshare_token, monkeypatch):
        """TINYSHARE_TOKEN 缺失时应抛出 EnvironmentError。"""
        # 确保 src 在路径中，直接 import
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        # 用 importlib 以便每次测试都重新执行模块级代码
        import tinyshare_auth
        importlib.reload(tinyshare_auth)
        with pytest.raises(EnvironmentError) as exc_info:
            tinyshare_auth.get_pro_api()
        msg = str(exc_info.value)
        assert "TINYSHARE_TOKEN" in msg, "错误消息应包含环境变量名"
        assert "export" in msg or ".env" in msg, "错误消息应包含设置指引"

    def test_raises_when_token_blank(self, monkeypatch):
        """TINYSHARE_TOKEN 设为空字符串时同样应抛出 EnvironmentError。"""
        monkeypatch.setenv("TINYSHARE_TOKEN", "   ")
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        import tinyshare_auth
        importlib.reload(tinyshare_auth)
        with pytest.raises(EnvironmentError):
            tinyshare_auth.get_pro_api()


# ---------------------------------------------------------------------------
# pyproject.toml 结构
# ---------------------------------------------------------------------------

class TestPyprojectToml:
    def test_pyproject_exists(self):
        assert (REPO_ROOT / "pyproject.toml").is_file(), "pyproject.toml 应存在于仓库根目录"

    def test_uv_lock_exists(self):
        assert (REPO_ROOT / "uv.lock").is_file(), "uv.lock 应存在（运行 uv lock 生成）"

    def test_python_version_file(self):
        pv = REPO_ROOT / ".python-version"
        assert pv.is_file(), ".python-version 文件应存在"
        content = pv.read_text().strip()
        assert content.startswith("3.11"), f".python-version 应钉 3.11，实际: {content}"

    def test_pyproject_requires_python(self):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # Python < 3.11 fallback
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        req = data.get("project", {}).get("requires-python", "")
        assert "3.11" in req, f"requires-python 应包含 3.11 约束，实际: {req}"

    def test_pyproject_uv_index_tinyshare(self):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        indexes = data.get("tool", {}).get("uv", {}).get("index", [])
        names = [idx.get("name") for idx in indexes]
        assert "tinyshare" in names, "pyproject.toml 应在 [[tool.uv.index]] 中声明 tinyshare 自定义源"
        ts_index = next(idx for idx in indexes if idx.get("name") == "tinyshare")
        assert ts_index.get("explicit") is True, "tinyshare index 应设置 explicit = true"

    def test_pyproject_uv_sources_tinyshare(self):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        assert "tinyshare" in sources, "pyproject.toml 应在 [tool.uv.sources] 中配置 tinyshare"


# ---------------------------------------------------------------------------
# requirements.txt 废弃标注
# ---------------------------------------------------------------------------

class TestUvLock:
    def test_tinyshare_resolved_from_custom_index(self):
        """
        uv.lock 中 tinyshare 应从自定义源解析，而不是 PyPI。
        验证 AC2：tinyshare 通过 tool.uv.index（explicit=true）+ tool.uv.sources 安装成功。
        """
        lock_path = REPO_ROOT / "uv.lock"
        assert lock_path.is_file(), "uv.lock 应存在"
        content = lock_path.read_text()
        assert "tinyshare" in content, "uv.lock 中应包含 tinyshare 包"
        assert "minidoc.pages.dev" in content, (
            "uv.lock 中 tinyshare 应来自 minidoc.pages.dev 自定义源，"
            "而不是 PyPI。请检查 [[tool.uv.index]] 和 [tool.uv.sources] 配置。"
        )


class TestRequirementsTxt:
    def test_requirements_txt_deprecated(self):
        req = REPO_ROOT / "requirements.txt"
        if not req.is_file():
            pytest.skip("requirements.txt 已被删除，满足要求")
        content = req.read_text()
        assert (
            "DEPRECATED" in content.upper() or "废弃" in content or "pyproject" in content.lower()
        ), (
            "requirements.txt 仍存在但未标注废弃。"
            "请在文件顶部添加废弃注释，或直接删除该文件。"
        )
