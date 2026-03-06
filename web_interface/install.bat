@echo off
chcp 65001
echo ========================================
echo 股票AI量化交易系统 - Web界面安装
echo ========================================
echo.

echo [1/3] 检查Python版本...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)
echo.

echo [2/3] 安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo 错误: 依赖安装失败
    pause
    exit /b 1
)
echo.

echo [3/3] 检查系统环境...
python check_system.py
if errorlevel 1 (
    echo 警告: 部分检查未通过，请查看上述信息
    echo.
)

echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 启动服务: 双击 start.bat
echo 或运行: python app.py
echo.
pause
