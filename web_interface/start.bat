@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ========================================
echo 股票AI量化交易系统 - Web操作界面
echo ========================================
echo.
echo 正在启动服务...
echo 访问地址: http://localhost:5000
echo.

cd /d %~dp0
python app.py

pause
