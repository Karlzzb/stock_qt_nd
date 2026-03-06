@echo off
chcp 65001 > nul
echo.
echo ========================================
echo 检查执行状态
echo ========================================
echo.

python check_status.py %*

echo.
pause
