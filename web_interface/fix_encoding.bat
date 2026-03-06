@echo off
echo ========================================
echo 编码问题修复工具
echo ========================================
echo.

echo [1/4] 设置代码页为UTF-8...
chcp 65001 >nul 2>&1
echo 完成

echo.
echo [2/4] 设置环境变量...
setx PYTHONIOENCODING "utf-8" >nul 2>&1
setx PYTHONUTF8 "1" >nul 2>&1
echo 完成

echo.
echo [3/4] 清理Python缓存...
if exist __pycache__ (
    rd /s /q __pycache__ >nul 2>&1
)
if exist ..\__pycache__ (
    rd /s /q ..\__pycache__ >nul 2>&1
)
del /s /q *.pyc >nul 2>&1
echo 完成

echo.
echo [4/4] 验证设置...
echo PYTHONIOENCODING=%PYTHONIOENCODING%
echo PYTHONUTF8=%PYTHONUTF8%

echo.
echo ========================================
echo 修复完成！
echo ========================================
echo.
echo 请关闭此窗口，重新打开命令行
echo 然后使用 start.bat 启动服务
echo.
pause
