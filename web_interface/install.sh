#!/bin/bash

echo "========================================"
echo "股票AI量化交易系统 - Web界面安装"
echo "========================================"
echo ""

echo "[1/3] 检查Python版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python，请先安装Python 3.7+"
    exit 1
fi
echo ""

echo "[2/3] 安装依赖包..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi
echo ""

echo "[3/3] 检查系统环境..."
python3 check_system.py
if [ $? -ne 0 ]; then
    echo "警告: 部分检查未通过，请查看上述信息"
    echo ""
fi

echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "启动服务: ./start.sh"
echo "或运行: python3 app.py"
echo ""
