#!/bin/bash

echo "========================================"
echo "股票AI量化交易系统 - Web操作界面"
echo "========================================"
echo ""
echo "正在启动服务..."
echo ""

cd "$(dirname "$0")"
python app.py
