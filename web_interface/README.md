# 股票AI量化交易系统 - Web操作界面

## 📋 功能说明

这是一个独立的Web操作界面，用于远程管理股票AI量化交易系统，支持手机端访问。

### 主要功能

1. **资金管理**
   - 查看当前可用资金
   - 在线编辑资金金额
   - 查看最后运行时间

2. **持仓管理**
   - 查看当前所有持仓
   - 可视化表单编辑持仓信息（替代JSON编辑）
   - 支持添加/删除/修改持仓
   - 自动验证股票代码格式和必填字段
   - 显示成本、止损、止盈等详细信息

3. **交易执行**
   - 一键远程执行 `daily_trading_executor.py`
   - 实时查看执行状态和进度
   - 查看执行日志

4. **报告查看**
   - 查看最近的交易报告
   - 在线浏览报告内容

## 🚀 快速开始

### 1. 安装依赖

```bash
cd web_interface
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app.py
```

服务将在 `http://0.0.0.0:5000` 启动

### 3. 访问界面

- **本地访问**: http://localhost:5000
- **局域网访问**: http://你的IP地址:5000 (例如: http://192.168.1.100:5000)
- **手机访问**: 确保手机和电脑在同一局域网，使用电脑IP地址访问

## 📱 手机端使用

1. 确保手机和运行服务的电脑在同一WiFi网络
2. 在电脑上运行 `ipconfig` (Windows) 或 `ifconfig` (Linux/Mac) 查看IP地址
3. 在手机浏览器输入: `http://电脑IP:5000`
4. 建议添加到手机主屏幕，方便快速访问

## 🌐 公网访问（可选）

### 方法1: 使用内网穿透工具

推荐使用 **ngrok** 或 **frp**:

```bash
# 使用 ngrok
ngrok http 5000
```

### 方法2: 云服务器部署

1. 将整个项目上传到云服务器
2. 配置防火墙开放5000端口
3. 使用 `nohup` 或 `systemd` 后台运行

```bash
nohup python app.py > server.log 2>&1 &
```

### 方法3: 使用 Gunicorn (生产环境)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🔒 安全建议

1. **添加身份验证**
   - 建议添加登录功能
   - 使用 Flask-Login 或 JWT

2. **使用HTTPS**
   - 配置SSL证书
   - 使用 Nginx 反向代理

3. **限制访问IP**
   - 在防火墙中限制访问来源
   - 使用白名单机制

4. **修改默认端口**
   - 在 `app.py` 中修改端口号
   - 避免使用常见端口

## 📂 文件结构

```
web_interface/
├── app.py              # Flask后端服务
├── requirements.txt    # Python依赖
├── README.md          # 说明文档
├── templates/         # HTML模板
│   └── index.html     # 主页面
└── logs/              # 执行日志目录（自动创建）
```

## 🔧 配置说明

在 `app.py` 中可以修改以下配置:

```python
# 端口号
app.run(host='0.0.0.0', port=5000, debug=True)

# 文件路径
CASH_FILE = "real_trading_data/investment_data/portfolio_cash.csv"
POSITION_FILE = "real_trading_data/investment_data/portfolio_positions.csv"
EXECUTOR_SCRIPT = "src/daily_trading_executor.py"
```

## 📝 使用流程

1. **启动服务**
   ```bash
   python app.py
   ```

2. **访问界面**
   - 打开浏览器访问 http://localhost:5000

3. **编辑资金和持仓**
   - 在"资金持仓"标签页点击"编辑"按钮
   - 修改资金金额或持仓信息
   - 点击"保存"

4. **执行交易**
   - 切换到"执行交易"标签页
   - 点击"执行每日交易"按钮
   - 实时查看执行日志

5. **查看报告**
   - 切换到"交易报告"标签页
   - 点击任意报告查看详情

## ⚠️ 注意事项

1. **不修改原项目代码**: 本界面完全独立，不会修改原有项目的任何代码
2. **数据安全**: 建议定期备份 `portfolio_cash.csv` 和 `portfolio_positions.csv`
3. **执行前检查**: 执行交易前务必确认资金和持仓信息正确
4. **网络稳定**: 执行交易时保持网络连接稳定
5. **日志查看**: 执行完成后及时查看日志，确认执行结果

## 🐛 故障排除

### 问题1: 无法访问界面

- 检查防火墙是否开放5000端口
- 确认服务是否正常启动
- 查看是否有端口冲突

### 问题2: 执行失败

- 检查 `daily_trading_executor.py` 路径是否正确
- 查看执行日志了解具体错误
- 确认Python环境和依赖是否完整

### 问题3: 文件保存失败

- 检查文件权限
- 确认文件路径存在
- 查看后端日志

## 📞 技术支持

如有问题，请查看:
- 执行日志: `web_interface/logs/`
- 后端日志: 终端输出
- 浏览器控制台: F12开发者工具

## 🎯 后续优化建议

1. 添加用户登录功能
2. 支持多用户管理
3. 添加数据可视化图表
4. 支持历史数据查询
5. 添加邮件/微信通知
6. 支持定时任务配置
