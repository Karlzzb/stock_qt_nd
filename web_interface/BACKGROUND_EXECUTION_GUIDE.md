# 后台执行与状态查询指南

## ✅ 关闭浏览器后继续执行

### 好消息：你的系统已经支持！

当你在Web界面点击"执行每日交易"后，**即使关闭浏览器，脚本也会继续在后台运行**。

## 工作原理

### 1. 后台线程执行

```python
# app.py 中的实现
thread = threading.Thread(target=run_executor, args=(log_file,))
thread.daemon = True  # 守护线程
thread.start()
```

- 脚本在**独立的后台线程**中运行
- 不依赖浏览器连接
- Flask服务器持续运行，脚本独立执行

### 2. 执行流程

```
用户点击执行
    ↓
Flask创建后台线程
    ↓
立即返回响应给浏览器
    ↓
后台线程继续运行
    ↓
【用户可以关闭浏览器】
    ↓
脚本继续执行
    ↓
完成后更新状态
```

## 如何查看执行状态

### 方法1: 重新打开Web界面（推荐）

1. **打开浏览器**，访问 `http://服务器IP:5000`
2. **切换到"执行交易"标签页**
3. **查看执行状态**：
   - 如果显示"运行中" → 脚本正在执行
   - 如果显示"就绪" → 脚本已完成或未运行
4. **查看实时进度**：
   - 状态栏下方显示当前执行步骤
   - 日志区域显示详细输出

### 方法2: 查看日志文件

#### Windows:
```cmd
# 进入日志目录
cd E:\train_models\stock_qt_nd\web_interface\logs

# 查看最新日志
type execution_*.log | more

# 或者用记事本打开
notepad execution_20260306_120224.log
```

#### Linux/Mac:
```bash
# 进入日志目录
cd /path/to/stock_qt_nd/web_interface/logs

# 查看最新日志
tail -f execution_*.log

# 查看最后100行
tail -n 100 execution_20260306_120224.log
```

### 方法3: 检查进程状态

#### Windows:
```cmd
# 查看Python进程
tasklist | findstr python

# 详细信息
wmic process where "name='python.exe'" get commandline,processid
```

#### Linux/Mac:
```bash
# 查看Python进程
ps aux | grep python

# 查看daily_trading_executor进程
ps aux | grep daily_trading_executor
```

### 方法4: 查看执行报告

执行完成后会生成报告：
```
real_trading_data/investment_reports/report_YYYYMMDD.txt
```

## 状态说明

### 执行状态

| 状态 | 说明 | 可以关闭浏览器 |
|------|------|----------------|
| 就绪 | 未运行或已完成 | ✅ 是 |
| 启动中... | 正在初始化 | ✅ 是 |
| 运行中 | 正在执行 | ✅ 是 |
| 正在停止... | 用户请求停止 | ✅ 是 |
| 已停止 | 已被停止 | ✅ 是 |
| 执行完成 | 成功完成 | ✅ 是 |
| 执行失败 | 出现错误 | ✅ 是 |

### 进度信息

实时显示当前执行步骤：
- "步骤 1/3: 执行 ST股票过滤"
- "步骤 2/3: 执行 特征工程"
- "步骤 3/3: 执行 预测与交易"
- "执行完成"

## 典型使用场景

### 场景1: 移动办公

```
1. 在办公室启动执行
2. 关闭电脑，下班回家
3. 到家后用手机/平板访问Web界面
4. 查看执行状态和结果
```

### 场景2: 长时间执行

```
1. 启动执行（可能需要30分钟）
2. 关闭浏览器，去做其他事情
3. 30分钟后重新打开浏览器
4. 查看执行结果和交易报告
```

### 场景3: 多设备监控

```
1. 在电脑A上启动执行
2. 在电脑B上打开Web界面查看状态
3. 在手机上也可以查看状态
4. 所有设备看到的状态是同步的
```

## 状态持久化

### 内存中的状态

```python
execution_status = {
    "running": True,           # 是否正在运行
    "start_time": "2026-03-06T12:00:00",  # 开始时间
    "log_file": "logs/execution_20260306_120000.log",  # 日志文件
    "progress": "步骤 2/3: 执行 特征工程",  # 当前进度
    "process": <Process object>  # 进程对象
}
```

### 状态查询API

```javascript
// 前端每2秒查询一次
GET /api/status

// 返回
{
    "running": true,
    "start_time": "2026-03-06T12:00:00",
    "log_file": "logs/execution_20260306_120000.log",
    "progress": "步骤 2/3: 执行 特征工程"
}
```

## 注意事项

### ⚠️ 重要提醒

1. **不要关闭Flask服务器**
   - 关闭浏览器 ✅ 可以
   - 关闭Flask服务器 ❌ 不可以
   - 关闭命令行窗口 ❌ 不可以（会关闭Flask）

2. **服务器必须持续运行**
   ```
   正确：浏览器关闭，Flask继续运行
   错误：关闭运行Flask的命令行窗口
   ```

3. **重启服务器会丢失状态**
   - 如果重启Flask服务器
   - 内存中的执行状态会丢失
   - 但日志文件会保留
   - 可以通过日志文件查看执行情况

### 💡 最佳实践

1. **使用后台运行方式启动Flask**

   Windows (使用start.bat):
   ```cmd
   # start.bat 会在新窗口中运行
   start.bat
   
   # 或者最小化窗口运行
   start /min python app.py
   ```

   Linux/Mac (使用nohup):
   ```bash
   # 后台运行，不受终端关闭影响
   nohup python app.py > flask.log 2>&1 &
   
   # 或使用screen
   screen -S flask
   python app.py
   # Ctrl+A, D 分离会话
   ```

2. **设置开机自启动**
   
   让Flask服务器开机自动启动，这样永远不用担心服务器关闭。

3. **定期检查日志**
   
   即使不打开Web界面，也可以通过日志文件了解执行情况。

## 故障排除

### 问题1: 重新打开浏览器后看不到执行状态

**可能原因**:
- Flask服务器被关闭了
- 执行已经完成

**解决方法**:
1. 检查Flask服务器是否运行
   ```cmd
   # Windows
   tasklist | findstr python
   
   # Linux/Mac
   ps aux | grep "app.py"
   ```

2. 查看最新的日志文件
   ```cmd
   cd web_interface\logs
   dir /od
   type execution_*.log
   ```

### 问题2: 状态显示"就绪"但脚本还在运行

**可能原因**:
- Flask服务器重启过，内存状态丢失
- 但脚本进程还在运行

**解决方法**:
1. 检查Python进程
   ```cmd
   tasklist | findstr python
   ```

2. 查看日志文件最后几行
   ```cmd
   # 如果日志还在更新，说明脚本还在运行
   ```

3. 等待脚本完成或手动终止进程

### 问题3: 想要真正的后台服务

**需求**: 即使关闭Flask服务器，脚本也继续运行

**解决方案**: 使用系统服务或任务调度

Windows任务计划程序:
```cmd
# 创建定时任务
schtasks /create /tn "StockTrading" /tr "python E:\train_models\stock_qt_nd\src\daily_trading_executor.py" /sc daily /st 09:00
```

Linux crontab:
```bash
# 每天9点执行
0 9 * * * cd /path/to/stock_qt_nd && python src/daily_trading_executor.py
```

## 监控建议

### 1. 实时监控（执行期间）

- 保持一个浏览器标签页打开
- 每隔几分钟刷新查看进度
- 观察日志输出是否正常

### 2. 定期检查（执行后）

- 查看交易报告
- 检查持仓变化
- 验证资金变动

### 3. 异常告警（可选）

可以添加邮件或微信通知：
```python
# 在run_executor函数中添加
if process.returncode != 0:
    send_notification("执行失败")
else:
    send_notification("执行成功")
```

## 总结

✅ **你的系统已经支持关闭浏览器后继续执行**

关键点：
1. 脚本在Flask服务器的后台线程中运行
2. 关闭浏览器不影响脚本执行
3. 重新打开浏览器可以查看状态
4. 日志文件永久保存执行记录
5. 不要关闭Flask服务器（命令行窗口）

查看状态的方法：
1. 重新打开Web界面（最简单）
2. 查看日志文件
3. 检查进程状态
4. 查看执行报告

## 相关文档

- [Web界面使用指南](WEB_INTERFACE_GUIDE.md)
- [停止按钮使用指南](STOP_BUTTON_GUIDE.md)
- [部署指南](DEPLOYMENT.md)
- [快速开始](QUICKSTART.md)
