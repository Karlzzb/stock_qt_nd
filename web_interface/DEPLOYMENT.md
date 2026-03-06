# 部署指南

## 🖥️ 本地部署（推荐新手）

### Windows系统

1. **安装依赖**
   ```cmd
   cd web_interface
   pip install -r requirements.txt
   ```

2. **启动服务**
   - 双击 `start.bat` 文件
   - 或在命令行运行: `python app.py`

3. **访问界面**
   - 浏览器打开: http://localhost:5000

### Linux/Mac系统

1. **安装依赖**
   ```bash
   cd web_interface
   pip install -r requirements.txt
   ```

2. **启动服务**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
   或直接运行: `python app.py`

3. **访问界面**
   - 浏览器打开: http://localhost:5000

## 📱 局域网访问（手机访问）

### 1. 查看电脑IP地址

**Windows:**
```cmd
ipconfig
```
找到 "IPv4 地址"，例如: 192.168.1.100

**Linux/Mac:**
```bash
ifconfig
# 或
ip addr show
```

### 2. 启动服务

确保使用 `0.0.0.0` 作为host（默认已配置）

### 3. 手机访问

- 确保手机和电脑在同一WiFi
- 手机浏览器输入: `http://192.168.1.100:5000`
- 建议添加到主屏幕

### 4. 防火墙设置

**Windows防火墙:**
```cmd
# 允许5000端口
netsh advfirewall firewall add rule name="Trading Web" dir=in action=allow protocol=TCP localport=5000
```

**Linux防火墙 (ufw):**
```bash
sudo ufw allow 5000
```

## 🌐 公网访问方案

### 方案1: 使用 ngrok（最简单）

1. **下载 ngrok**
   - 访问: https://ngrok.com/
   - 注册并下载

2. **启动隧道**
   ```bash
   ngrok http 5000
   ```

3. **获取公网地址**
   - ngrok会显示一个公网URL，例如: https://abc123.ngrok.io
   - 使用此URL即可从任何地方访问

**优点**: 简单快速，无需配置
**缺点**: 免费版URL会变化，需要每次重启

### 方案2: 使用 frp（稳定）

1. **准备一台有公网IP的服务器**

2. **服务器端配置**
   ```ini
   # frps.ini
   [common]
   bind_port = 7000
   ```

3. **客户端配置**
   ```ini
   # frpc.ini
   [common]
   server_addr = 你的服务器IP
   server_port = 7000

   [trading_web]
   type = tcp
   local_ip = 127.0.0.1
   local_port = 5000
   remote_port = 5000
   ```

4. **启动服务**
   ```bash
   # 服务器
   ./frps -c frps.ini

   # 客户端
   ./frpc -c frpc.ini
   ```

### 方案3: 云服务器部署（生产环境）

#### 3.1 准备工作

1. 购买云服务器（阿里云、腾讯云等）
2. 安装Python 3.8+
3. 上传项目文件

#### 3.2 安装依赖

```bash
cd /path/to/project/web_interface
pip install -r requirements.txt
pip install gunicorn  # 生产环境服务器
```

#### 3.3 配置 Gunicorn

创建 `gunicorn_config.py`:
```python
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 120
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
```

#### 3.4 启动服务

```bash
# 前台运行（测试）
gunicorn -c gunicorn_config.py app:app

# 后台运行
nohup gunicorn -c gunicorn_config.py app:app > server.log 2>&1 &
```

#### 3.5 配置 Nginx（可选）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### 3.6 配置 systemd（开机自启）

创建 `/etc/systemd/system/trading-web.service`:
```ini
[Unit]
Description=Trading Web Interface
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project/web_interface
ExecStart=/usr/bin/gunicorn -c gunicorn_config.py app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-web
sudo systemctl start trading-web
sudo systemctl status trading-web
```

#### 3.7 配置HTTPS（推荐）

使用 Let's Encrypt 免费证书:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 🔒 安全加固

### 1. 添加基础认证

修改 `app.py`，添加:
```python
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

auth = HTTPBasicAuth()

users = {
    "admin": generate_password_hash("your-password")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# 在需要保护的路由添加装饰器
@app.route('/api/execute', methods=['POST'])
@auth.login_required
def execute_trading():
    # ...
```

### 2. 限制访问IP

```python
from flask import request, abort

ALLOWED_IPS = ['192.168.1.100', '127.0.0.1']

@app.before_request
def limit_remote_addr():
    if request.remote_addr not in ALLOWED_IPS:
        abort(403)
```

### 3. 使用环境变量

创建 `.env` 文件:
```
SECRET_KEY=your-secret-key
ADMIN_PASSWORD=your-password
```

在代码中使用:
```python
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')
```

## 📊 性能优化

### 1. 使用缓存

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/portfolio/cash')
@cache.cached(timeout=60)
def get_cash():
    # ...
```

### 2. 异步处理

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

@app.route('/api/execute', methods=['POST'])
def execute_trading():
    executor.submit(run_executor, log_file)
    # ...
```

## 🔍 监控和日志

### 1. 日志配置

```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('logs/app.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
```

### 2. 监控脚本

创建 `monitor.sh`:
```bash
#!/bin/bash
while true; do
    if ! pgrep -f "python app.py" > /dev/null; then
        echo "Service down, restarting..."
        cd /path/to/web_interface
        nohup python app.py > server.log 2>&1 &
    fi
    sleep 60
done
```

## 🐳 Docker部署（高级）

### 1. 创建 Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

### 2. 创建 docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ../real_trading_data:/app/real_trading_data
      - ./logs:/app/logs
    restart: always
```

### 3. 启动容器

```bash
docker-compose up -d
```

## 📱 移动端优化

### 1. PWA支持

在 `templates/index.html` 添加:
```html
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#667eea">
```

创建 `manifest.json`:
```json
{
  "name": "股票交易系统",
  "short_name": "交易",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#667eea",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

## 🆘 故障排除

### 问题1: 端口被占用

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux
lsof -i :5000
kill -9 <PID>
```

### 问题2: 权限问题

```bash
# Linux
chmod +x start.sh
chmod 755 web_interface/
```

### 问题3: 依赖安装失败

```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 📞 获取帮助

- 查看日志: `web_interface/logs/`
- 检查服务状态: `systemctl status trading-web`
- 测试端口: `curl http://localhost:5000`
