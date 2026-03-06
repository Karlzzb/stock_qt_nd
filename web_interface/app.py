"""
股票AI量化交易系统 - Web操作界面
Flask后端服务
"""
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# 配置路径
BASE_DIR = Path(__file__).parent.parent
CASH_FILE = BASE_DIR / "real_trading_data" / "investment_data" / "portfolio_cash.csv"
POSITION_FILE = BASE_DIR / "real_trading_data" / "investment_data" / "portfolio_positions.csv"
EXECUTOR_SCRIPT = BASE_DIR / "src" / "grid_trading_realworld_v8.py"
LOG_DIR = BASE_DIR / "web_interface" / "logs"
REPORTS_DIR = BASE_DIR / "real_trading_data" / "investment_reports"

# 确保日志目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量存储执行状态
execution_status = {
    "running": False,
    "start_time": None,
    "log_file": None,
    "progress": "",
    "process": None  # 存储进程对象，用于停止
}

# 线程锁，保护execution_status
status_lock = threading.Lock()


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/portfolio/cash', methods=['GET'])
def get_cash():
    """获取资金信息"""
    try:
        if not CASH_FILE.exists():
            return jsonify({"error": "资金文件不存在"}), 404
        
        with open(CASH_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            if data:
                return jsonify({"success": True, "data": data[0]})
            return jsonify({"error": "资金文件为空"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio/cash', methods=['POST'])
def update_cash():
    """更新资金信息"""
    try:
        data = request.json
        cash = data.get('cash')
        
        if cash is None:
            return jsonify({"error": "缺少cash参数"}), 400
        
        # 读取现有数据
        existing_data = {}
        if CASH_FILE.exists():
            with open(CASH_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    existing_data = rows[0]
        
        # 更新数据
        existing_data['cash'] = cash
        existing_data['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 写入文件
        with open(CASH_FILE, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['cash', 'last_run_date', 'update_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(existing_data)
        
        return jsonify({"success": True, "message": "资金更新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio/positions', methods=['GET'])
def get_positions():
    """获取持仓信息"""
    try:
        if not POSITION_FILE.exists():
            return jsonify({"success": True, "data": []})
        
        with open(POSITION_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio/positions', methods=['POST'])
def update_positions():
    """更新持仓信息"""
    try:
        positions = request.json.get('positions', [])
        
        if not positions:
            # 如果为空，创建空文件
            with open(POSITION_FILE, 'w', encoding='utf-8', newline='') as f:
                fieldnames = ['code', 'avg_cost', 'shares', 'entry_date', 
                            'stop_loss_price', 'take_profit_price', 'should_sell_date',
                            'actual_sell_price', 'actual_sell_date', 'is_sold']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        else:
            # 写入持仓数据
            with open(POSITION_FILE, 'w', encoding='utf-8', newline='') as f:
                fieldnames = list(positions[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(positions)
        
        return jsonify({"success": True, "message": "持仓更新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/execute', methods=['POST'])
def execute_trading():
    """执行交易脚本"""
    global execution_status
    
    with status_lock:
        if execution_status["running"]:
            return jsonify({"error": "脚本正在运行中"}), 400
        
        try:
            # 获取日期参数
            data = request.json or {}
            predict_date = data.get('predict_date')
            feature_date = data.get('feature_date')
            
            # 验证日期格式
            if not predict_date or not feature_date:
                return jsonify({"error": "缺少日期参数"}), 400
            
            try:
                # 验证日期格式 YYYY-MM-DD
                datetime.strptime(predict_date, '%Y-%m-%d')
                datetime.strptime(feature_date, '%Y-%m-%d')
            except ValueError:
                return jsonify({"error": "日期格式错误，应为 YYYY-MM-DD"}), 400
            
            # 创建日志文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = LOG_DIR / f"execution_{timestamp}.log"
            
            execution_status = {
                "running": True,
                "start_time": datetime.now().isoformat(),
                "log_file": str(log_file),
                "progress": "启动中...",
                "process": None,
                "predict_date": predict_date,
                "feature_date": feature_date
            }
            
            # 在后台线程中执行
            thread = threading.Thread(target=run_executor, args=(log_file, predict_date, feature_date))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "success": True, 
                "message": "脚本开始执行",
                "log_file": str(log_file.name),
                "predict_date": predict_date,
                "feature_date": feature_date
            })
        except Exception as e:
            execution_status["running"] = False
            return jsonify({"error": str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_execution():
    """停止执行脚本"""
    global execution_status
    
    with status_lock:
        if not execution_status["running"]:
            return jsonify({"error": "没有正在运行的脚本"}), 400
        
        try:
            # 设置停止标志
            execution_status["running"] = False
            execution_status["progress"] = "正在停止..."
            
            # 如果进程存在，尝试终止
            process = execution_status.get("process")
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # 在锁外部终止进程，避免死锁
    if process:
        try:
            process.terminate()
            # 等待进程结束
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 如果5秒后还没结束，强制杀死
                process.kill()
                process.wait()
        except Exception as e:
            pass
    
    return jsonify({
        "success": True,
        "message": "停止信号已发送"
    })


def run_executor(log_file, predict_date, feature_date):
    """在后台运行执行脚本"""
    global execution_status
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 开始执行 {datetime.now()} ===\n")
            f.write(f"预测日期 (PREDICT_DATE): {predict_date}\n")
            f.write(f"特征日期 (FEATURE_DATE): {feature_date}\n\n")
            
            # 执行Python脚本，传递日期参数
            # Windows下需要特殊处理编码
            import sys
            import locale
            
            # 获取系统默认编码
            system_encoding = locale.getpreferredencoding()
            
            # 构建命令，传递日期参数
            cmd = [
                'python', 
                str(EXECUTOR_SCRIPT),
                '--predict-date', predict_date,
                '--feature-date', feature_date
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                # 不使用text模式，手动处理编码
                text=False
            )
            
            # 保存进程对象，用于停止
            with status_lock:
                execution_status["process"] = process
            
            # 实时读取输出，处理编码问题
            for line in iter(process.stdout.readline, b''):
                # 检查是否被请求停止
                with status_lock:
                    should_stop = not execution_status["running"]
                
                if should_stop:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    f.write(f"\n\n=== 用户手动停止 {datetime.now()} ===\n")
                    with status_lock:
                        execution_status["progress"] = "已停止"
                        execution_status["process"] = None
                    return
                
                try:
                    # 尝试多种编码方式解码
                    try:
                        decoded_line = line.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            decoded_line = line.decode('gbk')
                        except UnicodeDecodeError:
                            try:
                                decoded_line = line.decode(system_encoding)
                            except:
                                # 如果都失败，使用errors='ignore'
                                decoded_line = line.decode('utf-8', errors='ignore')
                    
                    f.write(decoded_line)
                    f.flush()
                    
                    # 更新进度，移除特殊字符
                    progress_text = decoded_line.strip()
                    # 移除emoji等特殊字符
                    progress_text = ''.join(char for char in progress_text if ord(char) < 0x10000)
                    with status_lock:
                        execution_status["progress"] = progress_text
                    
                except Exception as e:
                    # 记录解码错误但继续执行
                    f.write(f"[编码错误: {str(e)}]\n")
                    f.flush()
            
            process.wait()
            
            f.write(f"\n\n=== 执行完成 {datetime.now()} ===\n")
            f.write(f"退出码: {process.returncode}\n")
            
            with status_lock:
                execution_status["running"] = False
                execution_status["progress"] = "执行完成" if process.returncode == 0 else "执行失败"
                execution_status["process"] = None
            
    except Exception as e:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n错误: {str(e)}\n")
        except:
            pass
        with status_lock:
            execution_status["running"] = False
            execution_status["progress"] = f"错误: {str(e)}"
            execution_status["process"] = None


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取执行状态"""
    with status_lock:
        # 返回状态的副本，避免并发问题
        return jsonify({
            "running": execution_status["running"],
            "start_time": execution_status["start_time"],
            "log_file": execution_status["log_file"],
            "progress": execution_status["progress"]
        })


@app.route('/api/logs/<filename>', methods=['GET'])
def get_log(filename):
    """获取日志内容"""
    try:
        log_file = LOG_DIR / filename
        if not log_file.exists():
            return jsonify({"error": "日志文件不存在"}), 404
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({"success": True, "content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/logs', methods=['GET'])
def list_logs():
    """列出所有日志文件"""
    try:
        logs = []
        if LOG_DIR.exists():
            for log_file in sorted(LOG_DIR.glob("*.log"), reverse=True):
                logs.append({
                    "name": log_file.name,
                    "size": log_file.stat().st_size,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports', methods=['GET'])
def list_reports():
    """列出所有交易报告"""
    try:
        reports = []
        if REPORTS_DIR.exists():
            for report_file in sorted(REPORTS_DIR.glob("*.txt"), reverse=True)[:10]:
                reports.append({
                    "name": report_file.name,
                    "date": report_file.stem.replace('report_', ''),
                    "size": report_file.stat().st_size,
                    "modified": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat()
                })
        return jsonify({"success": True, "reports": reports})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports/<filename>', methods=['GET'])
def get_report(filename):
    """获取报告内容"""
    try:
        report_file = REPORTS_DIR / filename
        if not report_file.exists():
            return jsonify({"error": "报告文件不存在"}), 404
        
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({"success": True, "content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 启动服务器
    # host='0.0.0.0' 允许外部访问
    # port=5000 端口号
    app.run(host='0.0.0.0', port=5000, debug=True)
