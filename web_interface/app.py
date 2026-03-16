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
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

app = Flask(__name__)
CORS(app)

# 配置路径
BASE_DIR = Path(__file__).parent.parent
CASH_FILE = BASE_DIR / "real_trading_data" / "investment_data" / "portfolio_cash.csv"
POSITION_FILE = BASE_DIR / "real_trading_data" / "investment_data" / "portfolio_positions.csv"
ASSETS_FILE = BASE_DIR / "real_trading_data" / "investment_data" / "portfolio_assets.csv"
EXECUTOR_SCRIPT = BASE_DIR / "src" / "grid_trading_realworld_v8.py"
LOG_DIR = BASE_DIR / "web_interface" / "logs"
REPORTS_DIR = BASE_DIR / "real_trading_data" / "investment_reports"
REAL_TRADING_DIR = BASE_DIR / "real_trading_data"

# 辅助脚本路径
ST_FILTER_SCRIPT = BASE_DIR / "src" / "st_stock_filter.py"
STOCK_ND_SCRIPT = BASE_DIR / "src" / "stock_nd.py"

# 确保日志目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量存储执行状态
execution_status = {
    "running": False,
    "st_filter_running": False,
    "stock_nd_running": False,
    "start_time": None,
    "log_file": None,
    "progress": "",
    "process": None,
    "st_filter_process": None,
    "st_filter_log": None,
    "st_filter_progress": "",
    "stock_nd_process": None,
    "stock_nd_log": None,
    "stock_nd_progress": ""
}

# 线程锁，保护execution_status
status_lock = threading.Lock()


# ========== 资产更新函数 ==========
def update_portfolio_assets(cash):
    """更新资产数据"""
    try:
        predict_date = datetime.now().strftime('%Y-%m-%d')

        # 计算持仓市值
        positions_value = 0
        if POSITION_FILE.exists():
            positions_df = pd.read_csv(POSITION_FILE)
            active = positions_df[positions_df['is_sold'] == 'NO']
            if not active.empty:
                positions_value = (active['shares'] * active['avg_cost']).sum()

        total = cash + positions_value

        # 更新或创建资产文件
        if ASSETS_FILE.exists():
            assets_df = pd.read_csv(ASSETS_FILE)
            new_record = pd.DataFrame([{
                'predict_date': predict_date,
                'total': total,
                'cash': cash,
                'positions_value': positions_value
            }])
            assets_df = pd.concat([assets_df, new_record], ignore_index=True)
        else:
            assets_df = pd.DataFrame([{
                'predict_date': predict_date,
                'total': total,
                'cash': cash,
                'positions_value': positions_value
            }])

        assets_df.to_csv(ASSETS_FILE, index=False)
    except Exception as e:
        logging.warning(f"更新资产文件失败: {e}")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


# ========== 原有 API ==========
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

        existing_data = {}
        if CASH_FILE.exists():
            with open(CASH_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    existing_data = rows[0]

        existing_data['cash'] = cash
        existing_data['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
            # 只返回未卖出的持仓
            data = [p for p in data if p.get('is_sold', 'NO') != 'YES']
            return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio/positions', methods=['POST'])
def update_positions():
    """更新持仓信息"""
    try:
        positions = request.json.get('positions', [])

        if not positions:
            with open(POSITION_FILE, 'w', encoding='utf-8', newline='') as f:
                fieldnames = ['code', 'avg_cost', 'shares', 'entry_date',
                            'stop_loss_price', 'take_profit_price', 'should_sell_date',
                            'actual_sell_price', 'actual_sell_date', 'is_sold']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        else:
            with open(POSITION_FILE, 'w', encoding='utf-8', newline='') as f:
                fieldnames = list(positions[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(positions)

        return jsonify({"success": True, "message": "持仓更新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== 手动交易 API ==========
@app.route('/api/trading/sellable-stocks', methods=['GET'])
def get_sellable_stocks():
    """获取可卖出的股票列表"""
    try:
        if not POSITION_FILE.exists():
            return jsonify({"success": True, "data": []})

        positions_df = pd.read_csv(POSITION_FILE)
        active_positions = positions_df[positions_df['is_sold'] == 'NO']

        result = []
        for _, row in active_positions.iterrows():
            result.append({
                "code": row['code'],
                "shares": int(row['shares']),
                "avg_cost": float(row['avg_cost']),
                "entry_date": row.get('entry_date', '')
            })

        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/trading/buyable-stocks', methods=['GET'])
def get_buyable_stocks():
    """获取可买入的股票列表（过去N天的建议）"""
    try:
        days = int(request.args.get('days', 3))
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(days)]

        suggestions = []
        for date in dates:
            suggestion_file = REAL_TRADING_DIR / 'investment_data' / f'trade_suggestions_{date}.csv'
            if suggestion_file.exists():
                df = pd.read_csv(suggestion_file)
                for _, row in df.iterrows():
                    suggestions.append({
                        "code": row.get('code', ''),
                        "score": float(row.get('score', 0)),
                        "predict_date": f"{date[:4]}-{date[4:6]}-{date[6:]}"
                    })

        seen = {}
        for s in suggestions:
            if s['code'] not in seen:
                seen[s['code']] = s

        return jsonify({"success": True, "data": list(seen.values())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/trading/sell', methods=['POST'])
def manual_sell():
    """手动卖出"""
    try:
        data = request.json
        stock_code = data.get('stock_code')
        quantity = int(data.get('quantity', 0))
        price = float(data.get('price', 0))

        print(f"[SELL] 收到请求: stock={stock_code}, qty={quantity}, price={price}")
        print(f"[SELL] CASH_FILE路径: {CASH_FILE}")
        print(f"[SELL] CASH_FILE存在: {CASH_FILE.exists()}")

        if not stock_code or quantity <= 0 or price <= 0:
            return jsonify({"error": "参数错误"}), 400

        if not POSITION_FILE.exists():
            return jsonify({"error": "持仓文件不存在"}), 404

        positions_df = pd.read_csv(POSITION_FILE)
        mask = (positions_df['code'] == stock_code) & (positions_df['is_sold'] == 'NO')
        if not mask.any():
            return jsonify({"error": "股票不在持仓中"}), 404

        idx = positions_df[mask].index[0]
        current_shares = int(positions_df.loc[idx, 'shares'])

        if quantity > current_shares:
            return jsonify({"error": "卖出数量超过当前持仓"}), 400

        sell_date = datetime.now().strftime('%Y-%m-%d')

        if quantity == current_shares:
            positions_df.loc[idx, 'is_sold'] = 'YES'
            positions_df.loc[idx, 'actual_sell_price'] = price
            positions_df.loc[idx, 'actual_sell_date'] = sell_date
        else:
            positions_df.loc[idx, 'shares'] = current_shares - quantity
            sold_record = positions_df.loc[idx].copy()
            sold_record['shares'] = quantity
            sold_record['actual_sell_price'] = price
            sold_record['actual_sell_date'] = sell_date
            sold_record['is_sold'] = 'YES'
            positions_df = pd.concat([positions_df, pd.DataFrame([sold_record])], ignore_index=True)

        positions_df.to_csv(POSITION_FILE, index=False)
        print(f"[SELL] 持仓已更新")

        # 读取并更新现金 - 使用csv模块直接写入,避免pandas缓存问题
        with open(CASH_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        current_cash = float(rows[0]['cash'])
        new_cash = current_cash + quantity * price
        rows[0]['cash'] = str(new_cash)
        rows[0]['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"[SELL] 当前现金: {current_cash}, 卖出获得: {quantity * price}, 新现金: {new_cash}")

        with open(CASH_FILE, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['cash', 'last_run_date', 'update_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[SELL] 现金文件已保存到: {CASH_FILE}")

        # 验证保存
        with open(CASH_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"[SELL] 文件内容: {content[:100]}")

        update_portfolio_assets(new_cash)

        return jsonify({
            "success": True,
            "message": "卖出成功",
            "data": {
                "cash": new_cash,
                "sold_quantity": quantity
            }
        })
    except Exception as e:
        import traceback
        print(f"[SELL] 错误: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/trading/buy', methods=['POST'])
def manual_buy():
    """手动买入"""
    try:
        data = request.json
        stock_code = data.get('stock_code')
        price = float(data.get('price', 0))
        quantity = int(data.get('quantity', 0))

        print(f"[BUY] 收到请求: stock={stock_code}, qty={quantity}, price={price}")
        print(f"[BUY] CASH_FILE路径: {CASH_FILE}")

        if not stock_code or price <= 0 or quantity <= 0:
            return jsonify({"error": "参数错误"}), 400

        cash_df = pd.read_csv(CASH_FILE)
        print(f"[BUY] 读取现金文件: {cash_df.to_dict()}")
        current_cash = float(cash_df.iloc[0]['cash'])
        total_cost = price * quantity
        print(f"[BUY] 当前现金: {current_cash}, 花费: {total_cost}")

        if total_cost > current_cash:
            return jsonify({"error": "现金不足"}), 400

        if POSITION_FILE.exists():
            positions_df = pd.read_csv(POSITION_FILE)
        else:
            positions_df = pd.DataFrame(columns=[
                'code', 'avg_cost', 'shares', 'entry_date',
                'stop_loss_price', 'take_profit_price', 'should_sell_date',
                'actual_sell_price', 'actual_sell_date', 'is_sold'
            ])

        entry_date = datetime.now().strftime('%Y-%m-%d')

        existing_mask = (positions_df['code'] == stock_code) & (positions_df['is_sold'] == 'NO')
        if existing_mask.any():
            idx = positions_df[existing_mask].index[0]
            existing_shares = int(positions_df.loc[idx, 'shares'])
            existing_avg_cost = float(positions_df.loc[idx, 'avg_cost'])

            new_shares = existing_shares + quantity
            new_avg_cost = (existing_shares * existing_avg_cost + quantity * price) / new_shares

            positions_df.loc[idx, 'shares'] = new_shares
            positions_df.loc[idx, 'avg_cost'] = new_avg_cost
        else:
            new_position = {
                'code': stock_code,
                'avg_cost': price,
                'shares': quantity,
                'entry_date': entry_date,
                'stop_loss_price': '',
                'take_profit_price': '',
                'should_sell_date': '',
                'actual_sell_price': '',
                'actual_sell_date': '',
                'is_sold': 'NO'
            }
            positions_df = pd.concat([positions_df, pd.DataFrame([new_position])], ignore_index=True)

        positions_df.to_csv(POSITION_FILE, index=False)
        print(f"[BUY] 持仓已更新")

        # 使用csv模块直接写入,避免pandas缓存问题
        with open(CASH_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        new_cash = current_cash - total_cost
        rows[0]['cash'] = str(new_cash)
        rows[0]['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"[BUY] 当前现金: {current_cash}, 花费: {total_cost}, 新现金: {new_cash}")

        with open(CASH_FILE, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['cash', 'last_run_date', 'update_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[BUY] 现金文件已保存到: {CASH_FILE}")

        # 验证保存
        with open(CASH_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"[BUY] 文件内容: {content[:100]}")

        update_portfolio_assets(new_cash)

        return jsonify({
            "success": True,
            "message": "买入成功",
            "data": {
                "cash": new_cash,
                "new_position": {
                    "code": stock_code,
                    "shares": quantity,
                    "avg_cost": price
                }
            }
        })
    except Exception as e:
        import traceback
        print(f"[BUY] 错误: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ========== 主交易脚本执行 API ==========
@app.route('/api/execute', methods=['POST'])
def execute_trading():
    """执行交易脚本"""
    global execution_status

    with status_lock:
        if execution_status["running"]:
            return jsonify({"error": "脚本正在运行中"}), 400
        if execution_status["st_filter_running"] or execution_status["stock_nd_running"]:
            return jsonify({"error": "辅助脚本运行中，无法执行主交易"}), 400

        try:
            data = request.json or {}
            predict_date = data.get('predict_date')
            feature_date = data.get('feature_date')
            update_data = data.get('update_data', False)

            if not predict_date or not feature_date:
                return jsonify({"error": "缺少日期参数"}), 400

            datetime.strptime(predict_date, '%Y-%m-%d')
            datetime.strptime(feature_date, '%Y-%m-%d')

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = LOG_DIR / f"execution_{timestamp}.log"

            execution_status["running"] = True
            execution_status["start_time"] = datetime.now().isoformat()
            execution_status["log_file"] = str(log_file)
            execution_status["progress"] = "启动中..."
            execution_status["process"] = None

            thread = threading.Thread(target=run_executor, args=(log_file, predict_date, feature_date, update_data))
            thread.daemon = True
            thread.start()

            return jsonify({
                "success": True,
                "message": "脚本开始执行",
                "log_file": str(log_file.name),
                "predict_date": predict_date,
                "feature_date": feature_date,
                "update_data": update_data
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

        execution_status["running"] = False
        execution_status["progress"] = "正在停止..."
        process = execution_status.get("process")

    if process:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        except:
            pass

    return jsonify({
        "success": True,
        "message": "停止信号已发送"
    })


def run_executor(log_file, predict_date, feature_date, update_data=False):
    """在后台运行执行脚本"""
    global execution_status

    try:
        print(f"[EXECUTOR] 开始执行, log_file={log_file}")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 开始执行 {datetime.now()} ===\n")
            f.write(f"预测日期 (PREDICT_DATE): {predict_date}\n")
            f.write(f"特征日期 (FEATURE_DATE): {feature_date}\n")
            f.write(f"数据更新 (UPDATE_DATA): {update_data}\n\n")
            f.flush()

            import locale
            system_encoding = locale.getpreferredencoding()

            # 构建命令
            cmd = [
                'python',
                str(EXECUTOR_SCRIPT),
                '--predict-date', predict_date,
                '--feature-date', feature_date
            ]

            if update_data:
                cmd.append('--update-data')

            # 设置环境变量
            env = os.environ.copy()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                env=env,
                text=False
            )

            with status_lock:
                execution_status["process"] = process

            for line in iter(process.stdout.readline, b''):
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
                    try:
                        decoded_line = line.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            decoded_line = line.decode('gbk')
                        except UnicodeDecodeError:
                            decoded_line = line.decode(system_encoding)

                    f.write(decoded_line)
                    f.flush()

                    progress_text = decoded_line.strip()
                    progress_text = ''.join(char for char in progress_text if ord(char) < 0x10000)
                    with status_lock:
                        execution_status["progress"] = progress_text

                except Exception as e:
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


# ========== 辅助脚本执行 API ==========
@app.route('/api/execute/st-filter', methods=['POST'])
def execute_st_filter():
    """执行st_stock_filter脚本"""
    global execution_status

    with status_lock:
        if execution_status["st_filter_running"]:
            return jsonify({"error": "st_stock_filter 正在运行中"}), 400

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = LOG_DIR / f"st_filter_{timestamp}.log"

            execution_status["st_filter_running"] = True
            execution_status["st_filter_log"] = str(log_file.name)
            execution_status["st_filter_progress"] = "启动中..."
            execution_status["running"] = True

            thread = threading.Thread(target=run_st_filter, args=(log_file,))
            thread.daemon = True
            thread.start()

            return jsonify({
                "success": True,
                "message": "st_stock_filter 开始执行",
                "log_file": str(log_file.name)
            })
        except Exception as e:
            execution_status["st_filter_running"] = False
            execution_status["running"] = False
            return jsonify({"error": str(e)}), 500


def run_st_filter(log_file):
    """运行st_stock_filter脚本"""
    global execution_status
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 开始执行 st_stock_filter {datetime.now()} ===\n\n")

            cmd = ['python', str(ST_FILTER_SCRIPT)]
            env = os.environ.copy()

            f.write(f"[CONFIG] 命令: {' '.join(cmd)}\n\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                env=env,
                text=False
            )

            with status_lock:
                execution_status["st_filter_process"] = process

            for line in iter(process.stdout.readline, b''):
                try:
                    decoded_line = line.decode('utf-8', errors='ignore')
                except:
                    decoded_line = line.decode('gbk', errors='ignore')

                f.write(decoded_line)
                f.flush()

                with status_lock:
                    execution_status["st_filter_progress"] = decoded_line.strip()[:100]

            process.wait()
            f.write(f"\n\n=== 执行完成 {datetime.now()} ===\n")
            f.write(f"退出码: {process.returncode}\n")

            with status_lock:
                execution_status["st_filter_running"] = False
                execution_status["st_filter_process"] = None
                if not execution_status["stock_nd_running"]:
                    execution_status["running"] = False
                execution_status["st_filter_progress"] = "执行完成" if process.returncode == 0 else "执行失败"
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n错误: {str(e)}\n")
        with status_lock:
            execution_status["st_filter_running"] = False
            execution_status["st_filter_process"] = None
            if not execution_status["stock_nd_running"]:
                execution_status["running"] = False
            execution_status["st_filter_progress"] = f"错误: {str(e)}"


@app.route('/api/execute/st-filter/stop', methods=['POST'])
def stop_st_filter():
    """停止st_stock_filter脚本"""
    global execution_status

    with status_lock:
        if not execution_status["st_filter_running"]:
            return jsonify({"error": "没有正在运行的st_stock_filter"}), 400

        process = execution_status.get("st_filter_process")
        execution_status["st_filter_running"] = False
        execution_status["st_filter_progress"] = "正在停止..."

    if process:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        except:
            pass

    with status_lock:
        if not execution_status["stock_nd_running"]:
            execution_status["running"] = False
        execution_status["st_filter_progress"] = "已停止"

    return jsonify({"success": True, "message": "停止信号已发送"})


@app.route('/api/execute/stock-nd', methods=['POST'])
def execute_stock_nd():
    """执行stock_nd脚本"""
    global execution_status

    with status_lock:
        if execution_status["stock_nd_running"]:
            return jsonify({"error": "stock_nd 正在运行中"}), 400

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = LOG_DIR / f"stock_nd_{timestamp}.log"

            execution_status["stock_nd_running"] = True
            execution_status["stock_nd_log"] = str(log_file.name)
            execution_status["stock_nd_progress"] = "启动中..."
            execution_status["running"] = True

            thread = threading.Thread(target=run_stock_nd, args=(log_file,))
            thread.daemon = True
            thread.start()

            return jsonify({
                "success": True,
                "message": "stock_nd 开始执行",
                "log_file": str(log_file.name)
            })
        except Exception as e:
            execution_status["stock_nd_running"] = False
            execution_status["running"] = False
            return jsonify({"error": str(e)}), 500


def run_stock_nd(log_file):
    """运行stock_nd脚本"""
    global execution_status
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 开始执行 stock_nd {datetime.now()} ===\n\n")

            cmd = ['python', str(STOCK_ND_SCRIPT)]
            env = os.environ.copy()

            f.write(f"[CONFIG] 命令: {' '.join(cmd)}\n\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                env=env,
                text=False
            )

            with status_lock:
                execution_status["stock_nd_process"] = process

            for line in iter(process.stdout.readline, b''):
                try:
                    decoded_line = line.decode('utf-8', errors='ignore')
                except:
                    decoded_line = line.decode('gbk', errors='ignore')

                f.write(decoded_line)
                f.flush()

                with status_lock:
                    execution_status["stock_nd_progress"] = decoded_line.strip()[:100]

            process.wait()
            f.write(f"\n\n=== 执行完成 {datetime.now()} ===\n")
            f.write(f"退出码: {process.returncode}\n")

            with status_lock:
                execution_status["stock_nd_running"] = False
                execution_status["stock_nd_process"] = None
                if not execution_status["st_filter_running"]:
                    execution_status["running"] = False
                execution_status["stock_nd_progress"] = "执行完成" if process.returncode == 0 else "执行失败"
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n错误: {str(e)}\n")
        with status_lock:
            execution_status["stock_nd_running"] = False
            execution_status["stock_nd_process"] = None
            if not execution_status["st_filter_running"]:
                execution_status["running"] = False
            execution_status["stock_nd_progress"] = f"错误: {str(e)}"


@app.route('/api/execute/stock-nd/stop', methods=['POST'])
def stop_stock_nd():
    """停止stock_nd脚本"""
    global execution_status

    with status_lock:
        if not execution_status["stock_nd_running"]:
            return jsonify({"error": "没有正在运行的stock_nd"}), 400

        process = execution_status.get("stock_nd_process")
        execution_status["stock_nd_running"] = False
        execution_status["stock_nd_progress"] = "正在停止..."

    if process:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        except:
            pass

    with status_lock:
        if not execution_status["st_filter_running"]:
            execution_status["running"] = False
        execution_status["stock_nd_progress"] = "已停止"

    return jsonify({"success": True, "message": "停止信号已发送"})


# ========== 状态查询 API ==========
@app.route('/api/status', methods=['GET'])
def get_status():
    """获取执行状态"""
    with status_lock:
        print(f"[STATUS] running={execution_status['running']}, log_file={execution_status['log_file']}")
        return jsonify({
            "running": execution_status["running"],
            "start_time": execution_status["start_time"],
            "log_file": execution_status["log_file"],
            "progress": execution_status["progress"],
            "st_filter_running": execution_status["st_filter_running"],
            "stock_nd_running": execution_status["stock_nd_running"],
            "st_filter_progress": execution_status.get("st_filter_progress", ""),
            "stock_nd_progress": execution_status.get("stock_nd_progress", ""),
            "st_filter_log": execution_status.get("st_filter_log", ""),
            "stock_nd_log": execution_status.get("stock_nd_log", "")
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
    app.run(host='0.0.0.0', port=5000, debug=True)
