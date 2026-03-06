#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查执行状态脚本
可以在不打开浏览器的情况下查看执行状态
"""

import requests
import sys
from datetime import datetime
from pathlib import Path

# 配置
API_URL = "http://localhost:5000/api/status"
LOG_DIR = Path(__file__).parent / "logs"

def check_status():
    """检查执行状态"""
    print("=" * 60)
    print("📊 股票AI交易系统 - 状态检查")
    print("=" * 60)
    print()
    
    try:
        # 尝试连接API
        response = requests.get(API_URL, timeout=5)
        
        if response.status_code == 200:
            status = response.json()
            
            print("✅ Flask服务器: 运行中")
            print()
            
            # 显示执行状态
            if status.get("running"):
                print("🚀 执行状态: 运行中")
                print(f"⏰ 开始时间: {status.get('start_time', 'N/A')}")
                print(f"📝 日志文件: {status.get('log_file', 'N/A')}")
                print(f"📊 当前进度: {status.get('progress', 'N/A')}")
                
                # 计算运行时长
                if status.get('start_time'):
                    try:
                        start = datetime.fromisoformat(status['start_time'])
                        duration = datetime.now() - start
                        minutes = int(duration.total_seconds() / 60)
                        seconds = int(duration.total_seconds() % 60)
                        print(f"⏱️  运行时长: {minutes}分{seconds}秒")
                    except:
                        pass
            else:
                print("⏸️  执行状态: 就绪/已完成")
                
                # 查找最新的日志文件
                if LOG_DIR.exists():
                    log_files = sorted(LOG_DIR.glob("execution_*.log"), 
                                     key=lambda x: x.stat().st_mtime, 
                                     reverse=True)
                    if log_files:
                        latest_log = log_files[0]
                        mod_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
                        print(f"📝 最新日志: {latest_log.name}")
                        print(f"⏰ 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # 显示日志最后几行
                        print()
                        print("📄 日志最后10行:")
                        print("-" * 60)
                        try:
                            with open(latest_log, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                for line in lines[-10:]:
                                    print(line.rstrip())
                        except Exception as e:
                            print(f"读取日志失败: {e}")
                        print("-" * 60)
        else:
            print(f"❌ API响应错误: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Flask服务器: 未运行")
        print()
        print("请先启动Flask服务器:")
        print("  Windows: start.bat")
        print("  Linux/Mac: ./start.sh")
        
    except requests.exceptions.Timeout:
        print("❌ 连接超时")
        print("请检查Flask服务器是否正常运行")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    print()
    print("=" * 60)

def show_logs():
    """显示所有日志文件"""
    print("=" * 60)
    print("📝 执行日志列表")
    print("=" * 60)
    print()
    
    if not LOG_DIR.exists():
        print("日志目录不存在")
        return
    
    log_files = sorted(LOG_DIR.glob("execution_*.log"), 
                      key=lambda x: x.stat().st_mtime, 
                      reverse=True)
    
    if not log_files:
        print("暂无日志文件")
        return
    
    for i, log_file in enumerate(log_files[:10], 1):
        mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        size_kb = log_file.stat().st_size / 1024
        print(f"{i}. {log_file.name}")
        print(f"   时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   大小: {size_kb:.1f} KB")
        print()

def show_help():
    """显示帮助信息"""
    print("=" * 60)
    print("📖 使用说明")
    print("=" * 60)
    print()
    print("用法:")
    print("  python check_status.py          # 检查当前状态")
    print("  python check_status.py logs     # 显示日志列表")
    print("  python check_status.py help     # 显示帮助")
    print()
    print("功能:")
    print("  - 检查Flask服务器是否运行")
    print("  - 查看脚本执行状态")
    print("  - 显示实时进度")
    print("  - 查看最新日志")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "logs":
            show_logs()
        elif command == "help":
            show_help()
        else:
            print(f"未知命令: {command}")
            print("使用 'python check_status.py help' 查看帮助")
    else:
        check_status()
