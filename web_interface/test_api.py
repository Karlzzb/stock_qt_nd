"""
API测试脚本
用于测试Web界面的各个API接口
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_get_cash():
    """测试获取资金信息"""
    print("测试: 获取资金信息")
    response = requests.get(f"{BASE_URL}/api/portfolio/cash")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print("-" * 50)

def test_get_positions():
    """测试获取持仓信息"""
    print("测试: 获取持仓信息")
    response = requests.get(f"{BASE_URL}/api/portfolio/positions")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print("-" * 50)

def test_update_cash():
    """测试更新资金"""
    print("测试: 更新资金")
    data = {"cash": 100000}
    response = requests.post(
        f"{BASE_URL}/api/portfolio/cash",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print("-" * 50)

def test_get_status():
    """测试获取执行状态"""
    print("测试: 获取执行状态")
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print("-" * 50)

def test_list_logs():
    """测试列出日志"""
    print("测试: 列出日志")
    response = requests.get(f"{BASE_URL}/api/logs")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print("-" * 50)

def test_list_reports():
    """测试列出报告"""
    print("测试: 列出报告")
    response = requests.get(f"{BASE_URL}/api/reports")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print("-" * 50)

if __name__ == "__main__":
    print("=" * 50)
    print("开始测试API接口")
    print("=" * 50)
    print()
    
    try:
        test_get_cash()
        test_get_positions()
        test_get_status()
        test_list_logs()
        test_list_reports()
        
        # 可选: 测试更新资金（会修改数据）
        # test_update_cash()
        
        print("=" * 50)
        print("所有测试完成")
        print("=" * 50)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器")
        print("请确保服务已启动: python app.py")
    except Exception as e:
        print(f"错误: {e}")
