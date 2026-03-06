"""
测试状态同步修复
验证线程安全和状态一致性
"""
import requests
import time
import threading

BASE_URL = "http://localhost:5000"

def test_status_check():
    """测试状态检查"""
    print("=" * 50)
    print("测试1: 状态检查")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/api/status")
    status = response.json()
    
    print(f"运行状态: {status['running']}")
    print(f"进度信息: {status.get('progress', '无')}")
    print(f"开始时间: {status.get('start_time', '无')}")
    print()

def test_duplicate_execution():
    """测试重复执行保护"""
    print("=" * 50)
    print("测试2: 重复执行保护")
    print("=" * 50)
    
    # 第一次执行
    print("发送第一次执行请求...")
    response1 = requests.post(f"{BASE_URL}/api/execute")
    result1 = response1.json()
    print(f"第一次结果: {result1}")
    
    # 立即发送第二次执行请求
    time.sleep(0.5)
    print("\n立即发送第二次执行请求...")
    response2 = requests.post(f"{BASE_URL}/api/execute")
    result2 = response2.json()
    print(f"第二次结果: {result2}")
    
    if response2.status_code == 400:
        print("✅ 成功阻止重复执行！")
    else:
        print("❌ 未能阻止重复执行！")
    
    # 停止执行
    time.sleep(2)
    print("\n停止执行...")
    requests.post(f"{BASE_URL}/api/stop")
    time.sleep(2)
    print()

def test_concurrent_status_check():
    """测试并发状态检查"""
    print("=" * 50)
    print("测试3: 并发状态检查（线程安全）")
    print("=" * 50)
    
    results = []
    
    def check_status(thread_id):
        try:
            response = requests.get(f"{BASE_URL}/api/status")
            status = response.json()
            results.append((thread_id, status['running']))
            print(f"线程 {thread_id}: running={status['running']}")
        except Exception as e:
            print(f"线程 {thread_id} 错误: {e}")
    
    # 创建10个并发线程
    threads = []
    for i in range(10):
        t = threading.Thread(target=check_status, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 检查结果一致性
    if len(set(r[1] for r in results)) == 1:
        print("✅ 所有线程读取到一致的状态！")
    else:
        print("❌ 状态不一致！")
    print()

def test_page_refresh_scenario():
    """测试页面刷新场景"""
    print("=" * 50)
    print("测试4: 模拟页面刷新场景")
    print("=" * 50)
    
    # 启动执行
    print("1. 启动执行...")
    response = requests.post(f"{BASE_URL}/api/execute")
    if response.status_code == 200:
        print("✅ 执行已启动")
    
    time.sleep(2)
    
    # 模拟页面刷新 - 立即检查状态
    print("\n2. 模拟页面刷新 - 立即检查状态...")
    response = requests.get(f"{BASE_URL}/api/status")
    status = response.json()
    
    if status['running']:
        print("✅ 刷新后立即看到运行状态！")
        print(f"   进度: {status.get('progress', '无')}")
    else:
        print("❌ 刷新后未能看到运行状态")
    
    # 尝试重复执行
    print("\n3. 尝试重复执行...")
    response = requests.post(f"{BASE_URL}/api/execute")
    if response.status_code == 400:
        print("✅ 成功阻止重复执行！")
        print(f"   错误信息: {response.json().get('error')}")
    else:
        print("❌ 未能阻止重复执行")
    
    # 停止执行
    print("\n4. 停止执行...")
    requests.post(f"{BASE_URL}/api/stop")
    time.sleep(2)
    
    # 再次检查状态
    print("\n5. 停止后检查状态...")
    response = requests.get(f"{BASE_URL}/api/status")
    status = response.json()
    
    if not status['running']:
        print("✅ 状态已正确更新为停止")
    else:
        print("❌ 状态未正确更新")
    print()

def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("状态同步修复测试")
    print("=" * 50 + "\n")
    
    try:
        # 测试服务器连接
        print("检查服务器连接...")
        response = requests.get(BASE_URL, timeout=5)
        print("✅ 服务器连接正常\n")
        
        # 运行测试
        test_status_check()
        test_duplicate_execution()
        test_concurrent_status_check()
        test_page_refresh_scenario()
        
        print("=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("请确保Flask服务器正在运行: python web_interface/app.py")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()
