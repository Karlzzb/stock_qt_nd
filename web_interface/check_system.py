"""
系统检查脚本
检查环境配置是否正确
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("  需要Python 3.7或更高版本")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n检查依赖包...")
    required = ['flask', 'flask_cors']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n请安装缺失的包: pip install {' '.join(missing)}")
        return False
    return True

def check_files():
    """检查必要文件"""
    print("\n检查项目文件...")
    base_dir = Path(__file__).parent.parent
    
    files_to_check = [
        ("app.py", "web_interface/app.py"),
        ("index.html", "web_interface/templates/index.html"),
        ("requirements.txt", "web_interface/requirements.txt"),
        ("执行脚本", "src/daily_trading_executor.py"),
        ("资金文件", "real_trading_data/investment_data/portfolio_cash.csv"),
        ("持仓文件", "real_trading_data/investment_data/portfolio_positions.csv"),
    ]
    
    all_exist = True
    for name, path in files_to_check:
        full_path = base_dir / path
        if full_path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} 不存在")
            all_exist = False
    
    return all_exist

def check_port():
    """检查端口是否可用"""
    print("\n检查端口...")
    import socket
    
    port = 5000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        print(f"✗ 端口 {port} 已被占用")
        print(f"  请关闭占用端口的程序或修改配置使用其他端口")
        return False
    else:
        print(f"✓ 端口 {port} 可用")
        return True

def check_permissions():
    """检查文件权限"""
    print("\n检查文件权限...")
    base_dir = Path(__file__).parent.parent
    
    dirs_to_check = [
        "web_interface/logs",
        "real_trading_data/investment_data",
    ]
    
    all_ok = True
    for dir_path in dirs_to_check:
        full_path = base_dir / dir_path
        
        # 创建目录如果不存在
        if not full_path.exists():
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ 创建目录: {dir_path}")
            except Exception as e:
                print(f"✗ 无法创建目录 {dir_path}: {e}")
                all_ok = False
                continue
        
        # 检查写权限
        test_file = full_path / ".test_write"
        try:
            test_file.touch()
            test_file.unlink()
            print(f"✓ {dir_path} 可写")
        except Exception as e:
            print(f"✗ {dir_path} 无写权限: {e}")
            all_ok = False
    
    return all_ok

def get_local_ip():
    """获取本机IP地址"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "无法获取"

def main():
    """主函数"""
    print("=" * 60)
    print("股票AI量化交易系统 - 环境检查")
    print("=" * 60)
    print()
    
    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("项目文件", check_files),
        ("端口可用性", check_port),
        ("文件权限", check_permissions),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n检查 {name} 时出错: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✓ 所有检查通过！")
        print("\n可以启动服务了:")
        print("  python app.py")
        print("\n访问地址:")
        print(f"  本地: http://localhost:5000")
        print(f"  局域网: http://{get_local_ip()}:5000")
    else:
        print("✗ 部分检查未通过，请解决上述问题后再启动服务")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
