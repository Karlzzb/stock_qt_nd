"""
编码测试脚本
用于检测和验证编码设置
"""
import sys
import os
import locale

def test_encoding():
    """测试编码设置"""
    print("=" * 60)
    print("编码环境检测")
    print("=" * 60)
    print()
    
    # 1. Python版本
    print(f"Python版本: {sys.version}")
    print()
    
    # 2. 默认编码
    print("编码设置:")
    print(f"  默认编码: {sys.getdefaultencoding()}")
    print(f"  标准输出编码: {sys.stdout.encoding}")
    print(f"  标准错误编码: {sys.stderr.encoding}")
    print(f"  文件系统编码: {sys.getfilesystemencoding()}")
    print(f"  系统首选编码: {locale.getpreferredencoding()}")
    print()
    
    # 3. 环境变量
    print("环境变量:")
    print(f"  PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', '未设置')}")
    print(f"  PYTHONUTF8: {os.environ.get('PYTHONUTF8', '未设置')}")
    print(f"  LANG: {os.environ.get('LANG', '未设置')}")
    print()
    
    # 4. 测试特殊字符
    print("特殊字符测试:")
    test_strings = [
        ("ASCII字符", "Hello World 123"),
        ("中文字符", "你好世界"),
        ("Emoji表情", "🚀 📈 💰 ✅ ❌"),
        ("混合字符", "Stock Trading 股票交易 📊"),
    ]
    
    for name, text in test_strings:
        try:
            print(f"  {name}: {text}")
            # 尝试编码
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            print(f"    ✓ UTF-8编码测试通过")
        except Exception as e:
            print(f"    ✗ 错误: {e}")
    
    print()
    
    # 5. 文件读写测试
    print("文件读写测试:")
    test_file = "test_encoding_temp.txt"
    test_content = "测试内容 Test Content 🚀"
    
    try:
        # 写入测试
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"  ✓ UTF-8写入成功")
        
        # 读取测试
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content == test_content:
            print(f"  ✓ UTF-8读取成功")
        else:
            print(f"  ✗ 读取内容不匹配")
        
        # 清理
        os.remove(test_file)
        
    except Exception as e:
        print(f"  ✗ 文件操作错误: {e}")
        if os.path.exists(test_file):
            try:
                os.remove(test_file)
            except:
                pass
    
    print()
    
    # 6. 建议
    print("=" * 60)
    print("建议:")
    print("=" * 60)
    
    issues = []
    
    if sys.stdout.encoding.lower() not in ['utf-8', 'utf8']:
        issues.append("标准输出编码不是UTF-8")
    
    if os.environ.get('PYTHONIOENCODING', '').lower() not in ['utf-8', 'utf8']:
        issues.append("PYTHONIOENCODING环境变量未设置为utf-8")
    
    if os.environ.get('PYTHONUTF8') != '1':
        issues.append("PYTHONUTF8环境变量未设置为1")
    
    if issues:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("修复方法:")
        print("  1. 运行 fix_encoding.bat")
        print("  2. 或使用 start.bat 启动服务")
        print("  3. 或手动设置环境变量:")
        print("     set PYTHONIOENCODING=utf-8")
        print("     set PYTHONUTF8=1")
    else:
        print("✅ 编码设置正确，可以正常使用！")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_encoding()
    except Exception as e:
        print(f"测试过程出错: {e}")
        import traceback
        traceback.print_exc()
