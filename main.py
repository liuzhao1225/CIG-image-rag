#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创意广告图文搜索系统 - 主入口文件

启动方式：
    python main.py

作者：AI Assistant
日期：2024
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """应用主入口，启动Streamlit应用"""
    
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    # 设置Streamlit应用路径
    streamlit_app_path = project_root / "ui" / "streamlit_app.py"
    
    # 检查应用文件是否存在
    if not streamlit_app_path.exists():
        print(f"❌ 错误：找不到Streamlit应用文件: {streamlit_app_path}")
        print("请确保项目结构正确，并且已创建相应的文件。")
        sys.exit(1)
    
    # 设置Python路径，确保可以导入项目模块
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    print("🚀 启动创意广告图文搜索系统...")
    print(f"📁 项目根目录: {project_root}")
    print(f"🌐 Streamlit应用: {streamlit_app_path}")
    print("=" * 50)
    
    try:
        # 启动Streamlit应用
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path),
            "--browser.headless", "false"
        ]
        
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except FileNotFoundError:
        print("❌ 错误：未找到streamlit命令")
        print("请确保已安装streamlit: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
