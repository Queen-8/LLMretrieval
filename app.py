import sys
import os

# 将 src 目录添加到 sys.path 中，以便可以正确导入 src.langgraphagenticai
project_root = os.path.abspath(os.path.join(os.getcwd(), 'src'))  # 'src' 在当前工作目录下
sys.path.append(project_root)

print("Current sys.path:", sys.path)  # 打印 sys.path

# 导入模块
from langchainagenticai.main import load_langgraph_agenticai_app

if __name__ == "__main__":
    load_langgraph_agenticai_app()
