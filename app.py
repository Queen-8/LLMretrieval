# app.py
import sys
import os

# ====== 添加项目根目录和 src 到 sys.path ======
project_root = os.path.abspath(os.getcwd())
sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# ====== 导入 main.py 中的 Streamlit 入口 ======
from langchainagenticai.main import load_langgraph_agenticai_app

if __name__ == "__main__":
    load_langgraph_agenticai_app()


# 没问题
# import sys
# import os

# # 将项目根目录添加到 sys.path 中，以便可以正确导入模块
# project_root = os.path.abspath(os.getcwd())
# sys.path.insert(0, project_root)

# # 将 src 目录添加到 sys.path 中
# src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
# sys.path.insert(0, src_path)

# print("Current sys.path:", sys.path)  # 打印 sys.path

# # 导入 main.py 中的 app 主入口函数
# from langchainagenticai.main import load_langgraph_agenticai_app


# # 导入 main.py 中的 app 主入口函数
# # from langchainagenticai.main import main as load_langgraph_agenticai_app

# if __name__ == "__main__":
#     load_langgraph_agenticai_app()