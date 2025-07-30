import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from typing import Any
from pydantic import SecretStr
from langgraph.prebuilt import create_react_agent
# from langchain_groq import ChatGroq
from abc import ABC, abstractmethod

load_dotenv() # 加载 .env 文件中的 OPENROUTER_API_KEY
# from langchain_community.chat_models import ChatOllama
# from langchain_ollama import ChatOllama



# ========== 环境变量加载 ==========
root_dir = Path(__file__).parent
load_dotenv(dotenv_path=root_dir / ".env")
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key is None:
    raise ValueError("OPENROUTER_API_KEY not found in .env!")
print(f"✅ API key loaded: {api_key[:5]}...")



# 获取当前脚本所在目录路径
root_dir = Path(__file__).parent

# 加载项目根目录下的 .env 文件中的 OPENROUTER_API_KEY
load_dotenv(dotenv_path=root_dir / ".env")

# 测试输出，检查环境变量是否加载成功
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key is None:
    raise ValueError("OPENROUTER_API_KEY not found in .env!")
print(f"✅ API key loaded: {api_key[:5]}...")  # 👈 输出部分 API key 进行调试

# 输出调试信息，确保环境变量加载成功
print(f"Loaded OPENROUTER_API_KEY: {api_key}")

# 定义查询内容
query = "2024年世界职业院校分为哪些赛道？"
# # query = "xxe产生原因是什么？"

# 获取当前虚拟环境的 Python 路径（如果使用虚拟环境）
python_executable = sys.executable  # 获取当前虚拟环境的 Python 路径


async def main():
    connections: dict[str, Any] = {
        "OnlineRetrievalQA": {
            "command": "python",
            "args": [os.path.abspath("online_server.py")],
            "transport": "stdio",
        },
        "OfflineRetrievalQA": {
            "command": "python",
            "args": [os.path.abspath("offline_server.py")],
            "transport": "stdio",
        }
    }
    client = MultiServerMCPClient(connections)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    tools=await client.get_tools()
    print(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
    
    # openrouter
    model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr(api_key),
        max_tokens=4000,  # 限制最大 token 数量
        temperature=0.7,
    )

    # ollama
    # model = ChatOllama(
    #     model="deepseek-r1:8b",  # 本地已拉取的模型名，可根据你本地的模型修改
    #     base_url="http://localhost:11434",
    # )
    
    agent = create_react_agent(model, tools)
    
    localserver_response = await agent.ainvoke(
        {"messages":[{"role":"user","content":"2024年世界职业技能大赛共有几个赛道?"}]}
    )
    
    # print("localserver response:", localserver_response['message'][-1].content)
    # print("Raw localserver_response:", localserver_response) 
    print("Answer:", localserver_response["messages"][-1].content)

asyncio.run(main())
