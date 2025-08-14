import os
import sys
import asyncio
import logging
import streamlit as st
from dotenv import load_dotenv

# ====== 运行期环境（尽量最前）======
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USER_AGENT", "LangChain-AgenticAI/1.0 (contact: you@example.com)")

load_dotenv(override=True)

# 给只认 OPENAI_* 的库做别名（兜底）
if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))

# ====== Imports（放在环境准备之后）======
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchainagenticai.agent.agent import create_mcp_agent
from langchainagenticai.utils.loaddocs import init_retrieval_pipeline
from langchainagenticai.utils.loadurls import init_url_retrieval_pipeline
from langchainagenticai.ui.streamlit.load_ui import LoadStreamlitUI

# 项目根目录
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MCP 连接
MCP_CONNECTIONS = {
    "LocalRetrievalQA": {
        "command": "python",
        "args": ["src/langchainagenticai/mcp_server/local_server.py"],
        "transport": "stdio",
    },
    "CloudRetrievalQA": {
        "command": "python",
        "args": ["src/langchainagenticai/mcp_server/cloud_server.py"],
        "transport": "stdio",
    },
}

# --------- 构建/缓存 LLM & Agent ----------
def _build_llm(selected_llm: str):
    selected_llm = (selected_llm or "ollama").strip()
    if selected_llm.lower() == "openai":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 OPENROUTER_API_KEY，请在 .env 中设置")

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

        openrouter_headers = {
            "HTTP-Referer": os.getenv("OR_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OR_X_TITLE", "LangChain Agentic AI"),
            "User-Agent": os.getenv("USER_AGENT", "LangChain-AgenticAI/1.0"),
        }

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            default_headers=openrouter_headers,
            timeout=120,       # 防卡住
            max_retries=2,
        )
        logger.info(f"✅ OpenRouter LLM ready: {model_name}")
        return llm

    llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
    logger.info("✅ Ollama LLM ready: llama3.1:8b")
    return llm


def _get_cached_agent(selected_llm: str):
    key_llm = f"llm::{selected_llm}"
    key_agent = f"mcp_agent::{selected_llm}"

    if key_agent in st.session_state:
        return st.session_state[key_agent]

    llm = st.session_state.get(key_llm)
    if llm is None:
        llm = _build_llm(selected_llm)
        st.session_state[key_llm] = llm

    agent = asyncio.run(create_mcp_agent(llm, MCP_CONNECTIONS))
    st.session_state[key_agent] = agent
    return agent

# --------- UI 主流程 ----------
def load_langgraph_agenticai_app():
    st.title("🤖 LangChain Agentic AI")

    # 初始化知识库（只做一次）
    if "vector_index" not in st.session_state:
        with st.spinner("📂 正在初始化本地知识库索引..."):
            try:
                st.session_state.vector_index = init_retrieval_pipeline()
                st.success("📂✅ 本地知识库加载完成")
            except Exception as e:
                st.error(f"❌ 本地知识库初始化失败: {e}")
                return

    if "vector_index_urls" not in st.session_state:
        with st.spinner("🌐 正在初始化 URL 知识库索引..."):
            try:
                st.session_state.vector_index_urls = init_url_retrieval_pipeline()
                st.success("🌐✅ URL 知识库加载完成")
            except Exception as e:
                st.error(f"❌ URL 知识库初始化失败: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 侧边栏
    with st.sidebar:
        st.header("模型配置")
        ui = LoadStreamlitUI(set_page_config=False)
        st.session_state.user_input = ui.load_streamlit_ui()

        st.header("对话管理")
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.rerun()

    # 显示历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 输入
    if prompt := st.chat_input("请输入您的问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                user_input = st.session_state.get("user_input", {}) or {}
                selected_llm = user_input.get("selected_llm", "ollama")

                agent = _get_cached_agent(selected_llm)
                answer = asyncio.run(agent.ainvoke(prompt))

                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err = f"❌ 发生错误: {e}"
                logging.exception("assistant error")
                placeholder.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})









# import os
# import sys
# import asyncio
# import streamlit as st
# import logging
# from dotenv import load_dotenv
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# # ====== MCP & LangChain ======
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain.agents import initialize_agent, AgentType
# from langchain_community.chat_models import ChatOllama
# from langchain_openai import ChatOpenAI


# # ====== 知识库 ======
# from langchainagenticai.utils.loaddocs import init_retrieval_pipeline
# from langchainagenticai.utils.loadurls import init_url_retrieval_pipeline
# from langchainagenticai.ui.streamlit.load_ui import LoadStreamlitUI
# from langchainagenticai.agent.agent import create_mcp_agent

# load_dotenv(override=True)

# # 项目根目录加入路径
# project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
# sys.path.append(project_root)

# # 日志
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # MCP 多服务器客户端
# MCP_CONNECTIONS = {
#     "LocalRetrievalQA": {
#         "command": "python",
#         "args": ["src/langchainagenticai/mcp_server/local_server.py"],
#         "transport": "stdio",
#     },
#     "CloudRetrievalQA": {
#         "command": "python",
#         "args": ["src/langchainagenticai/mcp_server/cloud_server.py"],
#         "transport": "stdio",
#     },
# }

# async def load_all_mcp_tools():
#     """从所有 MCP 客户端加载工具"""
#     tools = []
#     for client in mcp_clients.values():
#         client_tools = await client.get_tools()
#         tools.extend(client_tools)
#     return tools

# def load_langgraph_agenticai_app():
#     st.title("🤖 LangChain Agentic AI")

#     # 初始化知识库
#     if "vector_index" not in st.session_state:
#         with st.spinner("📂 正在初始化本地知识库索引..."):
#             try:
#                 st.session_state.vector_index = init_retrieval_pipeline()
#                 st.success("📂✅ 本地知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ 本地知识库初始化失败: {e}")
#                 return

#     if "vector_index_urls" not in st.session_state:
#         with st.spinner("🌐 正在初始化 URL 知识库索引..."):
#             try:
#                 st.session_state.vector_index_urls = init_url_retrieval_pipeline()
#                 st.success("🌐✅ URL 知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ URL 知识库初始化失败: {e}")

#     # 初始化消息历史
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # 侧边栏
#     with st.sidebar:
#         st.header("模型配置")
#         ui = LoadStreamlitUI(set_page_config=False)
#         st.session_state.user_input = ui.load_streamlit_ui()

#         st.header("对话管理")
#         if st.button("🗑️ 清除对话历史"):
#             st.session_state.messages = []
#             st.rerun()

#     # 显示历史消息
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # 在收到用户输入时：
#     if prompt := st.chat_input("请输入您的问题"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()

#             try:
#                 user_input = st.session_state.user_input
#                 selected_llm = user_input.get("selected_llm", "ollama")

#                 # 1) 构建模型
#                 if selected_llm == "OpenAI":
#                     # —— 在线：OpenRouter → openai/gpt-4o-mini （function calling）
#                     api_key = os.getenv("OPENROUTER_API_KEY")
#                     base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
#                     model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
#                     model = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
#                     # model = ChatOpenAI(model="openai/gpt-4o-mini")   # 你的实际选择
#                 else:
#                     model = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")

#                 # 2) 创建 MCP Agent（连接本地与云端 server，Agent 自己决定用哪个工具）
#                 agent = asyncio.run(create_mcp_agent(model, MCP_CONNECTIONS))

#                 # 3) 交给 Agent（它会 Thought→Action→调用 MCP tool→Observation→…→Final Answer）
#                 answer = asyncio.run(agent.ainvoke(prompt))

#                 message_placeholder.markdown(answer)
#                 st.session_state.messages.append({"role": "assistant", "content": answer})

#             except Exception as e:
#                 error_msg = f"❌ 发生错误: {e}"
#                 message_placeholder.markdown(error_msg)
#                 st.session_state.messages.append({"role": "assistant", "content": error_msg})




# # src/langchainagenticai/main.py
# import os
# import sys
# import logging
# import streamlit as st

# # 保持你的 UI/检索初始化不变
# from src.langchainagenticai.utils.loaddocs import init_retrieval_pipeline
# from src.langchainagenticai.utils.loadurls import init_url_retrieval_pipeline
# from src.langchainagenticai.ui.streamlit.load_ui import LoadStreamlitUI

# # 新的 Agent（负责自动路由 MCP 工具）
# from src.langchainagenticai.agent.agent import MCPAgent 

# import logging
# import streamlit as st

# from src.langchainagenticai.utils.loaddocs import init_retrieval_pipeline
# from src.langchainagenticai.utils.loadurls import init_url_retrieval_pipeline
# from src.langchainagenticai.ui.streamlit.load_ui import LoadStreamlitUI
# from src.langchainagenticai.agent.agent import MCPAgent

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# def _init_states():
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "vector_index" not in st.session_state:
#         with st.spinner("📂 正在初始化本地知识库索引..."):
#             try:
#                 st.session_state.vector_index = init_retrieval_pipeline()
#                 st.success("📂✅ 本地知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ 本地知识库初始化失败: {e}")
#     if "vector_index_urls" not in st.session_state:
#         with st.spinner("🌐 正在初始化 URL 知识库索引..."):
#             try:
#                 st.session_state.vector_index_urls = init_url_retrieval_pipeline()
#                 st.success("🌐✅ URL 知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ URL 知识库初始化失败: {e}")


# def _render_sidebar():
#     st.sidebar.header("模型配置")
#     ui = LoadStreamlitUI(set_page_config=False)
#     st.session_state.user_input = ui.load_streamlit_ui()

#     st.sidebar.header("对话管理")
#     user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
#     assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
#     st.sidebar.metric("用户消息", user_messages)
#     st.sidebar.metric("助手回复", assistant_messages)

#     if st.sidebar.button("🗑️ 清除对话历史"):
#         st.session_state.messages = []
#         st.rerun()

#     if st.session_state.messages:
#         if st.sidebar.button("📥 导出对话"):
#             conversation_text = ""
#             for msg in st.session_state.messages:
#                 role = "用户" if msg["role"] == "user" else "助手"
#                 conversation_text += f"{role}: {msg['content']}\n\n"

#             st.sidebar.download_button(
#                 label="下载对话记录",
#                 data=conversation_text,
#                 file_name=f"conversation_{st.session_state.get('session_id', 'unknown')}.txt",
#                 mime="text/plain"
#             )


# def _render_history():
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])


# def _handle_chat():
#     prompt = st.chat_input("请输入您的问题")
#     if not prompt:
#         return

#     # 展示用户消息
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # 生成助手回复
#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         try:
#             user_input = st.session_state.get("user_input", {})
#             selected_llm = (user_input.get("selected_llm", "ollama") or "ollama").lower()
#             backend = "openai" if selected_llm == "openai" else "ollama"

#             agent = MCPAgent(llm_backend=backend)
#             answer = agent.ask(prompt, history=st.session_state.messages)

#             placeholder.markdown(answer)
#             st.session_state.messages.append({"role": "assistant", "content": answer})

#         except Exception as e:
#             err = f"❌ 发生错误: {e}"
#             logging.exception("assistant error")
#             placeholder.markdown(err)
#             st.session_state.messages.append({"role": "assistant", "content": err})


# def main():
#     st.set_page_config(page_title="LangChain Agentic AI", page_icon="🤖")
#     st.title("🤖 LangChain Agentic AI")
#     _init_states()
#     _render_sidebar()
#     _render_history()
#     _handle_chat()


# # 兼容老入口（app.py 里可能还在 import 这个）
# def load_langgraph_agenticai_app():
#     return main()


# __all__ = ["main", "load_langgraph_agenticai_app"]


# if __name__ == "__main__":
#     main()








# 没问题
# import os
# import sys
# import asyncio
# import streamlit as st
# from pydantic import SecretStr
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchainagenticai.utils.loaddocs import init_retrieval_pipeline
# from langchainagenticai.utils.loadurls import init_url_retrieval_pipeline
# from langchainagenticai.ui.streamlit.load_ui import LoadStreamlitUI
# from angchainagenticai.agent.agent import MCPAgent

# from langchainagenticai.ui.streamlit.display_result import DisplayResultStreamlit
# from langchain_openai import ChatOpenAI
# from langchainagenticai.mcp_server.local_server import local_answer_question
# from langchainagenticai.mcp_server.cloud_server import cloud_answer_question
# from pydantic import SecretStr
# import json
# import requests       
# from langchain_openai import OpenAI

# from langchain_community.chat_models import ChatOllama
# from langchain.chat_models import ChatOllama
# import logging

# # 将项目根目录添加到 sys.path
# project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 跳出 src/langchainagenticai/retrieval
# sys.path.append(project_root)

# # 配置日志
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# # MCP 客户端初始化（全局只创建一次）
# mcp_clients = {
#     "CloudRetrievalQA": MultiServerMCPClient(
#         connections={
#             "CloudRetrievalQA": {
#                 "command": "python",
#                 "args": ["src/langchainagenticai/mcp_server/cloud_server.py"],
#                 "transport": "stdio",
#             }
#         }
#     ),
#     "LocalRetrievalQA": MultiServerMCPClient(
#         connections={
#             "LocalRetrievalQA": {
#                 "command": "python",
#                 "args": ["src/langchainagenticai/mcp_server/local_server.py"],
#                 "transport": "stdio",
#             }
#         }
#     ),
# }

# async def ask_cloud_tool(client, query):
#     tools = await client.get_tools()
#     cloud_tool = next((t for t in tools if t.name == "cloud_answer_question"), None)
#     if not cloud_tool:
#         return "❌ 未找到 cloud_answer_question 工具"
#     result = await cloud_tool.ainvoke({"query": query})
#     return result

# async def ask_local_tool(client, query):
#     tools = await client.get_tools()
#     local_tool = next((t for t in tools if t.name == "local_answer_question"), None)
#     if not local_tool:
#         return "❌ 未找到 local_answer_question 工具"
#     result = await local_tool.ainvoke({"query": query})
#     return result

# def load_langgraph_agenticai_app():
#     st.title("🤖 LangChain Agentic AI")

#     # 初始化知识库索引（只执行一次）
#     if "vector_index" not in st.session_state:
#         with st.spinner("📂 正在初始化本地知识库索引..."):
#             try:
#                 st.session_state.vector_index = init_retrieval_pipeline()
#                 st.success("📂✅ 本地知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ 本地知识库初始化失败: {e}")
#                 return

#     if "vector_index_urls" not in st.session_state:
#         with st.spinner("🌐 正在初始化 URL 知识库索引..."):
#             try:
#                 st.session_state.vector_index_urls = init_url_retrieval_pipeline()
#                 st.success("🌐✅ URL 知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ URL 知识库初始化失败: {e}")

#     # 初始化消息历史
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # 侧边栏
#     with st.sidebar:
#         st.header("模型配置")
        
#         # 使用 LoadStreamlitUI 来显示完整的配置界面
#         ui = LoadStreamlitUI(set_page_config=False)
#         st.session_state.user_input = ui.load_streamlit_ui()

#         st.header("对话管理")
#         user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
#         assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
#         st.metric("用户消息", user_messages)
#         st.metric("助手回复", assistant_messages)

#         if st.button("🗑️ 清除对话历史"):
#             st.session_state.messages = []
#             st.rerun()
            
#         if st.session_state.messages:
#             if st.button("📥 导出对话"):
#                 conversation_text = ""
#                 for msg in st.session_state.messages:
#                     role = "用户" if msg["role"] == "user" else "助手"
#                     conversation_text += f"{role}: {msg['content']}\n\n"

#                 st.download_button(
#                     label="下载对话记录",
#                     data=conversation_text,
#                     file_name=f"conversation_{st.session_state.get('session_id', 'unknown')}.txt",
#                     mime="text/plain"
#                 )

#     # 显示历史消息
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     if prompt := st.chat_input("请输入您的问题"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         try:
#             user_input = st.session_state.user_input
#             selected_llm = (user_input.get("selected_llm", "ollama")).lower()

#             agent = MCPAgent(llm_backend=("openai" if selected_llm == "openai" else "ollama"))

#             # 把历史传给 Agent（选）
#             answer = agent.ask(prompt, history=st.session_state.messages)

#             message_placeholder.markdown(answer)
#             st.session_state.messages.append({"role": "assistant", "content": answer})
#         except Exception as e:
#             err = f"❌ 发生错误: {e}"
#             message_placeholder.markdown(err)
#             st.session_state.messages.append({"role": "assistant", "content": err})

#     # 用户输入的 没问题
#     # if prompt := st.chat_input("请输入您的问题"):
#     #     st.session_state.messages.append({"role": "user", "content": prompt})
#     #     with st.chat_message("user"):
#     #         st.markdown(prompt)

#     #     with st.chat_message("assistant"):
#     #         message_placeholder = st.empty()

#     #         try:
#     #             user_input = st.session_state.user_input
#     #             selected_llm = user_input.get("selected_llm", "ollama")

#     #             # 由于 Streamlit 本身是同步运行，调用异步要用 asyncio.run 或 asyncio.create_task
#     #             # 但 asyncio.run 在 Streamlit 里可能有事件循环冲突，推荐使用 asyncio.run if no running loop
#     #             # 这里做简单处理：

#     #             if selected_llm == "OpenAI":
#     #                 client = mcp_clients["CloudRetrievalQA"]
#     #                 answer = asyncio.run(ask_cloud_tool(client, prompt))
#     #             else:
#     #                 client = mcp_clients["LocalRetrievalQA"]
#     #                 answer = asyncio.run(ask_local_tool(client, prompt))

#     #             message_placeholder.markdown(answer)
#     #             st.session_state.messages.append({"role": "assistant", "content": answer})

#     #         except Exception as e:
#     #             error_msg = f"❌ 发生错误: {e}"
#     #             logging.error(error_msg)
#     #             message_placeholder.markdown(error_msg)
#     #             st.session_state.messages.append({"role": "assistant", "content": error_msg})

    