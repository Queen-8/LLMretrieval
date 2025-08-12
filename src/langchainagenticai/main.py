import os
import sys
import asyncio
import streamlit as st
from pydantic import SecretStr
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchainagenticai.utils.loaddocs import init_retrieval_pipeline
from langchainagenticai.utils.loadurls import init_url_retrieval_pipeline
from langchainagenticai.ui.streamlit.load_ui import LoadStreamlitUI
from langchainagenticai.ui.streamlit.display_result import DisplayResultStreamlit
from langchain_openai import ChatOpenAI
from langchainagenticai.mcp_server.local_server import local_answer_question
from langchainagenticai.mcp_server.cloud_server import cloud_answer_question
from pydantic import SecretStr
import json
import requests       
from langchain_openai import OpenAI

from langchain_community.chat_models import ChatOllama
from langchain.chat_models import ChatOllama
import logging

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 跳出 src/langchainagenticai/retrieval
sys.path.append(project_root)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# MCP 客户端初始化（全局只创建一次）
mcp_clients = {
    "CloudRetrievalQA": MultiServerMCPClient(
        connections={
            "CloudRetrievalQA": {
                "command": "python",
                "args": ["src/langchainagenticai/mcp_server/cloud_server.py"],
                "transport": "stdio",
            }
        }
    ),
    "LocalRetrievalQA": MultiServerMCPClient(
        connections={
            "LocalRetrievalQA": {
                "command": "python",
                "args": ["src/langchainagenticai/mcp_server/local_server.py"],
                "transport": "stdio",
            }
        }
    ),
}

async def ask_cloud_tool(client, query):
    tools = await client.get_tools()
    cloud_tool = next((t for t in tools if t.name == "cloud_answer_question"), None)
    if not cloud_tool:
        return "❌ 未找到 cloud_answer_question 工具"
    result = await cloud_tool.ainvoke({"query": query})
    return result

async def ask_local_tool(client, query):
    tools = await client.get_tools()
    local_tool = next((t for t in tools if t.name == "local_answer_question"), None)
    if not local_tool:
        return "❌ 未找到 local_answer_question 工具"
    result = await local_tool.ainvoke({"query": query})
    return result

def load_langgraph_agenticai_app():
    st.title("🤖 LangChain Agentic AI")

    # 初始化知识库索引（只执行一次）
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

    # 初始化消息历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 侧边栏
    with st.sidebar:
        st.header("模型配置")
        
        # 使用 LoadStreamlitUI 来显示完整的配置界面
        ui = LoadStreamlitUI(set_page_config=False)
        st.session_state.user_input = ui.load_streamlit_ui()

        st.header("对话管理")
        user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
        st.metric("用户消息", user_messages)
        st.metric("助手回复", assistant_messages)

        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.rerun()
            
        if st.session_state.messages:
            if st.button("📥 导出对话"):
                conversation_text = ""
                for msg in st.session_state.messages:
                    role = "用户" if msg["role"] == "user" else "助手"
                    conversation_text += f"{role}: {msg['content']}\n\n"

                st.download_button(
                    label="下载对话记录",
                    data=conversation_text,
                    file_name=f"conversation_{st.session_state.get('session_id', 'unknown')}.txt",
                    mime="text/plain"
                )

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("请输入您的问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                user_input = st.session_state.user_input
                selected_llm = user_input.get("selected_llm", "ollama")

                # 由于 Streamlit 本身是同步运行，调用异步要用 asyncio.run 或 asyncio.create_task
                # 但 asyncio.run 在 Streamlit 里可能有事件循环冲突，推荐使用 asyncio.run if no running loop
                # 这里做简单处理：

                if selected_llm == "OpenAI":
                    client = mcp_clients["CloudRetrievalQA"]
                    answer = asyncio.run(ask_cloud_tool(client, prompt))
                else:
                    client = mcp_clients["LocalRetrievalQA"]
                    answer = asyncio.run(ask_local_tool(client, prompt))

                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"❌ 发生错误: {e}"
                logging.error(error_msg)
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# # 初始化 MCP 客户端（全局，只创建一次）
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





# def load_langgraph_agenticai_app():
#     st.title("🤖 LangChain Agentic AI")

#     # 初始化向量索引，仅在首次加载时执行一次
#     if "vector_index" not in st.session_state:
#         with st.spinner("📂 正在初始化本地知识库索引..."):
#             try:
#                 st.session_state.vector_index = init_retrieval_pipeline()
#                 st.success("📂✅ 本地知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ 知识库初始化失败: {str(e)}")
#                 return  # 若初始化失败，终止启动流程
#     if "vector_index_urls" not in st.session_state:
#         with st.spinner("🌐 正在初始化 URL 知识库索引..."):
#             try:
#                 st.session_state.vector_index_urls = init_url_retrieval_pipeline()
#                 st.success("🌐✅ URL 知识库加载完成")
#             except Exception as e:
#                 st.error(f"❌ URL 知识库初始化失败: {str(e)}")
    
#     # 初始化消息历史
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     # 侧边栏功能
#     with st.sidebar:
#         st.header("模型配置")
        
#         # 加载UI配置
#         ui = LoadStreamlitUI()
#         user_input = ui.load_streamlit_ui()
        
#         st.header("对话管理")
        
#         # 显示消息统计
#         user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
#         assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
#         st.metric("用户消息", user_messages)
#         st.metric("助手回复", assistant_messages)
        
#         # 清除历史按钮
#         if st.button("🗑️ 清除对话历史"):
#             st.session_state.messages = []
#             st.rerun()
        
#         # 导出对话功能
#         if st.session_state.messages:
#             if st.button("📥 导出对话"):
#                 # 生成对话文本
#                 conversation_text = ""
#                 for msg in st.session_state.messages:
#                     role = "用户" if msg["role"] == "user" else "助手"
#                     conversation_text += f"{role}: {msg['content']}\n\n"
                
#                 # 提供下载
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
    
#     # 用户输入
#     if prompt := st.chat_input("请输入您的问题"):

#         # ✅ 优先构造 RAG 上下文
#         if "vector_index" in st.session_state:
#         # 使用向量索引做 semantic search
#             rag_results = st.session_state.vector_index.similarity_search(prompt, k=3)
#             context = "\n".join([doc.page_content for doc in rag_results])
#             prompt_with_context = f"根据以下知识内容回答问题：\n{context}\n\n问题：{prompt}"
#         else:
#             prompt_with_context = prompt

#         # 添加用户消息到历史
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # 显示用户消息
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # 显示助手消息
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
            
#             try:
#                 # 获取选择的模型信息
#                 selected_llm = user_input.get("selected_llm")
#                 selected_model = user_input.get("selected_model")
                
#                 if selected_llm == "OpenAI":
#                     api_key = (
#                         user_input.get("OPENAI_API_KEY")
#                         or os.getenv("OPENAI_API_KEY")
#                         or os.getenv("OPENROUTER_API_KEY")
#                     )
#                     if not api_key:
#                         st.error("⚠️ 请提供 OpenAI API Key")
#                         return
                    
#                     # 初始化 MCP 客户端
#                     client = MultiServerMCPClient(
#                         connections={
#                             "CloudRetrievalQA": {
#                                 "command": "python",
#                                 "args": ["src/langchainagenticai/mcp_server/cloud_server.py"],
#                                 "transport": "stdio",
#                             }
#                         }
#                     )
#                     async def ask_cloud_tool(history, context, api_key):
#                         tools = await client.get_tools()
#                         cloud_tool = next((t for t in tools if t.name == "cloud_answer_question"), None)
#                         if not cloud_tool:
#                             return "❌ 未找到 cloud_answer_question 工具"
#                         # 传递 api_key 作为参数
#                         result = await cloud_tool.ainvoke({"history": history, "context": context, "api_key": api_key})
#                         return result

#                     # 调用示例（同步）
#                     answer = asyncio.run(ask_cloud_tool(st.session_state.messages, context, api_key))
#                     message_placeholder.markdown(answer)
#                     st.session_state.messages.append({"role": "assistant", "content": answer})
                    
#                 elif selected_llm == "ollama":

#                     # 初始化 MCP 客户端
#                     client = MultiServerMCPClient(
#                         connections={
#                             "LocalRetrievalQA": {
#                                 "command": "python",
#                                 "args": ["src/langchainagenticai/mcp_server/local_server.py"],
#                                 "transport": "stdio",
#                             }
#                         }
#                     )

#                     async def ask_local_tool(query: str) -> str:
#                         # 获取工具
#                         tools = await client.get_tools()
#                         # 找到 local_answer_question 工具
#                         local_tool = None
#                         for tool in tools:
#                             if tool.name == "local_answer_question":
#                                 local_tool = tool
#                                 break
                        
#                         if local_tool:
#                             result = await local_tool.ainvoke({"query": query})
#                             return result
#                         else:
#                             return "❌ 未找到 local_answer_question 工具"

#                     # 在 Streamlit 或其他前端中调用
#                     answer = asyncio.run(ask_local_tool(prompt))

#                     message_placeholder.markdown(answer)
                    
#                     # 添加助手消息到历史
#                     st.session_state.messages.append({"role": "assistant", "content": answer})
                    
#                 else:
#                     st.error(f"⚠️ 不支持的LLM类型: {selected_llm}")
                
                    
#             except Exception as e:
#                 error_message = f"❌ 发生错误: {str(e)}"
#                 message_placeholder.markdown(error_message)
#                 st.session_state.messages.append({"role": "assistant", "content": error_message})
