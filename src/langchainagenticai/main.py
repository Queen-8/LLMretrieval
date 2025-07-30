import streamlit as st
from langchainagenticai.ui.streamlit.load_ui import LoadStreamlitUI
from langchainagenticai.ui.streamlit.display_result import DisplayResultStreamlit
from langchain_openai import ChatOpenAI
from langchainagenticai.mcp_server.local_server import local_answer_question
import asyncio
import os
from pydantic import SecretStr
import json
import requests
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 跳出 src/langchainagenticai/retrieval
sys.path.append(project_root)


def load_langgraph_agenticai_app():
    st.title("LangChain Agentic AI")
    
    # 初始化消息历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 侧边栏功能
    with st.sidebar:
        st.header("模型配置")
        
        # 加载UI配置
        ui = LoadStreamlitUI()
        user_input = ui.load_streamlit_ui()
        
        st.header("对话管理")
        
        # 显示消息统计
        user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
        st.metric("用户消息", user_messages)
        st.metric("助手回复", assistant_messages)
        
        # 清除历史按钮
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.rerun()
        
        # 导出对话功能
        if st.session_state.messages:
            if st.button("📥 导出对话"):
                # 生成对话文本
                conversation_text = ""
                for msg in st.session_state.messages:
                    role = "用户" if msg["role"] == "user" else "助手"
                    conversation_text += f"{role}: {msg['content']}\n\n"
                
                # 提供下载
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
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 显示助手消息
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # 获取选择的模型信息
                selected_llm = user_input.get("selected_llm")
                selected_model = user_input.get("selected_model")
                
                if selected_llm == "OpenAI":
                    api_key = user_input.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
                    if not api_key:
                        st.error("⚠️ 请提供 OpenAI API Key")
                        return
                    
                    model = ChatOpenAI(
                        model=selected_model,
                        base_url="https://openrouter.ai/api/v1",
                        api_key=SecretStr(api_key),
                        max_tokens=4000,  # 限制最大 token 数量
                        temperature=0.7,
                    )
                    
                    response = model.invoke(prompt)
                    response_content = getattr(response, "content", response)
                    message_placeholder.markdown(response_content)
                    
                    # 添加助手消息到历史
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    
                elif selected_llm == "ollama":
                    answer = asyncio.run(local_answer_question(prompt))
                    message_placeholder.markdown(answer)
                    
                    # 添加助手消息到历史
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                else:
                    st.error(f"⚠️ 不支持的LLM类型: {selected_llm}")
                    
            except Exception as e:
                error_message = f"❌ 发生错误: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
