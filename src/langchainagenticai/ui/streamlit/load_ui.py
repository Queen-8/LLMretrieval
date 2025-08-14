import streamlit as st
import os
from langchainagenticai.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self, set_page_config=True):
        self.config = Config()
        self.user_controls = {}
        self.set_page_config = set_page_config

    def load_streamlit_ui(self):
        """
        加载并展示 Streamlit UI
        """
        if self.set_page_config:
            page_title = self.config.get_page_title() or "Chatbot"
            st.set_page_config(page_title="🤖 " + page_title, layout="wide")
            st.header("🤖 " + page_title)

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            # 选择 LLM，默认选中 ollama
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options, index=llm_options.index("ollama") if "ollama" in llm_options else 0, key="llm_selectbox")

            # 根据所选 LLM 动态显示模型选项
            if self.user_controls["selected_llm"] == 'OpenAI':
                model_options = self.config.get_openai_model_options()
                self.user_controls["selected_model"] = st.selectbox("Select Model", model_options, key="openai_model_selectbox")
                # API key 从环境变量读取，不需要用户输入
                st.info("🔑 API Key 将从环境变量读取")
            elif self.user_controls["selected_llm"] == 'ollama':
                model_options = self.config.get_ollama_model_options()
                default_model = "llama3.1:8b" if "llama3.1:8b" in model_options else model_options[0]
                self.user_controls["selected_model"] = st.selectbox("Select Model", model_options, index=model_options.index(default_model), key="ollama_model_selectbox")
                # ollama 本地模型一般不需要 API Key，可根据实际需求添加
            # 完全移除 usecase 相关内容
            # 返回 user_controls
            return self.user_controls
