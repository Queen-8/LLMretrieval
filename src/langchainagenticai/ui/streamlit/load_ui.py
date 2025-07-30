import streamlit as st
import os
from langchainagenticai.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        """
        加载并展示 Streamlit UI
        """
        page_title = self.config.get_page_title() or "Chatbot"
        st.set_page_config(page_title="🤖 " + page_title, layout="wide")
        st.header("🤖 " + page_title)

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            # 选择 LLM，默认选中 ollama
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options, index=llm_options.index("ollama") if "ollama" in llm_options else 0)

            # 根据所选 LLM 动态显示模型选项
            if self.user_controls["selected_llm"] == 'OpenAI':
                model_options = self.config.get_openai_model_options()
                self.user_controls["selected_model"] = st.selectbox("Select Model", model_options)
                self.user_controls["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"] = st.text_input("OpenAI API Key", type="password")
                if not self.user_controls["OPENAI_API_KEY"]:
                    st.warning("⚠️ Please enter your OpenAI API key to proceed. Don't have? Refer to: https://platform.openai.com/account/api-keys ")
            elif self.user_controls["selected_llm"] == 'ollama':
                model_options = self.config.get_ollama_model_options()
                default_model = "deepseek-r1:8b" if "deepseek-r1:8b" in model_options else model_options[0]
                self.user_controls["selected_model"] = st.selectbox("Select Model", model_options, index=model_options.index(default_model))
                # ollama 本地模型一般不需要 API Key，可根据实际需求添加
            # 完全移除 usecase 相关内容
            # 返回 user_controls
            return self.user_controls
