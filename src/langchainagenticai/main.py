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