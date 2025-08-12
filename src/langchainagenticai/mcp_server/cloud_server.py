# cloud_server.py
from mcp.server.fastmcp import FastMCP
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.langchainagenticai.retrieval.cloud_retrieval import recall_documents, rerank, generate_answer_openai
from src.langchainagenticai.utils.loadurls import init_url_retrieval_pipeline

logging.basicConfig(level=logging.INFO)

# 创建 FastMCP 实例
mcp = FastMCP("CloudRetrievalQA")

# 全局变量
conversation_history = []
index = None


@mcp.tool(name="cloud_answer_question", description="Answer questions from cloud URL documents")
async def cloud_answer_question(query: str) -> str:
    """
    从云端 URL 文档中检索内容并回答用户问题，支持多轮对话上下文
    """
    global index, conversation_history

    try:
        logging.info(f"🌐 收到云端查询: {query}")

        # 如果索引未初始化，先初始化
        if index is None:
            logging.info("🌐 正在初始化云端检索系统...")
            index = init_url_retrieval_pipeline()
            if index is None:
                return "⚠️ 云端文档索引未能初始化，请检查 CLOUD_URLS 配置"
            logging.info("✅ 云端检索系统初始化完成")

        # 构造上下文
        context_prompt = ""
        for q, a in conversation_history:
            context_prompt += f"用户: {q}\n助手: {a}\n"
        context_prompt += f"用户: {query}\n助手:"

        # Step1: 召回
        logging.info("🌐 开始召回云端文档...")
        retrieved_docs = recall_documents(context_prompt, index, k=10)

        if not retrieved_docs:
            logging.warning("⚠️ 未找到相关云端文档")
            return "未能从云端文档中检索到相关信息。"

        # Step2: 重排
        logging.info("🌐 开始重排云端文档...")
        reranked_chunks = rerank(context_prompt, retrieved_docs, top_k=3)

        if not reranked_chunks:
            logging.warning("⚠️ 重排后无相关云端片段")
            return "检索到了内容，但无法识别最相关片段。"

        # Step3: 生成答案
        logging.info("🌐 调用 OpenRouter 生成答案...")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "⚠️ 未在环境变量中设置 OPENAI_API_KEY 或 OPENROUTER_API_KEY"

        answer = generate_answer_openai(context_prompt, reranked_chunks, api_key=api_key)
        
        if answer is None:
            return "❌ 生成答案失败，请检查 API key 是否正确或稍后再试"

        # 缓存对话
        conversation_history.append((query, answer))

        logging.info(f"✅ 云端答案生成完成: {answer[:100]}...")
        return answer

    except Exception as e:
        logging.error(f"❌ 处理云端问题时出错: {e}")
        return f"❌ 处理云端问题时出错: {e}"


if __name__ == "__main__":
    logging.info("🚀 启动云端 MCP 服务")
    mcp.run(transport="stdio")
