# local_server.py
import os
os.environ.setdefault("USER_AGENT", "LangChain-AgenticAI/1.0 (contact: 564752114@qq.com)")

from mcp.server.fastmcp import FastMCP
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_huggingface import HuggingFaceEmbeddings
from src.langchainagenticai.retrieval.local_retrieval import recall_documents, rerank, generate_answer
from src.langchainagenticai.utils.loaddocs import init_retrieval_pipeline

logging.basicConfig(level=logging.INFO)

# 创建 FastMCP 实例
mcp = FastMCP("LocalRetrievalQA")

# 全局变量
conversation_history = []
index = None
embedding_model = None  # 确保全局使用同一个模型



@mcp.tool(
    name="local_answer_question", 
    description=(
        "用途：回答与本地存储的 PDF 文档、内部资料、离线文件相关的问题。"
        "适合场景：当用户的问题涉及本地文件、公司内部手册、项目文档、培训资料等。"
        "输入：JSON 格式，包含 query 字段，例如：{{\"query\": \"用户问题\"}}"
        "限制：不适合回答实时互联网数据的问题。"
    )
)
async def local_answer_question(input_data: dict) -> str:
    """
    从本地 PDF 文档中检索内容并回答用户问题
    """
    try:
        # 处理输入参数
        if isinstance(input_data, dict) and "query" in input_data:
            query = input_data["query"]
        else:
            return "❌ 参数格式错误，期望包含 query 字段的字典"
        
        logging.info(f"🔧 收到查询: {query}")
        
        # 步骤1：召回相关文档
        logging.info("🔧 开始召回文档...")
        retrieved_docs = recall_documents(query, k=10)

        if not retrieved_docs:
            return "未能从本地文档中检索到相关信息。"

        # 步骤2：重排 Top-k 文档片段
        logging.info("🔧 开始重排文档...")
        reranked_chunks = rerank(query, retrieved_docs, top_k=3)

        if not reranked_chunks:
            return "检索到了内容，但无法识别最相关片段。"

        # 步骤3：生成答案
        logging.info("🔧 调用 Ollama 开始生成答案...")
        answer = generate_answer(query, reranked_chunks)

        return answer or "生成答案失败，请稍后再试。"

    except Exception as e:
        return f"❌ 处理问题时出错: {e}"


if __name__ == "__main__":
    logging.info("🚀 启动本地 MCP 服务")
    mcp.run(transport="stdio")
