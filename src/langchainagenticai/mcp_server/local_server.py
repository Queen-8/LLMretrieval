# local_server.py
from mcp.server.fastmcp import FastMCP
import logging
from langchainagenticai.retrieval.local_retrieval import local_retrieve

logging.basicConfig(level=logging.INFO)

# 创建 FastMCP 实例
mcp = FastMCP("LocalRetrievalQA")

@mcp.tool(name="local_answer_question", description="Answer questions from users based on local PDF documents")
async def local_answer_question(query: str) -> str:
    """
    从本地PDF文件中检索文档并回答问题
    """
    try:
        logging.info("✅ 开始从本地加载PDF文档...")

        # local_retrieve 现在直接返回字符串答案
        answer = local_retrieve(query)

        logging.info("✅ 完成本地文档检索并生成回答。")
        return answer
    except Exception as e:
        logging.error(f"❌ 出错：{e}")
        return f"⚠️ 系统异常：{str(e)}"

if __name__ == "__main__":
    logging.info("🚀 启动本地 MCP 服务")
    mcp.run(transport="stdio")
