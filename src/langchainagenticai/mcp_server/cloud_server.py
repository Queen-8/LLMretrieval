# cloud_server.py
from mcp.server.fastmcp import FastMCP
import logging
import os
from .retrieval.cloud_retrieval import cloud_retrieve
from .utils.base_retrieval import split_documents, build_index, recall_documents

logging.basicConfig(level=logging.INFO)

# 创建 FastMCP 实例
mcp = FastMCP("CloudRetrievalQA")

@mcp.tool(name="cloud_answer_question", description="Answer questions from users based on cloud documents")
async def cloud_answer_question(query: str) -> str:
    """
    从云端检索文档并回答问题
    """
    try:
        logging.info("✅ 开始从云端获取文档...")

        # 从环境变量中获取云端 URLs
        cloud_urls = os.getenv('CLOUD_URLS', '[]')  # 默认值为空数组
        cloud_urls = eval(cloud_urls)

        # 云端检索
        results = cloud_retrieve(query, cloud_urls)

        # 提取文本内容返回
        answer = "\n".join([doc.page_content for doc in results]) if results else "⚠️ 无法找到相关文档。"

        logging.info("✅ 完成云端文档检索并生成回答。")
        return answer
    except Exception as e:
        logging.error(f"❌ 出错：{e}")
        return f"⚠️ 系统异常：{str(e)}"

if __name__ == "__main__":
    logging.info("🚀 启动云端 MCP 服务")
    mcp.run(transport="stdio")
