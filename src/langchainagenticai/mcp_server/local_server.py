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


# @mcp.tool(name="local_answer_question", description="Answer questions from users based on local PDF documents")
# async def local_answer_question(query: str, filter_private: bool = True) -> str:
#     """
#     从本地 PDF 文档中检索内容并回答用户问题，支持多轮对话上下文
#     """
#     global index, conversation_history, embedding_model

#     try:
#         logging.info(f"🔧 收到查询: {query}")

#         # 如果索引未初始化，先初始化
#         if index is None:
#             logging.info("🔧 正在初始化检索系统...")
#             embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
#             index = init_retrieval_pipeline(embedding=embedding_model)
#             logging.info("✅ 检索系统初始化完成")

#             # 统计文档信息
#             all_docs = list(index.docstore._dict.values())
#             private_count = sum(1 for d in all_docs if d.metadata.get("private", False))
#             logging.info(f"📊 索引文档总数: {len(all_docs)}, 其中 private 文档数: {private_count}")

#         # 过滤函数
#         def filter_func(doc):
#             if filter_private:
#                 return not (doc.metadata and doc.metadata.get("private", False))
#             return True

#         # 召回时仅用用户 query，保证语义纯净
#         logging.info("🔧 开始召回文档...")
#         retrieved_docs = recall_documents(query, index, k=10, filter_func=filter_func)

#         if not retrieved_docs:
#             logging.warning("⚠️ 未找到相关文档")
#             return "未能从本地文档中检索到相关信息。"

#         # 将历史对话拼接到生成阶段
#         context_prompt = ""
#         for q, a in conversation_history:
#             context_prompt += f"用户: {q}\n助手: {a}\n"
#         context_prompt += f"用户: {query}\n助手:"

#         # 重排
#         logging.info("🔧 开始重排文档...")
#         reranked_chunks = rerank(context_prompt, retrieved_docs, top_k=3)

#         if not reranked_chunks:
#             logging.warning("⚠️ 重排后无相关片段")
#             return "检索到了内容，但无法识别最相关片段。"

#         # 生成答案
#         logging.info("🔧 开始生成答案...")
#         answer = generate_answer(context_prompt, reranked_chunks)

#         # 缓存对话
#         conversation_history.append((query, answer))
        
#         logging.info(f"✅ 生成答案完成: {answer[:100]}...")
#         return answer or "生成答案失败，请稍后再试。"

#     except Exception as e:
#         logging.error(f"❌ 处理问题时出错: {e}")
#         return f"❌ 处理问题时出错: {e}"


# if __name__ == "__main__":
#     logging.info("🚀 启动本地 MCP 服务")
#     mcp.run(transport="stdio")


#  没问题
@mcp.tool(name="local_answer_question", description="Answer questions from users based on local PDF documents")
async def local_answer_question(query: str) -> str:
    """
    从本地 PDF 文档中检索内容并回答用户问题
    """
    try:

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
