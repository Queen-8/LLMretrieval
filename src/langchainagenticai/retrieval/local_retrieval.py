# local_retrieval.py 代码如下
import sys  # 确保导入 sys 模块
import os
import glob
import logging
from ollama import Client
# from langchainagenticai.utils.base_retrieval import split_documents, build_index, recall_documents, rerank_documents
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOllama
from sentence_transformers import CrossEncoder
# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain.chat_models import ChatOllama
# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 跳出 src/langchainAgenticAi/retrieval
sys.path.append(project_root)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 全局缓存
index = None


from src.langchainagenticai.utils.loaddocs import init_retrieval_pipeline  # 用loaddocs.py



def get_index():
    global index
    if index is None:
        index = init_retrieval_pipeline()
    return index

def recall_documents(query: str, k: int = 5):
    """
    直接基于相似度检索，返回前 k 篇最相似文档
    """
    index = get_index()
    results = index.similarity_search(query, k=k)

    logger.info(f"🔍 检索到 {len(results)} 篇文档（Top {k}）：")
    for i, doc in enumerate(results, 1):
        logger.info(f"{i}. {doc.page_content[:50]}... 来源: {doc.metadata.get('source')}")
    return results


def rerank(query: str, retrieved_docs, top_k: int):
    """
    使用 CrossEncoder 对检索到的文档进行重排
    """
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]
    pairs = [(query, chunk) for chunk in retrieved_chunks]

    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"📊 重排后选取前 {top_k} 段")
    return [chunk for chunk, _ in scored_chunks][:top_k]


def generate_answer(query, reranked_chunks):
    """
    基于查询和重排后的 chunk 文本生成最终答案
    """
    documents_content = "\n".join(reranked_chunks)

    prompt = (
        "你是一个智能助手，请根据以下内容回答用户问题：\n\n"
        f"{documents_content}\n\n用户提问：{query}\n\n请用简洁准确的语言作答："
    )

    llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b")
    print(f"DEBUG: llm instance = {llm}")

    try:
        response = llm.invoke(prompt)
        print(f"DEBUG: raw response = {response}; type = {type(response)}")

        if isinstance(response, str):
            return response.strip()
        elif isinstance(response, list) and len(response) > 0:
            first = response[0]
            if isinstance(first, str):
                return first.strip()
            elif isinstance(first, dict):
                content = first.get("content") or first.get("text") or first.get("answer")
                if isinstance(content, str):
                    return content.strip()
                else:
                    return str(first).strip()
            else:
                return str(first).strip()
        elif hasattr(response, "content") and isinstance(response.content, str):
            return response.content.strip()
        elif hasattr(response, "text") and isinstance(response.text, str):
            return response.text.strip()
        else:
            return str(response).strip()

    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return None



# def recall_documents(query : str, index, k=5):
#     """
#     A similarity retrieval is performed and the k most similar documents to the query are returned
#     """
#     return index.similarity_search(query, k=k)

# def rerank(query, retrieved_docs, top_k: int):
#     """
#     Rerank the retrieved document chunks using a cross-encoder model and return the top_k contents
#     """
#     # 提取每个 Document 的内容
#     retrieved_chunks = [doc.page_content for doc in retrieved_docs]
    
#     # 创建 query-chunk 对
#     pairs = [(query, chunk) for chunk in retrieved_chunks]

#     # 使用 CrossEncoder 进行打分
#     cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
#     scores = cross_encoder.predict(pairs)

#     # 将 chunk 与分数组成元组，并按分数降序排序
#     scored_chunks = list(zip(retrieved_chunks, scores))
#     scored_chunks.sort(key=lambda x: x[1], reverse=True)

#     # 返回 Top-K 重排后的 chunk 文本
#     return [chunk for chunk, _ in scored_chunks][:top_k]


# def generate_answer(query, reranked_chunks):
#     """
#     基于查询和重排后的 chunk 文本生成最终答案
#     """
#     documents_content = "\n".join(reranked_chunks)

#     prompt = (
#         "你是一个智能助手，请根据以下内容回答用户问题：\n\n"
#         f"{documents_content}\n\n用户提问：{query}\n\n请用简洁准确的语言作答："
#     )

#     llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b")
#     print(llm)

#     try:
#         response = llm.invoke(prompt)
#         return response.content.strip()
#     except Exception as e:
#         print(f"❌ Error generating answer: {e}")
#         return None