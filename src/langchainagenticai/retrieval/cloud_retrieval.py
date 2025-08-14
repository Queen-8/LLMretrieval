import logging
from sentence_transformers import CrossEncoder
# from langchain.chat_models import ChatOpenAI  弃用
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI 弃用
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

def recall_documents(query: str, index, k=5, filter_func=None):
    """
    从指定向量索引中召回文档
    """
    docs = index.similarity_search(query, k=k)
    
    if filter_func:
        docs = [doc for doc in docs if filter_func(doc)]
    
    return docs

def rerank(query, retrieved_docs, top_k: int):
    """
    使用 CrossEncoder 对召回的文档片段进行重排
    """
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    scores = cross_encoder.predict(pairs)
    
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, _ in scored_chunks][:top_k]

def generate_answer_openai(query, reranked_chunks, api_key, model_name="gpt-3.5-turbo"):
    """
    基于重排后的 chunks 和 OpenRouter API 生成答案
    """
    documents_content = "\n".join(reranked_chunks)
    
    prompt = (
        "你是一个智能助手，请根据以下内容回答用户问题：\n\n"
        f"{documents_content}\n\n用户提问：{query}\n\n请用简洁准确的语言作答："
    )
    
    # 使用 OpenRouter API
    llm = ChatOpenAI(
        model=model_name, 
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7
    )
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"生成答案出错: {e}")
        return None
