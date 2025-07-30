import logging
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_documents(documents, chunk_size=1000, overlap=100):
    """
    将文档分片，返回分片后的文档列表。
    """
    doc_objects = [Document(page_content=doc) for doc in documents]
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    
    split_docs = text_splitter.split_documents(doc_objects)
    
    # 调试输出
    for idx, doc in enumerate(split_docs):
        if not doc.page_content.strip():
            logger.warning(f"警告：分片 {idx} 的内容为空！")
    
    return split_docs


def build_index(documents):
    """
    将文档转换为向量并构建索引。
    """
    embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
    doc_objects = [Document(page_content=str(doc)) for doc in documents]
    
    # 直接使用 Chroma.from_documents，它会自动处理嵌入
    index = Chroma.from_documents(doc_objects, embeddings)
    return index


def recall_documents(query, index, k=10):
    """
    执行高精度相似性检索并返回与查询最相关的 k 个文档。
    使用精确的相似度阈值和多重过滤策略。
    """
    # 使用更大的初始检索数量，然后进行精确过滤
    initial_k = min(k * 3, 50)  # 最多检索50个文档
    results_with_scores = index.similarity_search_with_score(query, k=initial_k)
    
    # 根据相似度分数进行精确过滤
    # 分数越低表示越相似，我们设置一个合理的阈值
    high_quality_results = []
    medium_quality_results = []
    
    for doc, score in results_with_scores:
        if score < 0.8:  # 高质量匹配
            high_quality_results.append((doc, score))
        elif score < 1.2:  # 中等质量匹配
            medium_quality_results.append((doc, score))
    
    # 优先返回高质量结果，如果不够则补充中等质量结果
    final_results = []
    
    # 添加高质量结果
    high_quality_results.sort(key=lambda x: x[1])  # 按分数排序
    final_results.extend([doc for doc, _ in high_quality_results[:k//2]])
    
    # 如果高质量结果不够，补充中等质量结果
    if len(final_results) < k:
        remaining_slots = k - len(final_results)
        medium_quality_results.sort(key=lambda x: x[1])
        final_results.extend([doc for doc, _ in medium_quality_results[:remaining_slots]])
    
    # 如果还是没有足够的结果，返回原始结果的前k个
    if len(final_results) < k//2:
        final_results = [doc for doc, _ in results_with_scores[:k]]
    
    return final_results[:k]

def rerank_documents(results, query, k=5):
    """
    对召回的文档进行高精度重排。
    使用多种策略进行排序，确保最相关的文档排在前面。
    """
    if not results:
        return []
    
    # 计算每个文档与查询的相关性分数
    scored_docs = []
    
    for doc in results:
        score = calculate_relevance_score(doc, query)
        scored_docs.append((doc, score))
    
    # 按相关性分数排序（分数越高越相关）
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前k个文档
    return [doc for doc, _ in scored_docs[:k]]

def calculate_relevance_score(doc, query):
    """
    计算文档与查询的相关性分数。
    使用多种指标综合评估相关性。
    """
    content = doc.page_content.lower()
    query_lower = query.lower()
    
    # 1. 关键词匹配分数
    query_words = set(query_lower.split())
    content_words = set(content.split())
    keyword_overlap = len(query_words.intersection(content_words))
    keyword_score = keyword_overlap / max(len(query_words), 1)
    
    # 2. 短语匹配分数
    phrase_score = 0
    for word in query_words:
        if len(word) > 2 and word in content:
            phrase_score += 1
    phrase_score = phrase_score / max(len(query_words), 1)
    
    # 3. 文档长度归一化分数（避免长文档获得不公平优势）
    length_penalty = min(len(content) / 1000, 1.0)  # 文档越长，分数稍微降低
    
    # 4. 综合分数
    total_score = (keyword_score * 0.6 + phrase_score * 0.4) * (1 - length_penalty * 0.1)
    
    return total_score
