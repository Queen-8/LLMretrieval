import os
import logging
from dotenv import load_dotenv
import json

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 .env
load_dotenv()

def load_urls_from_env():
    """从 .env 文件读取 CLOUD_URLS"""
    urls_str = os.getenv("CLOUD_URLS", "[]")
    try:
        urls = json.loads(urls_str) if urls_str.startswith("[") else [u.strip() for u in urls_str.split(",") if u.strip()]
        logger.info(f"🌐 从环境变量加载 {len(urls)} 个 URL")
        return urls
    except Exception as e:
        logger.error(f"❌ 解析 CLOUD_URLS 失败: {e}")
        return []


def fetch_url_content(urls):
    """抓取网页内容"""
    all_documents = []
    for url in urls:
        try:
            logger.info(f"🔍 抓取 URL: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"❌ 抓取 {url} 失败: {e}")
    return all_documents


def split_documents(documents, chunk_size=800, overlap=120):
    """对网页文档进行分片"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_documents(documents)


def build_index(documents):
    """向量化并构建临时索引（内存数据库）"""
    embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
    doc_objects = [Document(page_content=str(doc.page_content), metadata=doc.metadata) for doc in documents]
    index = Chroma.from_documents(
        documents=doc_objects,
        embedding=embeddings,
        collection_name="cloud_urls",
        persist_directory=None
    )
    logger.info(f"✅ URL 向量索引创建完成，共 {index._collection.count()} 个分片")
    return index


def init_url_retrieval_pipeline():
    """初始化 URL 检索流程（可在 app.py 启动时调用）"""
    logger.info("🚀 正在初始化 URL 检索系统...")
    urls = load_urls_from_env()
    if not urls:
        logger.warning("⚠️ 未配置任何 CLOUD_URLS")
        return None

    raw_docs = fetch_url_content(urls)
    if not raw_docs:
        logger.warning("⚠️ 没有抓取到任何网页内容")
        return None

    split_docs = split_documents(raw_docs)
    index = build_index(split_docs)
    logger.info(f"✅ URL 检索系统初始化完成，分片数: {len(split_docs)}")
    return index