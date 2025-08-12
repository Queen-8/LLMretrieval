# loaddocs.py代码如下
import logging
import os
import glob
from dotenv import load_dotenv  # 用于加载.env环境变量
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader

# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
# from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoder
from langchain.chat_models import ChatOllama

# 建立索引并存储在向量数据库
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 .env 中的配置
load_dotenv()

folder_path = os.getenv("LOCAL_KNOWLEDGE_BASE_PATH", "Local_knowledge_base")
print("📂 加载路径:", folder_path)

# 加载 .env 中的配置
load_dotenv()
folder_path = os.getenv("LOCAL_KNOWLEDGE_BASE_PATH", "Local_knowledge_base")
print("📂 加载路径:", folder_path)


# 1. 加载 PDF 文档
def load_pdf_from_folder(folder_path):
    all_documents = []
    pdf_files = glob.glob(f"{folder_path}/*.pdf")
    print(f"📄 匹配到 {len(pdf_files)} 个 PDF 文件：", pdf_files)

    for filename in pdf_files:
        loader = PyMuPDFLoader(filename)
        documents = loader.load()
        for doc in documents:
            # 只保留 source 信息，不添加 private 标记
            doc.metadata["source"] = filename
        all_documents.extend(documents)

    return all_documents


# 2. 文本分片
def split_documents(documents, chunk_size=1000, overlap=100):
    """
    对每个文档做分片，chunk_size 控制单片长度，overlap 保证上下文连贯
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_documents(documents)



# 3. 建立向量索引（内存模式）
def build_index(documents):
    embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
    index = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="default",
        persist_directory=None
    )
    logger.info(f"✅ 向量索引创建完成，包含 {index._collection.count()} 个文档")
    return index


# 4. 初始化检索系统
def init_retrieval_pipeline():
    logger.info("🚀 正在初始化检索系统...")
    raw_documents = load_pdf_from_folder(folder_path)
    split_docs = split_documents(raw_documents)
    index = build_index(split_docs)
    logger.info(f"✅ 检索系统初始化完成，共加载 {len(split_docs)} 个文档分片")
    return index