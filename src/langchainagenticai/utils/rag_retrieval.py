import os
import glob
import json
import logging
import fitz  
import asyncio
from pathlib import Path
from dotenv import load_dotenv  # 用于加载.env环境变量

from langchain.llms import OpenAI  # OpenAI LLM调用
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_community.vectorstores import Chroma # 向量数据库
from langchain_community.embeddings import SentenceTransformerEmbeddings # 向量化
from langchain_community.document_loaders import PyPDFLoader # PyMuPDF, 用于PDF文本提取
from langchain.text_splitter import CharacterTextSplitter # 文本分片
from langchain.schema import Document



# 加载本地.env文件中的环境变量（如API KEY）
load_dotenv()

# 读取OpenAI密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 设置使用的模型（OpenRouter上的GPT-4o-mini）
OPENAI_MODEL = "gpt-4o-mini"

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf_from_folder(folder_path):
    """
    Load all the PDF files in the specified folder, 
    returning the full text content of each file (string list).
    """
    all_documents = []
    # Iterate over all pdf files
    for filename in glob.glob(f"Local_knowledge_base/*.pdf"):
        document_text = ""
        # Open the PDF and iterate over each page, accumulating the text
        with fitz.open(filename) as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                document_text += page.get_text("text")
        all_documents.append(document_text)
    return all_documents



from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

def split_documents(documents, chunk_size=1000, overlap=100):
    """
    对每个文档做分片，chunk_size控制单片长度，overlap保证上下文连贯
    输入: 字符串list，输出: 分片list
    """
    # 将字符串列表转换为Document对象列表
    doc_objects = [Document(page_content=doc) for doc in documents]
    
    # 创建text_splitter对象
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    
    # 分片并返回结果
    return text_splitter.split_documents(doc_objects)




def build_index(documents):
    """
    Convert the sharded documents into vectors and index them
    """
    embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
    
    # 将文档列表转换为Document对象（确保doc是字符串类型）
    doc_objects = [Document(page_content=str(doc)) for doc in documents]
    
    # 使用Chroma创建向量索引
    index = Chroma.from_documents(doc_objects, embeddings)
    
    return index



def recall_documents(query, index, k=5):
    """
    A similarity retrieval is performed and the k most similar documents to the query are returned
    """
    return index.similarity_search(query, k=k)





# 加载项目根目录下的 .env 文件中的 OPENROUTER_API_KEY
root_dir = Path().resolve().parent
load_dotenv(dotenv_path=root_dir / ".env")
# 从 .env 读取 OPENROUTER_API_KEY，并设置成 OpenAI 兼容变量
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 获取 OPENROUTER_API_KEY 环境变量
api_key = os.getenv("OPENROUTER_API_KEY")

def rerank_documents(query, retrieved_docs):
    """
    Reorder retrieved documents using LLM or models
    """
    
    # 使用ChatOpenAI进行重排
    llm = ChatOpenAI(openai_api_key=api_key, model="openai/gpt-4o-mini")
    scores = []
    
    # 为每个文档生成相关性分数
    for doc in retrieved_docs:
        prompt = f"Query: {query}\nDocument: {doc.page_content}\n"
        
        # 构建适合ChatOpenAI的消息格式
        messages = [
            {"role": "system", "content": "You are an assistant helping to rank documents based on relevance."},
            {"role": "user", "content": prompt}
        ]
        
        # 获取相关性分数，确保传递正确的消息列表格式
        response = llm.invoke(messages)  # 确保使用 invoke 来调用
        
        # 提取模型生成的内容（直接访问 content 属性）
        score = response.content  # 获取生成的文本
        
        scores.append((score, doc))
    
    # 按照分数对文档进行排序
    sorted_docs = sorted(scores, key=lambda x: x[0], reverse=True)
    
    # 返回排序后的文档
    return [doc[1] for doc in sorted_docs]




# 加载项目根目录下的 .env 文件中的 OPENROUTER_API_KEY
root_dir = Path().resolve().parent
load_dotenv(dotenv_path=root_dir / ".env")

# 从 .env 读取 OPENROUTER_API_KEY，并设置成 OpenAI 兼容变量
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 获取 OPENROUTER_API_KEY 环境变量
api_key = os.getenv("OPENROUTER_API_KEY")

def generate_answer(query, top_docs):
    """
    基于查询和重排后的文档生成最终答案
    """
    # 将文档内容拼接为一个字符串（确保传递给模型的是一个长文本）
    documents_content = "\n".join([doc.page_content for doc in top_docs])
    
    # 格式化prompt，确保传递给模型的是一个字符串
    prompt = f"Answer the following question based on the retrieved documents:\n{documents_content}\n\nQuestion: {query}\nAnswer:"
    
    # 使用ChatOpenAI进行生成
    llm = ChatOpenAI(openai_api_key=api_key, model="openai/gpt-4o-mini")
    
    try:
        # 调用模型生成答案
        response = llm.invoke([{"role": "system", "content": "You are an assistant helping to answer questions based on documents."},
                               {"role": "user", "content": prompt}])
        
        # 获取生成的答案（通常是从 response.content 中提取）
        answer = response.content.strip()  # 提取生成的文本，去除多余的空格
        
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None
