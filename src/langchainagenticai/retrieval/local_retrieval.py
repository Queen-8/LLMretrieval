import sys  # 确保导入 sys 模块
import os
import glob
import logging
from ollama import Client
from langchainagenticai.utils.base_retrieval import split_documents, build_index, recall_documents, rerank_documents

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 跳出 src/langchainAgenticAi/retrieval
sys.path.append(project_root)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf_from_folder(folder_path):
    """
    从文件夹加载所有 PDF 文件，并提取文本内容。
    """
    all_documents = []
    for filename in glob.glob(os.path.join(folder_path, "*.pdf")):
        document_text = ""
        try:
            import fitz  # 用于PDF文件处理
            with fitz.open(filename) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    document_text += page.get_text("text")
        except Exception as e:
            logger.error(f"Error loading PDF {filename}: {e}")
            continue
            
        if not document_text.strip():  # 检查是否加载了有效文本
            logger.warning(f"警告：文件 {filename} 没有有效文本内容！")
        all_documents.append(document_text)
    return all_documents

def local_retrieve(query, model_name="deepseek-r1:8b", base_url="http://localhost:11434"):
    """
    使用本地知识库和 Ollama 模型进行 RAG 推理。
    """
    try:
        # 获取项目根目录的绝对路径
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, "../../.."))
        
        # 从环境变量获取本地知识库路径，如果没有则使用默认路径
        local_knowledge_base_path = os.getenv('LOCAL_KNOWLEDGE_BASE_PATH', os.path.join(project_root, 'Local_knowledge_base'))
        
        logger.info(f"使用本地知识库路径: {local_knowledge_base_path}")
        
        # 检查知识库路径是否存在
        if not os.path.exists(local_knowledge_base_path):
            logger.error(f"本地知识库路径不存在: {local_knowledge_base_path}")
            return "⚠️ 本地知识库路径不存在，请检查配置。"
        
        documents = load_pdf_from_folder(local_knowledge_base_path)
        
        if not documents:
            logger.warning("没有找到任何文档")
            return "⚠️ 本地知识库中没有找到任何文档。"
        
        logger.info(f"成功加载了 {len(documents)} 个文档")
        
        # 分片、构建索引并检索
        split_docs = split_documents(documents)
        logger.info(f"文档分片完成，共 {len(split_docs)} 个分片")
        
        index = build_index(split_docs)
        logger.info("向量索引构建完成")
        
        # 使用高精度检索
        results = recall_documents(query, index, k=8)  # 适中的检索数量，保证质量
        logger.info(f"检索到 {len(results)} 个相关文档")
        
        # 打印检索到的文档内容预览
        for i, doc in enumerate(results):
            logger.info(f"文档 {i+1} 预览: {doc.page_content[:200]}...")
        
        if not results:
            logger.warning("没有检索到相关文档")
            return "⚠️ 没有找到与问题相关的文档。"
        
        # 高精度重排
        ranked_results = rerank_documents(results, query, k=5)  # 适中的重排数量
        
        # 构建简洁的 RAG prompt，限制长度
        context_parts = []
        total_length = 0
        max_context_length = 8000  # 限制上下文长度
        
        for doc in ranked_results:
            doc_content = doc.page_content
            if total_length + len(doc_content) > max_context_length:
                # 如果添加这个文档会超出限制，就截断它
                remaining_length = max_context_length - total_length
                if remaining_length > 200:  # 确保至少有200个字符
                    doc_content = doc_content[:remaining_length] + "..."
                else:
                    break
            context_parts.append(doc_content)
            total_length += len(doc_content)
        
        context = "\n".join(context_parts)
        prompt = f"基于以下信息回答问题，请直接给出简洁的答案，不要包含思考过程：\n\n信息：{context}\n\n问题：{query}\n\n答案："
        
        logger.info(f"构建的 prompt 长度: {len(prompt)} 字符")

        # 使用 Ollama 模型进行推理
        client = Client(host=base_url)
        response = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
        
    except Exception as e:
        logger.error(f"本地检索过程中发生错误: {e}")
        return f"⚠️ 本地检索失败: {str(e)}"
