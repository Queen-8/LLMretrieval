import sys  # 确保导入 sys 模块
import logging
import requests
from langchain_openai import ChatOpenAI
from langchainagenticai.utils.base_retrieval import split_documents, build_index, recall_documents, rerank_documents

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 跳出 src/langchainAgenticAi/retrieval
sys.path.append(project_root)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cloud_retrieve(query, cloud_urls, api_key):
    """
    使用云端 URL 数组进行文档检索，API 密钥从前端传递。
    """
    documents = []
    for url in cloud_urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                documents.append(response.text)
            else:
                logger.warning(f"无法从 {url} 获取文档，状态码：{response.status_code}")
        except Exception as e:
            logger.error(f"从 {url} 获取文档时出错：{str(e)}")

    if not documents:
        logger.error("没有从云端检索到文档。")
        return []

    # 分片、构建索引并进行检索
    split_docs = split_documents(documents)
    index = build_index(split_docs)
    results = recall_documents(query, index)
    ranked_results = rerank_documents(results, query)
    return ranked_results

def call_cloud_model(results, api_key):
    """
    调用云端 OpenAI 模型进行推理，API 密钥由前端传递。
    """
    model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    input_text = "\n".join([doc.page_content for doc in results])
    try:
        response = model.invoke(input_text)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"调用云端模型时出错：{e}")
        return "⚠️ 云端模型调用失败"
