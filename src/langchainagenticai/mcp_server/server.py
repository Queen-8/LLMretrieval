
from flask import Flask, request, jsonify
import json
import os
import logging

# 本地/云端检索
from retrieval.local_retrieval import local_retrieve
from retrieval.cloud_retrieval import cloud_retrieve

# 确保环境变量优先加载
import langchainagenticai.config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"LOCAL_KNOWLEDGE_BASE_PATH: {os.getenv('LOCAL_KNOWLEDGE_BASE_PATH')}")
logger.info(f"CLOUD_URLS: {os.getenv('CLOUD_URLS')}")

app = Flask(__name__)

def is_network_available() -> bool:
    """检测网络是否可用"""
    try:
        import requests
        requests.get("http://www.google.com", timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False

def check_local_model_ready() -> bool:
    """检查本地模型是否就绪"""
    try:
        import requests
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query = data.get("query")
        selected_llm = data.get("selected_llm", None)
        api_key = data.get("api_key")  # OpenAI API key
        cloud_urls = json.loads(data.get('cloud_urls', '[]'))

        # 云端需要 API Key
        if selected_llm == "OpenAI" and not api_key:
            return jsonify({"error": "API key is required for OpenAI"}), 400

        # 自动选择推理路径
        if selected_llm is None:
            if is_network_available():
                if check_local_model_ready():
                    selected_llm = 'ollama'
                else:
                    selected_llm = 'OpenAI'
            else:
                selected_llm = 'ollama'

        answer = None

        if selected_llm == 'OpenAI':
            # 端云协同：只传输非私有文档
            logger.info("🔍 正在执行云端检索（过滤私有文档）...")
            results = cloud_retrieve(
                query,
                cloud_urls,
                api_key,
                filter_func=lambda doc: not doc.metadata.get("private", False)
            )
            answer = results if results else "⚠️ 云端无法找到相关文档。"

        elif selected_llm == 'ollama':
            logger.info("🔍 正在执行本地检索（包含私有文档）...")
            results = local_retrieve(query)  # 本地直接检索
            answer = results if results else "⚠️ 本地模型无法返回结果。"

        else:
            return jsonify({"error": "Invalid LLM selection"}), 400

        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"查询处理时发生错误: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)






# from flask import Flask, request, jsonify
# import json
# import os
# import logging
# from retrieval.local_retrieval import local_retrieve  # 本地检索
# from retrieval.cloud_retrieval import cloud_retrieve  # 云端检索

# # 确保在其他模块之前导入 config.py，以便环境变量被设置
# import langchainagenticai.config  # 导入config.py来设置环境变量

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # 检查环境变量
# logger.info(f"LOCAL_KNOWLEDGE_BASE_PATH: {os.getenv('LOCAL_KNOWLEDGE_BASE_PATH')}")
# logger.info(f"CLOUD_URLS: {os.getenv('CLOUD_URLS')}")

# app = Flask(__name__)

# def is_network_available() -> bool:
#     """
#     检测网络是否可用
#     """
#     try:
#         # 尝试连接到一个常见的外部服务器
#         import requests
#         requests.get("http://www.google.com", timeout=5)
#         return True
#     except requests.exceptions.RequestException:
#         return False

# def check_local_model_ready() -> bool:
#     """
#     检查本地模型是否就绪（例如，检查本地服务是否启动）
#     """
#     try:
#         import requests
#         # 假设本地模型服务在 11434 端口运行
#         response = requests.get("http://localhost:11434", timeout=5)
#         if response.status_code == 200:
#             return True
#     except requests.exceptions.RequestException:
#         return False
#     return False

# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         # 获取查询数据
#         data = request.get_json()
#         query = data.get("query")
#         selected_llm = data.get("selected_llm", None)  # 如果前端没有选择，则为 None
#         api_key = data.get("api_key")  # 从前端接收 API 密钥

#         # 检查 API 密钥是否存在
#         if selected_llm == "OpenAI" and not api_key:
#             return jsonify({"error": "API key is required for OpenAI"}), 400
        
#         # 判断使用的推理路径
#         if selected_llm is None:
#             if is_network_available():
#                 if check_local_model_ready():
#                     selected_llm = 'ollama'  # 本地模型可用，使用本地推理
#                 else:
#                     selected_llm = 'OpenAI'  # 本地模型不可用，使用云端推理
#             else:
#                 selected_llm = 'ollama'  # 如果网络不可用，默认使用本地推理
        
#         answer = None
        
#         if selected_llm == 'OpenAI':
#             # 使用云端 OpenAI 推理
#             cloud_urls = json.loads(data.get('cloud_urls', '[]'))  # 云端 URLs
#             results = cloud_retrieve(query, cloud_urls, api_key)
#             answer = results if results else "⚠️ 无法找到相关文档。"
        
#         elif selected_llm == 'ollama':
#             # 使用本地 Ollama 推理
#             results = local_retrieve(query)
#             answer = results if results else "⚠️ 本地模型无法返回结果。"
        
#         else:
#             return jsonify({"error": "Invalid LLM selection"}), 400
        
#         # 返回检索结果
#         return jsonify({"answer": answer})
    
#     except Exception as e:
#         logger.error(f"查询处理时发生错误: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)  # 启动 Flask 服务
