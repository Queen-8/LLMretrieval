from flask import Flask, request, jsonify
import json
from retrieval.local_retrieval import local_retrieve  # 本地检索
from retrieval.cloud_retrieval import cloud_retrieve  # 云端检索

# 确保在其他模块之前导入 config.py，以便环境变量被设置
import langchainagenticai.config  # 导入config.py来设置环境变量

import os

# 检查环境变量
print(f"LOCAL_KNOWLEDGE_BASE_PATH: {os.getenv('LOCAL_KNOWLEDGE_BASE_PATH')}")
print(f"CLOUD_URLS: {os.getenv('CLOUD_URLS')}")


app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    try:
        # 获取查询数据
        data = request.get_json()
        query = data.get("query")
        selected_llm = data.get("selected_llm", "ollama")  # 获取用户选择的模型，默认为 ollama
        api_key = data.get("api_key")  # 从前端接收 API 密钥

        if selected_llm == 'OpenAI':
            # 使用云端 OpenAI 推理
            cloud_urls = json.loads(data.get('cloud_urls', '[]'))  # 云端 URLs
            results = cloud_retrieve(query, cloud_urls, api_key)
            answer = cloud_retrieve.call_cloud_model(results, api_key)
        elif selected_llm == 'ollama':
            # 使用本地 Ollama 推理
            results = local_retrieve(query)
            answer = results  # 对于本地模型，直接返回检索到的内容
        else:
            return jsonify({"error": "Invalid LLM selection"}), 400
        
        # 返回检索结果
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # 启动 Flask 服务
