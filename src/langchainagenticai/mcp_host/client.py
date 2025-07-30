import os
import json
from flask import Flask, request, jsonify
from retrieval.local_retrieval import local_retrieve
from retrieval.cloud_retrieval import cloud_retrieve
from mcp_server.local_server import local_answer_question

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    try:
        # 获取前端请求数据
        data = request.get_json()
        query = data.get("query")
        use_local = data.get("use_local", True)
        answer = None
        if use_local:
            # 调用 async 的本地推理
            import asyncio
            answer = asyncio.run(local_answer_question(query))
        else:
            cloud_urls = json.loads(os.getenv('CLOUD_URLS', '[]'))  # 从环境变量中读取云端 URL 列表
            results = cloud_retrieve(query, cloud_urls)
        answer = "\n".join([doc.page_content for doc in results]) if results else "⚠️ 无法找到相关文档。"
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # 启动 Flask 服务
