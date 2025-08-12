import os
import json
import asyncio
import requests
from flask import Flask, request, jsonify
from collections import defaultdict
import logging

# 假设这两个接口都是 async 函数，且参数改成 history 列表形式传递
from mcp_server.local_server import local_answer_question
from mcp_server.cloud_server import cloud_answer_question

from retrieval.cloud_retrieval import cloud_retrieve  # 返回 Document list，支持filter_func参数

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 多轮对话缓存，key: session_id, value: [对话历史]
conversation_cache = defaultdict(list)

def is_network_available() -> bool:
    try:
        requests.get("http://www.baidu.com", timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False

def check_local_model_ready() -> bool:
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

async def ask_local_tool(query: str) -> str:
    # 获取工具
    tools = await mcp_clients["LocalRetrievalQA"].get_tools()
    
    # 找到 local_answer_question 工具
    local_tool = None
    for tool in tools:
        if tool.name == "local_answer_question":
            local_tool = tool
            break
    
    if local_tool:
        try:
            result = await local_tool.ainvoke({"query": query})
            return result
        except Exception as e:
            logging.error(f"调用 local_answer_question 工具时出错: {e}")
            return f"❌ 发生错误: {str(e)}"
    else:
        logging.error("未找到 local_answer_question 工具")
        return "❌ 未找到 local_answer_question 工具"

async def ask_cloud_tool(payload: dict) -> str:
    # 获取工具
    tools = await mcp_clients["CloudRetrievalQA"].get_tools()
    
    # 找到 cloud_answer_question 工具
    cloud_tool = None
    for tool in tools:
        if tool.name == "cloud_answer_question":
            cloud_tool = tool
            break
    
    if cloud_tool:
        try:
            result = await cloud_tool.ainvoke(payload)
            return result
        except Exception as e:
            logging.error(f"调用 cloud_answer_question 工具时出错: {e}")
            return f"❌ 发生错误: {str(e)}"
    else:
        logging.error("未找到 cloud_answer_question 工具")
        return "❌ 未找到 cloud_answer_question 工具"

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query = data.get("query", "")
        session_id = data.get("session_id", "default")  # 多轮会话ID
        use_local = data.get("use_local", None)  # 手动控制是否使用本地模型，None表示自动选择

        logging.info(f"Received query: {query}, session_id: {session_id}, use_local: {use_local}")
        
        # 自动推理路径选择
        if use_local is None:
            if is_network_available():
                use_local = check_local_model_ready()
            else:
                use_local = True  # 网络不可用，只能使用本地模型
        logging.info(f"Using local model: {use_local}")
        
        # 获取历史对话
        history = conversation_cache[session_id]
        history.append({"role": "user", "content": query})

        answer = None
        if use_local:
            # 本地模型，传递对话历史列表
            answer = asyncio.run(ask_local_tool(query))  # 调用本地工具
        else:
            # 云端检索先召回文档并过滤掉 private 的
            cloud_urls = json.loads(os.getenv('CLOUD_URLS', '[]'))
            
            # 过滤掉私有文档
            filter_func = lambda doc: not doc.metadata.get("private", False)
            retrieved_docs = cloud_retrieve(query, cloud_urls, filter_func=filter_func)

            # 构造上下文字符串
            context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""

            # 调用云端问答接口，传递历史和上下文
            answer = asyncio.run(ask_cloud_tool({"history": history, "context": context}))

        # 更新历史缓存，限制长度避免爆炸
        history.append({"role": "assistant", "content": answer})
        conversation_cache[session_id] = history[-10:]

        return jsonify({"answer": answer, "history": history})

    except Exception as e:
        logging.error(f"Error in query processing: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



    




# import os
# import json
# import asyncio
# import requests
# from flask import Flask, request, jsonify
# from retrieval.local_retrieval import local_retrieve
# from retrieval.cloud_retrieval import cloud_retrieve
# from mcp_server.local_server import local_answer_question

# app = Flask(__name__)

# import os
# import json
# import asyncio
# import requests
# from flask import Flask, request, jsonify
# from retrieval.local_retrieval import local_retrieve
# from retrieval.cloud_retrieval import cloud_retrieve
# from mcp_server.local_server import local_answer_question
# from mcp_server.cloud_server import cloud_answer_question  # 假设有 cloud_answer_question()
# from collections import defaultdict

# app = Flask(__name__)

# # 多轮对话缓存，key: session_id, value: [对话历史]
# conversation_cache = defaultdict(list)


# def is_network_available() -> bool:
#     try:
#         requests.get("http://www.baidu.com", timeout=5)
#         return True
#     except requests.exceptions.RequestException:
#         return False


# def check_local_model_ready() -> bool:
#     try:
#         response = requests.get("http://localhost:11434", timeout=5)
#         return response.status_code == 200
#     except requests.exceptions.RequestException:
#         return False


# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         data = request.get_json()
#         query = data.get("query", "")
#         session_id = data.get("session_id", "default")  # 多轮会话ID
#         use_local = data.get("use_local", None)  # 可手动控制是否使用本地模型

#         # 自动推理路径选择
#         if use_local is None:
#             if is_network_available():
#                 use_local = check_local_model_ready()
#             else:
#                 use_local = True  # 网络不可用，只能本地模型

#         # 取出历史对话
#         history = conversation_cache[session_id]
#         history.append({"role": "user", "content": query})

#         # 获取回答
#         if use_local:
#             answer = asyncio.run(local_answer_question(history))
#         else:
#             cloud_urls = json.loads(os.getenv('CLOUD_URLS', '[]'))
#             retrieved_docs = cloud_retrieve(query, cloud_urls)
#             context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
#             answer = cloud_answer_question(history, context)

#         # 更新缓存
#         history.append({"role": "assistant", "content": answer})
#         conversation_cache[session_id] = history[-10:]  # 限制最多10轮上下文，避免过长

#         return jsonify({"answer": answer, "history": history})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)