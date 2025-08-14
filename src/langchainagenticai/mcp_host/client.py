# client.py（关键思路示例）

from src.langchainagenticai.agent.agent import create_mcp_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

MCP_CONNECTIONS = {
    "LocalRetrievalQA": {
        "command": "python",
        "args": ["src/langchainagenticai/mcp_server/local_server.py"],
        "transport": "stdio",
    },
    "CloudRetrievalQA": {
        "command": "python",
        "args": ["src/langchainagenticai/mcp_server/cloud_server.py"],
        "transport": "stdio",
    },
}

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query = data.get("query", "")
        session_id = data.get("session_id", "default")
        prefer = data.get("prefer", "auto")  # "auto" / "local" / "cloud"

        # 根据 prefer 选择模型（不是选择工具！）
        if prefer == "local":
            model = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
        elif prefer == "cloud":
            model = ChatOpenAI(model="openai/gpt-4o-mini")
        else:
            # auto：先尝试云，不行退本地，或自行策略
            model = ChatOpenAI(model="openai/gpt-4o-mini")

        agent = asyncio.run(create_mcp_agent(model, MCP_CONNECTIONS))
        answer = asyncio.run(agent.ainvoke(query))

        # 维护会话历史（可选）
        history = conversation_cache[session_id]
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        conversation_cache[session_id] = history[-10:]

        return jsonify({"answer": answer, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
