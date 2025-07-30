
# LangChain Agentic AI - 端云协同RAG智能助理系统

## 项目简介

本项目是一个基于端云协同RAG（Retrieval Augmented Generation）架构的文本智能助理系统。系统支持本地大模型（如 Ollama）和云端大模型（如 OpenAI、OpenRouter），可根据用户选择动态切换推理和知识库来源，满足隐私保护、离线推理和复杂语义理解等多样化需求。

- **本地推理**：通过 Ollama 实现，适合隐私和离线场景。
- **云端推理**：通过 OpenAI/OpenRouter 实现，适合大规模知识库和复杂推理。
- **RAG 架构**：统一的检索增强生成流程，支持文档分片、向量检索、重排和答案生成。
- **MCP 协议**：端云功能封装为工具，支持灵活的端云协同。
- **Streamlit 前端**：现代化的交互界面，支持模型选择、历史消息、对话导出等功能。

---

## 主要功能

- ✅ 支持本地 Ollama 和云端 OpenAI/Router 大模型
- ✅ 动态选择 LLM 和具体模型
- ✅ 支持本地/云端知识库检索与推理
- ✅ 端云协同，MCP 协议工具化
- ✅ RAG 检索增强生成流程
- ✅ Streamlit 聊天界面，支持历史消息、导出、清除
- ✅ API Key 管理与安全
- ✅ 错误处理与日志追踪

---

## 安装与环境配置

1. **克隆项目**
   ```bash
   git clone https://github.com/Queen-8/LLMretrieval.git
   cd llm_retrieval
   ```

2. **创建虚拟环境并激活**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   - 在项目根目录下创建 `.env` 文件，添加你的 OpenAI 或 OpenRouter API Key：
     ```
     OPENROUTER_API_KEY=sk-xxxxxx
     ```

5. **准备本地知识库**
   - 将 PDF 文档放入 `Local_knowledge_base/` 目录。

6. **安装并启动 Ollama**
   - 参考 [Ollama 官方文档](https://ollama.com/) 安装并拉取所需模型，如：
     ```bash
     ollama pull deepseek-r1:8b
     ollama serve
     ```

---

## 启动与使用

1. **启动 Streamlit 前端**
   ```bash
   streamlit run app.py
   ```

2. **使用说明**
   - 在左侧栏选择 LLM（OpenAI 或 Ollama）和具体模型
   - 输入 OpenAI API Key（如使用云端模型）
   - 在输入框提问，系统会自动根据选择调用本地或云端 RAG 流程
   - 支持连续对话、历史消息查看、导出和清除

---

## 目录结构

```
llm_retrieval/
├── app.py                  # Streamlit 前端入口
├── client.py               # MCP 客户端
├── main.py                 # 主逻辑入口
├── requirements.txt        # 依赖列表
├── Local_knowledge_base/   # 本地知识库PDF
├── src/
│   └── langchainagenticai/
│       ├── main.py
│       ├── retrieval/      # RAG 检索与推理
│       ├── mcp_server/     # MCP 端云服务
│       ├── mcp_host/       # MCP 客户端
│       ├── ui/             # Streamlit UI与配置
│       └── utils/          # 工具与基础模块
└── ...
```

---

## 常见问题

- **Q: 本地模型无法推理？**
  - 检查 Ollama 是否已启动，模型是否已拉取，端口是否为 11434。
- **Q: OpenAI/Router 报 402 错误？**
  - 免费额度不足，请减少 max_tokens 或升级账户。
- **Q: 提问后历史消息消失？**
  - 已修复，现支持完整对话历史存储。
- **Q: 如何扩展知识库？**
  - 直接将 PDF 文档放入 `Local_knowledge_base/` 目录即可。

---

## 参考与致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [OpenRouter](https://openrouter.ai/)
- [Streamlit](https://streamlit.io/)


```
