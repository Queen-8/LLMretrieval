import os

# 设置环境变量
os.environ['LOCAL_KNOWLEDGE_BASE_PATH'] = '/Users/queen/Documents/VSCode/llm_retrieval/Local_knowledge_base'
os.environ['CLOUD_URLS'] = '["https://blog.csdn.net/zy15667076526/article/details/127069876", "https://mp.weixin.qq.com/s/C7GQEb5oNnKNKf_yU0db1w"]'  # 示例 URL 数组


os.environ["OLLAMA_BASEURL"] = "http://127.0.0.1:11434"
# os.environ["OLLAMA_MODEL"] = "deepseek-r1:8b"
os.environ["OLLAMA_MODEL"] = "llama3.1:8b"

os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"