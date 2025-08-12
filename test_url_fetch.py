#!/usr/bin/env python3
import os
import logging
from dotenv import load_dotenv
import json
from langchain_community.document_loaders import WebBaseLoader

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 .env
load_dotenv()

def test_url_fetch():
    """测试URL抓取"""
    urls_str = os.getenv("CLOUD_URLS", "[]")
    try:
        urls = json.loads(urls_str) if urls_str.startswith("[") else [u.strip() for u in urls_str.split(",") if u.strip()]
        logger.info(f"🌐 从环境变量加载 {len(urls)} 个 URL")
        
        for i, url in enumerate(urls):
            logger.info(f"\n🔍 测试抓取 URL {i+1}: {url}")
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                logger.info(f"✅ 成功抓取，获得 {len(docs)} 个文档")
                
                for j, doc in enumerate(docs):
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    logger.info(f"  文档 {j+1} 预览: {content_preview}")
                    logger.info(f"  文档 {j+1} 元数据: {doc.metadata}")
                    
            except Exception as e:
                logger.error(f"❌ 抓取失败: {e}")
                
    except Exception as e:
        logger.error(f"❌ 解析 CLOUD_URLS 失败: {e}")

if __name__ == "__main__":
    test_url_fetch()
