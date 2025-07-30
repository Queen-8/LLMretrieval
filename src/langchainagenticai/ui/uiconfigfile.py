import os
from configparser import ConfigParser

class Config:
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "uiconfigfile.ini")
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_llm_options(self):
        return self.config.get("Default", "LLM_OPTIONS", fallback="OpenAI").split(",")
    
    def get_usecase_options(self):
        return self.config.get("Default", "USECASE_OPTIONS", fallback="Agentic AI,Basic Chatbot").split(",")
    
    def get_page_title(self):
        return self.config.get("Default", "PAGE_TITLE", fallback="LangChain Agentic AI")
    
    def get_openai_model_options(self):
        return self.config.get("Default", "OpenAI_MODEL_OPTIONS", fallback="gpt-3.5-turbo").split(",")
    
    def get_ollama_model_options(self):
        return self.config.get("Default", "ollama_MODEL_OPTIONS", fallback="deepseek-r1-7b").split(",")
    

