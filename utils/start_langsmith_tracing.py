import os
import configparser
from dotenv import load_dotenv

def start_langsmith_tracing(project_name="default"):
        load_dotenv()
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGSMITH_API_KEY')