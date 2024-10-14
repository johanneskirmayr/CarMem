# from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
import os

from dotenv import load_dotenv
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)

from config.config_loader import config

load_dotenv()


# ==== Azure OpenAI ====
def get_llm_gpt35_azure_openai(temperature=0.0):
    load_dotenv()
    print("Model: gpt-3.5-turbo")
    return AzureChatOpenAI(
        temperature=temperature,
        azure_deployment="gpt-35-turbo-1106",
        azure_endpoint=os.getenv("AZURE__OPENAI_API_BASE"),
        openai_api_version="2023-07-01-preview",
        openai_api_key=os.getenv("AZURE__OPENAI_API_KEY"),
    )


def get_llm_gpt4o_azure_openai(temperature=0.0):
    load_dotenv()
    print("Model: gpt-4-o")
    return AzureChatOpenAI(
        temperature=temperature,
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("AZURE__OPENAI_API_BASE"),
        openai_api_version="2023-07-01-preview",
        openai_api_key=os.getenv("AZURE__OPENAI_API_KEY"),
    )


def get_embedding_azure_openai():
    load_dotenv()
    print("Model: text-embedding-ada-002")
    return AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        azure_endpoint=os.getenv("AZURE__OPENAI_API_BASE"),
        openai_api_version="2023-07-01-preview",
        openai_api_key=os.getenv("AZURE__OPENAI_API_KEY"),
    )


# ==== OpenAI ====
def get_llm_gpt35_openai(temperature=0.0):
    load_dotenv()
    print("Model: gpt-3.5-turbo")
    return ChatOpenAI(
        model="gpt-35-turbo",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_llm_gpt4o_openai(temperature=0.0):
    load_dotenv()
    print("Model: gpt-4-o")
    return ChatOpenAI(
        model="gpt-4o-2024-08-06",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_embedding_openai():
    load_dotenv()
    print("Model: text-embedding-ada-002")
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# get llm and embedding
def get_llm_gpt35(temperature=0.0):
    config_use_azure = config.get("use_azure", "use_azure_openai")
    if config_use_azure == "True":
        return get_llm_gpt35_azure_openai(temperature)
    else:
        return get_llm_gpt35_openai(temperature)


def get_llm_gpt4o(temperature=0.0):
    config_use_azure = config.get("use_azure", "use_azure_openai")
    if config_use_azure == "True":
        return get_llm_gpt4o_azure_openai(temperature)
    else:
        return get_llm_gpt4o_openai(temperature)


def get_embedding():
    config_use_azure = config.get("use_azure", "use_azure_openai")
    if config_use_azure == "True":
        return get_embedding_azure_openai()
    else:
        return get_embedding_openai()
