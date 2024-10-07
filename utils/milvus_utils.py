import configparser
from pymilvus import connections
import os
from dotenv import load_dotenv
def connect2milvusdb():

    # Creating Personalization VectorDB
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    load_dotenv()
    
    #MILVUS_HOST = "delphi.northeurope.cloudapp.azure.com"
    MILVUS_HOST = config.get('database', 'host')
    MILVUS_PORT = config.get('database', 'port')
    print("Testing connection to milvus...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, user="root", password=os.getenv("MILVUS_DB_PASSWORD"))
    print("Connected")