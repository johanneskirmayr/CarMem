import logging
from typing import Any, Optional

from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from pymilvus.client.types import DataType

from utils.milvus_utils import connect2milvusdb

logger = logging.getLogger(__name__)


class Milvus2PreferenceStore:
    def __init__(self):
        connect2milvusdb()
        self.embedding_dim = 1536
        self.metric_type = "IP"
        self.index_type = "IVF_FLAT"
        self.index_param = {"nlist": 16384}
        self.search_param = {"nprobe": 10}

    def _create_collection_and_index(
        self,
        collection_name: str,
        recreate_collection: Optional[bool] = False,
    ):

        has_collection = utility.has_collection(collection_name=collection_name)

        if has_collection and recreate_collection == True:
            print("Dropping Collection")
            utility.drop_collection(collection_name=collection_name)
            has_collection = False

        if not has_collection:
            fields = [
                FieldSchema(
                    name="pk",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    is_primary=True,
                    auto_id=False,
                ),  # "pk" is langchain convention
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim
                ),  # "vector" is langchain convention
                FieldSchema(
                    name="text", dtype=DataType.VARCHAR, max_length=50000
                ),  # "text" is langchain convention
                FieldSchema(
                    name="main_category", dtype=DataType.VARCHAR, max_length=50000
                ),
                FieldSchema(
                    name="subcategory", dtype=DataType.VARCHAR, max_length=50000
                ),
                FieldSchema(
                    name="detail_category", dtype=DataType.VARCHAR, max_length=50000
                ),
                FieldSchema(name="attribute", dtype=DataType.VARCHAR, max_length=50000),
                FieldSchema(
                    name="user_name",
                    dtype=DataType.VARCHAR,
                    description="",
                    max_length=512,
                    is_partition_key=True,
                ),
            ]

            collection_schema = CollectionSchema(
                fields=fields,
                enable_dynamic_field=True,
                partition_key_field="user_name",
            )
            collection = Collection(name=collection_name, schema=collection_schema)
        else:
            logger.warning(
                f"Collection {collection_name} already exists. Will not be recreated, please delete or rename."
            )
            collection = Collection(collection_name)

        has_index = collection.has_index()
        if not has_index:
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": self.index_param,
                },
            )

        collection.load()

        return collection

    def upload_preferences(
        self,
        collection,
        preferences: Any = None,
    ):

        if len(preferences) == 0:
            logger.warning("Calling DocumentStore.write_documents() with empty list")
            return

        mutation_result: Any = None

        # Only embedding(=vector) and content(=text) is mandatory, the metadata will be uploaded to Milvus in a "dynamic schema", this can then be accessed with $meta["<Your_Field>"]

        mutation_result = collection.insert(preferences)
        collection.flush()
        collection.compact()
        logger.info(f"Inserted {mutation_result.insert_count} entities")

        return mutation_result
