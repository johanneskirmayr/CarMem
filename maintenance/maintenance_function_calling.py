import os
from typing import Optional, Type

from dotenv import load_dotenv
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from config.config_loader import config
from utils.llm import get_embedding


class AppendInput(BaseModel):
    incoming_preference: str = Field(
        description="the attribute of the incoming preference"
    )


class PassInput(BaseModel):
    to_pass_incoming_preference: str = Field(
        description="the attribute of the incoming preference"
    )
    pk_of_equal_existing_preference: str = Field(
        description="the primary key (pk) of the existing preference that is equal to the incoming preference attribute"
    )


class UpdateInput(BaseModel):
    to_insert_incoming_preference: str = Field(
        description="the attribute of the incoming preference"
    )
    pk_of_to_delete_existing_preference: str = Field(
        description="the primary key (pk) of existing preference that should be deleted"
    )


class Append(BaseTool):
    name = "append_preference"
    description = "appends incoming preference to database and keep existing preferences. Call if incoming preference attribute is different to existing preferences attributes, it can be of the same category"
    args_schema: Type[BaseModel] = AppendInput

    def _run(
        self,
        incoming_preference,
        perform_function=True,
        run_manager: Optional[CallbackManagerForToolRun] = True,
    ) -> str:
        if perform_function:
            load_dotenv()
            milvus_client = MilvusClient(
                uri=f"http://{config.get('database', 'host')}:{config.get('database', 'port')}"
            )
            milvus_client.insert(
                collection_name=config.get("database", "collection_name"),
                data=incoming_preference,
            )

        return (
            f"Preference '{incoming_preference['detail_category']}: {incoming_preference['attribute']}' got appended",
            self.name,
        )


class Pass(BaseTool):
    name = "pass_preference"
    description = "passes incoming preference (so it is not inserted in database) and keep existing preferences. Call if incoming preference attribute is equal or very similar to one existing preference attribute"
    args_schema: Type[BaseModel] = PassInput

    def _run(
        self,
        incoming_preference,
        pk_of_equal_existing_preference=None,
        equal_existing_preference=None,
        perform_function=True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if perform_function:
            if pk_of_equal_existing_preference and equal_existing_preference:
                concat_sentences = (
                    incoming_preference["text"]
                    + "\n"
                    + equal_existing_preference["text"]
                )
                new_joint_preference = incoming_preference.copy()
                new_joint_preference["text"] = concat_sentences
                new_joint_preference["vector"] = get_embedding().embed_query(
                    concat_sentences
                )
                load_dotenv()
                milvus_client = MilvusClient(
                    uri=f"http://{config.get('database', 'host')}:{config.get('database', 'port')}"
                )
                milvus_client.delete(
                    collection_name=config.get("database", "collection_name"),
                    pks=pk_of_equal_existing_preference,
                )
                milvus_client.insert(
                    collection_name=config.get("database", "collection_name"),
                    data=new_joint_preference,
                )
        return (
            f"Preference '{incoming_preference['detail_category']}: {incoming_preference['attribute']}' got passed",
            self.name,
        )


class Update(BaseTool):
    name = "update_preference"
    description = "deletes one existing preference and insert incoming preference. Call if incoming preference attribute is updating or contradicting one existing preference attribute, either the text or the attribute"
    args_schema: Type[BaseModel] = UpdateInput

    def _run(
        self,
        incoming_preference,
        pk_to_delete_existing_preference,
        to_delete_existing_preference,
        perform_function=True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if perform_function:
            load_dotenv()
            milvus_client = MilvusClient(
                uri=f"http://{config.get('database', 'host')}:{config.get('database', 'port')}"
            )
            milvus_client.delete(
                collection_name=config.get("database", "collection_name"),
                pks=pk_to_delete_existing_preference,
            )
            milvus_client.insert(
                collection_name=config.get("database", "collection_name"),
                data=incoming_preference,
            )
        return (
            f"Preference '{to_delete_existing_preference['detail_category']}: {to_delete_existing_preference['attribute']}' got updated to '{incoming_preference['detail_category']}: {incoming_preference['attribute']}'",
            self.name,
        )
