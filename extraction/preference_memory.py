import datetime
import json
import os
import uuid
from typing import (
    Dict,
    Optional,
    Union,
    cast,
)

from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langmem._internal.utils import ID_T
from pydantic import BaseModel, ValidationError

from utils.custom_logger import log_debug, log_error, log_info

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm with the task to extract long-term user preferences that will persist after the conversation. "
            "Default is to not extract a preference. "
            "Only extract relevant user preferences of the user {user_name} from the in-car conversation between the user and the voice assistant. "
            "Only extract the preferences mentioned in the '{name_preference_function}' function, strictly follow the function parameters format. "
            "Users can delete categories for privacy reasons, so really make sure to only extract preferences that neatly fit into the category descriptions. "
            "Avoid interpreting preferences to other topics. "
            "If a category is not present, do not include it in the output. If no preference in the conversation or no fitting function parameter, return null. "
            "{custom_instructions}",
        ),
        # MessagesPlaceholder('examples'),
        (
            "human",
            "Conversation: \n===\n{conversation}\n===\n"
            "Only extract long-term preferences said or confirmed by the user {user_name}, never from text or assumptions from the assistant.",
        ),
    ]
)


def create_preference_extraction_chain_pydantic(
    pydantic_schema: BaseModel,
    data: dict,
    llm: BaseLanguageModel,
    prompt: Optional[Union[BasePromptTemplate, ChatPromptTemplate]] = None,
) -> Chain:
    """Creates a chain that extracts user preferences from a conversation using pydantic schema.

    Args:
        pydantic_schema: The pydantic schema of the entities to extract.
        data: includes information of preference extraction function.
        llm: The language model to use.
        prompt: The prompt to use for extraction.

    Returns:
        Chain that can be used to extract information from a passage.
    """

    function = cast(
        Dict,
        convert_pydantic_to_openai_function(
            model=pydantic_schema,
            name=data["schema"]["name"],
            description=data["schema"]["description"],
        ),
    )
    extraction_prompt = (
        prompt
        if type(prompt) == ChatPromptTemplate
        else ChatPromptTemplate.from_template(prompt)
    )
    extraction_prompt = extraction_prompt.partial(
        name_preference_function=data["schema"]["name"],
        custom_instructions=data["custom_instructions"],
    )

    chain = extraction_prompt | llm.bind(
        function_call={"name": data["schema"]["name"]}, functions=[function]
    )
    return chain


class PreferenceMemory:

    def __init__(self) -> None:
        pass

    async def create_memory_function(
        self,
        llm: BaseLanguageModel,
        parameters: BaseModel,
        *,
        target_type: str = "user_state",
        name: Optional[str] = None,
        description: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        function_id: Optional[ID_T] = None,
    ) -> Chain:

        params = parameters.model_json_schema()

        function_schema = {
            "name": name or params.pop("title", ""),
            "description": description or params.pop("description", ""),
            "parameters": params,
        }

        data = {
            "type": target_type,
            "custom_instructions": custom_instructions,
            "id": str(function_id) if function_id else str(uuid.uuid4()),
            "schema": function_schema,
        }

        extraction_chain = create_preference_extraction_chain_pydantic(
            pydantic_schema=parameters, llm=llm, data=data, prompt=extraction_prompt
        )

        return extraction_chain

    def validate_extraction(self, extraction_result, pydantic_schema: BaseModel):
        if not extraction_result:
            log_debug("The extraction_result is empty. Passing validation by default.")
            return None
        try:
            # Parse the JSON output using the Pydantic model
            validated_data = pydantic_schema(**extraction_result)
            log_debug("The JSON output is valid according to the Pydantic schema.")
            return None
        except ValidationError as e:
            log_info("The JSON output is not valid:")
            return e

    def retry_chain_with_validation_error(
        self, extraction_output: str, validation_error: str, chain: Chain
    ) -> Chain:
        chain_ = chain.copy()
        original_prompt = chain.first.messages[0].prompt.template
        chain_.first.messages[0].prompt.template = (
            chain.first.messages[0].prompt.template
            + "\n"
            + "\n# Errors from previous try:\n Your previous call did not produce a valid output format (possible reasons: category was skipped, non-existing key, subcategory corresponds to different parent category): "
            + str(extraction_output).replace("{", "{{").replace("}", "}}")
            + ". This failed because of the validation error:\n"
            + str(validation_error).replace("{", "{{").replace("}", "}}")
            + "Please correct the error and extract the same preference in the correct format."
        )
        return chain_, original_prompt

    def filter_none_values(self, d):
        if isinstance(d, dict):
            return {
                k: self.filter_none_values(v) for k, v in d.items() if v is not None
            }
        elif isinstance(d, list):
            return [self.filter_none_values(v) for v in d if v is not None]
        else:
            return d

    def validate_output_and_retry(
        self, output_extraction, pydantic_schema, chain, username, messages_string
    ):
        output_extraction_arguments = json.loads(
            output_extraction.additional_kwargs["function_call"]["arguments"]
        )
        validation_error = self.validate_extraction(
            output_extraction_arguments, pydantic_schema
        )

        if validation_error:
            extraction_chain_retry, original_prompt = (
                self.retry_chain_with_validation_error(
                    extraction_output=str(output_extraction_arguments),
                    validation_error=validation_error,
                    chain=chain,
                )
            )
            output_extraction = extraction_chain_retry.invoke(
                input={"user_name": username, "conversation": messages_string}
            )

            # remove validation error message again
            chain.first.messages[0].prompt.template = original_prompt

            output_extraction_arguments = json.loads(
                output_extraction.additional_kwargs["function_call"]["arguments"]
            )
            validation_error = self.validate_extraction(
                extraction_result=output_extraction_arguments,
                pydantic_schema=pydantic_schema,
            )
            if validation_error:
                valid_at_try = None
                log_error(
                    f"Repeating Validation of the extraction output w.r.t the pydantic schema: {validation_error}"
                )
                return output_extraction, valid_at_try
            else:
                valid_at_try = 2
                # filter None outputs
                filtered_extraction = json.dumps(
                    self.filter_none_values(
                        json.loads(
                            output_extraction.additional_kwargs["function_call"][
                                "arguments"
                            ]
                        )
                    )
                )
                output_extraction.additional_kwargs["function_call"][
                    "arguments"
                ] = filtered_extraction
                return output_extraction, valid_at_try
        else:
            valid_at_try = 1
            # filter None outputs
            filtered_extraction = json.dumps(
                self.filter_none_values(
                    json.loads(
                        output_extraction.additional_kwargs["function_call"][
                            "arguments"
                        ]
                    )
                )
            )
            output_extraction.additional_kwargs["function_call"][
                "arguments"
            ] = filtered_extraction
            return output_extraction, valid_at_try
