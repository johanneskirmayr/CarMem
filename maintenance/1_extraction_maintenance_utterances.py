"""
File to extract preferences from the maintenance utterances.
For the equal preference, and the different preference, the extraction can be artificially created as the correct preference and the sentence where the preference gets extracted is known.
For the negate preference, the extraction has to be performed as pference attribute can be negated in different ways.
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from langchain.callbacks import get_openai_callback
from rich.progress import track
from tqdm import tqdm

# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse


from extraction.mapping_category_to_label import convert_preference_to_labels
from extraction.mapping_category_to_pyd_category import (
    category_to_pyd_category,
)
from extraction.preference_memory import PreferenceMemory
from extraction.pydantic_schemas.categories_pydantic import (
    PreferencesFunctionOutput,
)
from utils.llm import get_llm_gpt4o
from utils.start_langsmith_tracing import start_langsmith_tracing


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extraction_result_dir",
        type=str,
        default="extraction/evaluation/gpt4o/eval_of_extraction_in_schema/dataset/eval_of_extraction.jsonl",
    )
    parser.add_argument("--trace_by_langsmith", type=bool, default=False)
    parser.add_argument(
        "--langsmith_project_name", type=str, default="extraction_negate_preference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="maintenance/evaluation/gpt4o/extraction_maintenance_utterances/dataset/",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="extraction_maintenance_utterances_test.jsonl",
    )
    return parser.parse_args()


async def main():

    args = parse_args()
    if args.trace_by_langsmith:
        start_langsmith_tracing(project_name=args.langsmith_project_name)

    os.makedirs(args.output_dir, exist_ok=True)

    preference_memory = PreferenceMemory()

    extraction_chain = await preference_memory.create_memory_function(
        llm=get_llm_gpt4o(),
        parameters=PreferencesFunctionOutput,
        target_type="user_state",
        name="extract_user_preferences",
        custom_instructions="Only extract long-term user preferences, no temporal desires in the current situation. It is better to not extract any preference than to extract temporal wishes.",
        description="A function that extracts long-term personal preferences of the user in the categories 'Points of Interest', 'Navigation and Routing', 'Vehicle Settings and Comfort', 'Entertainment and Media'.",
    )

    # read out train dataset
    train_dataset_lines = []
    with open(args.extraction_result_dir, "r") as file:
        for line in file:
            train_dataset_line = json.loads(line.strip())
            train_dataset_lines.append(train_dataset_line)

    for idx, line in enumerate(
        track(train_dataset_lines[:-1])
    ):  # one user in for loop, exclude line with overall results
        john_user_id = uuid.UUID(line["user_uuid"])
        john_username = f"john-{john_user_id.hex[:4]}"

        for conversation in tqdm(line["data"]):  # one conversation in for loop

            try:  # only score conversations where extraction is performed
                conversation_extraction = conversation[
                    "conversation_extracted_preferences"
                ]
            except KeyError as e:
                print(
                    f"Key Error: {e} \n Probably no extraction performed for conversation {conversation['conversation_uuid']}"
                )
                break

            # check if only one and the correct preference got extracted else skip
            if not (
                conversation_extraction["evaluation"]["conv_detail_accuracy"] == 1.0
            ):
                continue

            datapoint_strings = conversation["user_preference"].split(";")
            datapoint_user_preference_dict = {
                "main_category": datapoint_strings[0].strip(),
                "subcategory": datapoint_strings[1].strip(),
                "detail_category": datapoint_strings[2].strip(),
                "attribute": datapoint_strings[3].strip(),
            }

            keys_to_include_category_eval = [
                "main_category",
                "subcategory",
                "detail_category",
            ]

            maintenance_questions = conversation["maintenance_questions"]
            question_negate = f"user {john_username}: {maintenance_questions['question_negate_preference']}"

            # artificial extraction for equal preference
            preference_equal = datapoint_user_preference_dict.copy()
            preference_equal.update(
                {
                    "main_category": category_to_pyd_category(
                        datapoint_user_preference_dict["main_category"], "main_category"
                    ),
                    "subcategory": category_to_pyd_category(
                        datapoint_user_preference_dict["subcategory"], "subcategory"
                    ),
                    "detail_category": category_to_pyd_category(
                        datapoint_user_preference_dict["detail_category"],
                        "detail_category",
                    ),
                    "text": maintenance_questions["question_equal_preference"],
                    "user_name": str(john_user_id),
                }
            )
            preference_equal_categories_eval = {
                key: preference_equal[key]
                for key in keys_to_include_category_eval
                if key in preference_equal
            }
            preference_equal_dict = {
                "extracted_preference_equal_full": preference_equal,
                "extracted_preference_equal_categories_labels": convert_preference_to_labels(
                    preference_equal_categories_eval
                ),
            }

            # artificial extraction for different preference
            preference_different = datapoint_user_preference_dict.copy()
            preference_different.update(
                {
                    "main_category": category_to_pyd_category(
                        datapoint_user_preference_dict["main_category"], "main_category"
                    ),
                    "subcategory": category_to_pyd_category(
                        datapoint_user_preference_dict["subcategory"], "subcategory"
                    ),
                    "detail_category": category_to_pyd_category(
                        datapoint_user_preference_dict["detail_category"],
                        "detail_category",
                    ),
                    "text": maintenance_questions["question_different_preference"],
                    "attribute": maintenance_questions["different_attribute"],
                    "user_name": str(john_user_id),
                }
            )
            preference_different_categories_eval = {
                key: preference_different[key]
                for key in keys_to_include_category_eval
                if key in preference_different
            }
            preference_different_dict = {
                "extracted_preference_different_full": preference_different,
                "extracted_preference_different_categories_labels": convert_preference_to_labels(
                    preference_different_categories_eval
                ),
            }

            # perform extraction for negate preference
            preference_negate_dict = {}
            with get_openai_callback() as cb:
                output_extraction_maintenance_negate = extraction_chain.invoke(
                    input={"user_name": john_username, "conversation": question_negate}
                )
            output_extraction_maintenance_negate_retry, valid_at_try = (
                preference_memory.validate_output_and_retry(
                    output_extraction=output_extraction_maintenance_negate,
                    pydantic_schema=PreferencesFunctionOutput,
                    chain=extraction_chain,
                    messages_string=question_negate,
                    username=john_username,
                )
            )

            preference_negate_dict.update(
                {
                    "valid_at_try": valid_at_try,
                }
            )
            if valid_at_try == 1:
                pass
            elif valid_at_try == 2:
                preference_negate_dict.update(
                    {
                        "failed_extraction_1": json.loads(
                            output_extraction_maintenance_negate.additional_kwargs[
                                "function_call"
                            ]["arguments"]
                        ),
                    }
                )
                output_extraction_maintenance_negate = (
                    output_extraction_maintenance_negate_retry
                )
            else:
                preference_negate_dict.update(
                    {
                        "failed_extraction_1": json.loads(
                            output_extraction_maintenance_negate.additional_kwargs[
                                "function_call"
                            ]["arguments"]
                        ),
                        "failed_extraction_2": json.loads(
                            output_extraction_maintenance_negate_retry.additional_kwargs[
                                "function_call"
                            ][
                                "arguments"
                            ]
                        ),
                    }
                )

            function_json_extraction_maintenance_negate = json.loads(
                output_extraction_maintenance_negate.additional_kwargs["function_call"][
                    "arguments"
                ]
            )

            num_preference_counter = 0

            if function_json_extraction_maintenance_negate and not valid_at_try == None:
                for (
                    main_category,
                    rest,
                ) in (
                    function_json_extraction_maintenance_negate.items()
                ):  # in for loop are preferences of one main category
                    extracted_preferences_per_category = {
                        "main_category": main_category
                    }
                    for subcategory, rest in rest.items():
                        # discard no_or_other_preferences
                        if not (subcategory == "no_or_other_preferences"):
                            extracted_preferences_per_category.update(
                                {"subcategory": subcategory}
                            )
                            for detail_category, value in rest.items():
                                # discard no_or_other_preferences
                                if not (detail_category == "no_or_other_preferences"):
                                    extracted_preferences_per_category.update(
                                        {"detail_category": detail_category}
                                    )
                                    for preference in value:
                                        extracted_preference = (
                                            extracted_preferences_per_category.copy()
                                        )
                                        extracted_preference.update(
                                            {
                                                "text": preference[
                                                    "user_sentence_preference_revealed"
                                                ],
                                                "attribute": preference[
                                                    "user_preference"
                                                ],
                                                # "vector": get_embedding_recoro_ada002().embed_query(preference['sentence_preference_revealed']),
                                                "user_name": str(john_user_id),
                                            }
                                        )

                                        extracted_preference_categories_eval = {
                                            key: extracted_preference[key]
                                            for key in keys_to_include_category_eval
                                            if key in extracted_preference
                                        }

                                        preference_negate_dict.update(
                                            {
                                                f"extracted_preference_negate_{num_preference_counter}_full": extracted_preference,
                                                f"extracted_preference_negate_{num_preference_counter}_categories_label": convert_preference_to_labels(
                                                    extracted_preference_categories_eval
                                                ),
                                            }
                                        )
                                        num_preference_counter += 1

            preference_negate_dict.update(
                {
                    "number_preferences_negate_extracted": num_preference_counter,
                }
            )

            conversation["maintenance_questions"][
                "question_equal_extraction"
            ] = preference_equal_dict
            conversation["maintenance_questions"][
                "question_negate_extraction"
            ] = preference_negate_dict
            conversation["maintenance_questions"][
                "question_different_extraction"
            ] = preference_different_dict

        # write line extended with extraction results, filter conversations where no extraction is performed
        filtered_data = filter(
            lambda d: "question_equal_extraction" in d["maintenance_questions"],
            line["data"],
        )
        line["data"] = list(filtered_data)
        with open(os.path.join(args.output_dir, args.output_file), "a") as file:
            file.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
