import uuid
import json
import asyncio
import os
from tqdm import tqdm
from rich.progress import track

import sys
from pathlib import Path
# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from extraction.pydantic_schemas.categories_pydantic import PreferencesFunctionOutput, return_pydantic_schema, return_pydantic_schema_wo_category
from utils.start_langsmith_tracing import start_langsmith_tracing
from utils.llm import get_llm_gpt4o, get_embedding
import argparse

from extraction.preference_memory import PreferenceMemory
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from utils.general_utils import stringify_conversations
from extraction.mapping_category_to_label import convert_preference_to_labels

from utils.custom_logger import log_error, log_debug
from extraction.mapping_category_to_pyd_category import category_to_pyd_category_sub_extra, category_to_pyd_category

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset/dataset.jsonl", 
                        help="The directory + filename where to read the conversations from")
    parser.add_argument("--trace_by_langsmith", type=bool, default=False)
    parser.add_argument("--langsmith_project_name", type=str, default="extract_preferences")
    parser.add_argument("--experiment_type", type=str, default="in_schema", help="in_schema or out_of_schema")
    parser.add_argument("--output_dir_in_schema", type=str, default="extraction/evaluation/gpt4o/extraction_for_eval_in_schema/dataset")
    parser.add_argument("--output_dir_out_of_schema", type=str, default="extraction/evaluation/gpt4o/extraction_for_eval_out_of_schema/dataset")
    parser.add_argument("--output_file", type=str, default="extraction_for_eval.jsonl")
    return parser.parse_args()

async def main():

    # ====================================================================
    # make sure that the correct experiment type is chosen in the argument
    # ====================================================================

    args = parse_args()
    if args.trace_by_langsmith:
        start_langsmith_tracing(project_name=args.langsmith_project_name)

    if args.experiment_type=="in_schema":
        os.makedirs(args.output_dir_in_schema, exist_ok=True)
    elif args.experiment_type=="out_of_schema":
        os.makedirs(args.output_dir_out_of_schema, exist_ok=True)
    else:
        raise ValueError("Experiment type not supported, either in_schema or out_of_schema")
    
    preference_memory = PreferenceMemory()

    # read out train dataset
    train_dataset_lines = []
    with open(args.dataset_dir, 'r') as file:
        for line in file:
            train_dataset_line = json.loads(line.strip())
            train_dataset_lines.append(train_dataset_line)

    for idx, line in enumerate(track(train_dataset_lines[:50])): # evaluate on 50 dataset lines (= 50 user with 10 conversations each)
        john_user_id = uuid.UUID(line["user_uuid"])
        john_username = f"john-{john_user_id.hex[:4]}"

        for datapoint in tqdm(line["data"]): # one conversation in for loop

            datapoint_strings = datapoint["user_preference"].split(";")
            datapoint_user_preference_dict = {
                "main_category": datapoint_strings[0].strip(),
                "subcategory": datapoint_strings[1].strip(),
                "detail_category": datapoint_strings[2].strip(),
                "attribute": datapoint_strings[3].strip()
            }
            
            main_category_pyd_class = category_to_pyd_category_sub_extra(datapoint_strings[0].strip(), "main_category")
            main_category_pyd_variable = category_to_pyd_category(datapoint_strings[0].strip(), "main_category")
            subcategory_pyd_class = category_to_pyd_category_sub_extra(datapoint_strings[1].strip(), "subcategory")
            subcategory_pyd_variable = category_to_pyd_category(datapoint_strings[1].strip(), "subcategory")
            detail_category_pyd_variable = category_to_pyd_category_sub_extra(datapoint_strings[2].strip(), "detail_category")

            if args.experiment_type=="in_schema":
                # remove ground-truth preference from examples in schema
                ModifiedPreferencesBaseModel = return_pydantic_schema(detail_category_pyd_variable, datapoint_user_preference_dict["attribute"])
            elif args.experiment_type=="out_of_schema":
                # remove ground-truth subcategory and corresponding detail category from schema
                ModifiedPreferencesBaseModel = return_pydantic_schema_wo_category(main_category_pyd_variable, subcategory_pyd_variable, subcategory_pyd_class, detail_category_pyd_variable)
            else:
                raise ValueError("Experiment type not supported, either in_schema or out_of_schema")

            # create extraction function
            extraction_chain = await preference_memory.create_memory_function(
                llm = get_llm_gpt4o(),
                parameters=ModifiedPreferencesBaseModel,
                target_type="user_state",
                name="extract_user_preferences",
                custom_instructions="Only extract long-term user preferences, no temporal desires in the current situation. It is better to not extract any preference than to extract temporal wishes.",
                description="A function that extracts long-term personal preferences of the user in the categories 'Points of Interest', 'Navigation and Routing', 'Vehicle Settings and Comfort', 'Entertainment and Media' and its specified subcategories. It ignores preferences that don't fit into the categories. Don't generate new categories"
            )
            
            conversation_extraction = {}

            keys_to_include_category_eval = ['main_category', 'subcategory', 'detail_category']
            ground_truth_preference_categories = {key: datapoint_user_preference_dict[key] for key in keys_to_include_category_eval if key in datapoint_user_preference_dict}

            conversation_extraction.update({
                "ground_truth_preference": datapoint_user_preference_dict,
                "ground_truth_preference_categories_labels": convert_preference_to_labels(ground_truth_preference_categories)
                })
            
            # convert conversation into expected messages format
            conversation = datapoint["extraction_conversation"]
            messages = [{
                "content": str(list(turn.values())[0]),
                "role": "User" if str(list(turn.keys())[0])=="USER" else "Voice Assistant",
                "name": john_username if str(list(turn.keys())[0])=="USER" else None,
                "metadata": {
                    "user_id": str(john_user_id) if str(list(turn.keys())[0])=="USER" else None,
                },
            } for turn in conversation]

            # perform extraction on one conversation
            messages_string = stringify_conversations(messages)
            print("\nConversation: \n", messages_string)
            output_extraction = extraction_chain.invoke(input={"user_name": john_username, "conversation": messages_string})
            log_debug(f"Outout Extraction: {output_extraction}")

            # validate if output is valid according to preference schema and retry if necessary
            output_extraction_retry, valid_at_try = preference_memory.validate_output_and_retry(output_extraction=output_extraction, pydantic_schema=ModifiedPreferencesBaseModel, chain=extraction_chain, messages_string=messages_string, username=john_username)
            
            conversation_extraction.update({
                "valid_at_try": valid_at_try,
                })
            if valid_at_try==1:
                pass
            elif valid_at_try==2:
                conversation_extraction.update({
                    "failed_extraction_1": json.loads(output_extraction.additional_kwargs['function_call']['arguments']),
                })
                output_extraction = output_extraction_retry
            else:
                conversation_extraction.update({
                    "failed_extraction_1": json.loads(output_extraction.additional_kwargs['function_call']['arguments']),
                    "failed_extraction_2": json.loads(output_extraction_retry.additional_kwargs['function_call']['arguments']),
                })
                

            function_json_extraction = json.loads(output_extraction.additional_kwargs['function_call']['arguments'])
            log_debug(f"Function Json Extraction: {function_json_extraction}")


            # log extraction and extracted labels for evaluation        
            num_preference_counter = 0
            
            if function_json_extraction and not valid_at_try==None:
                for main_category, rest in function_json_extraction.items(): # in for loop are preferences of one main category
                    extracted_preferences_per_category = {"main_category": main_category}
                    for subcategory, rest in rest.items():
                        # discard if preference extracted in no_or_other_preferences
                        if not (subcategory=="no_or_other_preferences"):
                            extracted_preferences_per_category.update({"subcategory": subcategory})
                            for detail_category, value in rest.items():
                                # discard if preference extracted in no_or_other_preferences
                                if not (detail_category=="no_or_other_preferences"):
                                    extracted_preferences_per_category.update({"detail_category": detail_category})
                                    for preference in value:
                                        extracted_preference = extracted_preferences_per_category.copy()
                                        extracted_preference.update({
                                                            "text": preference['user_sentence_preference_revealed'], 
                                                            "attribute": preference['user_preference'],
                                                            "vector": get_embedding().embed_query(preference['user_sentence_preference_revealed']), # vector is replaced when uploading to database in load_extracted_to_database_vctr_dc_attr_text.py
                                                            "user_name": john_username})
                                        
                                        keys_to_include_preference_eval = ['main_category', 'subcategory', 'detail_category', 'attribute']
                                        extracted_preference_eval = {key: extracted_preference[key] for key in keys_to_include_preference_eval if key in extracted_preference}
                                        extracted_preference_categories_eval = {key: extracted_preference[key] for key in keys_to_include_category_eval if key in extracted_preference}

                                        conversation_extraction.update({
                                            f"extracted_preference_{num_preference_counter}_full": extracted_preference,
                                            f"extracted_preference_{num_preference_counter}_core": extracted_preference_eval,
                                            f"extracted_preference_{num_preference_counter}_categories_label": convert_preference_to_labels(extracted_preference_categories_eval)
                                            })
                                        num_preference_counter += 1

            conversation_extraction.update({
                "number_preferences_extracted": num_preference_counter,
                })
                    
            # add extraction result to datapoint
            datapoint["conversation_extracted_preferences"] = conversation_extraction

        # write line extended with extraction results, filter conversations where no extraction is performed
        filtered_data = filter(lambda d: "conversation_extracted_preferences" in d, line["data"])
        line["data"] = list(filtered_data)

        if args.experiment_type=="in_schema":
            with open(os.path.join(args.output_dir_in_schema, args.output_file), 'a') as file:
                file.write(json.dumps(line) + '\n')
        elif args.experiment_type=="out_of_schema":
            with open(os.path.join(args.output_dir_out_of_schema, args.output_file), 'a') as file:
                file.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    asyncio.run(main())