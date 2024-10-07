"""
File to call the maintenance functions for each extracted preference from the maintenance utterances.
"""

import uuid
import json
import asyncio
import os
import time
from tqdm import tqdm
from rich.progress import track
from langchain.callbacks import get_openai_callback

import sys
from pathlib import Path
# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from extraction.pydantic_schemas.categories_pydantic import PreferencesFunctionOutput
from utils.start_langsmith_tracing import start_langsmith_tracing
from utils.llm import get_llm_gpt4o, get_embedding
import argparse
from pymilvus import MilvusClient

from utils.custom_logger import log_debug
from config.config_loader import config
from maintenance.maintenance_functions import Maintenace
from dataset.utils.mapping_detail_category_to_type import detail_category_to_type
from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--extraction_maintenance_result_dir", type=str, default="maintenance/evaluation/gpt4o/extraction_maintenance_utterances/dataset/extraction_maintenance_utterances.jsonl")
    parser.add_argument("--trace_by_langsmith", type=bool, default=False)
    parser.add_argument("--langsmith_project_name", type=str, default="call_maintenance_function")
    parser.add_argument("--output_dir", type=str, default="maintenance/evaluation/gpt4o/call_maintenance_function_for_eval/dataset/")
    parser.add_argument("--output_file", type=str, default="call_maintenance_function.jsonl")
    parser.add_argument("--perform_function", type=bool, default=False, help="wether to actually perform the function in the database or just simulate")
    return parser.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    if args.trace_by_langsmith:
        start_langsmith_tracing(project_name=args.langsmith_project_name)
    
    milvus_client = MilvusClient(uri=f"http://{config.get('database', 'host')}:{config.get('database', 'port')}")
    maintenance = Maintenace()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset_lines = []
    with open(args.extraction_maintenance_result_dir, 'r') as file:
        for line in file:
            train_dataset_line = json.loads(line.strip())
            train_dataset_lines.append(train_dataset_line)

    for line in track(train_dataset_lines):
        for conversation_data in tqdm(line["data"]):
            conversation_extraction = conversation_data["conversation_extracted_preferences"]
            maintenance_questions = conversation_data["maintenance_questions"]
            preference_equal_dict = maintenance_questions["question_equal_extraction"]
            preference_equal = preference_equal_dict["extracted_preference_equal_full"]
            
            ground_truth_labels = conversation_extraction["ground_truth_preference_categories_labels"]
            num_preference_counter = maintenance_questions["question_negate_extraction"]["number_preferences_negate_extracted"]
            
            preference_negate_dict = maintenance_questions["question_negate_extraction"]
            if num_preference_counter==1:
                preference_labels = maintenance_questions["question_negate_extraction"][f"extracted_preference_negate_0_categories_label"]
                if preference_labels==ground_truth_labels:
                    preference_negate = preference_negate_dict[f"extracted_preference_negate_0_full"]
                else:
                    log_debug(f"negate preference not extracted on correct category")
                    preference_negate = {}
            else:
                log_debug(f"Either no or multiple preferences extracted from negate question")
                preference_negate = {}
                    
            
            preference_different_dict = maintenance_questions["question_different_extraction"]
            preference_different = preference_different_dict["extracted_preference_different_full"]
            
            dict_list = [preference_equal, preference_negate, preference_different]
            for idx, extracted_preference in enumerate(dict_list):
                eval_preference_dict = {}
                if idx == 0:
                    question = "equal"
                elif idx==1:
                    question = "negate"
                elif idx==2:
                    question = "different"

                if extracted_preference:
                    # time.sleep(0.1)
                    keys_to_include_category_eval = ['main_category', 'subcategory', 'detail_category', 'text', 'attribute', 'user_name']
                    extracted_preference_slim = {key: extracted_preference[key] for key in keys_to_include_category_eval if key in extracted_preference}
                    print("\nIncoming Preference:\n")
                    print(f"{question}: {extracted_preference_slim}")
                    expr = f"user_name=='{conversation_data['user_uuid']}' && main_category=='{extracted_preference['main_category']}' && subcategory=='{extracted_preference['subcategory']}' && detail_category=='{extracted_preference['detail_category']}'"
                    
                    # if existing preferences are empty, make sure you uploaded the extracted preferences to milvus with 'extraction/3_load_extracted_to_database.py' or 'extraction/3_load_extracted_to_database_vctr_dc_attr_text.py'
                    existing_preferences = milvus_client.query(
                        collection_name=config.get('database', 'collection_name'),
                        filter=expr,
                        output_fields=["pk", "main_category", "subcategory", "detail_category", "text", "attribute", "user_name"]
                    )
                    eval_preference_dict.update({
                        "number_preferences_existing": len(existing_preferences)
                    })

                    print("\nExisting Preferences within category:\n")
                    print(existing_preferences)
                    if not existing_preferences:
                        if args.perform_function:
                            milvus_client.insert(
                                collection_name=config.get('database', 'collection_name'),
                                data=extracted_preference
                            )
                            tool_call = 'insert_preference'
                        
                    else:
                        if detail_category_to_type(extracted_preference['detail_category'])=="MP":
                            print("\n Category Type 'MP': performing filter...")
                            with get_openai_callback() as cb:
                                run_tool_answer, tool_call = maintenance.filter_extracted_preference_mp(incoming_preference=extracted_preference, existing_preferences=existing_preferences, perform_function=args.perform_function)
                        elif detail_category_to_type(extracted_preference['detail_category'])=="MNP":
                            print("\n Category Type 'MNP': performing filter...")
                            with get_openai_callback() as cb:
                                run_tool_answer, tool_call = maintenance.filter_extracted_preference_mnp(incoming_preference=extracted_preference, existing_preferences=existing_preferences, perform_function=args.perform_function)
                        else:
                            raise ValueError(f"Detail category does not exist or mapping wrong")
                else:
                    tool_call = None

                eval_preference_dict.update({"maintenance_function_call": tool_call})
                if question=="equal":
                    preference_equal_dict["evaluation"] = eval_preference_dict
                elif question=="negate":
                    preference_negate_dict["evaluation"] = eval_preference_dict
                elif question=="different":
                    preference_different_dict["evaluation"] = eval_preference_dict

        # write line extended with extraction results, filter conversations where no extraction is performed
        filtered_data = filter(lambda d: "evaluation" in d["maintenance_questions"]["question_equal_extraction"], line["data"])
        line["data"] = list(filtered_data)
        with open(os.path.join(args.output_dir, args.output_file), 'a') as file:
            file.write(json.dumps(line) + '\n')

if __name__=="__main__":
    main()