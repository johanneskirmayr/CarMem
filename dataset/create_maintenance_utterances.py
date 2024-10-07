import argparse
import os
import json
import pandas as pd
import random
from tqdm import tqdm
from rich.progress import track

import sys
from pathlib import Path
# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from itertools import cycle

from dataset.chains.maintenance_utterance import MaintenanceQuestionChain
from utils.start_langsmith_tracing import start_langsmith_tracing
from dataset.utils.mapping_detail_category_to_type import detail_category_to_type

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset/dataset.jsonl", 
                        help="The directory + filename where to read the conversations from")
    parser.add_argument("--trace_by_langsmith", type=bool, default=False)
    parser.add_argument("--langsmith_project_name", type=str, default="create_maintenance_utterance")
    parser.add_argument("--output_dir", type=str, default="dataset/dataset/")
    parser.add_argument("--output_file", type=str, default="dataset_full.jsonl")
    return parser.parse_args()

def main():

    args = parse_args()
    if args.trace_by_langsmith:
        start_langsmith_tracing(project_name=args.langsmith_project_name)

    os.makedirs(args.output_dir, exist_ok=True)

    maintenance_question_chain = MaintenanceQuestionChain()

    # read out categories
    df_categories = pd.read_csv('dataset/categories_v4.csv', delimiter='#')
    # read out train dataset
    dataset_lines = []
    with open(args.dataset_dir, 'r') as file:
        for line in file:
            train_dataset_line = json.loads(line.strip())
            dataset_lines.append(train_dataset_line)

    for idx, train_dataset_line in enumerate(track(dataset_lines)):
        for conversation_data in tqdm(train_dataset_line["data"]):

            # sample different attribute from same detail_category
            detail_category = conversation_data["user_preference"].split(";")[2].strip()
            attribute = conversation_data["user_preference"].split(";")[-1].strip()
            same_category_attributes = df_categories.loc[df_categories['Detail Category'] == detail_category, 'Attributes']
            same_category_attributes = [attr.strip() for attr in same_category_attributes.iloc[0].split(",")]
            same_category_attributes.remove(attribute)
            if same_category_attributes:
                different_attribute = random.choice(same_category_attributes)
            elif detail_category=="Need for Handicapped Accessible Parking": # this is the only category where there is only one attribute
                different_attribute = "No"
            else:
                raise ValueError
            
            different_preference = ";".join(conversation_data["user_preference"].split(";")[:-1] + [" " + different_attribute])
            
            #sample_different_attribute = 
            maintenance_questions, metadata = maintenance_question_chain.generate_maintenance_questions(
                conversation=conversation_data["extraction_conversation"],
                user_preference=conversation_data["user_preference"],
                different_preference = different_preference,
                user_conversation_style=train_dataset_line["user_conversation_style"],
                detail_category_type=detail_category_to_type(detail_category)
                )
            
            maintenance_questions = json.loads(maintenance_questions.content)
            
            conversation_data["maintenance_questions"] = maintenance_questions
            conversation_data["maintenance_questions"]["different_attribute"] = different_attribute
            
        with open(os.path.join(args.output_dir, args.output_file), 'a') as file:
            file.write(json.dumps(train_dataset_line) + '\n')


if __name__ == "__main__":
    main()