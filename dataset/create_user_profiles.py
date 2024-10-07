import re
import pandas as pd
import random
import json
from datetime import datetime
import uuid
import argparse
import os

import sys
from pathlib import Path
# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local Imports
from dataset.chains.sample_user_profiles import sample_user_profiles
from dataset.chains.extraction_conversations import ExtractionConversationChain

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/userprofiles",
        help="The output directory where to store the user profile files"
    )
    parser.add_argument(
        "--path_to_categories",
        type=str,
        default="dataset/categories_v4.csv",
        help="The input file where the categories are located"
    )
    parser.add_argument(
        "--num_user",
        type=int,
        default=100,
        help="The number of user profiles to create, will be splitted in train, val, test set",
    )
    parser.add_argument(
        "--num_attributes_per_user",
        type=int,
        default=10,
        help="The number of attributes per user profile",
    )
    parser.add_argument(
        "--train_val_test_split",
        type=list[float],
        default=[1.0,0.0,0.0],
        help="The data split for train, val, test set",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    return parser.parse_args()

def dataframe_to_json(df):
    user_data = {
        "user_id": str(uuid.uuid4()),
        "user_profile": df.to_dict(orient='records')
    }
    return user_data

def main():

    args = parse_args()

    random.seed(args.random_seed)
    
    # Sample User Profiles
    train_user_profiles, _, _ = sample_user_profiles(args.path_to_categories, number_user_profiles=round(args.train_val_test_split[0]*args.num_user), number_attributes_per_user=args.num_attributes_per_user)
    # val_user_profiles, _, _ = sample_user_profiles(args.path_to_categories, number_user_profiles=round(args.train_val_test_split[1]*args.num_user), number_attributes_per_user=args.num_attributes_per_user)
    # test_user_profiles, _, _ = sample_user_profiles(args.path_to_categories, number_user_profiles=round(args.train_val_test_split[2]*args.num_user), number_attributes_per_user=args.num_attributes_per_user)
    
    # Create output direction if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert each pandas dataframe in the list to jsonl line with "user_id" and "user_profile"
    train_output_path = os.path.join(args.output_dir, 'user_profiles.jsonl')
    with open(train_output_path, 'w') as file:
        for train_user_profile in train_user_profiles:
            json_user_data = dataframe_to_json(train_user_profile)
            file.write(json.dumps(json_user_data) + '\n')
    
    # val_output_path = os.path.join(args.output_dir, 'val_user_profiles.jsonl')
    # with open(val_output_path, 'w') as file:
    #     for val_user_profile in val_user_profiles:
    #         json_user_data = dataframe_to_json(val_user_profile)
    #         file.write(json.dumps(json_user_data) + '\n')

    # test_output_path = os.path.join(args.output_dir, 'test_user_profiles.jsonl')
    # with open(test_output_path, 'w') as file:
    #     for test_user_profile in test_user_profiles:
    #         json_user_data = dataframe_to_json(test_user_profile)
    #         file.write(json.dumps(json_user_data) + '\n')

if __name__ == '__main__':
    main()