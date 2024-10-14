"""
File to load the extracted preferences to the Milvus database.
In this file the embeddings are created from the string of the sentence where the preference got extracted.
Only preferences with extraction accuracy of 1.0 are loaded to the database.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

from pymilvus import MilvusClient

# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

from config.config_loader import config
from document_store.milvus2_preference_store import Milvus2PreferenceStore


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extracted_prefs_dir",
        type=str,
        default="extraction/evaluation/gpt4o/eval_of_extraction_in_schema/dataset/eval_of_extraction.jsonl",
        help="The directory + filename where to read the conversations from",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    load_dotenv()
    milvus_preference_store = Milvus2PreferenceStore()
    collection_name = "user_preferences_vctr_text"
    milvus_preference_store._create_collection_and_index(
        collection_name=collection_name, recreate_collection=True
    )
    milvus_client = MilvusClient(
        uri=f"http://{config.get('database', 'host')}:{config.get('database', 'port')}"
    )

    # read out train dataset
    extraction_for_eval_lines = []
    with open(args.extracted_prefs_dir, "r") as file:
        for line in file:
            extraction_for_eval_line = json.loads(line.strip())
            extraction_for_eval_lines.append(extraction_for_eval_line)

    used_conversation_uuids = []
    for line in extraction_for_eval_lines:
        if "data" in line:
            for conversation in line["data"]:
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

                extracted_preference = conversation_extraction[
                    "extracted_preference_0_full"
                ]
                extracted_preference["user_name"] = line["user_uuid"]
                extracted_preference["pk"] = conversation["conversation_uuid"]
                used_conversation_uuids.append(conversation["conversation_uuid"])
                milvus_client.insert(
                    collection_name=collection_name, data=extracted_preference
                )
    with open(
        os.path.join(Path(__file__).parent, "used_conversation_uuids.pkl"), "wb"
    ) as file:
        pickle.dump(used_conversation_uuids, file)


if __name__ == "__main__":
    main()
