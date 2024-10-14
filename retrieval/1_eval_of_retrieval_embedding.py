"""
File to evaluate the retrieval of preferences based on the embedding of the retrieval utterance question.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from pymilvus import MilvusClient

from config.config_loader import config
from extraction.mapping_category_to_pyd_category import category_to_pyd_category
from utils.llm import get_embedding
from utils.start_langsmith_tracing import start_langsmith_tracing

# import tikzplotlib


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extraction_result_dir",
        type=str,
        default="extraction/evaluation/gpt4o/eval_of_extraction_in_schema/dataset/eval_of_extraction.jsonl",
    )
    parser.add_argument("--trace_by_langsmith", type=bool, default=False)
    parser.add_argument(
        "--langsmith_project_name", type=str, default="eval_of_retrieval_embedding"
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="text_attr_dc",
        help="The type of embedding to use for retrieval, either text_attr_dc or text",
    )
    parser.add_argument(
        "--output_dir_text_attr_dc",
        type=str,
        default="retrieval/evaluation/gpt4o/eval_of_rerieval_embedding/dataset/text_attr_dc/",
    )
    parser.add_argument(
        "--output_dir_text",
        type=str,
        default="retrieval/evaluation/gpt4o/eval_of_rerieval_embedding/dataset/text/",
    )
    parser.add_argument(
        "--output_file", type=str, default="eval_of_retrieval_embedding.jsonl"
    )
    parser.add_argument("--write_to_file", type=bool, default=True)
    parser.add_argument("--confusion_matrix", type=bool, default=False)
    parser.add_argument("--label_song_is_genre", type=bool, default=True)
    parser.add_argument(
        "--collection_name_text_attr_dc",
        type=str,
        default="user_preferences_vctr_dc_attr_text",
    )
    parser.add_argument(
        "--collection_name_text", type=str, default="user_preferences_vctr_text"
    )
    return parser.parse_args()


def main():

    args = parse_args()
    if args.trace_by_langsmith:
        start_langsmith_tracing(project_name=args.langsmith_project_name)

    if args.embedding_type == "text_attr_dc":
        args.output_dir = args.output_dir_text_attr_dc
        args.collection_name = args.collection_name_text_attr_dc
    elif args.embedding_type == "text":
        args.output_dir = args.output_dir_text
        args.collection_name = args.collection_name_text

    os.makedirs(args.output_dir, exist_ok=True)
    load_dotenv()
    milvus_client = MilvusClient(
        uri=f"http://{config.get('database', 'host')}:{config.get('database', 'port')}"
    )

    # read out train dataset
    extraction_for_eval_lines = []
    with open(args.extraction_result_dir, "r") as file:
        for line in file:
            extraction_for_eval_line = json.loads(line.strip())
            extraction_for_eval_lines.append(extraction_for_eval_line)

    all_ground_truth_preference_retrieved_top_ssc = []
    all_ground_truth_preference_retrieved_top_ssc_p1 = []
    all_ground_truth_preference_retrieved_top_ssc_p2 = []
    total_number_same_subcategory_preferences = 0
    total_number_questions = 0
    total_ratio_relevant_ssc = []
    total_ratio_relevant_ssc_p1 = []
    total_ratio_relevant_ssc_p2 = []
    for line in extraction_for_eval_lines[:-1]:  # last line is the evaluation scores
        user_id = line["user_uuid"]
        for conversation in line["data"]:
            start_time = time.time()
            try:  # only score conversations where extraction is performed
                conversation_extraction = conversation[
                    "conversation_extracted_preferences"
                ]
                if not (
                    conversation_extraction["evaluation"]["conv_detail_accuracy"] == 1.0
                ):
                    continue
            except KeyError as e:
                print(
                    f"Key Error: {e} \n Probably no extraction performed for conversation {conversation['conversation_uuid']}"
                )
                break
            conversation_uuid = conversation["conversation_uuid"]
            user_preference = conversation["user_preference"].split(";")
            subcategory = user_preference[1].strip()
            subcategory_pyd = category_to_pyd_category(
                input_string=subcategory, category="subcategory"
            )
            # get number of preference for same subcategory
            same_subcategory_preferences = milvus_client.query(
                collection_name=args.collection_name,
                filter=f"user_name=='{user_id}' && subcategory=='{subcategory_pyd}'",
            )
            number_same_subcategory_preferences = len(same_subcategory_preferences)
            total_number_same_subcategory_preferences += (
                number_same_subcategory_preferences
            )
            next_conversation_question = conversation["next_conversation_question"]
            next_conversation_question_embedding = get_embedding().embed_query(
                next_conversation_question
            )
            medium_time = time.time()
            embedding_time = medium_time - start_time
            print(embedding_time)
            retrieved_preferences = milvus_client.search(
                collection_name=args.collection_name,
                data=[next_conversation_question_embedding],
                filter=f"user_name=='{user_id}'",
                limit=10,
                search_params={"params": {"level": 3}},
                output_fields=["*"],
            )
            retrieved_ids = [
                retrieved_preference["id"]
                for retrieved_preference in retrieved_preferences[0]
            ]

            ratio_relevant_irrelevant_ssc = [0, 0, 0]
            # top-(number_same_subcategory)
            ground_truth_preference_retrieved_top_ssc = False
            for id, preference in zip(
                retrieved_ids[:number_same_subcategory_preferences],
                retrieved_preferences[0],
            ):
                if preference["entity"]["subcategory"] == subcategory_pyd:
                    ratio_relevant_irrelevant_ssc[0] += 1
                else:
                    ratio_relevant_irrelevant_ssc[1] += 1

                if id == conversation_uuid:
                    ground_truth_preference_retrieved_top_ssc = True

            ratio_relevant_irrelevant_ssc[2] = (
                ratio_relevant_irrelevant_ssc[0] / number_same_subcategory_preferences
            )
            total_ratio_relevant_ssc.append(ratio_relevant_irrelevant_ssc[2])
            all_ground_truth_preference_retrieved_top_ssc.append(
                ground_truth_preference_retrieved_top_ssc
            )

            # top-(number_same_subcategory + 1)
            ratio_relevant_irrelevant_ssc_p1 = [0, 0, 0]
            ground_truth_preference_retrieved_top_ssc_p1 = False
            for id, preference in zip(
                retrieved_ids[: number_same_subcategory_preferences + 1],
                retrieved_preferences[0],
            ):
                if preference["entity"]["subcategory"] == subcategory_pyd:
                    ratio_relevant_irrelevant_ssc_p1[0] += 1
                else:
                    ratio_relevant_irrelevant_ssc_p1[1] += 1

                if id == conversation_uuid:
                    ground_truth_preference_retrieved_top_ssc_p1 = True

            ratio_relevant_irrelevant_ssc[2] = ratio_relevant_irrelevant_ssc_p1[0] / (
                number_same_subcategory_preferences + 1
            )
            total_ratio_relevant_ssc_p1.append(ratio_relevant_irrelevant_ssc[2])
            all_ground_truth_preference_retrieved_top_ssc_p1.append(
                ground_truth_preference_retrieved_top_ssc_p1
            )

            # top-(number_same_subcategory + 2)
            ratio_relevant_irrelevant_ssc_p2 = [0, 0, 0]
            ground_truth_preference_retrieved_top_ssc_p2 = False
            for id, preference in zip(
                retrieved_ids[: number_same_subcategory_preferences + 2],
                retrieved_preferences[0],
            ):
                if preference["entity"]["subcategory"] == subcategory_pyd:
                    ratio_relevant_irrelevant_ssc_p2[0] += 1
                else:
                    ratio_relevant_irrelevant_ssc_p2[1] += 1

                if id == conversation_uuid:
                    ground_truth_preference_retrieved_top_ssc_p2 = True

            ratio_relevant_irrelevant_ssc[2] = ratio_relevant_irrelevant_ssc_p2[0] / (
                number_same_subcategory_preferences + 2
            )
            total_ratio_relevant_ssc_p2.append(ratio_relevant_irrelevant_ssc[2])
            all_ground_truth_preference_retrieved_top_ssc_p2.append(
                ground_truth_preference_retrieved_top_ssc_p2
            )
            # remove vectors
            for item in retrieved_preferences[0]:
                item["entity"]["vector"] = None
            conversation["evaluation_next_conversation_question_embedding"] = {
                # "next_conversation_question_embedding": next_conversation_question_embedding,
                "number_same_subcategory_preferences": number_same_subcategory_preferences,
                "retrieved_ids": retrieved_ids,
                "retrieved_preferences_top_ssc_p2": retrieved_preferences[0][
                    : number_same_subcategory_preferences + 2
                ],
                "ground_truth_preference_retrieved_top_ssc": ground_truth_preference_retrieved_top_ssc,
                "ratio_relevant_irrelevant_ssc": ratio_relevant_irrelevant_ssc[2],
                "ground_truth_preference_retrieved_top_ssc_p1": ground_truth_preference_retrieved_top_ssc_p1,
                "ratio_relevant_irrelevant_ssc_p1": ratio_relevant_irrelevant_ssc_p1[2],
                "ground_truth_preference_retrieved_top_ssc_p2": ground_truth_preference_retrieved_top_ssc_p2,
                "ratio_relevant_irrelevant_ssc_p2": ratio_relevant_irrelevant_ssc_p2[2],
            }
            total_number_questions += 1

            end_time = time.time()
            latency = end_time - start_time
            print(latency)

        if args.write_to_file:
            # write line extended with extraction results
            with open(os.path.join(args.output_dir, args.output_file), "a") as file:
                file.write(json.dumps(line) + "\n")

    # aggregate evaluations
    accuracy_ground_truth_preference_retrieved_top_ssc = sum(
        all_ground_truth_preference_retrieved_top_ssc
    ) / len(all_ground_truth_preference_retrieved_top_ssc)
    accuracy_ground_truth_preference_retrieved_top_ssc_p1 = sum(
        all_ground_truth_preference_retrieved_top_ssc_p1
    ) / len(all_ground_truth_preference_retrieved_top_ssc_p1)
    accuracy_ground_truth_preference_retrieved_top_ssc_p2 = sum(
        all_ground_truth_preference_retrieved_top_ssc_p2
    ) / len(all_ground_truth_preference_retrieved_top_ssc_p2)
    mean_ratio_relevant_ssc = sum(total_ratio_relevant_ssc) / len(
        total_ratio_relevant_ssc
    )
    mean_ratio_relevant_ssc_p1 = sum(total_ratio_relevant_ssc_p1) / len(
        total_ratio_relevant_ssc_p1
    )
    mean_ratio_relevant_ssc_p2 = sum(total_ratio_relevant_ssc_p2) / len(
        total_ratio_relevant_ssc_p2
    )
    all_evaluation_scores = {
        "accuracy_ground_truth_preference_retrieved_top_ssc": accuracy_ground_truth_preference_retrieved_top_ssc,
        "accuracy_ground_truth_preference_retrieved_top_ssc_p1": accuracy_ground_truth_preference_retrieved_top_ssc_p1,
        "accuracy_ground_truth_preference_retrieved_top_ssc_p2": accuracy_ground_truth_preference_retrieved_top_ssc_p2,
        "mean_number_same_subcategory_preferences": total_number_same_subcategory_preferences
        / total_number_questions,
        "mean_questions_per_user": total_number_questions
        / len(extraction_for_eval_lines),
        "mean_ratio_relevant_ssc": mean_ratio_relevant_ssc,
        "mean_ratio_relevant_ssc_p1": mean_ratio_relevant_ssc_p1,
        "mean_ratio_relevant_ssc_p2": mean_ratio_relevant_ssc_p2,
    }
    if args.write_to_file:
        with open(os.path.join(args.output_dir, args.output_file), "a") as file:
            file.write(json.dumps(all_evaluation_scores) + "\n")


if __name__ == "__main__":
    main()
