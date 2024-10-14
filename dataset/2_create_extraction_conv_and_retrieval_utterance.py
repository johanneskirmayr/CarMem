import argparse
import configparser
import json
import os
import random
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from tqdm import tqdm

# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local Imports
from dataset.chains.extraction_conversations import ExtractionConversationChain
from dataset.chains.retrieval_utterance import NextConversationQuestionChain
from utils.start_langsmith_tracing import start_langsmith_tracing


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--user_profiles_dir",
        type=str,
        default="dataset/userprofiles/user_profiles.jsonl",
        help="The directory + filename where to read the user profiles from",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--send_to_label_studio", type=bool, default=False)
    parser.add_argument("--trace_by_langchain", type=bool, default=False)
    parser.add_argument(
        "--langsmith_project_name",
        type=str,
        default="create_incar_conv_and_retrieval_utterance",
    )
    parser.add_argument("--mock_gpt4_output", type=bool, default=False)
    parser.add_argument("--mock_next_conversation_question", type=bool, default=False)
    parser.add_argument("--print_pretty", type=bool, default=True)
    parser.add_argument("--send_to_dataset", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="dataset/dataset")
    return parser.parse_args()


def main():

    args = parse_args()
    config = configparser.ConfigParser()
    config.read("config/config.ini")

    run = datetime.now().isoformat()

    if args.trace_by_langchain:
        start_langsmith_tracing(project_name=args.langsmith_project_name)

    extraction_conversation_chain = ExtractionConversationChain()
    next_conversation_questions_chain = NextConversationQuestionChain()

    user_profiles = []
    with open(args.user_profiles_dir, "r") as file:
        for line in file:
            json_object = json.loads(line.strip())
            user_profiles.append(json_object)

    counter = 0
    for idx, user_profile in tqdm(enumerate(user_profiles), total=len(user_profiles)):
        user_profile_df = pd.DataFrame(user_profile["user_profile"])
        data = []
        # random_user_preference = random.randint(1, len(user_profile_df))
        for _, row in user_profile_df.iterrows():
            topic = row["Main Category"] + "; " + row["Subcategory"]
            user_preference = (
                row["Main Category"]
                + "; "
                + row["Subcategory"]
                + "; "
                + row["Detail Category"]
                + "; "
                + row["Attributes"]
            )
            print("User Preference: ", user_preference)

            # ============================================================
            # Generate Extraction Conversation
            # ============================================================

            conversation, metadata_prompt = (
                extraction_conversation_chain.generate_one_conversation(
                    topic=topic, user_preference=user_preference, random_seed=idx
                )
            )  # random_seed so that per user the dynamic pieces are equal
            conversation_list = json.loads(conversation.content)["conversation"]
            print(conversation_list)

            # ============================================================
            # Generate Retrieval Utterance
            # ============================================================

            next_conversation_questions, metadata_prompt_question = (
                next_conversation_questions_chain.generate_next_conversation_questions(
                    topic=topic,
                    user_preference=user_preference,
                    conversation=conversation_list,
                    user_conversation_style=metadata_prompt["user_conversation_style"],
                )
            )
            next_conversation_questions_dict = json.loads(
                next_conversation_questions.content
            )
            print(next_conversation_questions_dict)

            # Parse output to pandas dataframe
            flattened_data = [
                (speaker, text)
                for turn in conversation_list
                for speaker, text in turn.items()
            ]
            df_conversation = pd.DataFrame(flattened_data, columns=["speaker", "text"])
            df_conversation.index.name = "turn"

            meta_info = {
                "user_uuid": user_profile["user_id"],
                "timestamp": datetime.now().isoformat(),
            }
            meta_info.update(metadata_prompt)
            # Convert to JSON format that label studio expects
            label_studio_task = {
                "data": {
                    "messages": [
                        {"role": row["speaker"], "content": row["text"]}
                        for _, row in df_conversation.iterrows()
                    ],
                    "next_conversation_questions": next_conversation_questions_dict,
                    "meta_info": meta_info,
                    "meta_info_str": str(meta_info),
                },
            }

            # print pretty
            if args.print_pretty:
                # Write to file
                os.makedirs(
                    f"dataset/generated_unlabeled_conversations/{run}/", exist_ok=True
                )
                with open(
                    f"dataset/generated_unlabeled_conversations/{run}/conversations.jsonl",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(json.dumps(label_studio_task, ensure_ascii=False) + "\n")
                    # file.flush()
                # Print to file pretty
                with open(
                    f"dataset/generated_unlabeled_conversations/{run}/conversations_print_pretty.txt",
                    "a",
                ) as file:
                    if idx == counter:
                        pretty_string = "=== NEW USER === \n"
                        counter += 1
                    else:
                        pretty_string = ""
                    pretty_string += "=== NEW CONVERSATION === \n"
                    pretty_string += f"Topic: {topic} \n"
                    pretty_string += f"User Preference: {user_preference} \n\n"

                    pretty_string += f"Extraction Conversation: \n"
                    for sentence in label_studio_task["data"]["messages"]:
                        pretty_string += (
                            f"{sentence['role']}: " + f"{sentence['content']}" + "\n"
                        )
                    pretty_string += "\n"

                    pretty_string += f"Next Conversation Questions: \n"
                    pretty_string += (
                        "Next conversation question: "
                        + str(
                            label_studio_task["data"]["next_conversation_questions"][
                                "next_conversation_question_user"
                            ]
                        )
                        + "\n"
                    )
                    # pretty_string += "Second next conversation question: " + str(label_studio_task["data"]["next_conversation_questions"]["next_conversation_question_user_2"]) + "\n"

                    pretty_string += "Meta Info: \n"
                    pretty_string += (
                        "user_uuid: "
                        + str(label_studio_task["data"]["meta_info"]["user_uuid"])
                        + "\n"
                    )
                    pretty_string += (
                        "user_profile: "
                        + str(label_studio_task["data"]["meta_info"]["user_profile"])
                        + "\n"
                    )
                    pretty_string += (
                        "user_conversation_style: "
                        + str(
                            label_studio_task["data"]["meta_info"][
                                "user_conversation_style"
                            ]
                        )
                        + "\n"
                    )
                    pretty_string += (
                        "car_location_city: "
                        + str(
                            label_studio_task["data"]["meta_info"]["car_location_city"]
                        )
                        + "\n"
                    )
                    pretty_string += (
                        "level_of_proactivity_assistant: "
                        + str(
                            label_studio_task["data"]["meta_info"][
                                "level_of_proactivity_assistant"
                            ]
                        )
                        + "\n"
                    )
                    pretty_string += (
                        "preference_strength_modulation: "
                        + str(
                            label_studio_task["data"]["meta_info"][
                                "preference_strength_modulation"
                            ]
                        )
                        + "\n"
                    )
                    pretty_string += (
                        "conversation_length: "
                        + str(
                            label_studio_task["data"]["meta_info"][
                                "conversation_length"
                            ]
                        )
                        + "\n"
                    )
                    pretty_string += (
                        "position_user_preference_in_conv: "
                        + str(
                            label_studio_task["data"]["meta_info"][
                                "position_user_preference_in_conv"
                            ]
                        )
                        + "\n"
                    )
                    pretty_string += (
                        "few_shot_example: "
                        + str(
                            label_studio_task["data"]["meta_info"]["few_shot_examples"]
                        )
                        + "\n"
                    )
                    pretty_string += "\n"

                    file.write(pretty_string + "\n\n")

            if args.send_to_dataset:
                # For Dataset
                datapoint = {
                    "conversation_uuid": str(uuid.uuid4()),
                    "user_uuid": user_profile["user_id"],
                    "user_preference": metadata_prompt["user_preference"],
                    "extraction_conversation": conversation_list,
                    "next_conversation_question": next_conversation_questions_dict[
                        "next_conversation_question_user"
                    ],
                    "meta_info": {
                        "level_of_proactivity_assistant": metadata_prompt[
                            "level_of_proactivity_assistant"
                        ],
                        "preference_strength_modulation": metadata_prompt[
                            "preference_strength_modulation"
                        ],
                        "conversation_length": metadata_prompt["conversation_length"],
                        "position_user_preference_in_conv": metadata_prompt[
                            "position_user_preference_in_conv"
                        ],
                        "few_shot_example": metadata_prompt["few_shot_examples"],
                    },
                }
                data.append(datapoint)

        if args.send_to_dataset:
            datapoint = {
                "user_uuid": user_profile["user_id"],
                "user_profile": metadata_prompt["user_profile"],
                "user_conversation_style": metadata_prompt["user_conversation_style"],
                "car_location_city": metadata_prompt["car_location_city"],
                "user_preferences": user_profile["user_profile"],
                "data": data,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f".jsonl")
            with open(output_path, "a") as file:
                file.write(json.dumps(datapoint) + "\n")

                # file.flush()


if __name__ == "__main__":
    main()
