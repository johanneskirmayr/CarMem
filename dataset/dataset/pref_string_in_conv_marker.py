"""
Script to mark if the preference string is in the conversation or in the user sentences only. Relevant information for named entity recognition.
"""

import json
import os

# Define relative paths
input_file_path = os.path.join(os.path.dirname(__file__), 'dataset.jsonl')
output_file_path = os.path.join(os.path.dirname(__file__), 'dataset_w_string.jsonl')

dataset_lines = []
with open(input_file_path, 'r') as file:
    for line in file:
        dataset_line = json.loads(line.strip())
        dataset_lines.append(dataset_line)

for line in dataset_lines:
    for conversation_data in line["data"]:
        concat_conversation = " ".join([str(list(turn.values())[0]) for turn in conversation_data["extraction_conversation"]])
        concat_conversation_user_only = " ".join([str(list(turn.values())[0]) for turn in conversation_data["extraction_conversation"][::2]])
        preference = conversation_data["user_preference"].split(";")[-1].strip()

        conversation_data["preference_string_in_conversation"] = False
        conversation_data["preference_string_in_user_sentences"] = False
        if preference.lower() in concat_conversation.lower():
            conversation_data["preference_string_in_conversation"] = True
        if preference.lower() in concat_conversation_user_only.lower():
            conversation_data["preference_string_in_user_sentences"] = True

    with open(output_file_path, 'a') as file:
        file.write(json.dumps(line) + '\n')