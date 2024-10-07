import uuid
import json
import asyncio
import os
import time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from rich.progress import track
import seaborn as sns
import numpy as np

import sys
from pathlib import Path
# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.start_langsmith_tracing import start_langsmith_tracing
from utils.llm import get_llm_gpt4o, get_embedding
import argparse
from pymilvus import MilvusClient

from utils.custom_logger import log_debug
from config.config_loader import config
from maintenance.maintenance_functions import Maintenace
from dataset.utils.mapping_detail_category_to_type import detail_category_to_type

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--call_maintenance_function_dir", type=str, default="maintenance/evaluation/gpt4o/call_maintenance_function_for_eval/dataset/call_maintenance_function.jsonl")
    parser.add_argument("--trace_by_langsmith", type=bool, default=False)
    parser.add_argument("--langsmith_project_name", type=str, default="eval_of_maintenance")
    parser.add_argument("--output_dir", type=str, default="maintenance/evaluation/gpt4o/eval_of_maintenance/dataset/")
    parser.add_argument("--output_file", type=str, default="eval_of_maintenance.jsonl")
    parser.add_argument("--write_to_file", type=bool, default=True)
    parser.add_argument("--confusion_matrix", type=bool, default=True)
    return parser.parse_args()

def custom_accuracy_different_mp(y_true, y_pred):
   correct_predictions = 0
   for true, pred in zip(y_true, y_pred):
       if pred == "update_preference" or pred == "append_preference":
           correct_predictions += 1
   return correct_predictions / len(y_pred)

def main():

    args = parse_args()
    if args.trace_by_langsmith:
        start_langsmith_tracing(project_name=args.langsmith_project_name)

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset_lines = []
    with open(args.call_maintenance_function_dir, 'r') as file:
        for line in file:
            train_dataset_line = json.loads(line.strip())
            train_dataset_lines.append(train_dataset_line)

    mp_label_equal, mp_label_negate, mp_label_different = [], [], []
    mnp_label_equal, mnp_label_negate, mnp_label_different = [], [], []

    mean_existing_preferences = []
    invalid_negate_counter = 0
    total_conversations_counter = 0
    for line in track(train_dataset_lines):
        for conversation_data in tqdm(line["data"]):
            
            total_conversations_counter += 1

            conversation_extraction = conversation_data["conversation_extracted_preferences"]
            ground_truth_labels = conversation_extraction["ground_truth_preference_categories_labels"]
            
            maintenance_questions = conversation_data["maintenance_questions"]
            detail_category_type = detail_category_to_type(conversation_extraction["ground_truth_preference"]["detail_category"])
            
            preference_equal_dict = maintenance_questions["question_equal_extraction"]
            preference_equal = preference_equal_dict["extracted_preference_equal_full"]
            preference_equal_tool_call = preference_equal_dict["evaluation"]["maintenance_function_call"]
            mean_existing_preferences.append(preference_equal_dict["evaluation"]["number_preferences_existing"])
            
            preference_negate_dict = maintenance_questions["question_negate_extraction"]
            number_preferences_extracted_negate = preference_negate_dict['number_preferences_negate_extracted']
            if number_preferences_extracted_negate==1:
                preference_negate = preference_negate_dict[f"extracted_preference_negate_0_full"]
                mean_existing_preferences.append(preference_equal_dict["evaluation"]["number_preferences_existing"])
            else:
                invalid_negate_counter += 1
            preference_negate_tool_call = preference_negate_dict["evaluation"]["maintenance_function_call"]

            preference_different_dict = maintenance_questions["question_different_extraction"]
            preference_different = preference_different_dict["extracted_preference_different_full"]
            preference_different_tool_call = preference_different_dict["evaluation"]["maintenance_function_call"]
            mean_existing_preferences.append(preference_equal_dict["evaluation"]["number_preferences_existing"])

            if detail_category_type=="MP":
                mp_label_equal.append(preference_equal_tool_call)
                mp_label_negate.append(preference_negate_tool_call)
                mp_label_different.append(preference_different_tool_call)

                if preference_equal_tool_call=="pass_preference":
                    correct_tool = True
                else:
                    correct_tool = False

                preference_equal_dict["evaluation"].update({
                    "correct_tool": correct_tool
                })

                if preference_negate_tool_call=="update_preference":
                    correct_tool = True
                else:
                    correct_tool = False
                    
                preference_negate_dict["evaluation"].update({
                    "correct_tool": correct_tool
                })

                if preference_different_tool_call=="append_preference":
                    correct_tool = True
                else:
                    correct_tool = False
                    
                preference_different_dict["evaluation"].update({
                    "correct_tool": correct_tool
                })

            else:
                mnp_label_equal.append(preference_equal_tool_call)
                mnp_label_negate.append(preference_negate_tool_call)
                mnp_label_different.append(preference_different_tool_call)

                if preference_equal_tool_call=="pass_preference":
                    correct_tool = True
                else:
                    correct_tool = False

                preference_equal_dict["evaluation"].update({
                    "correct_tool": correct_tool
                })

                if preference_negate_tool_call=="update_preference":
                    correct_tool = True
                else:
                    correct_tool = False
                    
                preference_negate_dict["evaluation"].update({
                    "correct_tool": correct_tool
                })

                if preference_different_tool_call=="append_preference":
                    correct_tool = True
                else:
                    correct_tool = False
                    
                preference_different_dict["evaluation"].update({
                    "correct_tool": correct_tool
                })
        if args.write_to_file:
            # write line extended with extraction results
            with open(os.path.join(args.output_dir, args.output_file), 'a') as file:
                file.write(json.dumps(line) + '\n')

    # overall evaluation
    maintenance_evaluation_dict = {}
    mean_number_existing_preferences = sum(mean_existing_preferences)/len(mean_existing_preferences)
    maintenance_evaluation_dict.update({
        "mean_number_existing_preferences": mean_number_existing_preferences,
        "invalid_negate_counter": invalid_negate_counter,
        "total_conversations_counter": total_conversations_counter,
        })
    
    # ==== evaluation mp ====
    # calculate accuracy
    number_mp = len(mp_label_equal)
    mp_equal_true = ["pass_preference" for i in range(number_mp)]
    mp_labels_equal_true = [0 for i in range(number_mp)]
    mp_equal_accuracy = accuracy_score(y_true=mp_equal_true, y_pred=mp_label_equal)

    # remove None values
    mp_label_negate_cleaned = list(filter(lambda x: x is not None, mp_label_negate))
    number_mp_negate = len(mp_label_negate_cleaned)
    mp_labels_negate_true = [1 for i in range(number_mp_negate)]
    mp_negate_true = ["update_preference" for i in range(number_mp_negate)]
    mp_negate_accuracy = accuracy_score(y_true=mp_negate_true, y_pred=mp_label_negate_cleaned)

    mp_labels_different_true = [2 for i in range(number_mp)]
    mp_different_true = ["append_preference" for i in range(number_mp)]
    mp_different_accuracy = accuracy_score(y_true=mp_different_true, y_pred=mp_label_different)
    mp_different_custom_accuracy = custom_accuracy_different_mp(y_true=mp_different_true, y_pred=mp_label_different)

    maintenance_evaluation_dict.update({
        "mp_number_convs": number_mp,
        "mp_number_negate_questions": number_mp_negate,
        "mp_equal_accuracy": mp_equal_accuracy,
        "mp_negate_accuracy": mp_negate_accuracy,
        "mp_different_accuracy": mp_different_accuracy,
        "mp_different_custom_accuracy": mp_different_custom_accuracy,
        })
    
    if args.confusion_matrix:
        # create heatmap
        concat_label_mp_true = mp_labels_equal_true + mp_labels_negate_true + mp_labels_different_true
        concat_label_mp_pred = mp_label_equal + mp_label_negate_cleaned + mp_label_different

        true_label_mapping = {'equal': 0, 'negate': 1, 'different': 2}
        predicted_label_mapping = {'pass_preference': 0, 'update_preference': 1, 'append_preference': 2}
        # Initialize a 3x3 matrix to zeros
        tally_matrix = np.zeros((3, 3))

        for true_label, pred_label in zip(concat_label_mp_true, concat_label_mp_pred):
            row = true_label  # Get the row index for the true label
            col = predicted_label_mapping[pred_label]  # Get the column index for the predicted label
            tally_matrix[row, col] += 1

        tally_matrix_normalized = tally_matrix / tally_matrix.sum(axis=1, keepdims=True)

        # Define the display labels for the tally matrix
        true_display_labels = ['equal', 'negate', 'different']
        predicted_display_labels = ['pass', 'update', 'append']

        # Plot the tally matrix as a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(tally_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=predicted_display_labels, yticklabels=true_display_labels)
        plt.xlabel('Predicted Labels', fontsize=16)
        plt.ylabel('True Labels', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Tally of Predicted Functions for Each Maintenance Query (mp)', fontsize=16)
        output_file = args.output_file.split(".")[0] + "_mp"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))

        # Plot the normalized tally matrix as a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(tally_matrix_normalized, vmin=0, vmax=1, annot=True, annot_kws={"size": 16}, fmt='.2f', cmap='Blues', xticklabels=predicted_display_labels, yticklabels=true_display_labels)
        plt.xlabel('Predicted Labels', fontsize=16)
        plt.ylabel('True Labels', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Normalized Tally of Predicted Functions for Each Maintenance Query (mp)', fontsize=16)
        output_file = args.output_file.split(".")[0] + "_mp_nm"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))
    # ========

    # ==== evaluation mnp ====
    number_mnp = len(mnp_label_equal)
    mnp_equal_true = ["pass_preference" for i in range(number_mnp)]
    mnp_labels_equal_true = [0 for i in range(number_mnp)]
    mnp_equal_accuracy = accuracy_score(y_true=mnp_equal_true, y_pred=mnp_label_equal)

    # remove None values
    mnp_label_negate_cleaned = list(filter(lambda x: x is not None, mnp_label_negate))
    number_mnp_negate = len(mnp_label_negate_cleaned)
    mnp_negate_true = ["update_preference" for i in range(number_mnp_negate)]
    mnp_labels_negate_true = [1 for i in range(number_mnp_negate)]
    mnp_negate_accuracy = accuracy_score(y_true=mnp_negate_true, y_pred=mnp_label_negate_cleaned)

    mnp_different_true = ["update_preference" for i in range(number_mnp)]
    mnp_labels_different_true = [2 for i in range(number_mnp)]
    mnp_different_accuracy = accuracy_score(y_true=mnp_different_true, y_pred=mnp_label_different)

    maintenance_evaluation_dict.update({
        "mnp_number_convs": number_mnp,
        "mnp_number_negate_questions": number_mnp_negate,
        "mnp_equal_accuracy": mnp_equal_accuracy,
        "mnp_negate_accuracy": mnp_negate_accuracy,
        "mnp_different_accuracy": mnp_different_accuracy,
        })
    
    # create heatmap
    if args.confusion_matrix:
        # Initialize a 3x3 matrix to zeros for mnp
        tally_matrix_mnp = np.zeros((3, 2))

        # Concatenate the true and predicted labels for mnp
        concat_label_mnp_true = mnp_labels_equal_true + mnp_labels_negate_true + mnp_labels_different_true
        concat_label_mnp_pred = mnp_label_equal + mnp_label_negate_cleaned + mnp_label_different

        for true_label, pred_label in zip(concat_label_mnp_true, concat_label_mnp_pred):
            row = true_label  # Get the row index for the true label
            col = predicted_label_mapping[pred_label]  # Get the column index for the predicted label
            tally_matrix_mnp[row, col] += 1

        true_display_labels_mnp = ['equal', 'negate', 'different']
        predicted_display_labels_mnp = ['pass', 'update']

        # Plot the tally matrix as a heatmap for mnp
        plt.figure(figsize=(10, 7))
        sns.heatmap(tally_matrix_mnp, annot=True, fmt='g', cmap='Blues', xticklabels=predicted_display_labels_mnp, yticklabels=true_display_labels_mnp)
        plt.xlabel('Predicted Labels', fontsize=16)
        plt.ylabel('True Labels', fontsize=16)
        plt.title('Tally of Predicted Functions for Each Maintenance Query (mnp)', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        output_file = args.output_file.split(".")[0] + "_mnp"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))

        # Normalize the tally matrix for mnp
        tally_matrix_mnp_normalized = tally_matrix_mnp / tally_matrix_mnp.sum(axis=1, keepdims=True)

        # Plot the normalized tally matrix as a heatmap for mnp
        plt.figure(figsize=(10, 7))
        sns.heatmap(tally_matrix_mnp_normalized, vmin=0, vmax=1, annot=True, annot_kws={"size": 16}, fmt='.2f', cmap='Blues', xticklabels=predicted_display_labels_mnp, yticklabels=true_display_labels_mnp)
        plt.xlabel('Predicted Labels', fontsize=16)
        plt.ylabel('True Labels', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Normalized Tally of Predicted Functions for Each Maintenance Query (mnp)', fontsize=16)
        output_file = args.output_file.split(".")[0] + "_mnp_nm"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))
        # ========

        # ==== bar plot of accuracies ====
        mp_accuracies = [mp_equal_accuracy, mp_negate_accuracy, mp_different_accuracy]
        mnp_accuracies = [mnp_equal_accuracy, mnp_negate_accuracy, mnp_different_accuracy]
        mp_labels = ['pass', 'update', 'append']
        mnp_labels = ['pass', 'update', 'update']

        # Accuracy types
        accuracy_types = ['equal pref.', 'negate pref.', 'different pref.']

        # Colors for each accuracy type
        colors = ['royalblue', 'cornflowerblue', 'lightsteelblue']

        # Setting the positions for the groups
        group_labels = ['Multiple Possible', 'Multiple Not Possible']
        group_pos = np.array([0,0.4])

        # Setting up the plot
        fig, ax = plt.subplots()

        # The width of the bars and the space between them
        bar_width = 0.1
        space_between_bars = 0.0

        # Function to create offset positions for each accuracy type
        def create_offset_positions(center_positions, num_bars, bar_width, space):
            offsets = np.linspace(-num_bars/2.0 * (bar_width + space) + (bar_width/2.0),
                                    num_bars/2.0 * (bar_width + space) - (bar_width/2.0),
                                    num_bars)
            return center_positions[:, None] + offsets

        # Creating offset positions for the bars
        offset_positions = create_offset_positions(group_pos, len(accuracy_types), bar_width, space_between_bars)

        # Plotting the bars
        for i, accuracy in enumerate(accuracy_types):
            # Plotting 'mp' group bars
            ax.bar(offset_positions[0, i], mp_accuracies[i], bar_width, label=accuracy if group_pos[0] == 0 else "", color=colors[i])
            ax.text(offset_positions[0, i], 0.4, mp_labels[i], ha='center', rotation='vertical')
            # Plotting 'mnp' group bars, if data is available
            if mnp_accuracies[i] is not None:
                ax.bar(offset_positions[1, i], mnp_accuracies[i], bar_width, color=colors[i])
                ax.text(offset_positions[1, i], 0.4, mnp_labels[i], ha='center', rotation='vertical')

        # Adding the labels, title, and legend
        ax.set_xlabel('Detail Category Type')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Scores by Group')
        ax.set_xticks(group_pos)
        ax.set_xticklabels(group_labels)
        ax.set_ylim([0,1.1])
        ax.legend(title='Maintenance Query', loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        output_file = args.output_file.split(".")[0] + "_accuracy"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))
        # Displaying the plot
        plt.show()

    if args.write_to_file:
        with open(os.path.join(args.output_dir, args.output_file), 'a') as file:
            file.write(json.dumps(maintenance_evaluation_dict) + '\n')


if __name__=="__main__":
    main()

