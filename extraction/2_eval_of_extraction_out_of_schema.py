"""
File to evaluate the extraction of preferences for the out-of-schema experiment.
The results for individual conversations are written in-line by enhancing the jsonl line with the evaluation results.
The overall evaluation results are written as the last line in the output file.
Additionally creates confusion matrices for the main, sub and detail category.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from utils.start_langsmith_tracing import start_langsmith_tracing


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extraction_for_eval_data_dir",
        type=str,
        default="extraction/evaluation/gpt4o/extraction_for_eval_out_of_schema/dataset/extraction_for_eval.jsonl",
    )
    parser.add_argument("--trace_by_langsmith", type=bool, default=False)
    parser.add_argument(
        "--langsmith_project_name", type=str, default="eval_of_extraction"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extraction/evaluation/gpt4o/eval_of_extraction_out_of_schema/dataset/",
    )
    parser.add_argument("--output_file", type=str, default="eval_of_extraction.jsonl")
    parser.add_argument("--write_to_file", type=bool, default=True)
    parser.add_argument("--confusion_matrix", type=bool, default=True)
    parser.add_argument("--label_song_is_genre", type=bool, default=True)
    return parser.parse_args()


def main():

    args = parse_args()
    if args.trace_by_langsmith:
        start_langsmith_tracing(project_name=args.langsmith_project_name)

    os.makedirs(args.output_dir, exist_ok=True)

    # read out train dataset
    extraction_for_eval_lines = []
    with open(args.extraction_for_eval_data_dir, "r") as file:
        for line in file:
            extraction_for_eval_line = json.loads(line.strip())
            extraction_for_eval_lines.append(extraction_for_eval_line)

    # calculate quantitative scores
    mlb_main = MultiLabelBinarizer(classes=range(0, 4))
    mlb_sub = MultiLabelBinarizer(classes=range(4, 15))
    mlb_detail = MultiLabelBinarizer(classes=range(15, 56))

    string_in_conversation_counter = 0
    string_in_conversation_counter_user_only = 0
    string_in_extracted_preference_counter = 0
    string_in_extracted_preference_counter_user_only = 0

    label_main_category, label_subcategory, label_detail_category = (
        ([], []),
        ([], []),
        ([], []),
    )  # ([[ground_truth_label_conv1], ...], [[predicted_labels_conv1], ...])
    label_main_category_empty, label_subcategory_empty, label_detail_category_empty = (
        ([], []),
        ([], []),
        ([], []),
    )  # ([[ground_truth_label_conv1], ...], [[predicted_labels_conv1], ...])
    (
        label_main_category_one_pref_only,
        label_subcategory_one_pref_only,
        label_detail_category_one_pref_only,
    ) = (
        ([], []),
        ([], []),
        ([], []),
    )  # ([[ground_truth_label_conv1], ...], [[predicted_labels_conv1], ...])
    (
        label_main_category_true_repeated,
        label_subcategory_true_repeated,
        label_detail_category_true_repeated,
    ) = (
        ([], []),
        ([], []),
        ([], []),
    )  # ([[ground_truth_label_conv1], ...], [[predicted_labels_conv1], ...])

    confusion_matrix_main = None
    confusion_matrix_sub = None
    confusion_matrix_detail = None

    counter = 0
    number_preferences_extracted_total = 0
    number_preferences_extracted_other_total = 0
    valid_at_try_1_counter = 0
    valid_at_try_2_counter = 0
    not_valid_extraction_counter = 0

    for i in list(range(0, 50)):  # + list(range(20, 50)) + list(range(60, 70)):
        line = extraction_for_eval_lines[i]
        # for line in extraction_for_eval_lines:
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

            number_preferences_valid_extracted = conversation_extraction[
                "number_preferences_extracted"
            ]
            # main_category
            label_main_true = [
                conversation_extraction["ground_truth_preference_categories_labels"][0]
            ]
            label_main_category[0].extend([label_main_true])
            label_main_true_empty = []
            label_main_category_empty[0].extend(
                [[]]
            )  # empty because we don't want to extract the ground truth

            label_main_pred = [
                conversation_extraction[f"extracted_preference_{idx}_categories_label"][
                    0
                ]
                for idx in range(number_preferences_valid_extracted)
            ]
            label_main_category[1].extend([label_main_pred])

            conv_main_accuracy = accuracy_score(
                y_true=mlb_main.fit_transform([label_main_true_empty]),
                y_pred=mlb_main.transform([label_main_pred]),
            )

            # subcategory
            label_sub_true = [
                conversation_extraction["ground_truth_preference_categories_labels"][1]
            ]
            label_subcategory[0].extend([label_sub_true])
            label_sub_true_empty = []
            label_subcategory_empty[0].extend([[]])

            label_sub_pred = [
                conversation_extraction[f"extracted_preference_{idx}_categories_label"][
                    1
                ]
                for idx in range(number_preferences_valid_extracted)
            ]
            label_subcategory[1].extend([label_sub_pred])

            conv_sub_accuracy = accuracy_score(
                y_true=mlb_sub.fit_transform([label_sub_true_empty]),
                y_pred=mlb_sub.transform([label_sub_pred]),
            )

            # detail_category
            label_detail_true = [
                conversation_extraction["ground_truth_preference_categories_labels"][2]
            ]
            label_detail_category[0].extend([label_detail_true])
            label_detail_true_empty = []
            label_detail_category_empty[0].extend([[]])

            label_detail_pred = [
                conversation_extraction[f"extracted_preference_{idx}_categories_label"][
                    2
                ]
                for idx in range(number_preferences_valid_extracted)
            ]
            if args.label_song_is_genre:
                for i in range(len(label_detail_pred)):
                    if label_detail_pred[i] == 50:
                        label_detail_pred[i] = 49
            label_detail_category[1].extend([label_detail_pred])

            conv_detail_accuracy = accuracy_score(
                y_true=mlb_detail.fit_transform([label_detail_true_empty]),
                y_pred=mlb_detail.transform([label_detail_pred]),
            )

            number_preferences_extracted = conversation_extraction[
                "number_preferences_extracted"
            ]
            number_preferences_extracted_total += number_preferences_extracted

            if conversation_extraction["valid_at_try"] == 1:
                valid_at_try_1_counter += 1
            elif conversation_extraction["valid_at_try"] == 2:
                valid_at_try_2_counter += 1
            else:
                not_valid_extraction_counter += 1

            conversation_extraction["evaluation"] = {
                "conv_main_accuracy": conv_main_accuracy,
                "conv_sub_accuracy": conv_sub_accuracy,
                "conv_detail_accuracy": conv_detail_accuracy,
                "number_preference_extracted": number_preferences_extracted,
            }
            if "number_other_extracted" in conversation_extraction:
                number_preferences_extracted_other = conversation_extraction[
                    "number_other_extracted"
                ]
                number_preferences_extracted_other_total += (
                    number_preferences_extracted_other
                )
                conversation_extraction["evaluation"].update(
                    {"number_other_extracted": number_preferences_extracted_other}
                )
            counter += 1

        if args.write_to_file:
            # write line extended with extraction results
            with open(os.path.join(args.output_dir, args.output_file), "a") as file:
                file.write(json.dumps(line) + "\n")

    # ==== evaluation over all read out lines ====
    all_main_accuracy = accuracy_score(
        y_true=mlb_main.transform(label_main_category_empty[0]),
        y_pred=mlb_main.transform(label_main_category[1]),
    )

    all_sub_accuracy = accuracy_score(
        y_true=mlb_sub.transform(label_subcategory_empty[0]),
        y_pred=mlb_sub.transform(label_subcategory[1]),
    )

    # print("binarized_labels_example: \n", label_subcategory[0], label_subcategory[1])
    # print("binarized_labels_example: \n", mlb_sub.transform(label_subcategory[0]), mlb_sub.transform(label_subcategory[1]))

    all_detail_accuracy = accuracy_score(
        y_true=mlb_detail.transform(label_detail_category_empty[0]),
        y_pred=mlb_detail.transform(label_detail_category[1]),
    )
    # ========

    # ==== add evaluation scores and write as last line ====
    all_evaluation_scores = {
        "all_main_accuracy": all_main_accuracy,
        "all_sub_accuracy": all_sub_accuracy,
        "all_detail_accuracy": all_detail_accuracy,
        "number_preferences_extracted_total": number_preferences_extracted_total,
        "valid_at_try_1_counter": valid_at_try_1_counter,
        "valid_at_try_2_counter": valid_at_try_2_counter,
        "not_valid_extraction_counter": not_valid_extraction_counter,
    }
    if args.write_to_file:
        with open(os.path.join(args.output_dir, args.output_file), "a") as file:
            file.write(json.dumps(all_evaluation_scores) + "\n")

    print("string_in_conversation_counter: ", string_in_conversation_counter)
    print(
        "string_in_conversation_counter_user_only: ",
        string_in_conversation_counter_user_only,
    )
    print(
        "string_in_extracted_preference_counter: ",
        string_in_extracted_preference_counter,
    )
    print(
        "string_in_extracted_preference_counter_user_only: ",
        string_in_extracted_preference_counter_user_only,
    )

    if args.confusion_matrix:

        display_labels_main = [
            "points_of_interest",
            "navigation_and_routing",
            "vehicle_settings_and_comfort",
            "entertainment_and_media",
        ]
        display_labels_main = [label.replace("_", " ") for label in display_labels_main]
        display_labels_sub = [
            "restaurant",
            "gas_station",
            "charging_station",
            "grocery_shopping",
            "routing",
            "traffic_and_conditions",
            "parking",
            "climate_control",
            "lighting_and_ambience",
            "music",
            "radio_and_podcast",
        ]
        display_labels_sub = [label.replace("_", " ") for label in display_labels_sub]
        display_labels_detail = [
            "favourite_cuisine",
            "restaurant_type",
            "fast_food_preference",
            "desired_price_range",
            "dietary_preference",
            "payment_method",
            "preferred_gas_station",
            "pay_extra_for_green_fuel",
            "price_sensitivity_for_fuel",
            "preferred_charging_network",
            "type_of_charging_traveling",
            "type_of_charging_everyday_points",
            "charging_station_amenities",
            "supermarket_chain",
            "local_markets_or_supermarket",
            "avoidance_of_road_types",
            "shortest_time_or_distance",
            "tolerance_for_traffic",
            "traffic_source",
            "longer_route_to_avoid_traffic",
            "preferred_parking_type",
            "price_sensitivity_parking",
            "distance_from_parking",
            "covered_parking",
            "handicapped_parking",
            "parking_with_security",
            "temperature",
            "fan_speed",
            "airflow_direction",
            "seat_heating",
            "interior_lighting_brightness",
            "interior_lighting_ambient",
            "interior_lighting_color",
            "genres",
            "artists_or_bands",
            "favorite_songs",
            "music_streaming_service",
            "radio_station",
            "podcast_genres",
            "podcast_shows",
            "general_news_source",
        ]
        display_labels_detail = [
            label.replace("_", " ") for label in display_labels_detail
        ]
        display_labels_main_mlcm = deepcopy(display_labels_main)
        display_labels_main_mlcm.append("NPL/NTL")
        display_labels_sub_mlcm = deepcopy(display_labels_sub)
        display_labels_sub_mlcm.append("NPL/NTL")
        display_labels_detail_mlcm = deepcopy(display_labels_detail)
        display_labels_detail_mlcm.append("NPL/NTL")

        from mlcm import mlcm

        ml_confusion_matrix_main, ml_confusion_matrix_nm_main = mlcm.cm(
            label_true=mlb_main.transform(label_main_category[0]),
            label_pred=mlb_main.transform(label_main_category[1]),
        )
        fig, ax = plt.subplots(figsize=(10, 7), sharex=True, sharey=True, squeeze=True)
        cm = ConfusionMatrixDisplay(
            ml_confusion_matrix_nm_main / 100, display_labels=display_labels_main_mlcm
        )
        cm.plot(ax=ax, xticks_rotation="vertical")
        for text in ax.texts:
            if text.get_text().startswith("0."):
                text.set_text(text.get_text()[1:])
        # cbar = cm.im_.colorbar
        # bbox = ax.get_position()
        # cbar.ax.set_position([bbox.x1 + 0.01, bbox.y0, bbox.x1 + 0.02, bbox.y1])
        plt.title("Normalized Multi-Label Confusion Matrix for Main Category")
        plt.tight_layout()
        output_file = args.output_file.split(".")[0] + "_nm_mlcm_main"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))
        # tikzplotlib.save(os.path.join(args.output_dir, output_file + ".tex"))

        ml_confusion_matrix_sub, ml_confusion_matrix_nm_sub = mlcm.cm(
            label_true=mlb_sub.transform(label_subcategory[0]),
            label_pred=mlb_sub.transform(label_subcategory[1]),
        )
        fig, ax = plt.subplots(figsize=(10, 7), sharex=True, sharey=True, squeeze=True)
        cm = ConfusionMatrixDisplay(
            ml_confusion_matrix_nm_sub / 100, display_labels=display_labels_sub_mlcm
        )
        cm.plot(ax=ax, xticks_rotation="vertical")
        for text in ax.texts:
            if text.get_text().startswith("0."):
                text.set_text(text.get_text()[1:])
        # cbar = cm.im_.colorbar
        # bbox = ax.get_position()
        # cbar.ax.set_position([bbox.x1 + 0.01, bbox.y0, bbox.x1 + 0.02, bbox.y1])
        plt.title("Normalized Multi-Label Confusion Matrix for Sub Category")
        plt.tight_layout()
        output_file = args.output_file.split(".")[0] + "_nm_mlcm_sub"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))
        # tikzplotlib.save(os.path.join(args.output_dir, output_file + ".tex"))

        ml_confusion_matrix_detail, ml_confusion_matrix_nm_detail = mlcm.cm(
            label_true=mlb_detail.transform(label_detail_category[0]),
            label_pred=mlb_detail.transform(label_detail_category[1]),
        )
        fig, ax = plt.subplots(figsize=(20, 14), sharex=True, sharey=True, squeeze=True)
        cm = ConfusionMatrixDisplay(
            ml_confusion_matrix_nm_detail / 100,
            display_labels=display_labels_detail_mlcm,
        )
        cm.plot(ax=ax, xticks_rotation="vertical", values_format=".1f")
        for text in ax.texts:
            if text.get_text().startswith("0."):
                text.set_text(text.get_text()[1:])
        # cbar = cm.im_.colorbar
        # bbox = ax.get_position()
        # cbar.ax.set_position([bbox.x1 + 0.01, bbox.y0, bbox.x1 + 0.02, bbox.y1])
        plt.title("Normalized Multi-Label Confusion Matrix for Detail Category")
        plt.tight_layout()
        output_file = args.output_file.split(".")[0] + "_nm_mlcm_detail"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))
        # tikzplotlib.save(os.path.join(args.output_dir, output_file + ".tex"))

        fig, ax = plt.subplots(figsize=(10, 7), sharex=True, sharey=True, squeeze=True)
        cm = ConfusionMatrixDisplay(
            ml_confusion_matrix_main, display_labels=display_labels_main_mlcm
        )
        cm.plot(ax=ax, xticks_rotation="vertical")
        for text in ax.texts:
            if text.get_text().startswith("0."):
                text.set_text(text.get_text()[1:])
        plt.title("Multi-Label Confusion Matrix for Main Category")
        plt.tight_layout()
        output_file = args.output_file.split(".")[0] + "_mlcm_main"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))

        fig, ax = plt.subplots(figsize=(10, 7))
        cm = ConfusionMatrixDisplay(
            ml_confusion_matrix_sub, display_labels=display_labels_sub_mlcm
        )
        cm.plot(ax=ax, xticks_rotation="vertical")
        for text in ax.texts:
            if text.get_text().startswith("0."):
                text.set_text(text.get_text()[1:])
        plt.title("Multi-Label Confusion Matrix for Sub Category")
        plt.tight_layout()
        output_file = args.output_file.split(".")[0] + "_mlcm_sub"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))

        fig, ax = plt.subplots(figsize=(20, 14))
        cm = ConfusionMatrixDisplay(
            ml_confusion_matrix_detail, display_labels=display_labels_detail_mlcm
        )
        cm.plot(ax=ax, xticks_rotation="vertical")
        for text in ax.texts:
            if text.get_text().startswith("0."):
                text.set_text(text.get_text()[1:])
        plt.title("Multi-Label Confusion Matrix for Detail Category")
        plt.tight_layout()
        output_file = args.output_file.split(".")[0] + "_mlcm_detail"
        plt.savefig(os.path.join(args.output_dir, output_file + ".pdf"))
        plt.show()


if __name__ == "__main__":
    main()
