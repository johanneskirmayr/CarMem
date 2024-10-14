import random

import matplotlib.pyplot as plt
import pandas as pd


# Function to expand a row if it's an MP row
def expand_row_if_mp(row):
    if row["Preference Type"] == "MP":
        # Split the Options/Attributes into a list
        attributes = row["Attributes"].split(",")
        # Create a DataFrame for each attribute
        expanded_rows = [row.to_dict() for attr in attributes]
        for expanded_row, attr in zip(expanded_rows, attributes):
            expanded_row["Attributes"] = attr.strip()
        return expanded_rows
    else:
        return [row.to_dict()]


def sample_user_profiles(
    path_to_categories: str,
    number_user_profiles: int,
    number_attributes_per_user: int,
    random_seed: int = None,
):
    """
    Samples attributes from the categories for user profiles.
    Sampling Strategy is that every "Detail Category" has the same probability of being sampled,
    but it is still possible that multiple attributes of same detail category are sampled if "MP": Multiple attributes Possible.
    Ouput:
    user_profiles: list of pandas dataframes that include sampled user preferences
    sampled_indices: list of number of times each indice(= attribute) was sampled
    sampled_categories: dict of number of times each 'Detail Category' was sampled
    """

    df = pd.read_csv(path_to_categories, delimiter="#")

    # Applying the expansion
    expanded_data = []
    for _, row in df.iterrows():
        expanded_data.extend(expand_row_if_mp(row))

    # Creating a new DataFrame with the expanded data
    expanded_df = pd.DataFrame(expanded_data)

    # Initialize
    user_profiles = []
    sampled_indices = []
    sampled_categories = {}

    # Repeat the sampling process 100 times
    for _ in range(number_user_profiles):
        df = expanded_df.copy(deep=True)
        # Randomly choose either Gas Station or Charging Station Preference since its vehicle dependent
        fuel_type = random.choice(
            ["Gas Station Preferences", "Charging Station Preferences"]
        )
        df = df[df["Subcategory"] != fuel_type]

        # Counting unique values in the 'Detail Category' column
        unique_categories_count = expanded_df["Detail Category"].nunique()
        # weighting rows so that 'Detail Category' is uniformly distributed
        df["Weight"] = df["Detail Category"].transform(
            lambda x: 1
            / (df["Detail Category"].value_counts()[x] * unique_categories_count)
        )

        sampled_df = df.sample(
            n=number_attributes_per_user,
            replace=False,
            weights="Weight",
            random_state=random_seed,
        ).sort_index()
        for idx, row in sampled_df[sampled_df["Preference Type"] == "MNP"].iterrows():
            sampled_df.loc[idx, "Attributes"] = random.choice(
                row["Attributes"].split(", ")
            )

        user_profiles.append(sampled_df)

        sampled_indices.extend(sampled_df.index.tolist())

        for category in sampled_df["Detail Category"]:
            if category in sampled_categories:
                sampled_categories[category] += 1
            else:
                sampled_categories[category] = 1
    sampled_indices.sort()
    sampled_indices_dict = pd.Series(sampled_indices).value_counts(sort=False).to_dict()
    return user_profiles, sampled_indices_dict, sampled_categories


def main():
    test_user_profiles, test_sampled_indices, test_sampled_categories = (
        sample_user_profiles("dataset/categories_v4.csv", 10, 10)
    )

    print("User Profiles: \n\n", test_user_profiles)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Histogram of sampled indices
    axs[0].bar(
        test_sampled_indices.keys(),
        test_sampled_indices.values(),
        color="orange",
        edgecolor="black",
    )
    axs[0].set_xlabel("Row Index")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Histogram of Sampled Row Indices (100 iterations)")

    # Histogram of sampled 'Detail Category'
    axs[1].bar(
        test_sampled_categories.keys(),
        test_sampled_categories.values(),
        color="skyblue",
        edgecolor="black",
    )
    axs[1].set_xlabel("Detail Category")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title(
        "Histogram of Sampled Values per Unique 'Detail Category' (100 iterations)"
    )
    axs[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
