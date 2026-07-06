import os
import pandas as pd
import mlcroissant as mlc
from match_matrix import get_match_matrix
from utils_dating import subsample_to_current_us_distribution

# Fetch the Croissant JSON-LD
croissant_dataset = mlc.Dataset(
    "https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset"
    "/croissant/download"
)

# Check what record sets are in the dataset
record_sets = croissant_dataset.metadata.record_sets

# Fetch the records and put them in a DataFrame
record_set_df = pd.DataFrame(
    croissant_dataset.records(record_set=record_sets[0].uuid)
)
record_set_df.head()

# rename the columns to remove the beginning dating_app_behavior_dataset.csv/
record_set_df.columns = [
    col.replace("dating_app_behavior_dataset.csv/", "")
    for col in record_set_df.columns
]

# all string entries of the df start with b'entry_of_the_cell'.
# Remove the b and the ''
record_set_df = record_set_df.applymap(
    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
)

# drop the columns that are not relevant for the matching task, such as those
# related to app usage and behavior, and the match outcome

columns_to_drop = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "swipe_right_label",
    "mutual_matches",
    "profile_pics_count",
    "message_sent_count",
    "last_active_hour",
    "match_outcome",
]

record_set_df = record_set_df.drop(columns=columns_to_drop)

match_matrix = get_match_matrix()

valid_combinations = [
    tuple(label.strip("()").replace("'", "").split(", "))
    for label in match_matrix.index
]
record_set_df = record_set_df[
    record_set_df.apply(
        lambda row: (row["gender"], row["sexual_orientation"])
        in valid_combinations,
        axis=1,
    )
]

# encode the income level bracket and the education level as numerical
# variables, with a value given that reflects proximity and logical ordering

income_brackets = record_set_df["income_bracket"].unique()

income_brackets_ordered = [
    "Very Low",
    "Low",
    "Lower-Middle",
    "Middle",
    "Upper-Middle",
    "High",
    "Very High",
]

education_levels = record_set_df["education_level"].unique()
education_levels_ordered = [
    "No Formal Education",
    "Diploma",
    "High School",
    "Associate’s",
    "Bachelor’s",
    "Master’s",
    "MBA",
    "PhD",
    "Postdoc",
]

# Encode the education levels as numerical values based on the ordered list
education_level_mapping = {
    level: 20 * i for i, level in enumerate(education_levels_ordered)
}
record_set_df["education_level_encoded"] = record_set_df[
    "education_level"
].map(education_level_mapping)
# Same for the income brackets

income_bracket_mapping = {
    bracket: 60 * i for i, bracket in enumerate(income_brackets)
}
record_set_df["income_bracket_encoded"] = record_set_df["income_bracket"].map(
    income_bracket_mapping
)

# create a custom function of distance between income brackets and education
# levels


record_set_df, diag = subsample_to_current_us_distribution(record_set_df)

# save as a csv
os.makedirs("exps/exp_dating/data/", exist_ok=True)
record_set_df.to_csv(
    "exps/exp_dating/data/processed_dating_data.csv", index=True
)
