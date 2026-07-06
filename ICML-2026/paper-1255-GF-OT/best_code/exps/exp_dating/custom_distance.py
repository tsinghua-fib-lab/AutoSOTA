import torch
import numpy as np


def table_distance(
    x, y, id_x, id_y, match_matrix, return_sum=False, constant=5
):

    if match_matrix[id_x, id_y] == 0:
        if return_sum:
            return 1e6 * 9
        else:
            return 1e6 * np.ones(9)

    interest_x = x["interest_tags"]
    interest_y = y["interest_tags"]

    # count the matching words in both interest tags
    interest_x_words = set(interest_x.split())
    interest_y_words = set(interest_y.split())
    matching_words = interest_x_words.intersection(interest_y_words)

    interest_distance = (
        constant / len(matching_words) if matching_words else constant
    )

    location_distance = (
        constant if x["location_type"] != y["location_type"] else 0
    )

    app_use_distance = (
        constant
        if x["app_usage_time_label"] != y["app_usage_time_label"]
        else 0
    )

    swipe_time_distance = (
        constant if x["swipe_time_of_day"] != y["swipe_time_of_day"] else 0
    )
    # select remaining columns
    remaining_cols_x = x.drop(
        labels=[
            "interest_tags",
            "location_type",
            "app_usage_time_label",
            "swipe_time_of_day",
            "education_level",
            "income_bracket",
            "gender",
            "sexual_orientation",
        ]
    )
    remaining_cols_y = y.drop(
        labels=[
            "interest_tags",
            "location_type",
            "app_usage_time_label",
            "swipe_time_of_day",
            "education_level",
            "income_bracket",
            "gender",
            "sexual_orientation",
        ]
    )
    distance = np.abs(remaining_cols_x - remaining_cols_y)

    all_distances = [
        interest_distance,
        location_distance,
        app_use_distance,
        swipe_time_distance,
    ] + distance.tolist()
    if return_sum:
        return sum(all_distances)
    else:
        return all_distances


def pre_compute_distance_matrix(
    x, y, match_matrix, idx_x=None, idx_y=None, constant=5
):
    num_x = len(x)
    num_y = len(y)
    if idx_x is None:
        idx_x = np.arange(num_x)
    if idx_y is None:
        idx_y = np.arange(num_y)

    # Match mask: (num_x, num_y), True where pair is invalid
    mm = np.asarray(match_matrix)
    no_match = mm[np.ix_(idx_x, idx_y)] == 0  # (num_x, num_y)

    # Interest tag distance: pre-compute word sets once, then loop
    sets_x = [set(s.split()) for s in x["interest_tags"]]
    sets_y = [set(s.split()) for s in y["interest_tags"]]
    interest_dist = np.empty((num_x, num_y), dtype=np.float32)
    for i, sx in enumerate(sets_x):
        for j, sy in enumerate(sets_y):
            common = sx & sy
            interest_dist[i, j] = (
                constant / len(common) if common else constant
            )

    # Categorical distances via broadcasting: (num_x, num_y, 3)
    cat_cols = ["location_type", "app_usage_time_label", "swipe_time_of_day"]
    cat_x = x[cat_cols].values  # (num_x, 3)
    cat_y = y[cat_cols].values  # (num_y, 3)
    cat_dist = (cat_x[:, None, :] != cat_y[None, :, :]).astype(
        np.float32
    ) * constant

    # Numeric remaining columns via broadcasting: (num_x, num_y, k)
    drop_cols = [
        "interest_tags",
        "location_type",
        "app_usage_time_label",
        "swipe_time_of_day",
        "education_level",
        "income_bracket",
        "gender",
        "sexual_orientation",
    ]
    num_x_vals = x.drop(columns=drop_cols).values.astype(
        np.float32
    )  # (num_x, k)
    num_y_vals = y.drop(columns=drop_cols).values.astype(
        np.float32
    )  # (num_y, k)
    num_dist = np.abs(num_x_vals[:, None, :] - num_y_vals[None, :, :])

    # Stack all features: [interest, location, app_use, swipe_time, ...numeric]
    distance_matrix = np.concatenate(
        [interest_dist[:, :, None], cat_dist, num_dist], axis=-1
    )  # (num_x, num_y, 9)
    distance_matrix /= distance_matrix.max()

    # Apply no-match mask
    distance_matrix[no_match] = 1e1

    return torch.tensor(distance_matrix, dtype=torch.float32)
