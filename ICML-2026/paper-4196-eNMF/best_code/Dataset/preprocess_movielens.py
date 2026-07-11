"""Preprocess Movielens dataset. No need to rerun."""

import numpy as np
import pandas as pd
import itertools


## Mask1: put mask on each user's lastest 20% ratings (latest: sort values by 'Date')
def get_latest_ratings(x):
    training_percent = 0.8
    sort_res = x.sort_values(by="Timestamp")
    num_ele = len(sort_res.index)
    rmv_idx = int(num_ele * training_percent)
    latest_ratings_idx = sort_res.index.tolist()[rmv_idx:]
    return latest_ratings_idx


if __name__ == "__main__":
    # Column headers for the dataset.
    # format according to Movielens: http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
    data_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    item_cols = ["MovieID", "Title", "Genres"]
    user_cols = ["UserID", "Gender", "Age", "Occupation", "zip code"]

    # Import the data files onto dataframes.
    users = pd.read_csv(
        "../Dataset/ml-1m/users.dat", sep="::", names=user_cols, encoding="latin-1"
    )
    movies = pd.read_csv(
        "../Dataset/ml-1m/movies.dat", sep="::", names=item_cols, encoding="latin-1"
    )
    ratings = pd.read_csv(
        "../Dataset/ml-1m/ratings.dat", sep="::", names=data_cols, encoding="latin-1"
    )

    dataset = pd.merge(pd.merge(movies, ratings), users)
    all_data = pd.concat(
        [dataset.MovieID, dataset.UserID, dataset.Rating, dataset.Timestamp], axis=1
    )

    # Get index for masked ratings.
    new_mask_idx = all_data.groupby(["UserID"]).apply(get_latest_ratings)
    index_list = list(itertools.chain.from_iterable(new_mask_idx.tolist()))
    # Put mask on the original movielens data accordingly.
    masked_data = all_data.copy()
    masked_data.loc[index_list, "Rating"] = np.nan
    train_matrix = masked_data.pivot(index="MovieID", columns="UserID", values="Rating")
    full_matrix = all_data.pivot(index="MovieID", columns="UserID", values="Rating")
    # Convert nan entries to 0.
    full_matrix[np.isnan(full_matrix)] = 0
    movie_train_mat = train_matrix.values
    movie_train_mat[np.isnan(movie_train_mat)] = 0
    # Save final result (once only)
    # np.save("../Dataset/movielens1m_0.8training.npy", movie_train_mat.T)
    # np.save("../Dataset/movielens1m.npy", full_matrix.values.T)
