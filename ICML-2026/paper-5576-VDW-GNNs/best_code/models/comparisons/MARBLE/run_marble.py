import pickle
import sys
import numpy as np
from macaque_reaching_helpers import *
import MARBLE

def main():

    """fitting model for each day + pca embedding"""
    
    data_file = "data/rate_data_20ms.pkl"
    metadata_file = "data/trial_ids.pkl"
    
    rates = pickle.load(open(data_file, "rb"))
    trial_ids = pickle.load(open(metadata_file, "rb"))

    # defining the set of conditions
    conditions = ["DownLeft", "Left", "UpLeft", "Up", "UpRight", "Right", "DownRight"]

    # list of days
    days = rates.keys()

    # define some parameters
    pca_n = 5
    filter_data = True

    # storing all distance matrices
    embeddings = []
    distance_matrices = []
    times = [] # to store the time point of each node in the trajectory
    all_condition_labels = [] # to store the condition label for each node
    all_trial_ids = [] # trial ids for each node
    all_sampled_ids = [] # to store all the nodes sampled by marble
    
    # loop over each day
    for day in days:

        # first stack all trials from that day together and fit pca
        print(day)
        pca = fit_pca(rates, day, conditions, filter_data=filter_data, pca_n=pca_n)
        pos, vel, timepoints, condition_labels, trial_indexes = format_data(
            rates, 
            trial_ids,
            day, 
            conditions, 
            pca=pca,
            filter_data=filter_data
        )
        # construct data for marble
        data = MARBLE.construct_dataset(
            anchor=pos,
            vector=vel,
            k=30,
            spacing=0.0,
            delta=1.5,
        )

        params = {
            "epochs": 120,  # optimisation epochs
            "order": 2,  # order of derivatives
            "hidden_channels": 100,  # number of internal dimensions in MLP
            "out_channels": 3, # 3 for Fig3
            "inner_product_features": False,
            "diffusion": True,
        }

        model = MARBLE.net(data, params=params)

        model.fit(data, outdir="data/session_{}_20ms".format(day))
        data = model.transform(data)

        n_clusters = 50
        data = MARBLE.distribution_distances(data, n_clusters=n_clusters)

        embeddings.append(data.emb)
        distance_matrices.append(data.dist)
        times.append(np.hstack(timepoints))
        all_condition_labels.append(data.y)
        all_trial_ids.append(np.hstack(trial_indexes))
        all_sampled_ids.append(data.sample_ind)

        # save over after each session (incase computations crash)
        with open("data/marble_embeddings_20ms_out3.pkl", "wb") as handle:
            pickle.dump(
                [
                    distance_matrices,
                    embeddings,
                    times,
                    all_condition_labels,
                    all_trial_ids,
                    all_sampled_ids,
                ],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        break

    # final save
    with open("data/marble_embeddings_20ms_out3.pkl", "wb") as handle:
        pickle.dump(
            [
                distance_matrices,
                embeddings,
                times,
                all_condition_labels,
                all_trial_ids,
                all_sampled_ids,
            ],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    sys.exit(main())
