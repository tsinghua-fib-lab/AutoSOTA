import numpy as np

from itertools import product
from tqdm import tqdm

from templates import flooding
from templates import flooding_utils

import json 
from pathlib import Path
from datetime import datetime

import itertools
import argparse


from initialization_utils import flooding_final_messages, create_node_features
from generate_tree import color_square_of_tree, generate_tree_adjacency_matrices

from saving_utils import save_test_samples, save_problem_cases, save_test_cases

eps = 1e-10

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate flooding test cases")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--l", type=int, required=True)
    args = parser.parse_args()

    n = args.n
    D = args.D
    l = args.l

    print("Generate flooding test cases")

    d = D+1
    file_path = f"l{l}_D{D}_n{n}"
    X, Y_train = flooding.get_dataset(l,d)
    X_train = np.eye(len(X))
    Y_train = np.array(Y_train, dtype=np.float64)
    n0 = len(X)

    # create the test graph
    trees = generate_tree_adjacency_matrices(n,D)
    num_trees = len(trees)
    num_messages = 2**l
    total_cases = num_trees * num_messages

    counter = 1  
    test_cases=[]

    for t_idx, A in enumerate(trees):
        
        # compute true 2-hop local ids
        local_ids = color_square_of_tree(A, D)  # values in 1..D^2+1
        # iterate all binary messages
        for bits_idx, bits in enumerate(itertools.product([0, 1], repeat=l)):

            all_original_of_case = []
            all_encoded_of_case = []
            all_target_of_case = []

            seen = set()

            message = list(bits)
            # source is the 0-th node
            s = 0
            final_messages = flooding_final_messages(A, s, message)
            X_test_list = create_node_features(n, s, l, d, message, local_ids)

            for _ in range(65):
                X_test_encoded_l = []
                X_test_encoded_nonorm = []
                for x_test in X_test_list:
                    x_e_no_norm, x_enc = flooding_utils.encode_data(x_test, X)
                    X_test_encoded_l.append(x_enc)
                    X_test_encoded_nonorm.append(x_e_no_norm)
                
                X_test = np.array(X_test_encoded_l, dtype=np.float64)
                Y_pred = X_test @ X_train @ Y_train

                Y_pred_round = np.where(Y_pred > eps, 1.0, 0.0)


                for x_t, x_e, y_t in zip(X_test_list, X_test_encoded_nonorm, Y_pred_round):
                    tuple_e = tuple(x_e)
                    if tuple_e not in seen:
                        all_original_of_case.append( x_t)
                        all_encoded_of_case.append(x_e)
                        all_target_of_case.append(y_t)
                        seen.add(tuple_e)                        

                # Get dimensions
                n_nodes = Y_pred_round.shape[0]  # Number of nodes
                n_features = Y_pred_round.shape[1]  # Total number of features
                split_point = l + 3*d*l  # The column where we split

                # Create P_c (selects columns 0 to split_point-1)
                P_c = np.zeros((n_features, n_features))
                P_c[:split_point, :split_point] = np.eye(split_point)

                # Create P_m (selects columns from split_point onward)
                P_m = np.zeros((n_features, n_features))
                P_m[split_point:, split_point:] = np.eye(n_features - split_point)

                # Compute X_test according to the formula
                X_test = Y_pred_round @ P_c + A @ Y_pred_round @ P_m

                X_test_list = []

                for e, row in enumerate(X_test):
                    x_test = flooding.unflatten_sample(row, l, d)
                    X_test_list.append(x_test)

            test_cases.append((all_original_of_case, all_encoded_of_case, all_target_of_case))
                  
            print(f"finished processing {counter} out of {total_cases}")
            counter += 1

    # save final versions with final metadata:
    save_test_cases(
        test_cases,
        file_path,
        n = n,
        D = D,
        l = l
    )
