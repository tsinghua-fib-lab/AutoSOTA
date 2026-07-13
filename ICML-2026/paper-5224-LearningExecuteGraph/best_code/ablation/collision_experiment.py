import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt

from neural_tangents import stax
from itertools import product
from tqdm import tqdm

from templates import flooding_collision
from templates import flooding_utils

import json 
from pathlib import Path
from datetime import datetime

import itertools

eps = 1e-10

jax.config.update("jax_enable_x64", True)

## In this script the index 1 next to the variables stands for the main flooding templates, which use cascade chain mechanism
## and the index 2 stands for the flooding templates without cascade chain mechanism
def test_w_collisions(n, D, l): 

    d = D+1
    num_slots = d

    # doing the training for flooding with cascade chain
    X1, Y_train1 = flooding_collision.get_dataset(l,d, num_slots)
    X_train1 = jnp.eye(len(X1))
    Y_train1 = jnp.array(Y_train1, dtype=jnp.float64)
    n1 = len(X1)

    _, _, kernel_fn1  = stax.serial(
        stax.Dense(1024),   # First dense layer with 128 units
        stax.Relu(),       # ReLU activation
        stax.Dense(n1)      # Output dense layer with 1 unit
    )
    predict_fn1 = nt.predict.gradient_descent_mse_ensemble(kernel_fn1, X_train1, Y_train1)

    # doing the training for flooding without cascade chain
    X2, Y_train2 = flooding_collision.get_dataset_collision(l,d, num_slots)
    X_train2 = jnp.eye(len(X2))
    Y_train2 = jnp.array(Y_train2, dtype=jnp.float64)
    n2 = len(X2)

    _, _, kernel_fn2  = stax.serial(
        stax.Dense(1024),   # First dense layer with 128 units
        stax.Relu(),       # ReLU activation
        stax.Dense(n2)      # Output dense layer with 1 unit
    )
    predict_fn2 = nt.predict.gradient_descent_mse_ensemble(kernel_fn2, X_train2, Y_train2)

    A = flooding_collision.star_graph_adjacency_matrix(n)

    local_ids = [i for i in  range(1,n+1)]  # values in 1..D^2+1

    message = [1]

    # source
    s = 1
    # target
    t = 0

    X_test_list1 = flooding_collision.create_node_features(n, s, l, d, message, local_ids, num_slots)
    X_test_list2 = flooding_collision.create_node_features(n, s, l, d, message, local_ids, num_slots)

    f_message1 = []
    f_message2 = []

    for _ in range(150):

        X_test_encoded_l1 = []
        for x_test in X_test_list1:
            _, x_enc = flooding_utils.encode_data(x_test, X1)
            X_test_encoded_l1.append(x_enc)

        X_test_encoded_l2 = []
        for x_test in X_test_list2:
            _, x_enc = flooding_utils.encode_data(x_test, X2)
            X_test_encoded_l2.append(x_enc)

        X_test_list1 = []
        X_test_list2 = []


        X_test1 = jnp.array(X_test_encoded_l1, dtype=jnp.float64)
        X_test2 = jnp.array(X_test_encoded_l2, dtype=jnp.float64)


        Y_pred1 = predict_fn1(x_test=X_test1, get='ntk', compute_cov=True)
        Y_pred2 = predict_fn2(x_test=X_test2, get='ntk', compute_cov=True)

        Y_pred1 = Y_pred1.mean
        Y_pred2 = Y_pred2.mean

        ylist1 = []
        for e, row in enumerate(Y_pred1):
            y_test = flooding_collision.unflatten_sample(row, l,d, num_slots)
            ylist1.append(y_test)

        ylist2 = []
        for e, row in enumerate(Y_pred2):
            y_test = flooding_collision.unflatten_sample(row, l,d, num_slots)
            ylist2.append(y_test)

        f_message1.append(ylist1[0]['final_message'][0])
        f_message2.append(ylist2[0]['final_message'][0])



        Y_pred_round1 = np.where(Y_pred1 > eps, 1, 0)
        Y_pred_round2 = np.where(Y_pred2 > eps, 1, 0)

        # Get dimensions
        n_nodes = Y_pred_round1.shape[0]  # Number of nodes
        n_features = Y_pred_round1.shape[1]  # Total number of features
        split_point = 2*l + 3*d*l  # The column where we split

        # Create P_c (selects columns 0 to split_point-1)
        P_c = np.zeros((n_features, n_features))
        P_c[:split_point, :split_point] = np.eye(split_point)

        # Create P_m (selects columns from split_point onward)
        P_m = np.zeros((n_features, n_features))
        P_m[split_point:, split_point:] = np.eye(n_features - split_point)

        # Compute X_test according to the formula
        X_test1 = Y_pred_round1 @ P_c + A @ Y_pred_round1 @ P_m
        X_test2 = Y_pred_round2 @ P_c + A @ Y_pred_round2 @ P_m


        for e, row in enumerate(X_test1):
            x_test = flooding_collision.unflatten_sample(row, l,d, num_slots)
            X_test_list1.append(x_test)

        for e, row in enumerate(X_test2):
            x_test = flooding_collision.unflatten_sample(row, l,d, num_slots)
            X_test_list2.append(x_test)

    last_three1 = f_message1[-3:]
    last_three2 = f_message2[-3:]

    average1 = np.mean(np.array(last_three1))
    average2 = np.mean(np.array(last_three2))

    f_m1 = 1 if average1 > eps else 0
    f_m2 = 1 if average2 > eps else 0

    return average1, f_m1, average2, f_m2



import csv
import os

if __name__ == "__main__":
    print("Ablation for using the cascade chain trick to avoid growing collisions")

    l = 1

    # Create results folder if it doesn't exist
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Created folder: {results_folder}")
    
    # Define the file path
    file_path = os.path.join(results_folder, 'collision_effect.csv')
    
    # open CSV file for writing
    with open(file_path, 'w', newline='') as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['n', 'Value with cascade chain', 'Binary value with cascade chain', 'Value without cascade chain', 'Binary value without cascade chain'])
        
        for n in range(3, 21):
            print(f"==============this is the n: {n}===========\n\n")
            D = n-1
            
            avg1, f_m1, avg2, f_m2 = test_w_collisions(n, D, l)
            
            # Write results to CSV
            writer.writerow([n, avg1, f_m1, avg2, f_m2])
        
    print(f"Results saved to {file_path}")

    