import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt

from neural_tangents import stax
from itertools import product
from tqdm import tqdm

from templates import flooding_ablation
from templates import flooding_utils

import json 
from pathlib import Path
from datetime import datetime

import itertools

eps = 1e-10

jax.config.update("jax_enable_x64", True)

## In this script the index 1 next to the variables, stands for using encoded input for training an inference
## and index 2 stands for using un-encoded input for training and inference
## and the index 3 stands for encoded inference but Heaviside after aggregation
## and the index 4 stands for un-enoded inference with Heaviside after aggregation.
## and index 5 stands for encoded case but without Heaviside except for the final comparison
## and index 6 stands for un-encoded inference without Heaviside, except for final comparison
def test_three_rounding_cases(n, D, l): 

    d = D+1

    X1, Y_train1 = flooding_ablation.get_dataset(l,d)
    n0 = len(X1)
    
    pad = Y_train1.shape[1]-Y_train1.shape[0]

    pad_enc=0
    pad_unenc=0
    if pad > 0:
        pad_enc = pad
    elif pad < 0:
        pad_unenc = abs(pad)

    X_train1 = np.eye(n0)
    X_train1 = np.pad(X_train1, ((0, 0), (0, pad_enc)), mode='constant', constant_values=0)
    X_train1 = jnp.array(X_train1, dtype=jnp.float64)
    Y_train1 = jnp.array(Y_train1, dtype=jnp.float64)

    _, _, kernel_fn1  = stax.serial(
        stax.Dense(1024),   # First dense layer with 128 units
        stax.Relu(),       # ReLU activation
        stax.Dense(n0)      # Output dense layer with 1 unit
    )
    predict_fn1 = nt.predict.gradient_descent_mse_ensemble(kernel_fn1, X_train1, Y_train1)

    # doing the training for the uncoded case
    X_train2, Y_train2 = flooding_ablation.get_dataset_uncoded(l,d)
    

    X_train2 = np.pad(X_train2, ((0, 0), (0, pad_unenc)), mode='constant', constant_values=0)


    X_train2 = jnp.array(X_train2, dtype=jnp.float64)
    Y_train2 = jnp.array(Y_train2, dtype=jnp.float64)

    _, _, kernel_fn2  = stax.serial(
        stax.Dense(1024),   # First dense layer with 128 units
        stax.Relu(),       # ReLU activation
        stax.Dense(n0)      # Output dense layer with 1 unit
    )
    predict_fn2 = nt.predict.gradient_descent_mse_ensemble(kernel_fn2, X_train2, Y_train2)
    num_messages = 2**l-1

    # Initialize a list to store problematic cases
    correct_cases1 = []
    correct_cases2 = []
    correct_cases3 = []
    correct_cases4 = []
    correct_cases5 = []
    correct_cases6 = []

    penultimate_message1 = []
    penultimate_message2 = []
    penultimate_message3 = []
    penultimate_message4 = []
    penultimate_message5 = []
    penultimate_message6 = []

    ultimate_message1 = []
    ultimate_message2 = []
    ultimate_message3 = []
    ultimate_message4 = []
    ultimate_message5 = []
    ultimate_message6 = []

    final_message = []

    counter=1


    A = adj_matrix = np.ones((n, n), dtype=int)

    local_ids = [i for i in range(1,n+1)]
    s = 0
    t = 1
    # iterate all binary messages
    for bits_idx, bits in enumerate(itertools.product([0, 1], repeat=l)):
        # Skip if all bits are 0
        if not any(bits):  # all bits are 0
            continue

        message = list(bits)
        print(message)
        numpy_message = np.array(message)
        final_message.append(numpy_message)


        X_test_list1 = flooding_ablation.create_node_features(n, s, t, l, d, message, local_ids)
        X_test_list2 = flooding_ablation.create_node_features(n, s, t, l, d, message, local_ids)
        X_test_list3 = flooding_ablation.create_node_features(n, s, t, l, d, message, local_ids)
        X_test_list4 = flooding_ablation.create_node_features(n, s, t, l, d, message, local_ids)
        X_test_list5 = flooding_ablation.create_node_features(n, s, t, l, d, message, local_ids)
        X_test_list6 = flooding_ablation.create_node_features(n, s, t, l, d, message, local_ids)


        X_test_flattened_l2 = []
        for x_test in X_test_list2:
            x_flt = flooding_utils.flatten_sample(x_test, normalize= True)
            X_test_flattened_l2.append(x_flt)

        X_test_flattened_l4 = []
        for x_test in X_test_list4:
            x_flt = flooding_utils.flatten_sample(x_test, normalize= True)
            X_test_flattened_l4.append(x_flt)

        X_test_flattened_l6 = []
        for x_test in X_test_list6:
            x_flt = flooding_utils.flatten_sample(x_test, normalize= True)
            X_test_flattened_l6.append(x_flt)

        X_test2 = np.array(X_test_flattened_l2, dtype=np.float64)
        X_test4 = np.array(X_test_flattened_l4, dtype=np.float64)
        X_test6 = np.array(X_test_flattened_l6, dtype=np.float64)

        X_test_list2 = []
        X_test_list4 = []
        X_test_list6 = []

        iterations = 150

        for iter in range(iterations):

            X_test_encoded_l1 = []
            for x_test in X_test_list1:
                _, x_enc = flooding_ablation.encode_data(x_test, X1,pad_enc)
                X_test_encoded_l1.append(x_enc)

            X_test_list1 = []

            X_test_encoded_l3 = []
            for x_test in X_test_list3:
                _, x_enc = flooding_ablation.encode_data(x_test, X1, pad_enc)
                X_test_encoded_l3.append(x_enc)

            X_test_list3 = []

            X_test_encoded_l5 = []
            for x_test in X_test_list5:
                _, x_enc = flooding_ablation.encode_data(x_test, X1, pad_enc)
                X_test_encoded_l5.append(x_enc)

            X_test_list5 = []

            X_test1 = jnp.array(X_test_encoded_l1, dtype=jnp.float64)
            X_test3 = jnp.array(X_test_encoded_l3, dtype=jnp.float64)
            X_test5 = jnp.array(X_test_encoded_l5, dtype=jnp.float64)

            X_test2 = np.pad(X_test2, ((0, 0), (0, pad_unenc)), mode='constant', constant_values=0)
            X_test4 = np.pad(X_test4, ((0, 0), (0, pad_unenc)), mode='constant', constant_values=0)
            X_test6 = np.pad(X_test6, ((0, 0), (0, pad_unenc)), mode='constant', constant_values=0)

            X_test2 = flooding_utils.normalize_rows_l2(X_test2)
            X_test4 = flooding_utils.normalize_rows_l2(X_test4)
            X_test6 = flooding_utils.normalize_rows_l2(X_test6)

            X_test2 = jnp.array(X_test2, dtype=jnp.float64)
            X_test4 = jnp.array(X_test4, dtype=jnp.float64)
            X_test6 = jnp.array(X_test6, dtype=jnp.float64)

            Y_pred1 = predict_fn1(x_test=X_test1, get='ntk', compute_cov=True)
            Y_pred3 = predict_fn1(x_test=X_test3, get='ntk', compute_cov=True)
            Y_pred5 = predict_fn1(x_test=X_test5, get='ntk', compute_cov=True)

            Y_pred2 = predict_fn2(x_test=X_test2, get='ntk', compute_cov=True)
            Y_pred4 = predict_fn2(x_test=X_test4, get='ntk', compute_cov=True)
            Y_pred6 = predict_fn2(x_test=X_test6, get='ntk', compute_cov=True)


            Y_pred1 = Y_pred1.mean
            Y_pred1 = np.where(Y_pred1 > eps, 1, 0)

            Y_pred3 = Y_pred3.mean
            Y_pred5 = Y_pred5.mean

            Y_pred2 = Y_pred2.mean
            Y_pred2 = np.where(Y_pred2 > eps, 1, 0)

            Y_pred4 = Y_pred4.mean
            Y_pred6 = Y_pred6.mean

            # Get dimensions
            n_nodes = Y_pred1.shape[0]  # Number of nodes
            n_features = Y_pred1.shape[1]  # Total number of features
            split_point = 2*l + 3*d*l  # The column where we split

            # Create P_c (selects columns 0 to split_point-1)
            P_c = np.zeros((n_features, n_features))
            P_c[:split_point, :split_point] = np.eye(split_point)

            # Create P_m (selects columns from split_point onward)
            P_m = np.zeros((n_features, n_features))
            P_m[split_point:, split_point:] = np.eye(n_features - split_point)

            # Compute X_test according to the formula
            X_test1 = Y_pred1 @ P_c + A @ Y_pred1 @ P_m

            X_test3 = Y_pred3 @ P_c + A @ Y_pred3 @ P_m
            X_test3 = np.where(X_test3 > eps, 1, 0)

            X_test5 = Y_pred5 @ P_c + A @ Y_pred5 @ P_m
            ###########################################
            X_test2 = Y_pred2 @ P_c + A @ Y_pred2 @ P_m

            X_test4 = Y_pred4 @ P_c + A @ Y_pred4 @ P_m
            X_test4 = np.where(X_test4 > eps, 1, 0)

            X_test6 = Y_pred6 @ P_c + A @ Y_pred6 @ P_m


            for e, row in enumerate(X_test1):
                x_test = flooding_ablation.unflatten_sample(row, l,d)
                X_test_list1.append(x_test)

            for e, row in enumerate(X_test3):
                x_test = flooding_ablation.unflatten_sample(row, l,d)
                X_test_list3.append(x_test)

            for e, row in enumerate(X_test5):
                x_test = flooding_ablation.unflatten_sample(row, l,d)
                X_test_list5.append(x_test)


            if iter == iterations-2:

                penultimate_message1.append( X_test_list1[0]['final_message'])
                penultimate_message3.append( X_test_list3[0]['final_message'])
                penultimate_message5.append( np.where (X_test_list5[0]['final_message']> eps, 1, 0))

                for e, row in enumerate(X_test2):
                    x_test = flooding_ablation.unflatten_sample(row, l,d)
                    X_test_list2.append(x_test)
                penultimate_message2.append( X_test_list2[0]['final_message'])
                X_test_list2=[]

                for e, row in enumerate(X_test4):
                    x_test = flooding_ablation.unflatten_sample(row, l,d)
                    X_test_list4.append(x_test)
                penultimate_message4.append( X_test_list4[0]['final_message'])
                X_test_list4=[]

                for e, row in enumerate(X_test6):
                    x_test = flooding_ablation.unflatten_sample(row, l,d)
                    X_test_list6.append(x_test)
                penultimate_message6.append( np.where(X_test_list6[0]['final_message'] > eps, 1, 0))
                X_test_list6=[]


            if iter == iterations-1:

                ultimate_message1.append( X_test_list1[0]['final_message'])
                ultimate_message3.append( X_test_list3[0]['final_message'])
                ultimate_message5.append( np.where (X_test_list5[0]['final_message']> eps, 1, 0))

                for e, row in enumerate(X_test2):
                    x_test = flooding_ablation.unflatten_sample(row, l,d)
                    X_test_list2.append(x_test)
                ultimate_message2.append( X_test_list2[0]['final_message'])
                X_test_list2=[]

                for e, row in enumerate(X_test4):
                    x_test = flooding_ablation.unflatten_sample(row, l,d)
                    X_test_list4.append(x_test)
                ultimate_message4.append( X_test_list4[0]['final_message'])
                X_test_list4=[]

                for e, row in enumerate(X_test6):
                    x_test = flooding_ablation.unflatten_sample(row, l,d)
                    X_test_list6.append(x_test)
                ultimate_message6.append( np.where(X_test_list6[0]['final_message'] > eps, 1, 0))
                X_test_list6=[]

        print(f"finished processing {counter} out of {num_messages}")
        counter += 1

    acc1 = flooding_ablation.calculate_accuracy(penultimate_message1, ultimate_message1, final_message)
    acc2 = flooding_ablation.calculate_accuracy(penultimate_message2, ultimate_message2, final_message)
    acc3 = flooding_ablation.calculate_accuracy(penultimate_message3, ultimate_message3, final_message)
    acc4 = flooding_ablation.calculate_accuracy(penultimate_message4, ultimate_message4, final_message)
    acc5 = flooding_ablation.calculate_accuracy(penultimate_message5, ultimate_message5, final_message)
    acc6 = flooding_ablation.calculate_accuracy(penultimate_message6, ultimate_message6, final_message)
    

    return acc1, acc2, acc3, acc4, acc5, acc6


import csv
import os

if __name__ == "__main__":
    print("Ablation for orthogonality encoding and Heaviside")

    l = 3

    # Create results folder if it doesn't exist
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Created folder: {results_folder}")
    
    # Define the file path
    file_path = os.path.join(results_folder, 'ablation_result.csv')
    
    # open CSV file for writing
    with open(file_path, 'w', newline='') as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['n','Main variant','Unencoded inputs','Heaviside after aggregation','Unencoded with Heaviside after aggregation','No Heaviside','Unencoded with no Heaviside'])
        
        for n in range(3,16):  
            print(f"==========this is for n = {n} nodes========")
            D=n-1
            acc1, acc2, acc3, acc4, acc5, acc6 = test_three_rounding_cases(n, D, l)
            
            # Write results to CSV
            writer.writerow([n, acc1, acc2, acc3, acc4, acc5, acc6])
                
    print(f"Results saved to {file_path}")

    