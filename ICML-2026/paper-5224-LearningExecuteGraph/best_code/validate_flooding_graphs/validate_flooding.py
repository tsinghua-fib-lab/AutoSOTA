import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt

from neural_tangents import stax
from itertools import product
from tqdm import tqdm

from templates import flooding
from templates import flooding_utils

import json
from pathlib import Path
from datetime import datetime

import itertools
import argparse

from initialization_utils import (
    two_hop_coloring,
    flooding_final_messages,
    create_node_features,
    graph_diameter
)

from generate_graph import generatee_graphs


jax.config.update("jax_enable_x64", True)
eps = 1e-10


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validate flooding algorithm")
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--l", type=int, required=True)
    args = parser.parse_args()

    print("Validating flooding")

    D = args.D
    l = args.l
    num_graphs = 1000

    d = D**2 + 1
    X, Y_train = flooding.get_dataset(l, d)
    X_train = jnp.eye(len(X))
    Y_train = jnp.array(Y_train, dtype=jnp.float64)
    n0 = len(X)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1024),
        stax.Relu(),
        stax.Dense(n0)
    )

    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn, X_train, Y_train
    )

    # starting the test cases
    graphs = generatee_graphs(num_graphs, D)
    num_graphs = len(graphs)
    num_messages = 2 ** l
    total_cases = num_graphs * num_messages

    problematic_cases = []
    counter = 1

    for g_idx, A in enumerate(graphs):
        d_G = graph_diameter(A)
        n = A.shape[0]
        local_ids = two_hop_coloring(A, D)

        for bits_idx, bits in enumerate(itertools.product([0, 1], repeat=l)):
            message = list(bits)
            s = 0

            final_messages = flooding_final_messages(A, s, message)
            X_test_list = create_node_features(
                n, s, l, d, message, local_ids
            )

            for _ in range((d_G + 1) * (d + 3)):
                X_test_encoded_l = []

                for x_test in X_test_list:
                    _, x_enc = flooding_utils.encode_data(x_test, X)
                    X_test_encoded_l.append(x_enc)

                X_test_list = []

                X_test = jnp.array(X_test_encoded_l, dtype=jnp.float64)
                Y_pred = predict_fn(
                    x_test=X_test, get="ntk", compute_cov=True
                )

                Y_pred_round = np.where(Y_pred.mean > eps, 1.0, 0.0)

                n_nodes = Y_pred_round.shape[0]
                n_features = Y_pred_round.shape[1]
                split_point = l + 3 * d * l

                P_c = np.zeros((n_features, n_features))
                P_c[:split_point, :split_point] = np.eye(split_point)

                P_m = np.zeros((n_features, n_features))
                P_m[split_point:, split_point:] = np.eye(
                    n_features - split_point
                )

                X_test = Y_pred_round @ P_c + A @ Y_pred_round @ P_m

                for e, row in enumerate(X_test):
                    x_test = flooding.unflatten_sample(row, l, d)
                    X_test_list.append(x_test)

            all_match = True
            for i in range(len(X_test_list)):
                final_msg_array = np.array(
                    final_messages[i],
                    dtype=X_test_list[i]["message"].dtype
                )

                if not np.array_equal(
                    X_test_list[i]["message"], final_msg_array
                ):
                    all_match = False
                    print(
                        f"Mismatch at index {i}: "
                        f"{X_test_list[i]['message']} != {final_messages[i]}"
                    )
                    break

            if not all_match:
                problematic_cases.append({
                    "graph_index": g_idx,
                    "bits_index": bits_idx,
                    "source_node": s,
                    "X_test_message": [x["message"] for x in X_test_list],
                    "final_message": final_messages,
                    "local_ids": local_ids.copy(),
                    "adjacency_matrix": A.copy(),
                    "original_message": message.copy()
                })

            print(f"finished processing {counter} out of {total_cases}")
            print(f"This is all_match {all_match}")
            counter += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the directory if it doesn't exist
    output_dir = Path("failure_cases")
    output_dir.mkdir(exist_ok=True)

    filename = f"failure_cases/problematic_cases_l{l}_D{D}_n5to20_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "problem_cases": problematic_cases,
                "metadata": {
                    "timestamp": timestamp,
                    "total_cases": len(problematic_cases),
                },
            },
            f,
            indent=2,
        )

    print(
        f"Found {len(problematic_cases)} problematic cases. "
        f"Saved to {filename}"
    )
