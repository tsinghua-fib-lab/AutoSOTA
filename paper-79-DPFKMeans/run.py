import argparse
import numpy as np
import os
import pickle
from sklearn.cluster import kmeans_plusplus, KMeans

from algorithms import (FederatedOuterProduct, FederatedServerPointWeighting, FederatedInitFromProjectedCenters,
                        FederatedClusterInitHyperParams, FederatedLloyds, FederatedLloydsHyperParams,
                        KFed, KFedHyperParams, add_algorithms_arguments)
from data import add_data_arguments, make_data, set_data_args
from models import (LloydsModel, LloydsModelHyperParams,
                    FedClusterInitModel, FedClusterInitModelHyperParams, KFedModel, KFedModelHyperParams)
from privacy import get_mechanism
from utils import (set_seed, add_utils_arguments, post_evaluation, make_results_path,
                   kmeans_initialise_sphere_packing, maybe_inject_arguments_from_config)

from pfl.aggregate.simulate import SimulatedBackend
from pfl.callback import AggregateMetricsToDisk
from pfl.internal.ops.selector import set_framework_module
from pfl.internal.ops import numpy_ops, pytorch_ops


os.environ['PFL_PYTORCH_DEVICE'] = 'cpu'


def main():
    maybe_inject_arguments_from_config()

    parser = argparse.ArgumentParser()
    parser = add_algorithms_arguments(parser)
    parser = add_data_arguments(parser)
    parser = add_utils_arguments(parser)
    args = parser.parse_args()

    set_data_args(args)
    set_framework_module(numpy_ops, old_module=pytorch_ops)

    train_clients, val_clients, server_dataset, central_data = make_data(args)  # data seed is set here
    set_seed(args.seed)  # training seed is set here

    privacy_type = 'data_point_level' if args.datapoint_privacy else 'client_level'
    results_path = make_results_path(privacy_type, args.dataset)

    executed_privacy_mechanisms = []
    num_compositions = []
    sampling_probs = []

    # Run Initialization
    if args.initialization_algorithm in ['FederatedClusterInit', 'FederatedClusterInitExact']:
        if args.initialization_algorithm == 'FederatedClusterInitExact':
            args.center_init_send_sums_and_counts = True
        else:
            args.center_init_send_sums_and_counts = False
        init_model = FedClusterInitModel()
        init_model_params = FedClusterInitModelHyperParams(
            K=args.K,
        )

        init_algo_params = FederatedClusterInitHyperParams(
            K=args.K,
            center_init_send_sums_and_counts=args.center_init_send_sums_and_counts,
            server_dataset=server_dataset,
            num_iterations_svd=args.num_iterations_svd,
            num_iterations_weighting=args.num_iterations_weighting,
            num_iterations_center_init=args.num_iterations_center_init,
            multiplicative_margin=args.multiplicative_margin,
            minimum_server_point_weight=args.minimum_server_point_weight,
            train_cohort_size=args.num_train_clients*args.cohort_fraction,
            val_cohort_size=args.num_val_clients,
            datapoint_privacy=args.datapoint_privacy,
            outer_product_data_clipping_bound=args.outer_product_clipping_bound
        )

        print()
        print('#########')
        print('Running Outer Product Computation')
        print('#########')
        print()

        outer_product_results_path = os.path.join(results_path, 'outer_product_metrics.csv')
        mechanism_name = 'outer_product' if args.outer_product_privacy else 'no_privacy'
        outer_product_privacy = get_mechanism(args, mechanism_name)
        backend = SimulatedBackend(training_data=train_clients,
                                   val_data=val_clients,
                                   postprocessors=[outer_product_privacy])
        outer_prod_algo = FederatedOuterProduct()
        init_model = outer_prod_algo.run(
            backend=backend,
            model=init_model,
            algorithm_params=init_algo_params,
            model_train_params=init_model_params,
            model_eval_params=init_model_params,
            callbacks=[AggregateMetricsToDisk(outer_product_results_path)]
        )

        executed_privacy_mechanisms.append(outer_product_privacy.underlying_mechanism)
        num_compositions.append(args.num_iterations_svd)
        sampling_probs.append(args.cohort_fraction)

        print()
        print('#########')
        print('Running Point Weighting Computation')
        print('#########')
        print()

        point_weighting_results_path = os.path.join(results_path, 'point_weighting_metrics.csv')
        mechanism_name = 'point_weighting' if args.point_weighting_privacy else 'no_privacy'
        point_weighting_privacy = get_mechanism(args, mechanism_name)
        backend = SimulatedBackend(training_data=train_clients,
                                   val_data=val_clients,
                                   postprocessors=[point_weighting_privacy])
        weighting_algo = FederatedServerPointWeighting()
        init_model = weighting_algo.run(
            backend=backend,
            model=init_model,
            algorithm_params=init_algo_params,
            model_train_params=init_model_params,
            model_eval_params=init_model_params,
            callbacks=[AggregateMetricsToDisk(point_weighting_results_path)]
        )

        executed_privacy_mechanisms.append(point_weighting_privacy.underlying_mechanism)
        num_compositions.append(args.num_iterations_weighting)
        sampling_probs.append(args.cohort_fraction)

        print()
        print('#########')
        print('Running Center Computation')
        print('#########')
        print()

        center_init_results_path = os.path.join(results_path, 'center_init_metrics.csv')
        mechanism_name = 'center_init' if args.center_init_privacy else 'no_privacy'
        center_init_privacy = get_mechanism(args, mechanism_name)
        backend = SimulatedBackend(training_data=train_clients,
                                   val_data=val_clients,
                                   postprocessors=[center_init_privacy])
        center_init_algo = FederatedInitFromProjectedCenters()
        init_model = center_init_algo.run(
            backend=backend,
            model=init_model,
            algorithm_params=init_algo_params,
            model_train_params=init_model_params,
            model_eval_params=init_model_params,
            callbacks=[AggregateMetricsToDisk(center_init_results_path)]
        )

        executed_privacy_mechanisms.append(center_init_privacy.underlying_mechanism)
        num_compositions.append(args.num_iterations_center_init)
        sampling_probs.append(args.cohort_fraction)

        initial_centers = init_model.initial_centers
        assert initial_centers is not None

    elif args.initialization_algorithm == 'ServerKMeans++':
        init_model = FedClusterInitModel()
        server_X = server_dataset.raw_data[0]
        initial_centers, _ = kmeans_plusplus(server_X, args.K)
        init_model.initial_centers = initial_centers

    elif args.initialization_algorithm == 'ServerLloyds':
        init_model = FedClusterInitModel()
        server_X = server_dataset.raw_data[0]
        server_kmeans = KMeans(n_clusters=args.K)
        server_kmeans.fit(server_X)
        initial_centers = server_kmeans.cluster_centers_
        init_model.initial_centers = initial_centers

    elif args.initialization_algorithm == 'SpherePacking':
        init_model = FedClusterInitModel()
        server_X = server_dataset.raw_data[0]
        initial_centers = kmeans_initialise_sphere_packing(np.max(np.abs(server_X)), server_X.shape[1],
                                                           args.K, 1000, 0.0001,
                                                           1000)

        assert len(initial_centers) == args.K
        init_model.initial_centers = initial_centers

    elif args.initialization_algorithm == 'KFed':
        centers = None
        init_model = KFedModel(centers, args.K, args.K_client)
        model_params = KFedModelHyperParams(args.K, args.K_client)
        all_train_users = train_clients.get_cohort(args.num_train_clients)  # only works with minimize reuse sampler

        all_train_user_ids = [d.user_id for d, _ in all_train_users]

        userid_to_idx = dict(zip(all_train_user_ids, range(args.num_train_clients)))
        algo_params = KFedHyperParams(args.K, args.K_client, userid_to_idx,
                                      args.multiplicative_margin, args.num_train_clients, args.num_val_clients)

        backend = SimulatedBackend(training_data=train_clients,
                                   val_data=val_clients)
        algo = KFed()
        init_model = algo.run(
            backend=backend,
            model=init_model,
            algorithm_params=algo_params,
            model_train_params=model_params,
            model_eval_params=model_params
        )

    else:
        raise ValueError("Initialization algorithm not recognized.")

    results_dict = {}
    if args.clustering_algorithm == 'None' or args.fedlloyds_num_iterations == 0:
        when = 'Clustering'
    else:
        when = 'Initialization'

    post_evaluation(results_dict, args, init_model, central_data, when,
                    executed_privacy_mechanisms, num_compositions, sampling_probs, True)

    if args.clustering_algorithm != 'None' and args.fedlloyds_num_iterations != 0:
        print()
        print('#########')
        print('Running Clustering')
        print('#########')
        print()

        if args.clustering_algorithm == 'FederatedLloyds':
            args.send_sums_and_counts = True
            args.fedlloyds_cohort_size = args.num_train_clients # For simplicity sample all clients per round

            clustering_model = LloydsModel(initial_centers)
            model_params = LloydsModelHyperParams(
                K=args.K
            )

            algo_params = FederatedLloydsHyperParams(
                K=args.K,
                send_sums_and_counts=args.send_sums_and_counts,
                central_num_iterations=args.fedlloyds_num_iterations,
                evaluation_frequency=1,
                train_cohort_size=args.fedlloyds_cohort_size,
                val_cohort_size=args.num_val_clients
            )

            fedlloyds_results_path = os.path.join(results_path, 'fedlloyds_metrics.csv')
            mechanism_name = 'fedlloyds' if args.fedlloyds_privacy else 'no_privacy'
            fedlloyds_privacy = get_mechanism(args, mechanism_name)
            backend = SimulatedBackend(training_data=train_clients,
                                       val_data=val_clients,
                                       postprocessors=[fedlloyds_privacy])
            algo = FederatedLloyds()
            clustering_model = algo.run(
                backend=backend,
                model=clustering_model,
                algorithm_params=algo_params,
                model_train_params=model_params,
                model_eval_params=model_params,
                callbacks=[AggregateMetricsToDisk(fedlloyds_results_path)]
            )

            executed_privacy_mechanisms.append(fedlloyds_privacy.underlying_mechanism)
            num_compositions.append(args.fedlloyds_num_iterations)
            sampling_probs.append(args.fedlloyds_cohort_size / args.num_train_clients)

        else:
            raise ValueError("Clustering algorithm not recognized.")

        post_evaluation(results_dict, args, clustering_model, central_data, 'Clustering',
                        executed_privacy_mechanisms, num_compositions, sampling_probs, True)

    optimal_central = KMeans(n_clusters=args.K)
    optimal_central.fit(central_data[0].raw_data[0])
    central_clustering_model = LloydsModel(optimal_central.cluster_centers_)

    post_evaluation(results_dict, args, central_clustering_model, central_data, 'Optimal',
                    [], [], sampling_probs, True)

    with open(os.path.join(results_path, 'summary_results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == '__main__':
    main()
