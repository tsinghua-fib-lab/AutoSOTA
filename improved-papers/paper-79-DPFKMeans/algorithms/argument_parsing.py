import argparse
from utils import str2bool


def add_algorithms_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--initialization_algorithm", type=str,
                        choices=['FederatedClusterInit', 'FederatedClusterInitExact',
                                 'ServerKMeans++', 'KFed', 'SpherePacking', 'ServerLloyds'],
                        default='FederatedClusterInitExact')
    parser.add_argument("--clustering_algorithm", type=str,
                        choices=['FederatedLloyds', 'None'],
                        default='FederatedLloyds')

    parser.add_argument("--overall_target_delta", type=float, default=1e-6)
    parser.add_argument("--datapoint_privacy", type=str2bool, default=False)

    parser = add_fedclusterinit_arguments(parser)
    parser = add_fedlloyds_arguments(parser)
    parser = add_kfed_arguments(parser)

    return parser


def add_fedclusterinit_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument("--num_iterations_svd", type=int, default=1)
    parser.add_argument("--num_iterations_weighting", type=int, default=1)
    parser.add_argument("--num_iterations_center_init", type=int, default=1)
    parser.add_argument("--cohort_fraction", type=float, default=1)
    parser.add_argument("--multiplicative_margin", type=float, default=1)
    parser.add_argument("--minimum_server_point_weight", type=float, default=0)

    parser.add_argument("--outer_product_privacy", type=str2bool, default=True)
    parser.add_argument("--outer_product_clipping_bound", type=float, default=1510)
    parser.add_argument("--outer_product_epsilon", type=float, default=1)
    parser.add_argument("--outer_product_delta", type=float, default=1e-6)

    parser.add_argument("--point_weighting_privacy", type=str2bool, default=True)
    parser.add_argument("--weighting_epsilon", type=float, default=1)
    parser.add_argument("--weighting_clipping_bound", type=float, default=50)

    parser.add_argument('--center_init_send_sums_and_counts', type=str2bool, default=False)
    parser.add_argument("--center_init_privacy", type=str2bool, default=True)
    parser.add_argument("--center_init_clipping_bound", type=float, default=110)
    parser.add_argument("--center_init_laplace_clipping_bound", type=float, default=50)
    parser.add_argument("--center_init_gaussian_epsilon", type=float, default=1)
    parser.add_argument("--center_init_delta", type=float, default=1e-6)
    parser.add_argument("--center_init_epsilon_split", type=float, default=0.5)
    parser.add_argument("--center_init_contributed_components_epsilon", type=float, default=10)
    parser.add_argument("--center_init_contributed_components_clipping_bound", type=float, default=30)

    parser.add_argument("--initialization_target_delta", type=float, default=1e-6)

    return parser


def add_fedlloyds_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument('--send_sums_and_counts', type=str2bool, default=True)
    parser.add_argument("--fedlloyds_cohort_size", type=int, default=100)
    parser.add_argument("--fedlloyds_num_iterations", type=int, default=15)

    parser.add_argument("--fedlloyds_privacy", type=str2bool, default=True)
    parser.add_argument("--fedlloyds_epsilon", type=float, default=1)
    parser.add_argument("--fedlloyds_epsilon_split", type=float, default=0.5,
                        help='Assigns this fraction of the epsilon to the sum points noising')
    parser.add_argument("--fedlloyds_delta", type=float, default=1e-6)
    parser.add_argument("--fedlloyds_clipping_bound", type=float, default=85)
    parser.add_argument("--fedlloyds_laplace_clipping_bound", type=float, default=50)
    parser.add_argument("--fedlloyds_contributed_components_epsilon", type=float, default=0.2)
    parser.add_argument("--fedlloyds_contributed_components_clipping_bound", type=float, default=10)

    return parser


def add_kfed_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument('--K_client', type=int, default=10)

    return parser
