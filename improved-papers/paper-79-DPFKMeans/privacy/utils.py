import argparse

import numpy as np
from dp_accounting.pld import privacy_loss_distribution
from typing import List

from pfl.hyperparam import get_param_value
from pfl.privacy import (CentrallyAppliedPrivacyMechanism, GaussianMechanism,
                         LaplaceMechanism, PLDPrivacyAccountant, NoPrivacy, PrivacyMechanism)

from privacy import (MultipleMechanisms, SymmetricGaussianMechanism,
                     DataPrivacyGaussianMechanism, DataPrivacySymmetricGaussianMechanism, DataPrivacyLaplaceMechanism)


def get_mechanism(args: argparse.Namespace, mechanism_name):
    if mechanism_name == 'outer_product':
        mechanism_cls = DataPrivacySymmetricGaussianMechanism if args.datapoint_privacy else SymmetricGaussianMechanism
        mechanism = CentrallyAppliedPrivacyMechanism(
            mechanism_cls.construct_single_iteration(
                clipping_bound=args.outer_product_clipping_bound,
                epsilon=args.outer_product_epsilon,
                delta=args.outer_product_delta)
        )

    elif mechanism_name == 'point_weighting':
        mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
        mechanism = CentrallyAppliedPrivacyMechanism(
            mechanism_cls(args.weighting_clipping_bound, args.weighting_epsilon)
        )

    elif mechanism_name == 'center_init':
        if args.center_init_send_sums_and_counts:
            sum_points_mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            sum_points_privacy = sum_points_mechanism_cls.construct_single_iteration(
                clipping_bound=args.center_init_clipping_bound,
                epsilon=args.center_init_gaussian_epsilon * args.center_init_epsilon_split,
                delta=args.center_init_delta)

            num_points_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            num_points_privacy = num_points_mechanism_cls(
                args.center_init_laplace_clipping_bound, args.center_init_gaussian_epsilon * (1 - args.center_init_epsilon_split)
            )
            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [sum_points_privacy, num_points_privacy],
                [('sum_points_per_component',), ('num_points_per_component',)]))
        else:
            mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            mean_points_privacy = mechanism_cls.construct_single_iteration(
                clipping_bound=args.center_init_clipping_bound,
                epsilon=args.center_init_gaussian_epsilon,
                delta=args.center_init_delta)

            components_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            contributed_components_privacy = components_mechanism_cls(args.center_init_contributed_components_clipping_bound,
                                                              args.center_init_contributed_components_epsilon)

            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [mean_points_privacy, contributed_components_privacy],
                [('mean_points_per_component',), ('contributed_components',)]))

    elif mechanism_name == 'fedavg_kmeans':
        sampling_probability = args.fedavg_cohort_size / args.num_train_clients
        fedavg_accountant = PLDPrivacyAccountant(
            num_compositions=args.fedavg_num_iterations,
            sampling_probability=sampling_probability,
            mechanism='gaussian',
            epsilon=args.fedavg_epsilon,
            delta=args.fedavg_delta)

        fedavg_gaussian_noise_mechanism = GaussianMechanism.from_privacy_accountant(
            accountant=fedavg_accountant, clipping_bound=args.fedavg_clipping_bound)

        mechanism = CentrallyAppliedPrivacyMechanism(fedavg_gaussian_noise_mechanism)

    elif mechanism_name == 'fedlloyds':
        if args.send_sums_and_counts:
            sampling_probability = args.fedlloyds_cohort_size / args.num_train_clients
            sum_points_accountant = PLDPrivacyAccountant(
                num_compositions=args.fedlloyds_num_iterations,
                sampling_probability=sampling_probability,
                mechanism='gaussian',
                epsilon=args.fedlloyds_epsilon * args.fedlloyds_epsilon_split,
                delta=args.fedlloyds_delta)

            num_points_accountant = PLDPrivacyAccountant(
                num_compositions=args.fedlloyds_num_iterations,
                sampling_probability=sampling_probability,
                mechanism='laplace',
                epsilon=args.fedlloyds_epsilon * (1 - args.fedlloyds_epsilon_split),
                delta=args.fedlloyds_delta)

            sum_points_mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            sum_points_privacy = sum_points_mechanism_cls.from_privacy_accountant(
                accountant=sum_points_accountant, clipping_bound=args.fedlloyds_clipping_bound)

            num_points_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            laplace_noise_param = num_points_accountant.cohort_noise_parameter
            num_points_privacy = num_points_mechanism_cls(args.fedlloyds_laplace_clipping_bound,
                                                  1 / laplace_noise_param)

            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [sum_points_privacy, num_points_privacy],
                [('sum_points_per_component',), ('num_points_per_component',)]))

        else:
            sampling_probability = args.fedlloyds_cohort_size / args.num_train_clients
            fedlloyds_accountant = PLDPrivacyAccountant(
                num_compositions=args.fedlloyds_num_iterations,
                sampling_probability=sampling_probability,
                mechanism='gaussian',
                epsilon=args.fedlloyds_epsilon,
                delta=args.fedlloyds_delta)

            mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            fedlloyds_gaussian_noise_mechanism = mechanism_cls.from_privacy_accountant(
                accountant=fedlloyds_accountant, clipping_bound=args.fedlloyds_clipping_bound)

            components_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            contributed_components_privacy = components_mechanism_cls(args.fedlloyds_contributed_components_clipping_bound, args.fedlloyds_contributed_components_epsilon)

            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [fedlloyds_gaussian_noise_mechanism, contributed_components_privacy],
                [('mean_points_per_component',), ('contributed_components',)]))

    elif mechanism_name == 'no_privacy':
        mechanism = CentrallyAppliedPrivacyMechanism(NoPrivacy())

    else:
        raise ValueError('Mechanism name not recognized.')

    return mechanism


def compute_privacy_accounting(mechanisms: List[PrivacyMechanism], target_delta: float,
                               num_compositions: List[int] = None, sampling_probs: List[float] = None):
    if len(mechanisms) == 0:
        return 0

    if num_compositions is None:
        num_compositions = [1] * len(mechanisms)

    if sampling_probs is None:
        sampling_probs = [1] * len(mechanisms)

    unpacked_mechanisms = []
    unpacked_num_compositions = []
    unpacked_sampling_probs = []
    for mechanism, n, p in zip(mechanisms, num_compositions, sampling_probs):
        if type(mechanism).__name__ == "MultipleMechanisms":
            for sub_mechanism in mechanism.mechanisms:
                unpacked_mechanisms.append(sub_mechanism)
                unpacked_num_compositions.append(n)
                unpacked_sampling_probs.append(p)
        else:
            unpacked_mechanisms.append(mechanism)
            unpacked_num_compositions.append(n)
            unpacked_sampling_probs.append(p)

    dp_accounting_mechanisms = []
    for mechanism, n, p in zip(unpacked_mechanisms, unpacked_num_compositions, unpacked_sampling_probs):
        if type(mechanism).__name__ in ['GaussianMechanism', 'SymmetricGaussianMechanism',
                                        'DataPrivacySymmetricGaussianMechanism', 'DataPrivacyGaussianMechanism']:
            dp_accounting_mechanism = privacy_loss_distribution.from_gaussian_mechanism(
                mechanism.relative_noise_stddev,
                value_discretization_interval=1e-3,
                sampling_prob=p
            )

        elif type(mechanism).__name__ in ['LaplaceMechanism', 'DataPrivacyLaplaceMechanism']:
            noise_scale = get_param_value(mechanism._clipping_bound) / mechanism._epsilon
            dp_accounting_mechanism = privacy_loss_distribution.from_laplace_mechanism(
                noise_scale,
                sensitivity=get_param_value(mechanism._clipping_bound),
                value_discretization_interval=1e-3,
                sampling_prob=p
            )

        elif type(mechanism).__name__ == 'NoPrivacy':
            return np.inf

        else:
            raise ValueError('Mechanism must be Gaussian or Laplace.')

        composed_dp_accounting_mechanism = dp_accounting_mechanism.self_compose(n)
        dp_accounting_mechanisms.append(composed_dp_accounting_mechanism)

    full_composed_mechanism = dp_accounting_mechanisms[0]

    for mechanism in dp_accounting_mechanisms[1:]:
        full_composed_mechanism = full_composed_mechanism.compose(mechanism)

    epsilon = full_composed_mechanism.get_epsilon_for_delta(target_delta)
    return epsilon
