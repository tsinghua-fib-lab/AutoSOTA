import argparse


def add_data_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser = add_stackoverflow_arguments(parser)
    parser = add_folktables_arguments(parser)

    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--dataset", type=str,
                        choices=["GaussianMixtureUniform", "stackoverflow", "folktables"],
                        default="GaussianMixtureUniform")
    parser.add_argument("--num_train_clients", type=int, default=100)
    parser.add_argument("--num_val_clients", type=int, default=100)
    parser.add_argument("--samples_per_client", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--variance", type=float, default=0.5)
    parser.add_argument("--samples_per_mixture_server", type=int, default=20)
    parser.add_argument("--num_uniform_server", type=int, default=100)

    return parser


def add_stackoverflow_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument("--topics_list", type=str,
                        choices=['fb-hb',  'gith-pdf',  'ml-math',  'plt-cook'],
                        default='gith-pdf')

    return parser


def add_folktables_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument("--filter_label", type=int,
                        choices=[2, 5, 6],
                        default=5)

    return parser


def set_data_args(args):
    if args.dataset in ['GaussianMixture', 'GaussianMixtureUniform']:
        data_file_arg_names = ['samples_per_mixture_server', 'num_uniform_server', 'num_train_clients']

    elif args.dataset == 'stackoverflow':
        num_clients_lookup = {
            'plt-cook': (2720, 328),
            'iph-pd': (32457, 3651),
            'gith-pdf': (9237, 1157),
            'fb-hb': (23266, 2736),
            'ml-math': (10394, 1265)
        }
        args.num_train_clients, args.num_val_clients = num_clients_lookup[args.topics_list]
        data_file_arg_names = ['samples_per_mixture_server', 'num_uniform_server', 'topics_list']

    elif args.dataset == 'folktables':
        args.num_train_clients = 51
        args.num_val_clients = 51

        data_file_arg_names = ['samples_per_mixture_server', 'num_uniform_server', 'filter_label']

    else:
        raise ValueError('Dataset not recognized.')

    return data_file_arg_names
