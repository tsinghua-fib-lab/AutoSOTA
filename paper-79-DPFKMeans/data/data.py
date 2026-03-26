import argparse
import os
import pathlib
import h5py
import numpy as np

from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import get_user_sampler

from utils import set_seed

from .folktables import make_folktables_datasets


def make_data(args: argparse.Namespace):
    set_seed(args.data_seed)
    dataset = args.dataset

    if dataset == 'GaussianMixtureUniform':
        K, d = args.K, args.dim
        means = make_uniform_gaussian_means(K, d)
        covs = np.stack([args.variance * np.eye(d) for _ in range(args.K)])
        mixture_weights = np.ones(K) / K

        def num_samples_fn():
            return args.samples_per_client

        all_clients = []
        central_data = []
        for num_clients in [args.num_train_clients, args.num_val_clients]:
            client_datasets = []
            all_x, all_y = [], []
            for i in range(num_clients):
                n = num_samples_fn()
                x, y = generate_gaussian_mixture_data(n, mixture_weights, means, covs)
                client_datasets.append(Dataset((x, y), user_id=str(i)))
                all_x.append(x)
                all_y.append(y)

            all_x = np.vstack(all_x)
            all_y = np.hstack(all_y)
            central_data.append(Dataset((all_x, all_y)))

            def make_dataset_fn(user_id, datasets=client_datasets):
                return datasets[user_id]

            user_sampler = get_user_sampler('minimize_reuse', list(range(num_clients)))
            all_clients.append(FederatedDataset(make_dataset_fn, user_sampler))

        samples_per_mixture = args.samples_per_mixture_server * np.ones(K, dtype=int)
        server_x, server_y = [], []
        for k in range(K):
            mixture_weights = np.zeros(K)
            mixture_weights[k] = 1
            x, y = generate_gaussian_mixture_data(samples_per_mixture[k], mixture_weights, means, covs)
            assert y.sum() == len(y) * k
            server_x.append(x)
            server_y.append(y)

        num_uniform_server = args.num_uniform_server
        server_x.append(np.random.uniform(size=(num_uniform_server, d)))
        server_y.append(K * np.ones(num_uniform_server))
        server_dataset = Dataset((np.vstack(server_x), np.hstack(server_y)))

    elif dataset == 'stackoverflow':
        path_to_data_file = pathlib.Path(__file__).parent.absolute()
        path_to_data = os.path.join(path_to_data_file, 'stackoverflow', 'topic_extracted_data', args.topics_list)
        if not os.path.exists(path_to_data):
            raise NotADirectoryError('Please download stackoverflow dataset following the provided instuctions.')

        topic_abreviations = {'machine-learning': 'ml',
                              'math': 'math',
                              'facebook': 'fb',
                              'hibernate': 'hb',
                              'github': 'gith',
                              'pdf': 'pdf',
                              'plot': 'plt',
                              'cookies': 'cook'
                              }
        abreviations_lookup = {abr: topic for topic, abr in topic_abreviations.items()}
        topics = [abreviations_lookup[abr] for abr in args.topics_list.split('-')]

        all_clients = []
        central_data = []
        for split in ['train', 'val']:
            num_clients = args.num_train_clients if split == 'train' else args.num_val_clients
            client_datasets = []
            filename = f'{split}_users.hdf5'
            with h5py.File(os.path.join(path_to_data, filename), 'r') as f:
                d = f[split]
                user_ids = d.keys()
                print(f'Number of {split} users: {len(user_ids)}')
                for user_id in user_ids:
                    user_data = d[user_id]
                    x = user_data['embeddings'][()]
                    y = user_data['labels'][()]
                    client_datasets.append(Dataset((x, y), user_id=user_id))

                all_x = np.vstack([d.raw_data[0] for d in client_datasets])
                all_y = np.hstack([d.raw_data[1] for d in client_datasets])
                centralized_dataset = Dataset((all_x, all_y))
                central_data.append(centralized_dataset)

                def make_dataset_fn(user, datasets=client_datasets):
                    return datasets[user]

                user_sampler = get_user_sampler('minimize_reuse', list(range(num_clients)))
                all_clients.append(FederatedDataset(make_dataset_fn, user_sampler))

        path_to_server_data = os.path.join(path_to_data, 'server_data.hdf5')
        if not os.path.exists(path_to_server_data):

            train_x, train_y = central_data[0].raw_data
            sampled_idxs = np.random.choice(np.arange(len(train_y)), size=args.samples_per_mixture_server * len(topics))
            server_x = train_x[sampled_idxs]
            server_y = train_y[sampled_idxs]

            uniform_server_x = []
            uniform_server_y = []
            path_to_uniform_file = os.path.join('data', 'stackoverflow', 'embedded_data', 'embedded_stackoverflow_train_0.hdf5')
            with h5py.File(path_to_uniform_file, 'r') as f:
                done = False
                user_ids = list(f['train'].keys())
                for user_id in user_ids:
                    user_data = f['train'][user_id]
                    tags = user_data['tags'][()].astype(str)
                    embeddings = user_data['embeddings'][()]
                    for embedding, tag in zip(embeddings, tags):
                        if not set(tag.split('|')).intersection(set(topics)):
                            uniform_server_x.append(embedding)
                            uniform_server_y.append(len(topics))

                        if len(uniform_server_y) == args.num_uniform_server:
                            done = True
                            break

                    if done:
                        break

            uniform_server_x = np.vstack(uniform_server_x)
            uniform_server_y = np.hstack(uniform_server_y)

            server_x = np.vstack([server_x, uniform_server_x])
            server_y = np.hstack([server_y, uniform_server_y])

            with h5py.File(path_to_server_data, 'w') as server_f:
                server_f.create_dataset('server_x', data=server_x)
                server_f.create_dataset('server_y', data=server_y)
        else:
            with h5py.File(path_to_server_data, 'r') as server_f:
                server_x = server_f['server_x'][()]
                server_y = server_f['server_y'][()]
        server_dataset = Dataset((server_x, server_y))

    elif dataset == 'folktables':
        path_to_data_file = pathlib.Path(__file__).parent.absolute()
        path_to_data = os.path.join(path_to_data_file, 'folktables', 'cow_extracted_data', f'fl-{args.filter_label}')
        if not os.path.exists(path_to_data):
            make_folktables_datasets(args)
        all_clients, central_data = [], []
        for split in ['train', 'val']:
            client_datasets = []
            with h5py.File(os.path.join(path_to_data, f'{split}_users.hdf5'), 'r') as f:
                user_ids = list(f.keys())
                for user_id in user_ids:
                    features = f[user_id]['features'][()].astype(int)
                    labels = f[user_id]['labels'][()]

                    client_datasets.append(Dataset((features, labels), user_id=user_id))

            def make_dataset_fn(user, datasets=client_datasets):
                return datasets[user]

            user_sampler = get_user_sampler('minimize_reuse', list(range(len(user_ids))))
            all_clients.append(FederatedDataset(make_dataset_fn, user_sampler))

            all_x = np.vstack([d.raw_data[0] for d in client_datasets])
            all_y = np.hstack([d.raw_data[1] for d in client_datasets])
            central_data.append(Dataset((all_x, all_y)))

        with h5py.File(os.path.join(path_to_data, 'server_data.hdf5'), 'r') as f:
            train_x, train_y = central_data[0].raw_data
            in_dist_idxs = np.random.choice(range(len(train_x)), size=2 * args.samples_per_mixture_server)

            total_num_datapoints = 0
            for state in f.keys():
                total_num_datapoints += len(f[state]['labels'])

            uniform_features = []
            uniform_labels = []
            for state in f.keys():
                features, labels = f[state]['features'][()].astype(int), f[state]['labels'][()].reshape(-1)
                idxs = np.random.choice(range(len(features)),
                                        size=int(args.num_uniform_server * (len(features) / total_num_datapoints)) + 1,
                                        replace=False
                                        )
                uniform_features.append(features[idxs])
                uniform_labels.append(labels[idxs])

            uniform_features = np.vstack(uniform_features)
            uniform_labels = np.hstack(uniform_labels)
            uniform_idxs = np.random.choice(range(len(uniform_features)), size=args.num_uniform_server, replace=False)

            server_x = np.vstack([train_x[in_dist_idxs], uniform_features[uniform_idxs]])
            server_y = np.hstack([train_y[in_dist_idxs], uniform_labels[uniform_idxs]])
            server_dataset = Dataset((server_x, server_y))

    else:
        raise ValueError("Dataset not recognized.")

    return all_clients[0], all_clients[1], server_dataset, central_data


def make_uniform_gaussian_means(k, d, x_min=0, x_max=1):
    """
    Generates k uniformly random vectors of dimension d in [x_min, x_max] hypercube.

    :param k: number of vectors.
    :param d: dimension.
    :param x_min: hypercube lower bound.
    :param x_max: hypercube upper bound.
    :return: np.array of shape (k, d)
    """
    return np.random.uniform(low=x_min, high=x_max, size=(k, d)).astype(np.float32)


def generate_gaussian_mixture_data(num_samples: int, mixture_weights: np.array, means: np.array, covs: np.array):
    """
    Generates samples from a mixture of Gaussians.

    :param num_samples: number of samples to generate
    :param mixture_weights: array of length k, where k is the number of gaussians
    :param means: array of shape (k, d), where d is the dimension
    :param covs:: array of shape (k, d, d), each (d, d) array is the covariance matrix of a gaussian
    :return: (x, y) arrays of shape (num_samples, d) and (num_samples). Samples and labels of each sample.
    """
    d = len(means[0])
    cumulative_mixture_sum = np.cumsum(mixture_weights)
    z = np.random.uniform(size=(num_samples, 1))
    components_of_all_samples = (z <= cumulative_mixture_sum).argmax(axis=1)
    unique_components, per_component_counts = np.unique(components_of_all_samples, return_counts=True)

    x = np.zeros((num_samples, d), dtype=np.float32)
    y = np.zeros(num_samples, dtype=int)
    for k, count in zip(unique_components, per_component_counts):
        x[components_of_all_samples == k] = np.random.multivariate_normal(mean=means[k], cov=covs[k], size=count).astype(np.float32)
        y[components_of_all_samples == k] = k * np.ones(count)

    return x, y
