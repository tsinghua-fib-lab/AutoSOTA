import random
import logging
import numpy as np
from typing import Dict, List, Union
from copy import deepcopy

import torch
import torch.utils
import torch.utils.data

from .utils import get_dirichlet_distribution, check_distribution_lengths


__all__ = [
    'split',
    'shard_split',
    'homogenous_partition',
    'shard_partition',
    'dirichlet_partition',
    'check_clients_map',
]

def balanced_split(num_clients, num_samples_per_client) -> Union[List , int]:
    """ 
    Computes a balanced split 
    
    adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
    amount_samples_per_client = (np.ones(num_clients) * num_samples_per_client).astype(int)
    return amount_samples_per_client


def unbalanced_split(num_clients, num_samples, num_samples_per_client, unbalanced_sigma):
    """ 
    Computes a balanced split 
    
    adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
    assert unbalanced_sigma != 0 
    amount_samples_per_client = np.random.lognormal(
        mean=np.log(num_samples_per_client),
        sigma=unbalanced_sigma,
        size=num_clients
    )
    amount_samples_per_client = (
        amount_samples_per_client / amount_samples_per_client.sum() * num_samples
    ).astype(int) # scale
    diff = amount_samples_per_client.sum() - num_samples
    if diff != 0:
        for cid in range(num_clients):
            if amount_samples_per_client[cid] > diff:
                amount_samples_per_client[cid] -= diff
                break
    
    return amount_samples_per_client


def unbalanced_split_with_min_samples(
    num_clients: int, 
    num_samples: int, 
    min_samples_per_client: int,
    ) -> List[int]:
    """ Poor, dont use """
    amount_samples_per_client = []
    max_samples_per_client = (num_samples - min_samples_per_client * num_clients) // num_clients + min_samples_per_client
    for _ in range(num_clients-1):
        amount_samples_per_client.append(random.randint(min_samples_per_client, max_samples_per_client)) # Generate random number of samples for each bucket within the bounds
    amount_samples_per_client.append(num_samples - sum(amount_samples_per_client))

    return amount_samples_per_client

def split(
    num_clients: int, 
    num_samples: int, 
    balanced: bool,
    unbalanced_sigma: float=None, 
    min_samples_per_client: int=None,
    ) -> List[int]:
    """
    General split function to handle balance of data samples across clients
    """
    num_samples_per_client = int(num_samples / num_clients)

    if balanced:
        return balanced_split(num_clients, num_samples_per_client)
    else:
        if unbalanced_sigma != 0.0 and unbalanced_sigma is not None:
            return unbalanced_split(num_clients, num_samples, num_samples_per_client, unbalanced_sigma)
        elif min_samples_per_client is not None:
            return unbalanced_split_with_min_samples(num_clients, num_samples, min_samples_per_client)
        else:
            raise RuntimeError


def balanced_shard_split(
        num_clients: int,
        num_shards: int,
        num_shards_per_client: int=None,
    ) -> int:
    max_number_of_shards_per_client = num_shards // num_clients
    num_shards_per_client = max_number_of_shards_per_client if num_shards_per_client is None else num_shards_per_client

    if int(max_number_of_shards_per_client) < num_shards_per_client:
        raise RuntimeError(f'Number of shards set ({num_shards_per_client}) exceed maximum possible ({int(max_number_of_shards_per_client)})')
    return num_shards_per_client


def unbalanced_shard_split(
    num_clients: int=100, 
    num_shards: int=200, 
    min_shards_per_client: int=1,
    max_shards_per_client: int=5,   
    ) -> List[int]:
    assert min_shards_per_client is not None and max_shards_per_client is not None
    num_shards_per_client = []
    
    local_num_shards = num_shards
    correction = None
    for i in range(num_clients-1):
        # ensure, each client gets min number of shards
        if local_num_shards == (num_clients-i)*min_shards_per_client or correction:
            rand_shard = 1
        elif local_num_shards > (num_clients-i)*min_shards_per_client:
            # randomly allocate a number of shards to a client
            rand_shard = random.randint(min_shards_per_client, max_shards_per_client)
            # ensure min number of shards and no negative allocation of shards
            if (local_num_shards - rand_shard) <= (num_clients-i)*min_shards_per_client:
                rand_shard = min(1, abs((local_num_shards - rand_shard) - (num_clients-i)*min_shards_per_client - 1))
                correction = True
        elif local_num_shards < (num_clients-i)*min_shards_per_client:
            raise RuntimeError(f'{local_num_shards} < {(num_clients-i)*min_shards_per_client}')
        
        local_num_shards -= rand_shard
        num_shards_per_client.append(rand_shard)
    
    num_shards_per_client.append(num_shards - sum(num_shards_per_client))
    assert sum(num_shards_per_client) == num_shards and min(num_shards_per_client) == min_shards_per_client, f'Something went wrong! {sum(num_shards_per_client), num_shards, min(num_shards_per_client), min_shards_per_client}'
    random.shuffle(num_shards_per_client) # shuffle 
    return num_shards_per_client


def shard_split(
    balanced: bool, 
    num_clients: int=100, 
    num_shards: int=200, 
    num_shards_per_client: int=None, 
    min_shards_per_client: int=1,
    max_shards_per_client: int=5, 
    ) -> Union[int, List[int]]:
    """
    Split function for shard partition.
    """
    if balanced:
        return balanced_shard_split(num_clients=num_clients, num_shards=num_shards, num_shards_per_client=num_shards_per_client)
    else:
        return unbalanced_shard_split(num_clients=num_clients, num_shards=num_shards, 
            min_shards_per_client=min_shards_per_client, max_shards_per_client=max_shards_per_client)


def homogenous_partition(
    num_clients: int, 
    num_samples: int,
    num_samples_per_client: List[int], 
    logger: logging.getLogger=None,
    ) -> Dict[int, List[int]]:
    """
    I.i.d partition of the data indices given the sample numbers for each client.
    """
    print('num_samples', num_samples)
    all_indices =[i for i in range(num_samples)]
    client_data_indices_map =  {} # will contain client id and the corresponding indice
    for i in range(num_clients):
        client_data_indices_map[i] = set(np.random.choice(all_indices, num_samples_per_client[i], replace=False))
        all_indices = list(set(all_indices) - client_data_indices_map[i])
    
    check_clients_map(client_data_indices_map, logger=logger)
    
    return client_data_indices_map


def shard_partition(
    train_dataset: torch.utils.data.Dataset,
    num_clients: int=100, 
    num_shards: int=200, 
    num_samples: int=300, 
    num_shards_per_client: int=2,
    logger: logging.getLogger=None,
    ) -> Dict[int, List[int]]:
    """
    Non-iid partition used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_.
    """
    idx_shard = [i for i in range(num_shards)]
        
    # general
    client_data_indices_map = {i: torch.tensor([]) for i in range(num_clients)}
    targets = train_dataset.targets.type(torch.long)
    indices = torch.arange(len(train_dataset), dtype=torch.long)

    # allocate indices to labels and sort labels
    indices_labels = torch.vstack((indices, targets)) # [2, len(indices)] [0, :] indices, [1, :] labels
    indices_labels_sorted = indices_labels[:, indices_labels[1, :].argsort()] # sort by labels
    indices_sorted = indices_labels_sorted[0, :] # get sorted indices
    #labels_sorted = indices_labels_sorted[1, :] # get sorted labels
    
    # assign the shards and indicies to the clients
    for i in range(num_clients):
        if isinstance(num_shards_per_client, list):
            rand_set_shards = set(np.random.choice(idx_shard, num_shards_per_client[i], replace=False))
        elif isinstance(num_shards_per_client, int):
            rand_set_shards = set(np.random.choice(idx_shard, num_shards_per_client, replace=False))
        else:
            raise RuntimeError(f'{type(num_shards_per_client)} should be list | int')
        
        idx_shard = list(set(idx_shard) - rand_set_shards) # remove used shards
        for rand_shard in rand_set_shards:
            client_data_indices_map[i] = torch.concatenate(
                (client_data_indices_map[i], indices_sorted[rand_shard*num_samples:(rand_shard+1)*num_samples]), 
                axis=0
            ).type(torch.long)
    
    # check
    for k, v in client_data_indices_map.items():
        assert v.size(0) != 0, f'Size cannot be {v.size()}'
        #logger.info(k, v.size(), torch.unique(labels_sorted[v.type(torch.long)]))
    
    client_data_indices_map = {k: v.tolist() for k, v in client_data_indices_map.items()}

    # check 
    check_clients_map(client_data_indices_map, logger=logger)

    return client_data_indices_map


def dirichlet_partition(
    targets: Union[np.ndarray, List],
    num_clients: int,
    num_classes: int, 
    amount_samples_per_client: Union[np.ndarray, List],
    dir_alpha: int,
    logger: logging.getLogger=None
    ) -> Dict[int, List[int]]:
    """
    Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    """
    verbose = True if logger is not None else False
    if isinstance(targets, list):
        targets = np.array(targets)

    idx_list = [np.where(targets == i)[0] for i in range(num_classes)] # get all samples per class
    class_amount_available = [len(idx_list[i]) for i in range(num_classes)] # get number of samples per class
    #print('class_amount_available', class_amount_available)
    class_priors_dirichlet = get_dirichlet_distribution(
        dir_alpha=dir_alpha, 
        num_classes=num_classes,
        num_clients=num_clients,
        amount_samples_per_client=amount_samples_per_client
    ) # compute dirichlet distribution
    class_priors_dirichlet = check_distribution_lengths(class_priors_dirichlet, amount_samples_per_client) # check dirichlet distribution samples are matching with given splits
    #print('class_priors_dirichlet', class_priors_dirichlet)
    client_data_indices_map = {k: [] for k in range(num_clients)} # output: indices map #np.zeros(amount_samples_per_client[k]).astype(np.int64) 
    clients_not_processed = []

    # Inital allocation, adds remainder clients to be processed
    for client_id in range(num_clients):
        client_class_dirichlet_dist = class_priors_dirichlet[client_id]
        class_ids = np.where(client_class_dirichlet_dist != 0.0)[0].astype(np.int64)
        for class_id in class_ids:
            if (class_amount_available[class_id]) >= client_class_dirichlet_dist[class_id]: # check if available amount of allocated samples available, if not add to remainder
                for _ in range(client_class_dirichlet_dist[class_id]): # loop throught range of samples
                    class_amount_available[class_id] -= 1 # subtract -1 from length, used as index
                    amount_samples_per_client[client_id] -= 1 # subtract -1 from length, used as index
                    
                    client_data_indices_map[client_id].append(
                        idx_list[class_id][class_amount_available[class_id]]
                    )
                
            else:
                clients_not_processed.append(client_id)

    if verbose: 
        logger.info('\nAfter Initial allocation\n')
        logger.info('-'*45)
        logger.info(f'class_amount_available:\t{class_amount_available}')
        logger.info(f'amount_samples_per_client:\t{amount_samples_per_client}')
        logger.info(f'clients_not_processed:\t\t{clients_not_processed}')

    # Handle Remainder
    # Resample distribution for not processed clients
    if verbose: 
        logger.info('\nResample\n')
        logger.info('-'*45)

    while np.sum(amount_samples_per_client) != 0.0:
        # get new dist 
        dirichlet_dist_new = get_dirichlet_distribution(
            dir_alpha=dir_alpha, 
            num_classes=num_classes, 
            num_clients=len(clients_not_processed), 
            amount_samples_per_client=amount_samples_per_client[clients_not_processed]
        )

        for client_id_new, client_id in enumerate(clients_not_processed):
            if verbose:
                logger.info('Remaining Data: %d' % np.sum(amount_samples_per_client))
            client_class_dirichlet_dist = dirichlet_dist_new[client_id_new]
            class_ids = np.where(client_class_dirichlet_dist != 0.0)[0].astype(np.int64)
            for class_id in class_ids:
                while True:
                    if class_amount_available[class_id] != 0 and amount_samples_per_client[client_id] != 0:
                        # if sample per class and samples per client are still available,
                        # select the minimum and allocate the samples 
                        if verbose: logger.info(f'samples available (client {client_id}): {amount_samples_per_client[client_id]}')
                        min_value = min(class_amount_available[class_id], amount_samples_per_client[client_id])
                        for _ in range(min_value):
                            class_amount_available[class_id] -= 1
                            amount_samples_per_client[client_id] -= 1
                            client_data_indices_map[client_id].append(
                                idx_list[class_id][class_amount_available[class_id]]
                            )
                    elif class_amount_available[class_id] == 0 and amount_samples_per_client[client_id] != 0:
                        # if class is not available but still missing samples per client, fill with other classes
                        # manuel assignment
                        if verbose: logger.info(f'samples available (client {client_id}): {amount_samples_per_client[client_id]}')
                        for manuel_class_id, class_amount in enumerate(class_amount_available):
                            min_value = min(class_amount, amount_samples_per_client[client_id])
                            if amount_samples_per_client[client_id] == 0:
                                break

                            elif class_amount != 0:
                                for _ in range(min_value):
                                    class_amount_available[manuel_class_id] -= 1
                                    amount_samples_per_client[client_id] -= 1
                                    client_data_indices_map[client_id].append(
                                        idx_list[manuel_class_id][class_amount_available[manuel_class_id]]
                                    )
                            else:
                                continue
                        break
                    else:
                        break
        
        clients_not_processed = [client_id for amount in amount_samples_per_client if amount!= 0]
    
    # checks 
    if verbose: 
        logger.info('\nFinished\n')
        logger.info('-'*45)
        logger.info(f'{class_amount_available}: class_amount_available')
        logger.info(f'{amount_samples_per_client}: amount_samples_per_client')
        logger.info(f'{clients_not_processed}: clients_not_processed')
        logger.info('-'*45)
    # all samples per client and per class are allocated
    print(len(targets) % num_clients)
    if len(targets) % num_clients != 0:
        remainder = len(targets) % num_clients
        assert np.sum(class_amount_available) == remainder and np.sum(amount_samples_per_client) == 0., f'class_amount_available_tmp:\t{class_amount_available}\amount_samples_per_client_tmp:\t{amount_samples_per_client}'
    else:
        assert np.sum(class_amount_available) == 0. and np.sum(amount_samples_per_client) == 0., f'class_amount_available_tmp:\t{class_amount_available}\amount_samples_per_client_tmp:\t{amount_samples_per_client}'
    # now clients are left for processing
    assert not bool(clients_not_processed)

    check_clients_map(client_data_indices_map, logger=logger)

    return client_data_indices_map


def check_clients_map(client_data_indices_map, logger: logging.getLogger=None):
    verbose = True if logger is not None else False
    tot = []
    for client_id, indices in client_data_indices_map.items():
        if verbose: logger.info(f'Client {client_id}:\t{len(indices)} samples')
        for idx in indices:
            assert idx not in tot, f'{idx} already assigned'
        tot.extend(indices)
    if verbose: 
        logger.info('-'*30)
        logger.info(f'Total:\t{len(tot)} samples')


def partition_report():
    return 



