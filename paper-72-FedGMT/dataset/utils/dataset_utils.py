import os
import ujson
import numpy as np
import torch


def check(config_path, train_path, test_path, num_clients, num_classes, niid, imb,seed,shard_per_user,args):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if args.LDA and not args.shard:
            if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid_alpha'] == niid and \
                config['shard_per_user'] == None and \
                config['imb_factor'] == imb and \
                config['seed'] == seed  :
                print("\nDataset already generated.\n")
                return True
        elif not args.LDA and args.shard:
          if  config['num_classes'] == num_classes and \
            config['imb_factor'] == imb and \
            config['seed'] == seed and \
            config['non_iid_alpha'] == None and \
            config['shard_per_user'] == shard_per_user :
            print("\nDataset already generated.\n")
            return True
        else:
            print("check LDA and shard input!!!!")
            exit()

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1

def show_clients_data_distribution(dataset_label, clients_indices: list, num_classes):
    dict_per_client = []
    total_class= [0 for _ in range(num_classes)]
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = int(dataset_label[idx])
            nums_data[label] += 1
        new_num_data = []
        total = sum(nums_data)
        for i in range(num_classes):
            new_num_data.append((i,nums_data[i]))
            total_class[i]+=nums_data[i]
        dict_per_client.append((total,new_num_data))

        print(f'client:{client}:  {total} {new_num_data}')
    print(f"total:{total_class}")
    return dict_per_client,total_class



def save_file(config_path, train_path, test_path,train_index, train_data, test_data, num_clients, num_classes, niid, imb,seed,statistic,total,shard_per_user,data_partition):
    new_total = []
    for i,n in enumerate(total):
        new_total.append((i,n))
    # sorted_total = sorted(new_total, key=lambda x: x[1],reverse=True)

    if data_partition == 'LDA':
        config = {
            
            'num_clients': num_clients, 
            'num_classes': num_classes, 
            'data_partition':data_partition,
            'non_iid_alpha': niid, 
            'imb_factor':imb, 
            'shard_per_user':None,
            'seed': seed,
            'Total class number':new_total,
            'Size of samples for labels in clients': statistic, 

        }
    elif data_partition == 'shard':
        config = {
            'num_clients': num_clients, 
            'num_classes': num_classes, 
            'data_partition':data_partition,
            'non_iid_alpha': None, 
            'shard_per_user':shard_per_user,
            'imb_factor':imb, 
            'seed': seed,
            'Total class number':new_total,
            'Size of samples for labels in clients': statistic, 

        }

    # gc.collect()
    print("Saving to disk.\n")

    for client in range(num_clients):
        traindata = []
        for i in train_index[client]:
            traindata.append(train_data[i])
        
        torch.save(traindata,train_path + str(client) + '.pkl')

    torch.save(test_data,test_path+ 'test.pkl')

    with open(config_path, 'w') as f:
        ujson.dump(config, f,indent=1)

    print("Finish generating dataset.\n")


import random
def sharding_partition(all_targets, n_clients, shard_per_user, rand_set_all=[]):
    net_dataidx_map = {i: [] for i in range(n_clients)}
    idxs_dict = {}

    for i in range(len(all_targets)):
        label = torch.tensor(all_targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(all_targets))
    shard_per_class = int(shard_per_user * n_clients / num_classes)

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((n_clients, -1))

    # divide and assign
    for i in range(n_clients):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        net_dataidx_map[i] = np.concatenate(rand_set).astype("int")

    # test = []
    # for key, value in net_dataidx_map.items():
    #     x = np.unique(torch.tensor(all_targets)[value])
    #     assert (len(x)) <= shard_per_user
    #     test.append(value)
    # test = np.concatenate(test)
    # assert len(test) == len(all_targets)
    # assert len(set(list(test))) == len(all_targets)

    return net_dataidx_map, rand_set_all



