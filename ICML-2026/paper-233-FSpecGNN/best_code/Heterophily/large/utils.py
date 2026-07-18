import torch
import math
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits_eachclass(data, num_classes, train_rate, val_rate, seed=42):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)

    val_idx=[]

    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        total_class_num = class_idx.shape[0]
        print(f'class_id:{c} size:{total_class_num}')
        train_idx.extend(rnd_state.choice(class_idx, int(round(train_rate * total_class_num )), replace=False))
        rest_index = [i for i in class_idx if i not in train_idx]
        val_idx.extend(rnd_state.choice(rest_index, int(round(val_rate * total_class_num )), replace=False))
        # if len(class_idx)<percls_trn:
        #     train_idx.extend(class_idx)
        # else:
        #     train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))

    # rest_index = [i for i in index if i not in train_idx]
    # val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    remove_train_idx = [i for i in index if i not in train_idx]
    test_idx = [i for i in remove_train_idx if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def fixed_splits(data, num_classes, percls_trn, val_lb, name, seed=42):
    if name in ["Chameleon","Squirrel"]:
        seed = 1941488137
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    return data

