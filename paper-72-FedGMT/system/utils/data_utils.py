
import os
import torch
import ujson
from torch.utils.data.dataset import Dataset
import numpy as np

def read_data(dataset, idx,is_train = True):

    if is_train:
        train_data_dir = os.path.join('../dataset/data', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.pkl'

        train_data = torch.load(train_file)
        return train_data
    else:
        test_data_dir = os.path.join('../dataset/data', dataset, 'test/')
        test_file = test_data_dir + 'test.pkl'
        test_data = torch.load(test_file)
        return test_data



def read_client_data(dataset, idx):

    train_data = read_data(dataset, idx)
    return train_data

def read_client_json(dataset, idx):
    json_file = os.path.join('../dataset/data', dataset, 'config.json')
    with open(json_file, 'r') as f:
        config = ujson.load(f)
    trainsamples = config['Size of samples for labels in clients'][idx]
    return trainsamples

def read_total_json(dataset):
    json_file = os.path.join('../dataset/data', dataset, 'config.json')
    with open(json_file, 'r') as f:
        config = ujson.load(f)
    total = config['Total class number']
    return total

from torch.utils.data import DataLoader
def class_dataloader_for_MAS(dataset, idx,batch_size):
    client_data = read_client_data(dataset, idx)
    client_dict_per_class= read_client_json(dataset, idx)[1]
    client_data_num = []
    for i in client_dict_per_class:
        client_data_num.append(float(i[1]))

    #client_data_num = [x for x in client_data_num if x != 0]

    class_client_data = {}
    for i in range(len(client_data_num)):
        class_client_data[i] = []
    for input,label in client_data:
        class_client_data[label].append([input,label])

    dataloader = []
    for i in range(len(client_data_num)):  
        if len(class_client_data[i])!=0:
            dataloader.append(DataLoader(class_client_data[i], batch_size,pin_memory=True))

    client_data_num = [x for x in client_data_num if x != 0]
    # print(client_data_num)
    client_data_num = torch.tensor(client_data_num)
    client_data_num = client_data_num/sum(client_data_num)
    client_data_weight= -client_data_num.log()

 
    return dataloader,client_data_weight




class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def repair_cov(matrix, factor=0.001):
    matrix = torch.tensor(matrix).cuda()
    matrix = (matrix+matrix.T)/2
    w, v = torch.linalg.eig(matrix)
    w = w.real
    v= v.real
    if torch.all(w >= factor):
        m = matrix
    else:
        w[w < factor] = factor
        m = torch.matmul(torch.matmul(v, torch.diag(w)), v.T)
    
    return m


def get_head_class(dataset,head_ratio):
    total = read_total_json(dataset)
    sum = 0
    for t in total:
        sum+=t[1]
    cut = 0
    for t in total:
        cut+=t[1]
        if cut>=sum*head_ratio:
            return t[0]+1

def remix(image, label,num_class_list,device):
    r"""
    Reference:
        Chou et al. Remix: Rebalanced Mixup, ECCV 2020 workshop.
    The difference between input mixup and remix is that remix assigns lambdas of mixed labels
    according to the number of images of each class.
    Args:
        tau (float or double): a hyper-parameter
        kappa (float or double): a hyper-parameter
        See Equation (10) in original paper (https://arxiv.org/pdf/2007.03943.pdf) for more details.
    """
    assert num_class_list is not None, "num_class_list is required"
    class_num_list = []
    for i in num_class_list:
        class_num_list.append(i[1])
    num_class_list = torch.FloatTensor(class_num_list).to(device)
    alpha = 1
    remix_tau = 0.5
    remix_kappa = 3
    l = np.random.beta(alpha, alpha)
    idx = torch.randperm(image.size(0))
    image_a, image_b = image, image[idx]
    label_a, label_b = label, label[idx]
    mixed_image = l * image_a + (1 - l) * image_b
    mixed_image = mixed_image.to(device)

    #what remix does
    l_list = torch.empty(image.shape[0]).fill_(l).float().to(device)
    n_i, n_j = num_class_list[label_a], num_class_list[label_b].float()

    if l < remix_tau:
        l_list[n_i/n_j >= remix_kappa] = 0
    if 1 - l < remix_tau:
        l_list[(n_i*remix_kappa)/n_j <= 1] = 1

    label_a = label_a.to(device)
    label_b = label_b.to(device)
    # loss = l_list * criterion(output, label_a) + (1 - l_list) * criterion(output, label_b)
    # loss = loss.mean()

    return mixed_image,label_a,label_b,l_list


