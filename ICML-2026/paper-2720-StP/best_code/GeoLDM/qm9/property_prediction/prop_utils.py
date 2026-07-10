import os
import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt

def create_folders(args):
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_recon')
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_gen')
    except OSError:
        pass

def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def normalize_res(res, keys=[]):
    for key in keys:
        if key != 'counter':
            res[key] = res[key] / res['counter']
    del res['counter']
    return res

def plot_coords(coords_mu, path, coords_logvar=None):
    if coords_mu is None:
        return 0
    if coords_logvar is not None:
        coords_std = torch.sqrt(torch.exp(coords_logvar))
    else:
        coords_std = torch.zeros(coords_mu.size())
    coords_size = (coords_std ** 2) * 1

    plt.scatter(coords_mu[:, 0], coords_mu[:, 1], alpha=0.6, s=100)


    #plt.errorbar(coords_mu[:, 0], coords_mu[:, 1], xerr=coords_size[:, 0], yerr=coords_size[:, 1], linestyle="None", alpha=0.5)

    plt.savefig(path)
    plt.clf()


def filter_nodes(dataset, n_nodes):
    new_graphs = []
    for i in range(len(dataset.graphs)):
        if len(dataset.graphs[i].nodes) == n_nodes:
            new_graphs.append(dataset.graphs[i])
    dataset.graphs = new_graphs
    dataset.n_nodes = n_nodes
    return dataset


def adjust_learning_rate(optimizer, epoch, lr_0, factor=0.5, epochs_decay=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_0 * (factor ** (epoch // epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars