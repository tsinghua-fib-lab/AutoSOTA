import concurrent.futures
import os
import subprocess as sp
from datetime import datetime
from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
import secrets
from string import ascii_uppercase, digits

from evaluation.mmd import compute_mmd, gaussian, gaussian_emd, compute_nspdk_mmd, gaussian_tv
from utils.graph_utils import adjs_to_graphs
from utils.print_colors import red, blue

try:
    import graph_tool.all as gt
    from scipy.stats import chi2
except ModuleNotFoundError:
    pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


PRINT_TIME = False 
# -------- the relative path to the orca dir --------
ORCA_DIR = 'evaluation/orca'  


##### code adapted from https://github.com/KarolisMart/SPECTRE/blob/main/util/eval_helper.py
def degree_worker(G):
    return np.array(nx.degree_histogram(G))

# -------- Compute degree MMD --------
def degree_stats(graph_ref_list, graph_pred_list, KERNEL='tv', is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
            G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    
    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(
                    nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    if KERNEL=='emd':
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elif KERNEL=='tv':
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    else:
        raise NotImplementedError(f'Kernel {KERNEL} not implemented.')

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())  
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals+1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


# -------- Compute spectral MMD --------
def spectral_stats(graph_ref_list, graph_pred_list, KERNEL='tv', is_parallel=True, n_eigvals=-1):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
            G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty, [n_eigvals for i in graph_ref_list]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    if KERNEL=='emd':
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elif KERNEL=='tv':
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    else:
        raise NotImplementedError(f'Kernel {KERNEL} not implemented.')
    
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


# -------- Compute clustering coefficients MMD --------
def clustering_stats(graph_ref_list, graph_pred_list, KERNEL='tv', bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
            G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                    nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if KERNEL=='emd':
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    elif KERNEL=='tv':
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)
    else:
        raise NotImplementedError(f'Kernel {KERNEL} not implemented.')

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# -------- maps motif/orbit name string to its corresponding list of indices from orca output --------
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_file_path = os.path.join(ORCA_DIR, f'tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt')
    f = open(tmp_file_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_file_path, 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list, KERNEL='tv'):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
            G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    if KERNEL=='emd':
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)
    elif KERNEL=='tv':
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, sigma=30.0)
    else:
        raise NotImplementedError(f'Kernel {KERNEL} not implemented.')
    return mmd_dist


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """
    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps): 
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    
    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
            return False
    
    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra)**2 / (est_p_intra * (1-est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter)**2 / (est_p_inter * (1-est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9 # p value < 10 %
    else:
        return p


def is_sbm_graph_verbose(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """
    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps): 
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    
    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        node_problem = (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0
        block_problem = n_blocks > 5 or n_blocks < 2
        if node_problem or block_problem:
            verbose = f'Node cnt={node_counts}  ' * bool(node_problem) + f'Block cnt={n_blocks}  ' * block_problem
            print(verbose)
            return False
    
    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra)**2 / (est_p_intra * (1-est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter)**2 / (est_p_inter * (1-est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        if p <= 0.9:
            print(f'{p:.2f}')
        return p > 0.9 # p value < 10 %
    else:
        return p


def eval_acc_sbm_graph(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000, is_parallel=True):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
                for prob in executor.map(is_sbm_graph, [gg for gg in G_list], 
                                            [p_intra for i in range(len(G_list))], [p_inter for i in range(len(G_list))],
                                            [strict for i in range(len(G_list))], [refinement_steps for i in range(len(G_list))]):
                    count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict, refinement_steps=refinement_steps)
    return count / float(len(G_list))


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]

def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique


def eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func=(lambda x: True)):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(len(fake_graphs))  
    # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (float(len(fake_graphs)) - count_non_unique - count_isomorphic) / float(len(fake_graphs))  
    # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(len(fake_graphs))  
    # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


def eval_planar(graph_ref_list, graph_pred_list):
    validity_func = lambda g: is_planar_graph(g)
    uniq, uniq_non_iso, uniq_non_iso_val = eval_fraction_unique_non_isomorphic_valid(graph_pred_list, graph_ref_list, validity_func)
    return uniq, uniq_non_iso, uniq_non_iso_val


def eval_sbm(graph_ref_list, graph_pred_list):
    validity_func = lambda g: is_sbm_graph(g, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=100)
    uniq, uniq_non_iso, uniq_non_iso_val = eval_fraction_unique_non_isomorphic_valid(graph_pred_list, graph_ref_list, validity_func)
    return uniq, uniq_non_iso, uniq_non_iso_val

def eval_sbm_verbose(graph_ref_list, graph_pred_list):
    validity_func = lambda g: is_sbm_graph_verbose(g, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=100)
    uniq, uniq_non_iso, uniq_non_iso_val = eval_fraction_unique_non_isomorphic_valid(graph_pred_list, graph_ref_list, validity_func)
    return uniq, uniq_non_iso, uniq_non_iso_val


##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list, KERNEL=None):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=20)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def connected_stats(graph_ref_list, graph_pred_list):
    import copy
    G_list = [copy.deepcopy(gg) for gg in graph_pred_list]
    count = 0
    for graph in G_list:
        gg = nx.Graph(graph)
        gg.remove_nodes_from(list(nx.isolates(gg)))
        if nx.is_empty(gg):
            continue
        if nx.is_connected(gg):
            count += 1

    return count / float(len(G_list))


METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_all,
    'spectral': spectral_stats,
    'nspdk': nspdk_stats,
    'eval_sbm': eval_sbm,
    'eval_planar': eval_planar, 
    'connected': connected_stats,
    'eval_sbm_verbose': eval_sbm_verbose
}


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    graph_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, graph_pred_list, methods=methods)
    return results


# -------- Evaluate generated generic graphs --------
def eval_graph_list(graph_ref_list, graph_pred_list, methods=None, kernels=None):
    if methods is None:
        methods = ['degree', 'cluster', 'orbit', 'spectral']
    results = {}
    NUM=4
    for method in methods:
        if 'eval' in method: # sbm, planar
            uniq, uniq_non_iso, uniq_non_iso_val = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list)
            if uniq < 0.999:
                results['unique'] = round(uniq, NUM)
                print(red(f'{"unique":10s}') + ' : ' + blue(f'{results["unique"]:.4f}'))
            if uniq_non_iso < 0.999:
                results['uniq+nov'] = round(uniq_non_iso, NUM)
                print(red(f'{"uniq+nov":10s}') + ' : ' + blue(f'{results["uniq+nov"]:.4f}'))
            results['V.U.N.'] = round(uniq_non_iso_val, NUM)
            print(red(f'{"V.U.N.":10s}') + ' : ' + blue(f'{results["V.U.N."]:.4f}'))
        else:
            if method in ['degree', 'cluster', 'orbit', 'spectral']: # MMD requires kernel
                result = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, kernels[method])
            else: 
                result = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list)
            results[method] = round(result, NUM)
            print(red(f'{method:10s}') + ' : ' + blue(f'{results[method]:.4f}'))
    return results
