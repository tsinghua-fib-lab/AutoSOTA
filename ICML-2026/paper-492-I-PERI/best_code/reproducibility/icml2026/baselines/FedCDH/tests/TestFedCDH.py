import sys
sys.path.append("")
import numpy as np
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci
from causallearn.utils.data_utils import set_random_seed, simulate_dag
from causallearn.utils.data_utils import my_simulate_general_hetero, my_simulate_linear_gaussian
from causallearn.utils.data_utils import count_skeleton_accuracy
from causallearn.utils.data_utils import get_cpdag_from_cdnod, get_dag_from_pdag, count_dag_accuracy
import time 
import argparse
np.set_printoptions(suppress=True, precision=3)

import rpy2
import rpy2.robjects as robjects
print(rpy2.__version__)
robjects.r('R.version')

# Simulation 
def test_fedCHD(i, n, K, d, s0, model):
    set_random_seed(i)
    print(f"Runing instance {i}.")

    print(n, K)
    c_indx = np.asarray(list(range(K)))
    c_indx = np.repeat(c_indx, n) 
    c_indx = np.reshape(c_indx, (n*K,1)) 

    graph_type, sem_type = 'ER', 'gauss'
    true_DAG_bin = simulate_dag(d, s0, graph_type) # ground-truth binary matrix
    
    if model=='linear':
        # Linear Gaussian model. 
        X, _ = my_simulate_linear_gaussian(true_DAG_bin, K, n*K, sem_type)
    else:
        # General functional model.
        X, _ = my_simulate_general_hetero(true_DAG_bin, K, n*K, sem_type)
    
    start = time.time()  
    print(X.shape, c_indx.shape)
    cg = cdnod(X, c_indx, K, 0.05, kci, True, 0, -1)
    end = time.time()

    est_graph = np.zeros((d,d))
    est_graph = cg.G.graph[0:d, 0:d]
    est_cpdag = get_cpdag_from_cdnod(est_graph) # est_graph[i,j]=-1 & est_graph[j,i]=1  ->  est_graph_cpdag[i,j]=1
    est_dag_from_pdag = get_dag_from_pdag(est_cpdag) # return a DAG from a PDAG in causaldag.

    # Undirected skeleton: F1, recall, precision, SHD 
    ret_skeleton = count_skeleton_accuracy(true_DAG_bin, est_cpdag) 

    # Directed graph: F1, recall, precision, SHD
    ret_diretion = count_dag_accuracy(true_DAG_bin, est_dag_from_pdag)

    result = {}
    result.update(ret_skeleton)
    result.update(ret_diretion)
    result['time'] = end-start
    print('')
    return result 

def main(args):
    res_list = []
    for i in range(args.N):
        res = test_fedCHD(i, args.n, args.K, args.d, args.d, args.model) # a dictionary
        res_val = list(res.values())
        if None in res_val:
            print("Error! None in results!")
        else:
            res_list.append(res_val)
    res_list = np.array(res_list)
    avg = np.mean(res_list, axis=0) # skeleton, orientation
    std = np.std(res_list, axis=0)

    print("########## Measurement: ", list(res.keys()))
    print("########## Average:     ", avg)
    print("########## Std:         ", std)

def main_without_args(N, d, K, n, model):
    res_list = []
    for i in range(N):
        res = test_fedCHD(i, n, K, d, d, model) # a dictionary
        res_val = list(res.values())
        if None in res_val:
            print("Error! None in results!")
        else:
            res_list.append(res_val)
    res_list = np.array(res_list)
    avg = np.mean(res_list, axis=0) # skeleton, orientation
    std = np.std(res_list, axis=0)

    print("########## Measurement: ", list(res.keys()))
    print("########## Average:     ", avg)
    print("########## Std:         ", std)

if __name__=='__main__':    
    parser = argparse.ArgumentParser(description='Federated Causal Discovery')    # Mode
    parser.add_argument('--N', default=10, type=int,
                        help='number of instances')
    parser.add_argument('--d', default=6, type=int, 
                        help='number of variables')
    parser.add_argument('--K', default=10, type=int, 
                        help='number of clients')
    parser.add_argument('--n', default=100, type=int, 
                        help='number of samples in one client')
    parser.add_argument('--model', default='linear', type=str, 
                        help='generated functional model, linear or general')
    args = parser.parse_args()
    main(args)

    # N = 10
    # d = 6
    # K = 10
    # n = 100
    # model = 'linear'
    # main_without_args(N, d, K, n, model)