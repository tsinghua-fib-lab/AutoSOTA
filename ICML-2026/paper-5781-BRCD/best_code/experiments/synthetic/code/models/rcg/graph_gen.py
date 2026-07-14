import os
import copy
import random
import pickle

import numpy as np
import networkx as nx

import pyAgrum as gum

from utils import base_utils as bu
from config import GraphGenConf, load_config, dump_config

DATA_DIR = 'individual-graphs'
DEFAULT_CONFIG = 'graph_gen.yaml'

def draw_and_save(bn, target, samples, nodes):
    
    generator = gum.BNDatabaseGenerator(bn)
    generator.drawSamples(samples)

    # Don't specify the order of nodes. This gives us natural permutation of nodes
    # var_order = [bu.get_node_name(node) for node in range(nodes)]
    # generator.setVarOrder(var_order)
    generator.toCSV(target)

def save_nx_graph(G, name):
    A = nx.nx_agraph.to_agraph(G)
    A.layout('dot')
    A.draw(name)

def create_CPT(bn, node, method='random'):
    if method=='random':
        bn.generateCPT(node)
        return

    parent_names = [bn.variable(x).name() for x in bn.parents(node)]
    node_states = bn.variable(node).domainSize()
    if len(parent_names) == 0:
        # Both methods intervene the same way on a no-parent node
        assert method in ['Dirichlet', 'Meek']
        alpha = np.ones(node_states)
        new_probs = np.random.dirichlet(alpha).tolist()
        bn.cpt(node)[:] = new_probs
        return

    parent_states = [range(bn.variable(p).domainSize()) for p in parent_names]
    if method == 'Dirichlet':
        alpha = np.ones(node_states)
        # Iterate over every row of CPT
        for state_combination in np.array(np.meshgrid(*parent_states)).T.reshape(-1, len(parent_names)):
            # Map the parents' states to their names
            parent_state_dict = {parent_names[i]: int(state_combination[i]) for i in range(len(parent_names))}
            new_probs = np.random.dirichlet(alpha).tolist()
            bn.cpt(node)[parent_state_dict] = new_probs
    elif method == 'Meek':
        base = 1. / np.arange(1, node_states + 1)
        base /= np.sum(base)
        alpha = 10 * base
        for counter, state_combination in enumerate(np.array(np.meshgrid(*parent_states)).T.reshape(-1, len(parent_names))):
            parent_state_dict = {parent_names[i]: int(state_combination[i]) for i in range(len(parent_names))}
            alpha_shifted = np.roll(alpha, counter)
            new_probs = np.random.dirichlet(alpha_shifted).tolist()
            bn.cpt(node)[parent_state_dict] = new_probs

# def randomBNwithSpecificStates(nodes,arcs, states, p):
#     g=gum.BNGenerator()
#     tmp=g.generate(nodes,arcs,2)
#     bn=gum.BayesNet()
#     # Nodes
#     v=list(tmp.names())
#     random.shuffle(v)

#     _map = {}

#     # h=len(v)//2
#     for i, name in enumerate(v):
#         _map[name] = bu.get_node_name(i)
#         #np.random.seed(fixseed)
#         s = np.random.choice(a=np.array(states), size=1, p=p)
#         state_num = s[0]
#         bn.add(_map[name], int(state_num))
#         id = bn.ids([_map[name]])

#     # arcs
#     bn.beginTopologyTransformation()
#     for a,b in tmp.arcs():
#         bn.addArc(_map[tmp.variable(a).name()], _map[tmp.variable(b).name()])
#     bn.endTopologyTransformation()
#     bn.generateCPTs()
#     # output_dict = {value: key for key, value in table.items()}
#     return bn, list(_map.values())

# n is the number of nodes
def generate_random_dag(n):
    # p is the probability of an edge between any two nodes
    # p = n ^ -(k.log(n)), where k < 0 is the rate of decay
    p = n ** (-0.1 * (np.log(n)))
    # print(f'{p=}')
    # Create an upper triangular matrix with random values
    adj_matrix = np.triu(np.random.rand(n, n) < p, 1)
    # Create a DAG from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return G.edges()

def add_backdoors(bn, an_node, n):
    bn.add("X_star")
    bn.addArc("X_star", an_node)
    ls_nodes_to_add = random.sample(list(bn.names()), int(n/2))
    for nodename in ls_nodes_to_add:
        try:
            bn.addArc("X_star", nodename)
        except:
            # if there is a cycle, we move on
            continue
    ls_nodes_to_addch = random.sample(list(bn.names()), int(n/2))
    for nodename in ls_nodes_to_addch:
        try:
            bn.addArc(an_node, nodename)
        except:
            # if there is a cycle, we move on
            continue
    bn.generateCPT("X_star")
    return bn

def get_random_dag(cfg: GraphGenConf, an_node, add_backdoor):
    _n = cfg.nodes - 1 if add_backdoor else cfg.nodes

    names = [bu.get_node_name(x) for x in range(_n)]
    # bn = gum.fastBN("X0->X1->X2->X3->X4; X0->X3")
    arc_ratio = 1.4 if _n <= 4 else 2
    bn = gum.randomBN(n=_n, ratio_arc=arc_ratio,
                      domain_size=cfg.states, names=names)

    # n = cfg.nodes
    # list_of_states = list(range(2, 11))
    # states = np.random.choice(list_of_states, size=n, replace=True)
    # p_states = [1/len(states)] * n
    # bn, names = randomBNwithSpecificStates(n, int(n * 1.2), states, p_states)

    # edges = generate_random_dag(cfg.nodes)
    # bn = gum.BayesNet('BN')
    # for node in range(cfg.nodes):
    #     bn.add(gum.RangeVariable(bu.get_node_name(node), str(node), 0, cfg.states - 1))
    # for e in edges:
    #     bn.addArc(bu.get_node_name(e[0]), bu.get_node_name(e[1]))
    # bn.generateCPTs()

    if add_backdoor:
        bn = add_backdoors(bn, an_node, cfg.nodes)
        names.append("X_star")

    # G = dag.to_nx()
    # G = nx.relabel_nodes(G, {i: bu.get_node_name(i) for i in G.nodes})
    # bn = gum.BayesNet('BN')
    # for node in G.nodes:
    #     bn.add(gum.RangeVariable(node, node, 0, cfg.states - 1))
    # for e in G.edges():
    #     bn.addArc(e[0], e[1])

    G = nx.DiGraph()
    for i in names:
        G.add_node(i)
    for e in bn.arcs():
        G.add_edge(bn.variable(e[0]).name(), bn.variable(e[1]).name())
    bn.generateCPTs()
    # for n in bn.names():
    #     create_CPT(bn, n, method='Dirichlet')
    return bn, G

def inject_failure(bn, an_node):
    # Change the distribution of the anomalous node
    create_CPT(bn, an_node, method='random')

def generate_graph(cfg: GraphGenConf, src=None):
    if src is None:
        src_dir = os.path.join(os.path.dirname(__file__), DATA_DIR, bu.readable_time())
    else:
        src_dir = src
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    dump_config(cfg, src_dir)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    gum.initRandom(cfg.seed)

    add_backdoors = False
    # if np.random.uniform(0, 1) < 0.5:
    #     add_backdoors = True
    _n = cfg.nodes - 1 if add_backdoors else cfg.nodes

    an_node = bu.get_node_name(np.random.choice(_n))
    if cfg.verbose:
        print(f"Randomly assigned anomalous node {an_node}")

    # Create a random DAG
    normal_bn, nx_graph = get_random_dag(cfg, an_node, add_backdoors)
    if cfg.verbose:
        nx_graph.nodes[an_node]['style'] = 'filled'
        nx_graph.nodes[an_node]['fillcolor'] = 'red'
        if cfg.verbose:
            save_nx_graph(nx_graph, f'{src_dir}/{bu.GROUND_TRUTH_PDF}')
    draw_and_save(normal_bn, f'{src_dir}/{bu.NORMAL_DATA}', cfg.samples, cfg.nodes)
    anomalous_bn = copy.deepcopy(normal_bn)
    inject_failure(anomalous_bn, an_node)
    draw_and_save(anomalous_bn, f'{src_dir}/{bu.ANOMALOUS_DATA}', cfg.samples, cfg.nodes)
    if cfg.verbose:
        print(f"Data is saved at {src_dir}")
        gum.saveBN(normal_bn, f'{src_dir}/{bu.NORMAL_BN}')
        gum.saveBN(anomalous_bn, f'{src_dir}/{bu.ANOMALOUS_BN}')

    with open(f'{src_dir}/{bu.GROUND_TRUTH_NX_GRAPH}', 'wb') as f:
        pickle.dump(nx_graph, f)
    with open(f'{src_dir}/{bu.GRAPH_GEN_INFO}', 'wb') as f:
        pickle.dump({bu.ANOMALOUS_NODE: an_node}, f)
    return src_dir, an_node


if __name__ == '__main__':
    cfg: GraphGenConf = load_config(DEFAULT_CONFIG, GraphGenConf)
    generate_graph(cfg)
