import numpy as np
import copy
import networkx as nx
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
import pyAgrum as gum
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GraphNode import GraphNode# Visualization of the DAG
import matplotlib.pyplot as plt
import os
from graphical_models.classes.dags.pdag import PDAG
from graphical_models.classes.dags.dag import DAG
from mcs_enum import *
import cliquepicking as cp
import random
import pickle
import os
import shutil
import pandas as pd
from typing import Any, Dict, Hashable, Iterable, Tuple, List 

def pdag_to_causallearn_generalgraph(pdag: Any,
                                     node_name_fn=lambda x: str(x)) -> GeneralGraph:
    """
    Convert a PDAG from uhlerlab/graphical_models to a causal-learn GeneralGraph.

    Expected PDAG attributes:
      - pdag.nodes : iterable of node labels
      - pdag.arcs  : iterable of directed edges as (u, v)
      - pdag.edges : iterable of undirected edges as (u, v) OR an unordered 2-set {u, v}
    Fallback attribute names `directed` / `undirected` are also tried.
    """
    # --- Nodes
    try:
        raw_nodes: Iterable[Hashable] = pdag.nodes
    except AttributeError:
        raise AttributeError("Expected attribute 'nodes' on PDAG-like object.")

    node_map: Dict[Hashable, GraphNode] = {
        v: GraphNode(node_name_fn(v)) for v in raw_nodes
    }

    # IMPORTANT: GeneralGraph needs nodes in the constructor
    g = GeneralGraph(list(node_map.values()))

    # --- Edge containers with fallbacks
    arcs: Iterable[Tuple[Hashable, Hashable]] = getattr(pdag, "arcs", None)
    if arcs is None:
        arcs = getattr(pdag, "directed", [])

    undirs_any: Iterable = getattr(pdag, "edges", None)
    if undirs_any is None:
        undirs_any = getattr(pdag, "undirected", [])

    # --- Add directed edges u -> v  (TAIL -> ARROW)
    for u, v in arcs:
        head = node_map[v]
        tail = node_map[u]
        # Avoid double-adding if it already exists
        try:
            existing = g.get_edge(tail, head)
        except AttributeError:
            existing = None
        if existing is None:
            g.add_edge(Edge(tail, head, Endpoint.TAIL, Endpoint.ARROW))

    # --- Add undirected edges u -- v  (TAIL -- TAIL)
    for e in undirs_any:
        if isinstance(e, tuple) and len(e) == 2:
            u, v = e
        else:
            u, v = tuple(e)  # e.g., frozenset({u, v})

        a, b = node_map[u], node_map[v]

        # If there is already some edge (possibly directed), replace with undirected.
        # Different causal-learn versions expose different helpers; try a few.
        existing = None
        try:
            existing = g.get_edge(a, b)
        except AttributeError:
            pass
        if existing is None:
            try:
                existing = g.get_edge(b, a)
            except AttributeError:
                pass

        if existing is not None:
            try:
                g.remove_edge(existing)
            except Exception:
                # Fall back: some versions support remove_edge(u, v)
                try:
                    g.remove_edge(a, b)
                except Exception:
                    try:
                        g.remove_edge(b, a)
                    except Exception:
                        pass  # If we can't remove, we'll just try to add below.

        # Finally add the undirected edge (dedup if already adjacent)
        try:
            is_adj = g.is_adjacent_to(a, b)
        except AttributeError:
            # If not available, assume not adjacent
            is_adj = False
        if not is_adj:
            g.add_edge(Edge(a, b, Endpoint.TAIL, Endpoint.TAIL))

    return g

def export_nx_dags_to_pdf(nx_dags):
    """
    Exports each nx.DiGraph in the list to a PDF file.
    Ensures that files are not overwritten by using a counter.
    """
    # Create the figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')

    for index, nx_dag in enumerate(nx_dags):
        # Create a unique filename
        file_name = f"figures/dag_{index}.pdf"  # Save in the figures folder
        
        # Check if the file already exists
        if os.path.isfile(file_name):
            # print(f"File {file_name} already exists. Skipping export for index {index}.")
            continue  # Skip to the next index if the file exists
        
        # Draw the graph
        plt.figure()
        pos = nx.spring_layout(nx_dag)  # positions for all nodes
        nx.draw(nx_dag, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black')
        plt.title(f"Directed Acyclic Graph {index}")
        
        # Save the figure as a PDF
        plt.savefig(file_name)
        plt.close()  # Close the figure to free memory


def cartesianProduct(set_a, set_b): 
    result =[] 
    for i in range(0, len(set_a)): 
        for j in range(0, len(set_b)): 
  
            # for handling case having cartesian 
            # prodct first time of two sets 
            if type(set_a[i]) != list:          
                set_a[i] = [set_a[i]] 
                  
            # coping all the members 
            # of set_a to temp 
            temp = [num for num in set_a[i]] 
              
            # add member of set_b to  
            # temp to have cartesian product      
            temp.append(set_b[j])              
            result.append(temp)   
              
    return result 

# Function to do a cartesian  
# product of N sets  
def Cartesian(list_a, n): 
    # result of cartesian product 
    # of all the sets taken two at a time 
    if len(list_a)==0:
        return []
    else:
        temp = list_a[0] 

        # do product of N sets  
        for i in range(1, n): 
            temp = cartesianProduct(temp, list_a[i]) 
    return temp

def createCPT(bn,var_name,no_of_states_dict,option, specified_nodes, intervene=False, vec=None):

    if option=='random':
        bn.generateCPT(var_name)

    elif option=='logistic_binary':
        #print(var_name)
        parent_names=list(bn.cpt(var_name).names)
        for j in parent_names:
            assert no_of_states_dict[j]==2, "logistic_binary can only be used with binary variables"
        parent_names.remove(var_name)
        #print(parent_names)
        assert(len(parent_names)+1== len(vec)), "Length of the vector of coefficients mis matched with the number of parents"
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        #print(parent_states)
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=[vec[k]*int(j[k]) for k in range(len(parent_names))]
            logit=np.sum(np.array(my_dist))+vec[-1]
            #print(logit)
            bn.cpt(var_name)[my_dict] = np.array([expit(logit),1-expit(logit)])

    elif option=='deterministic':
        alpha=np.zeros((no_of_states_dict[var_name],))
        #print(no_of_states_dict[var_name])
        alpha[1]=1
        #print(alpha)

        parent_names=list(bn.cpt(var_name).names)
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            alpha_shifted=np.roll(alpha,counter)
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=alpha_shifted
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1

    elif option=='almost-deterministic':
        alpha=np.zeros((no_of_states_dict[var_name],))
        #print(no_of_states_dict[var_name])
        alpha[1]=1
        #print(alpha)

        parent_names=list(bn.cpt(var_name).names)
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            alpha_shifted=np.roll(alpha,counter)
            alpha_shifted = 0.99*alpha_shifted+0.01*np.random.rand(no_of_states_dict[var_name])
            alpha_shifted = alpha_shifted/np.sum(alpha_shifted)
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=alpha_shifted            
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    
    elif option=='Dirichlet':
        alpha=np.ones((no_of_states_dict[var_name],))
        parent_names=list(bn.cpt(var_name).names)
        parent_names.remove(var_name)

        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=list(np.random.dirichlet(tuple(alpha), 1)[0])
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    elif option=='Meek':
        base=1./np.arange(1,no_of_states_dict[var_name]+1)
        base=base/np.sum(base)
        # equivalent sample size = sum of ai's in Dirichlet
        alpha=10*base
        parent_names=bn.cpt(var_name).names
        parent_names.remove(var_name)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))

        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            alpha_shifted=np.roll(alpha,counter)
    
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}

            my_dist=list(np.random.dirichlet(tuple(alpha_shifted), 1)[0])

            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1

def refillCPT_Dirichlet(bn,node_names, specified_nodes, intervene, option='Dirichlet'):
    no_of_states_dict = {}
    for i in node_names:
        no_of_states_dict[i] = bn.variable(i).domainSize()
    for count, var_name in enumerate(node_names):
        if var_name not in specified_nodes and intervene == True:
            continue
        else:
            createCPT(bn,var_name,no_of_states_dict,option, specified_nodes, intervene=intervene)



def get_joint_prob(BN):
    joint_prob = 1
    for var_name in list(BN.names()):
        joint_prob = joint_prob * BN.cpt(var_name)
    return joint_prob



def nodeset_to_nameset(nodeset: set):
    nameset = set()
    for v in list(nodeset):
        nameset = nameset.union( set([str(v)]) )
    return nameset

# Calculate the conditional probability from a given joint probability
def get_conditional_prob(joint_prob, v_name:set, Condition_v_names:set):
    copy_joint = copy.copy(joint_prob)
    # No condition, return marg prob
    if len(Condition_v_names)==0:
        return copy_joint.margSumOut(list(set(joint_prob.names)-v_name))
    else:
        marg_nominator_names = set(joint_prob.names)-v_name-Condition_v_names
        marg_denominator_names = set(joint_prob.names) - Condition_v_names
        P_denominator = copy_joint.margSumOut(list(marg_denominator_names))
        # No need to marg out
        if marg_nominator_names == set():
            P_nominator = joint_prob
        # Marg out rest
        else:
            P_nominator = copy_joint.margSumOut(list(marg_nominator_names))
        return P_nominator/P_denominator
    

def convert_to_nx_pdag(cpdag):
    G = nx.DiGraph()
    for edge in cpdag.get_graph_edges():
        e1 = edge.get_endpoint1()
        e2 = edge.get_endpoint2()
        n1 = cpdag.node_map[edge.get_node1()]
        n2 = cpdag.node_map[edge.get_node2()]
        if e1 == Endpoint.TAIL and e2 ==  Endpoint.TAIL:
            G.add_edge(n1, n2)
            G.add_edge(n2, n1)
        if e1 ==  Endpoint.TAIL and e2 ==  Endpoint.ARROW:
            G.add_edge(n1, n2)
        if e1 ==  Endpoint.ARROW and e2 ==  Endpoint.TAIL:
            G.add_edge(n2, n1)
    return G




def convert_nx_graph_to_cg(nx_dag):
    nodes = [GraphNode(node) for node in nx_dag.nodes()]
    G = GeneralGraph(nodes)
    for edge in nx_dag.edges():
        node1, node2 = edge
        G.add_edge(Edge(nodes[node1], nodes[node2], Endpoint.TAIL, Endpoint.ARROW))
    return G


def addArcs(nx_dag, bn):
    for u, v in nx_dag.edges():
        bn.addArc(str(u), str(v))  # Add directed edges to the Bayesian network

def convert_nx_to_bayes_net(nx_dag, colnames):
    bn = gum.BayesNet()  # Create a new Bayesian network
    for node in nx_dag.nodes():
        bn.add(str(node))  # Add nodes to the Bayesian network
    for u, v in nx_dag.edges():
        bn.addArc(str(u), str(v))  # Add directed edges to the Bayesian network
    return bn

def convert_bn_to_nx(obs_bn):
    G = nx.DiGraph()
    id_to_name = {obs_bn.idFromName(str(i)):i for i in obs_bn.names()}

    # Add nodes
    for node in obs_bn.names():
        if node == 'F':
            G.add_node(node)
        else:
            G.add_node(int(node))
    
    # Add edges
    for i in range(obs_bn.size()):
        iname = id_to_name[i]
        for j in obs_bn.parents(str(iname)):
            if iname == 'F':
                G.add_edge(int(id_to_name[j]), iname)
            elif id_to_name[j] == 'F':
                G.add_edge(id_to_name[j], int(iname))
            else:
                G.add_edge(int(id_to_name[j]), int(iname))
    
    return G


def get_intervened_BN(BN, int_type, name_list, target_index):
    BN_inter_V = copy.copy(BN)
    if int_type =='hard':
        for v_name in name_list:
            p_v = BN_inter_V.cpt(v_name)
            a = np.zeros(p_v.shape)
            a[..., :] = [0, 1]
            BN_inter_V.cpt(v_name)[:] = a
    elif int_type == 'soft':
        target = name_list[target_index]
        BN_inter_V.generateCPT(target)
    else:
        NotImplementedError
    return BN_inter_V


def are_graphs_equal(graph1, graph2):
    return (graph1.nodes() == graph2.nodes()) and (graph1.edges() == graph2.edges())


def convert_to_nx_pdag(cpdag):
    G = nx.DiGraph()
    for edge in cpdag.get_graph_edges():
        e1 = edge.get_endpoint1()
        e2 = edge.get_endpoint2()
        n1 = cpdag.node_map[edge.get_node1()]
        n2 = cpdag.node_map[edge.get_node2()]
        if e1 == Endpoint.TAIL and e2 ==  Endpoint.TAIL:
            G.add_edge(n1, n2)
            G.add_edge(n2, n1)
        if e1 ==  Endpoint.TAIL and e2 ==  Endpoint.ARROW:
            G.add_edge(n1, n2)
        if e1 ==  Endpoint.ARROW and e2 ==  Endpoint.TAIL:
            G.add_edge(n2, n1)
    return G

def visualize_graph(v_G):
    """
    Visualizes the directed graph v_G using matplotlib and saves it to a PDF file.
    """
    # Create the figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')

    plt.figure(figsize=(10, 8))  # Set the figure size
    pos = nx.spring_layout(v_G)  # positions for all nodes
    nx.draw(v_G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black', arrows=True)
    plt.title("Visualization of Directed Graph")
    
    # Save the figure as a PDF
    plt.savefig('figures/true_cpdag.pdf')
    plt.close()  # Close the figure to free memory



def get_cpdag_from_cd_object(dag):
    nodenames = list(dag.nodes)
    node_map = {x: GraphNode(x) for x in nodenames}
    cl_dag: Dag = Dag(list(node_map.values()))
    for u, v in dag.arcs:
        cl_dag.add_edge(Edge(node_map[u], node_map[v], Endpoint.TAIL, Endpoint.ARROW))
    G = dag2cpdag(cl_dag)
    arcs = []
    edges = []
    for edge in G.get_graph_edges():
        e1 = edge.get_endpoint1()
        e2 = edge.get_endpoint2()
        n1 = G.node_map[edge.get_node1()]
        n2 = G.node_map[edge.get_node2()]
        if e1 == Endpoint.TAIL and e2 ==  Endpoint.TAIL:
            edges.append((n1, n2))
        if e1 ==  Endpoint.TAIL and e2 ==  Endpoint.ARROW:
            arcs.append((n1, n2))
        if e1 ==  Endpoint.ARROW and e2 ==  Endpoint.TAIL:
            arcs.append((n2, n1))
    cpdag_gm = PDAG(nodes=list(dag.nodes), edges=edges, arcs=arcs)
    return cpdag_gm 



def get_cpdag_from_nx(dag):
    nodenames = list(dag.nodes)
    node_map = {x: GraphNode(x) for x in nodenames}
    cl_dag: Dag = Dag(list(node_map.values()))
    for u, v in dag.edges:
        cl_dag.add_edge(Edge(node_map[u], node_map[v], Endpoint.TAIL, Endpoint.ARROW))
    G = dag2cpdag(cl_dag)
    arcs = []
    edges = []
    for edge in G.get_graph_edges():
        e1 = edge.get_endpoint1()
        e2 = edge.get_endpoint2()
        n1 = nodenames[G.node_map[edge.get_node1()]]
        n2 = nodenames[G.node_map[edge.get_node2()]]
        if e1 == Endpoint.TAIL and e2 ==  Endpoint.TAIL:
            edges.append((n1, n2))
        if e1 ==  Endpoint.TAIL and e2 ==  Endpoint.ARROW:
            arcs.append((n1, n2))
        if e1 ==  Endpoint.ARROW and e2 ==  Endpoint.TAIL:
            arcs.append((n2, n1))
    cpdag_gm = PDAG(nodes=list(dag.nodes), edges=edges, arcs=arcs)
    return cpdag_gm

def convert_cl_cpdag_to_gm_cpdag(G):
    arcs = []
    edges = []
    nodes = set()
    for edge in G.get_graph_edges():
        e1 = edge.get_endpoint1()
        e2 = edge.get_endpoint2()
        n1 = G.node_map[edge.get_node1()]
        n2 = G.node_map[edge.get_node2()]
        nodes.add(n1)
        nodes.add(n2)
        if e1 == Endpoint.TAIL and e2 ==  Endpoint.TAIL:
            edges.append((n1, n2))
        if e1 ==  Endpoint.TAIL and e2 ==  Endpoint.ARROW:
            arcs.append((n1, n2))
        if e1 ==  Endpoint.ARROW and e2 ==  Endpoint.TAIL:
            arcs.append((n2, n1))
    cpdag_gm = PDAG(nodes= nodes, edges=edges, arcs=arcs)
    return cpdag_gm


def get_cpdag(bn):
    """
        Get the CPDAG from a Bayesian network.
    """
    id_to_name = {bn.idFromName(i):i for i in bn.names()}
    #print("id_from_name")
    #print(id_to_name)
    arcs = set()
    for uid, vid in bn.arcs():
       u_name = id_to_name[uid]
       v_name = id_to_name[vid]
       arcs.add((u_name, v_name))
    g = DAG(arcs=arcs)
    return g.cpdag()


def check_pdf_exists(file_path):
    if os.path.isfile(file_path):
        #print(f"The file '{file_path}' exists.")
        return True
    else:
        #print(f"The file '{file_path}' does not exist.")
        return False

def check_duplicate_dags(ls_dags):
    for i in range(len(ls_dags)):
        for j in range(i + 1, len(ls_dags)):
            if are_graphs_equal(ls_dags[i], ls_dags[j]):
                print(f"Duplicate graphs found: DAG {i} and DAG {j}")

def precompute_node_probs(nx_dag, obs_bn_ie):
    precomputed = {}
    for v in nx_dag.nodes:
        v_name = str(v)
        Pa_v = set(nx_dag.predecessors(v))
        Pa_v_names = nodeset_to_nameset(Pa_v)
        precomputed[v_name] = obs_bn_ie.evidenceImpact(v_name, list(Pa_v_names))
    return precomputed

def gm_to_nx_Digraph(cpdag):
    nx_graph = nx.DiGraph()
    # add nodes to the graph
    nx_graph.add_nodes_from(list(cpdag.nodes))
    for u, v in cpdag.edges:
        nx_graph.add_edge(u, v)
        nx_graph.add_edge(v, u)
    for u, v in cpdag.arcs:
        nx_graph.add_edge(u, v)
    return nx_graph


def sampleDAGs(cpdag, k=1, full_enumeration=False, distinct_dags=False):
    nx_graph =  gm_to_nx_Digraph(cpdag)
    nodes = nx_graph.nodes()
    nx_dags = []
    if full_enumeration:
        sampled_dags = cp.mec_list_dags(list(nx_graph.edges))
    else:
        if distinct_dags:
            hash_map = {}
            while len(hash_map) < k:
                sampled_dag = cp.mec_sample_dags(list(nx_graph.edges), 1)
                g = nx.DiGraph(sampled_dag[0])
                key  = tuple(sorted(g.edges()))
                if key not in hash_map:
                    g.add_nodes_from(list(nodes))
                    hash_map[key] = g
            return list(hash_map.values())
        else:
            sampled_dags = cp.mec_sample_dags(list(nx_graph.edges), k)
    for g in sampled_dags:
        dg = nx.DiGraph(g)
        dg.add_nodes_from(list(nodes)) # add back any nodes in case there are any disjoint variables
        nx_dags.append(dg)
    return nx_dags
    

def add_rc_to_bn(bn, root_cause_name, df):
    bn.add('F')
    bn.addArc('F', root_cause_name)
    learner = gum.BNLearner(df)
    learner.useSmoothingPrior()
    learned_bn = learner.learnParameters(bn.dag())
    return learned_bn
    


def topological_sort(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    ordering = list(nx.topological_sort(G))
    return ordering


def generate_dag_with_constraints(n_nodes, intervened_node, edge_prob=0.3, confounder_prob=0.5):
    """
    Generates a DAG with specified constraints:
    - The intervened_node has at least one descendant.
    - There are confounders between the intervened_node and its descendants,
      with the number of confounders controlled by confounder_prob.

    Parameters:
    - n_nodes (int): Total number of nodes in the DAG.
    - intervened_node (int): The node to be intervened upon.
    - edge_prob (float): Probability of adding an extra edge between other nodes.
    - confounder_prob (float): Probability of including each possible confounder.

    Returns:
    - adj_matrix (np.ndarray): Adjacency matrix of the DAG.
    - G (networkx.DiGraph): The DAG as a NetworkX graph.
    - descendants (list): List of descendants of the intervened_node.
    - confounders (list): List of confounder nodes.
    """
    adj_matrix = np.zeros((n_nodes, n_nodes))
    nodes = list(range(n_nodes))

    # Ensure the intervened_node is not the last node
    if intervened_node >= n_nodes - 1:
        raise ValueError("Intervened node must not be the last node to have descendants.")

    # Step 1: Ensure the intervened_node has descendants
    possible_descendants = nodes[intervened_node + 1:]
    num_descendants = random.randint(1, len(possible_descendants))
    descendants = random.sample(possible_descendants, num_descendants)
    for desc in descendants:
        adj_matrix[intervened_node, desc] = 1  # Edge from intervened_node to descendant
    # Step 2: Add confounders between intervened_node and its descendants
    possible_confounders = nodes[:intervened_node]
    confounders = []
    for conf in possible_confounders:
        if np.random.rand() < confounder_prob:
            confounders.append(conf)
            # Add edges from confounder to intervened_node and its descendants
            adj_matrix[conf, intervened_node] = 1  # Confounder -> intervened_node
            for desc in descendants:
                adj_matrix[conf, desc] = 1  # Confounder -> descendant

    # Step 3: Add random edges among other nodes
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Skip if edge already exists or conflicts with constraints
            if adj_matrix[i, j] == 0 and i != intervened_node and j not in descendants:
                if np.random.rand() < edge_prob:
                    adj_matrix[i, j] = 1  # Edge from i to j

    # Create the DAG
    G = nx.DiGraph(adj_matrix)
    assert nx.is_directed_acyclic_graph(G), "Generated graph is not a DAG!"
    return adj_matrix, G, descendants, confounders

def perform_soft_intervention(noise_means, noise_stds, intervened_node):
    # Example soft intervention: Change the mean and std of the noise term
    new_noise_means = noise_means.copy()
    new_noise_stds = noise_stds.copy()

    # For demonstration, let's shift the mean and increase the variance
    new_noise_means[intervened_node] += np.random.uniform(-10, 10) 
    new_noise_stds[intervened_node] *= np.random.uniform(1.5, 10) 

    return new_noise_means, new_noise_stds


def simulate_data(adj_matrix, n_samples, noise_means, noise_stds):
    n_nodes = adj_matrix.shape[0]
    data = np.zeros((n_samples, n_nodes))
    ordering = topological_sort(adj_matrix)

    # Generate random weights for edges
    weights = np.multiply(adj_matrix, np.random.uniform(0.5, 1.5, size=adj_matrix.shape))
    
    for idx in ordering:
        parents = np.where(adj_matrix[:, idx] != 0)[0]
        if len(parents) == 0:
            # Exogenous node
            data[:, idx] = np.random.normal(loc=noise_means[idx], scale=noise_stds[idx], size=n_samples)
        else:
            # Endogenous node
            parent_data = data[:, parents]
            parent_weights = weights[parents, idx]
            deterministic_part = np.dot(parent_data, parent_weights)
            noise = np.random.normal(loc=noise_means[idx], scale=noise_stds[idx], size=n_samples)
            data[:, idx] = deterministic_part + noise

    return data, weights, ordering


    
def compute_mutual_information_ranking(joint_df):
    """
    Compute mutual information between each column and FNODE, and rank columns by MI.
    
    Args:
        joint_df (pd.DataFrame): DataFrame containing all variables including FNODE
        
    Returns:
        list: Column names ranked by mutual information with FNODE (highest to lowest)
    """
    mi_scores = []
    for col in joint_df.columns:
        if col != 'FNODE':
            # Create contingency table
            contingency = pd.crosstab(joint_df[col], joint_df['FNODE'])
            
            # Convert to probabilities
            p_xy = contingency / contingency.sum().sum()
            
            # Compute marginal probabilities
            p_x = p_xy.sum(axis=1)
            p_y = p_xy.sum(axis=0)
            
            # Compute mutual information
            mi = 0
            for i in p_xy.index:
                for j in p_xy.columns:
                    if p_xy.loc[i,j] > 0:  # Avoid log(0)
                        mi += p_xy.loc[i,j] * np.log(p_xy.loc[i,j] / (p_x[i] * p_y[j]))
            
            mi_scores.append((col, mi))
    
    # Sort by mutual information (highest to lowest)
    ranked_columns = [col for col, _ in sorted(mi_scores, key=lambda x: x[1], reverse=True)]
    return ranked_columns

    

def _safe_qcut_from_reference(ref_series: pd.Series, new_series: pd.Series, q: int):
    """
    Build bin edges from ref_series (quantiles), then apply to new_series.
    Handles duplicates (flat regions) by dropping duplicate edges.
    Returns pd.Series of integer bins (0..B-1) and the edges actually used.
    """
    # quantile edges from reference (normal data)
    qs = np.linspace(0, 1, q + 1)
    edges = np.unique(ref_series.quantile(qs, interpolation="linear").to_numpy())

    # if all values identical → single bin
    if len(edges) <= 2 and np.allclose(edges.min(), edges.max()):
        return pd.Series(np.zeros(len(new_series), dtype=int), index=new_series.index), np.array([edges.min(), edges.max()])

    # ensure strictly increasing edges (drop duplicates)
    if len(edges) < 2:
        # fallback: use min/max of ref
        mn, mx = float(ref_series.min()), float(ref_series.max())
        if mn == mx:
            return pd.Series(np.zeros(len(new_series), dtype=int), index=new_series.index), np.array([mn, mx])
        edges = np.array([mn, mx])

    # apply to new series using pd.cut; include_right=True to mimic KBinsDiscretizer's coverage
    binned = pd.cut(new_series, bins=edges, include_lowest=True, right=True, labels=False)
    # pd.cut returns NaN if value is outside edges due to numeric drift; extend edges slightly if needed
    if binned.isna().any():
        eps = 1e-9 * max(1.0, np.nanstd(ref_series))
        edges[0] = edges[0] - eps
        edges[-1] = edges[-1] + eps
        binned = pd.cut(new_series, bins=edges, include_lowest=True, right=True, labels=False)

    # cast to int (NaNs become -1 if any remain; but we try to avoid NaNs by padding above)
    binned = binned.astype("Int64").fillna(0).astype(int)
    # compress to 0..B-1 in case some bins ended empty in ref (rare when dropping dup edges)
    unique_bins = np.unique(binned)
    remap = {val: i for i, val in enumerate(sorted(unique_bins))}
    binned = binned.map(remap).astype(int)
    # rebuild edges to match the number of active bins (len(unique_bins))
    B = len(unique_bins)
    if B + 1 <= len(edges):
        # pick a subset of edges if some bins collapsed
        edges = edges[:B+1]
    return binned, edges


def _mi_discrete(x: pd.Series, y: pd.Series) -> float:
    """Simple MI(X;Y) for discrete pandas Series."""
    ct = pd.crosstab(x, y)
    pxy = ct / ct.values.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in pxy.index:
        for j in pxy.columns:
            pij = pxy.loc[i, j]
            if pij > 0:
                mi += float(pij * np.log(pij / (px[i] * py[j])))
    return float(mi)


def discretize_adaptive(
    normal_df: pd.DataFrame,
    anomalous_df: pd.DataFrame,
    n_bins_grid=(5, 3, 2),
    min_per_bin=15,
    prefer_higher_mi=True,
):
    """
    Adaptive, per-variable discretization:
      - bin edges learned from NORMAL ONLY (quantiles)
      - applied to both normal & failure
      - for each variable, choose the LARGEST bin count in n_bins_grid
        that keeps >= min_per_bin in EVERY bin for BOTH regimes
      - ties broken by higher MI(FNODE; X_binned) if prefer_higher_mi=True

    Returns:
      df_obs_binned, df_int_binned, joint_df_binned (with FNODE), bin_info (dict)
    """
    cols = list(normal_df.columns)
    # build FNODE to evaluate MI if needed
    fn_normal = pd.Series(np.zeros(len(normal_df), dtype=int), name="FNODE")
    fn_failure = pd.Series(np.ones(len(anomalous_df), dtype=int), name="FNODE")

    df_obs_binned = pd.DataFrame(index=normal_df.index)
    df_int_binned = pd.DataFrame(index=anomalous_df.index)
    bin_info = {}

    for col in cols:
        ref = normal_df[col].astype(float)
        obs_vals = normal_df[col].astype(float)
        int_vals = anomalous_df[col].astype(float)

        # constant feature short-circuit
        if np.allclose(ref.min(), ref.max()) and np.allclose(int_vals.min(), int_vals.max()):
            df_obs_binned[col] = 0
            df_int_binned[col] = 0
            bin_info[col] = {"bins": 1, "edges": np.array([ref.min(), ref.max()]), "reason": "constant"}
            continue

        candidates = []
        for b in n_bins_grid:
            # build edges on NORMAL and apply to both
            obs_b, edges = _safe_qcut_from_reference(ref, obs_vals, q=b)
            int_b, _ = _safe_qcut_from_reference(ref, int_vals, q=b)

            # check per-bin support in both regimes
            ok = True
            for s in [obs_b, int_b]:
                counts = s.value_counts().reindex(range(s.max() + 1), fill_value=0)
                if (counts < min_per_bin).any():
                    ok = False
                    break

            if not ok:
                continue

            # candidate accepted; compute MI if needed
            if prefer_higher_mi:
                joint_col = pd.concat([obs_b.reset_index(drop=True), int_b.reset_index(drop=True)], ignore_index=True)
                fnode = pd.concat([fn_normal, fn_failure], ignore_index=True)
                mi = _mi_discrete(joint_col, fnode)
            else:
                mi = 0.0

            candidates.append((b, mi, edges, obs_b, int_b))

        if not candidates:
            # fallback to the smallest bin count with best MI (even if min_per_bin fails)
            fallback = []
            for b in sorted(set(n_bins_grid)):
                obs_b, edges = _safe_qcut_from_reference(ref, obs_vals, q=b)
                int_b, _ = _safe_qcut_from_reference(ref, int_vals, q=b)
                if prefer_higher_mi:
                    joint_col = pd.concat([obs_b.reset_index(drop=True), int_b.reset_index(drop=True)], ignore_index=True)
                    fnode = pd.concat([fn_normal, fn_failure], ignore_index=True)
                    mi = _mi_discrete(joint_col, fnode)
                else:
                    mi = 0.0
                fallback.append((b, mi, edges, obs_b, int_b))
            # pick (smallest b first, then highest MI)
            fallback.sort(key=lambda t: (t[0], -t[1]))
            b, mi, edges, obs_b, int_b = fallback[0]
            reason = f"fallback_bins_{b}"
        else:
            # pick the candidate with largest b; break ties by MI
            candidates.sort(key=lambda t: (t[0], t[1] if prefer_higher_mi else 0.0), reverse=True)
            b, mi, edges, obs_b, int_b = candidates[0]
            reason = "meets_min_per_bin"

        df_obs_binned[col] = obs_b.astype(int).values
        df_int_binned[col] = int_b.astype(int).values
        bin_info[col] = {"bins": int(b), "mi": float(mi), "edges": edges, "reason": reason}

    # build joint with FNODE for convenience
    joint_df_binned = pd.concat([df_obs_binned.reset_index(drop=True),
                                 df_int_binned.reset_index(drop=True)], ignore_index=True)
    joint_df_binned["FNODE"] = np.r_[np.zeros(len(df_obs_binned), dtype=int),
                                     np.ones(len(df_int_binned), dtype=int)]

    return df_obs_binned.astype(int), df_int_binned.astype(int), joint_df_binned.astype(int), bin_info


def reconcile_edges(
    G: nx.DiGraph,
    directed_edges: Iterable[Tuple[Hashable, Hashable]],
    undirected_edges: Iterable[Tuple[Hashable, Hashable]],
) -> tuple[List[Tuple[Hashable, Hashable]], List[Tuple[Hashable, Hashable]]]:
    """
    Reconcile (directed, undirected) edge lists against a reference DiGraph G.

    Behavior
    --------
    1) Directed edges:
       - Drop (u, v) only if G asserts the *opposite only*: G has (v, u) and NOT (u, v).
       - Otherwise keep (u, v) (including when G has both or neither).

    2) Undirected edges:
       - If G orients the pair (x, y) in exactly one direction:
           * Only x->y  in G  => move (x, y) to directed as (x, y)
           * Only y->x  in G  => move (x, y) to directed as (y, x)
       - If G has both directions (x->y and y->x), treat as *undirected* and keep (x, y).
       - If G has neither direction, keep (x, y).

    Notes
    -----
    - Pairs with nodes not present in G are left as-is.
    - Output directed edges are de-duplicated while preserving insertion order.
    """
    G_nodes = set(G.nodes)

    # 1) Filter directed edges that contradict G
    new_directed: List[Tuple[Hashable, Hashable]] = []
    seen_directed = set()
    for u, v in directed_edges:
        if u in G_nodes and v in G_nodes:
            # contradiction only if G has strictly the reverse
            if G.has_edge(v, u) and not G.has_edge(u, v):
                continue  # drop (u, v)
        if (u, v) not in seen_directed:
            new_directed.append((u, v))
            seen_directed.add((u, v))

    # 2) Orient undirected edges using G (respecting the "both => keep undirected" rule)
    new_undirected: List[Tuple[Hashable, Hashable]] = []
    for x, y in undirected_edges:
        oriented = False
        if x in G_nodes and y in G_nodes:
            xy = G.has_edge(x, y)
            yx = G.has_edge(y, x)
            if xy ^ yx:  # exactly one is True
                # Move to directed in the existing orientation
                d = (x, y) if xy else (y, x)
                if d not in seen_directed:
                    new_directed.append(d)
                    seen_directed.add(d)
                oriented = True
            elif xy and yx:
                # Both directions in G => treat as undirected; keep (x, y)
                oriented = False

        if not oriented:
            new_undirected.append((x, y))

    return new_directed, new_undirected





    
    






