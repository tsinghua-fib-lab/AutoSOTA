from re import X
import networkx as nx
import numpy as np


def color_protected(G, protected_nodes, print_=False):
    # protected_nodes.sort()
    
    # for node_i in range(len(G.nodes())):
    for node_i in G.nodes():
        if node_i in protected_nodes:
            nx.set_node_attributes(G, {node_i: 'tab:red'}, name="color")
        else:
            nx.set_node_attributes(G, {node_i: 'tab:blue'}, name="color")

    red_nodes = [x for x,y in G.nodes(data=True) if y["color"]=="tab:red"]
    if print_==True:
        print("Protected (Red) Nodes: ", red_nodes)
    blue_nodes = [x for x,y in G.nodes(data=True) if y["color"]=="tab:blue"]
    if print_==True:
        print("Un-Protected (Blue) Nodes: ", blue_nodes)

    return blue_nodes, red_nodes

def find_complement(omega_vec, x_vec):
    x_set = set(x_vec)
    omega_set = set(omega_vec)
    return list(omega_set - x_set)

def find_common(x_vec, y_vec):
    x_set = set(x_vec)
    y_set = set(y_vec)

    # Find the common elements by taking the intersection of the sets
    common_elements = x_set.intersection(y_set)
    
    return common_elements

def find_num_common(x_vec, y_vec):
    # Find the common elements by taking the intersection of the sets
    common_elements = find_common(x_vec, y_vec)

    # Get the number of common elements
    num_common = len(common_elements)
    
    return num_common

def find_protected_portion(x_vec, protected_vec):
    num_common = find_num_common(x_vec, protected_vec)
    return num_common / len(x_vec)

def compute_density(S, G=None, weight=None):
    if type(S) is not nx.classes.graph.Graph:
        S = G.subgraph(S)

    return S.size(weight) / S.number_of_nodes()

