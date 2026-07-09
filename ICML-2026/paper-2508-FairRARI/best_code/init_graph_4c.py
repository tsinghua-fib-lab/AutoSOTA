import os
import copy
import numpy as np
import networkx as nx
import csv

def init_graph_4c(dataset_name, source_path):

    if "twitch" in dataset_name:    
        if "DE" in dataset_name:
            graph_path = source_path+"twitch/DE/musae_DE_edges.csv"
            color_path = source_path+"twitch/DE/musae_DE_target.csv"
        elif "ENGB" in dataset_name:
            graph_path = source_path+"twitch/ENGB/musae_ENGB_edges.csv"
            color_path = source_path+"twitch/ENGB/musae_ENGB_target.csv"
        elif "ES" in dataset_name:
            graph_path = source_path+"twitch/ES/musae_ES_edges.csv"
            color_path = source_path+"twitch/ES/musae_ES_target.csv"
        elif "FR" in dataset_name:
            graph_path = source_path+"twitch/FR/musae_FR_edges.csv"
            color_path = source_path+"twitch/FR/musae_FR_target.csv"
        elif "PTBR" in dataset_name:
            graph_path = source_path+"twitch/PTBR/musae_PTBR_edges.csv"
            color_path = source_path+"twitch/PTBR/musae_PTBR_target.csv"
        elif "RU" in dataset_name:
            graph_path = source_path+"twitch/RU/musae_RU_edges.csv"
            color_path = source_path+"twitch/RU/musae_RU_target.csv"


        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                            nodetype=int, data=(('weight', float),))

        color_file = open(color_path, "r")
        next(color_file, None)
        data = list(csv.reader(color_file, delimiter=","))
        color_file.close()
        colors = np.array(data)

        for idx, node_i in enumerate(colors[:,5]):
            color_value1 = colors[idx,2] # mature
            color_value2 = colors[idx,4] # partner
            nx.set_node_attributes(G, {int(node_i): color_value1}, name="value1")
            nx.set_node_attributes(G, {int(node_i): color_value2}, name="value2")

    ### Color Graph's Protected Nodes***
    if "twitch" in dataset_name:
        # Initialize empty lists for the 4 categories
        protected_nodes_0 = []  # value1=True, value2=True
        protected_nodes_1 = []  # value1=True, value2=False
        protected_nodes_2 = []  # value1=False, value2=True
        protected_nodes_3 = []  # value1=False, value2=False

        # Iterate through nodes and categorize them
        for node in G.nodes():
            value1 = G.nodes[node].get('value1')  # Get value1 for the node
            value2 = G.nodes[node].get('value2')  # Get value2 for the node

            if value1 == 'True' and value2 == 'False':
                protected_nodes_0.append(node)
                nx.set_node_attributes(G, {int(node): 0}, name="value")
            elif value1 == 'True' and value2 == 'True':
                protected_nodes_1.append(node)
                nx.set_node_attributes(G, {int(node): 1}, name="value")
            elif value1 == 'False' and value2 == 'True':
                protected_nodes_2.append(node)
                nx.set_node_attributes(G, {int(node): 2}, name="value")
            elif value1 == 'False' and value2 == 'False':
                protected_nodes_3.append(node)
                nx.set_node_attributes(G, {int(node): 3}, name="value")
       
    protected_nodes_indices_0 = protected_nodes_0
    protected_nodes_indices_1 = protected_nodes_1
    protected_nodes_indices_2 = protected_nodes_2
    protected_nodes_indices_3 = protected_nodes_3
    protected_nodes_0 = np.where(np.isin(list(G.nodes()), protected_nodes_0))[0]   
    protected_nodes_1 = np.where(np.isin(list(G.nodes()), protected_nodes_1))[0]   
    protected_nodes_2 = np.where(np.isin(list(G.nodes()), protected_nodes_2))[0]  
    protected_nodes_3 = np.where(np.isin(list(G.nodes()), protected_nodes_3))[0]    
   
    return G, protected_nodes_indices_0, protected_nodes_indices_1, protected_nodes_indices_2, protected_nodes_indices_3