import os
import copy
import numpy as np
import networkx as nx
import utils
import csv

def init_graph(dataset_name, source_path):
    
    if dataset_name == "polbooks":
        
        graph_path = source_path+"pol_books/polbooks.gml"
        orig_G = nx.read_gml(graph_path)
        all_books = list(orig_G.nodes())
        neutral_books = [x for x,y in orig_G.nodes(data=True) if y['value']=='n']

        G_ = copy.deepcopy(orig_G)
        G_.remove_nodes_from(neutral_books)
        lib_cons_books = list(G_.nodes())
        G = copy.deepcopy(G_)

        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)

    elif "deezer" in dataset_name:    

        graph_path = source_path+"deezer_europe/deezer_europe_edges.csv"
        color_path = source_path+"deezer_europe/deezer_europe_target.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                              nodetype=int, data=(('weight', float),))

        color_file = open(color_path, "r")
        next(color_file, None)
        data = list(csv.reader(color_file, delimiter=","))
        color_file.close()
        colors = np.array(data).astype(int)

        for idx, node_i in enumerate(colors[:,0]):
            color_value = colors[idx,1]
            nx.set_node_attributes(G, {node_i: color_value}, name="value")
    
    elif "github" in dataset_name:    

        graph_path = source_path+"git_web_ml/musae_git_edges.csv"
        color_path = source_path+"git_web_ml/musae_git_target.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                            nodetype=int, data=(('weight', float),))

        color_file = open(color_path, "r")
        next(color_file, None)
        data = list(csv.reader(color_file, delimiter=","))
        color_file.close()
        colors_ = np.array(data)
        colors = np.delete(colors_, 1, 1)
        colors = colors.astype(int)

        for idx, node_i in enumerate(colors[:,0]):
            color_value = colors[idx,1]
            nx.set_node_attributes(G, {node_i: color_value}, name="value")
        
    elif "twitch" in dataset_name:    
        if "gamers" not in dataset_name:
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
                color_value = colors[idx,2] # mature
                nx.set_node_attributes(G, {int(node_i): color_value}, name="value")
            
        else:
            graph_path = source_path+"twitch_gamers/large_twitch_edges.csv"
            color_path = source_path+"twitch_gamers/large_twitch_features.csv"

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
                color_value = colors[idx,1] # mature
                nx.set_node_attributes(G, {int(node_i): color_value}, name="value")

    elif "blogs" in dataset_name:    

        graph_path = source_path+"blogs/out_graph.txt"
        color_path = source_path+"blogs/out_community.txt"

        G = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), nodetype = int)
        
        colors = np.loadtxt(color_path)
        colors = colors.astype(int)
        
        for idx, node_i in enumerate(colors[:,0]):
            color_value = colors[idx,1]
            nx.set_node_attributes(G, {node_i: color_value}, name="value")
    
    elif "twitter" in dataset_name:    

        graph_path = source_path+"twitter/out_graph.txt"
        color_path = source_path+"twitter/out_community.txt"

        G = nx.read_edgelist(graph_path, create_using=nx.Graph(), nodetype = int)
        
        colors = np.loadtxt(color_path)
        colors = colors.astype(int)
        
        for idx, node_i in enumerate(colors[:,0]):
            color_value = colors[idx,1]
            nx.set_node_attributes(G, {node_i: color_value}, name="value")
        
    elif "dblp" in dataset_name:    

        graph_path = source_path+dataset_name+"/out_graph.txt"
        color_path = source_path+dataset_name+"/out_community.txt"

        G = nx.read_edgelist(graph_path, create_using=nx.Graph(), nodetype = int)
        
        colors = np.loadtxt(color_path)
        colors = colors.astype(int)
        
        for idx, node_i in enumerate(colors[:,0]):
            color_value = colors[idx,1]
            nx.set_node_attributes(G, {node_i: color_value}, name="value")
    
    elif "erdos" in dataset_name:    

        graph_path = source_path+"Erdos02"+"/out_graph.txt"
        color_path = source_path+"Erdos02"+"/out_community.txt"

        # Initialize the graph
        G = nx.Graph()
        protected_nodes = []

        # Load the graph edges
        with open(graph_path, "r") as f:
            for line in f:
                node1, node2 = map(int, line.split())
                G.add_edge(node1, node2)

        # Load the node communities
        with open(color_path, "r") as f:
            for line in f:
                node, community = map(int, line.split())
                if node not in G:
                    G.add_node(node)  # Add node if not present in the graph
                G.nodes[node]["value"] = community
                if community == 1:
                    protected_nodes.append(node)

    elif "slashdot" in dataset_name:    

        graph_path = source_path+"slashdot/out_graph.txt"
        color_path = source_path+"slashdot/out_community.txt"

        G = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), nodetype = int)
        
        colors = np.loadtxt(color_path)
        colors = colors.astype(int)
        
        for idx, node_i in enumerate(colors[:,0]):
            color_value = colors[idx,1]
            nx.set_node_attributes(G, {node_i: color_value}, name="value")

    ### Color Graph's Protected Nodes
    if dataset_name == "polbooks":
        values_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in values_list if item[1] == 'l'] # protected = 'liberal'
    
    elif "deezer" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 1
    
    elif "github" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 1
                    
    elif "twitch" in dataset_name:
        if "gamers" not in dataset_name:
            colors_list = list(G.nodes(data='value'))
            protected_nodes = [item[0] for item in colors_list if item[1] == 'True'] # (for "mature" label)
        else: 
            colors_list = list(G.nodes(data='value'))
            protected_nodes = [item[0] for item in colors_list if item[1] == '1'] # (for "mature" label)

    elif "blogs" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 0] # protected = 0

    elif "twitter" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 1

    elif "dblp" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 1 (FEMALE)

    elif "slashdot" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 1

    protected_nodes_indices = protected_nodes
    protected_nodes = np.where(np.isin(list(G.nodes()), protected_nodes))[0]   
    blue_nodes, red_nodes = utils.color_protected(G, protected_nodes_indices) # red = protected
   
    return G, protected_nodes_indices, blue_nodes, red_nodes