import math
import networkx as nx
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
# import warnings
# warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)

options = {
    'node_size': 5,
    'edge_color' : 'black',
    'linewidths': 1,
    'width': 0.5
}

def plot_graphs_list(graphs, title='title', max_num=16, save_dir=None, N=0):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        idx = i + max_num*N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        pos = nx.spring_layout(G, iterations=50)

        try:
            # Set node colors based on the eigenvectors
            w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(G).toarray())
            vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
            m = max(np.abs(vmin), vmax)
            vmin, vmax = -m, m
            nx.draw(G, pos, with_labels=False, vmin=vmin, vmax=vmax, node_color=U[:, 1],
                    cmap=plt.cm.coolwarm, **options)
        except:
            nx.draw(G, pos, with_labels=False, **options)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_fig(save_dir=None, title='fig', dpi=300):
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(*['samples', 'fig', save_dir])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title+'.png'),
                    bbox_inches='tight', dpi=dpi, 
                    transparent=False)
        plt.close()
    return


def save_graph_list(log_folder_name, exp_name, gen_graph_list):

    if not(os.path.isdir(f'./samples/pkl/{log_folder_name}')):
        os.makedirs(os.path.join(f'./samples/pkl/{log_folder_name}'))
    with open(f'./samples/pkl/{log_folder_name}/{exp_name}.pkl', 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = f'./samples/pkl/{log_folder_name}/{exp_name}.pkl'
    return save_dir
