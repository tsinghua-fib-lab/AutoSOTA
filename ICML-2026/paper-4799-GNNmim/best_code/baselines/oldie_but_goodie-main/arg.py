import argparse

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="Indexes of gpu to run program on", default=1)
    parser.add_argument("--dataset", type=str, help="Name of Dataset", default='Cora', choices=["Cora","CiteSeer","PubMed","Photo","Computers","wikics", "cs", "physics", "OGBN-Arxiv","OGBN-Products","MixHopSynthetic"])
    parser.add_argument("--mask_type", type=str, help="Type of missing feature mask", default="structural", choices=["uniform", "structural"])
    parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.9999)
    parser.add_argument("--lp_alpha", type=float, help="Alpha parameter of label propagation", default=0.99) 
    parser.add_argument("--lamb", type=float, help="Control loss between metric loss and cross entropy loss", default=1.0) # 0.0
    parser.add_argument("--filling_method",type=str,help="Method to solve the missing feature problem",default="fp",choices=["random", "zero", "mean", "neighborhood_mean", "fp"])
    parser.add_argument("--embedder",type=str,help="Type of model to make a prediction on the downstream task",default="GOODIE", choices=["LP", "MLP", "GNN", "GCNMF", "PaGNN", "Correct_Smooth", "GCN_LPA", "Node2Vec", "Node2Vec_x", "Node2Vec_GNN", "GOODIE"])
    if parser.parse_known_args()[0].embedder in ['GNN', 'GOODIE', 'Node2Vec_GNN']:
        parser.add_argument("--gnn",type=str,help="Type of model to make a prediction on the downstream task",default="GCN", choices=["SGC", "SAGE", "GCN", "GAT"])
    
    parser.add_argument("--scaled", type=str2bool, help="Wheter to utilize scaled PseudoCon loss, True for large datasets", default=False)
    parser.add_argument("--p", type=float, help="Likelihood of immediately revisiting a node in the walk", default=1)
    parser.add_argument("--q", type=float, help="Control parameter to interpolate between BFS and DFS", default=1)
    parser.add_argument("--leaky_alpha", type=float, help="Control slope of leaky relu", default=0.3)
    parser.add_argument("--temp", type=float, help="Temperature for Contrastive Learning", default=0.01)

    parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.005)
    parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
    parser.add_argument("--n_runs", type=int, help="Max number of runs", default=10)
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=2)
    parser.add_argument("--num_heads", type=int, help="Number of GAT heads", default=2)
    parser.add_argument("--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=40)
    parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
    parser.add_argument("--batch_size", type=int, help="Batch size for models trained with neighborhood sampling", default=1024)
    parser.add_argument("--batch_norm",help="Applying Batch Normalizetion",action="store_true",default=True)
    parser.add_argument("--homophily", type=float, help="Level of homophily for synthetic datasets", default=None)

    return parser

# python main.py --missing_rate 0.0 --gpu 0