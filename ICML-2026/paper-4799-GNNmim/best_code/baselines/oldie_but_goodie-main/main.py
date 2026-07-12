import time
import arg
import os
import yaml

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

def main():
    parser = arg.get_parser()
    args = parser.parse_args()
    print(args)
    with open("hyperparameters.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        dataset = args.dataset + f'_{args.mask_type}'
        if dataset in hyperparams:
            for k, v in hyperparams[dataset].items():
                setattr(args, k, v)

    if args.embedder == 'LP':
        from models import LP
        embedder = LP(args)

    elif args.embedder == 'GCNMF': 
        from models import GCNMF
        embedder = GCNMF(args)
    
    elif args.embedder == 'PaGNN':
        from models import PaGNN
        embedder = PaGNN(args)

    elif args.embedder == 'Node2Vec':
        from models import node2vec
        embedder = node2vec(args)

    elif args.embedder == 'Node2Vec_x': 
        from models import node2vec_x
        embedder = node2vec_x(args)

    elif args.embedder == 'Node2Vec_GNN':
        from models import node2vec_gnn
        embedder = node2vec_gnn(args)

    elif args.embedder == 'MLP':
        from models import mlp
        embedder = mlp(args)

    elif args.embedder == 'GNN': # Filling Variants: "zero", "neighbor_mean", "fp"
        from models import gnn
        embedder = gnn(args)
    
    elif args.embedder == 'GCN_LPA': 
        from models import GCN_LPA
        embedder = GCN_LPA(args)

    elif args.embedder == 'Correct_Smooth':
        from models import Correct_Smooth
        embedder = Correct_Smooth(args)

    elif args.embedder == 'GOODIE':
        from models import GOODIE
        embedder = GOODIE(args)

    t_total = time.time()
    embedder.training()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if __name__ == '__main__':
    main()
