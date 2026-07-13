
import argparse
import torch
import numpy as np
import torch.optim as optim
import sys
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

sys.path.append('./src/')

from AICoG import LSM


parser = argparse.ArgumentParser(description='Aitchison Compositional Graph Embeddings')

parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs for training (default: 5K)')

parser.add_argument('--scaling_epochs', type=int, default=500, metavar='N',
                    help='number of epochs for learning initial scale for the random effects (default: 500)')


parser.add_argument('--cuda', type=eval, 
                      choices=[True, False],  default=True,
                    help='CUDA training')



parser.add_argument('--LP',type=eval, 
                      choices=[True, False], default=False,
                    help='performs link prediction')

parser.add_argument('--clas',type=eval, 
                      choices=[True, False], default=False,
                    help='performs node classification')


parser.add_argument('--euclidean',type=eval, 
                      choices=[True, False], default=False,
                    help='learns embeddings on the simplex without ILR projection')


parser.add_argument('--K', type=int, default=9, metavar='N',
                    help='number of compoments of the embeddings (default: 9)')

parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                    help='learning rate for the ADAM optimizer, for large values of delta 0.01 is more stable (default: 0.1)')



parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset to apply AICoG on')

parser.add_argument('--neg_ratio', type=float, default=5.0,
                    help='negative sampling ratio (M_neg = ratio * |E|, default: 5.0)')

parser.add_argument('--seed', type=int, default=None,
                    help='random seed for reproducibility')



args = parser.parse_args()

if  args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')


    
    


if __name__ == "__main__":

    # Deterministic seeding for reproducibility (Idea CODE-09)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)


    latent_dims=[args.K]
   
    datasets=[args.dataset]
    

    for dataset in datasets:
        for latent_dim in latent_dims:
            rocs=[]
            prs=[]
            micros=[]
            macros=[]
            for run in range(1):


                sparse_i_=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_i.txt')).long().to(device)     
                sparse_j_=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_j.txt')).long().to(device)    
                
            
                sparse_i=torch.cat((sparse_i_,sparse_j_))
                sparse_j=torch.cat((sparse_j_,sparse_i_))
                
                if args.LP:
                    # file denoting rows i of missing links, with i<j 
                    sparse_i_rem=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/sparse_i_rem.txt')).long().to(device)
                    # file denoting columns j of missing links, with i<j
                    sparse_j_rem=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/sparse_j_rem.txt')).long().to(device)
                    # file denoting negative sample rows i, with i<j
                    non_sparse_i=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/non_sparse_i.txt')).long().to(device)
                    # file denoting negative sample columns, with i<j
                    non_sparse_j=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/non_sparse_j.txt')).long().to(device)
                
                else:
                    non_sparse_i=None
                    non_sparse_j=None
                    sparse_i_rem=None
                    sparse_j_rem=None
                    
                N=int(sparse_j.max()+1)
                

                model = LSM(sparse_i,sparse_j,N,latent_dim=latent_dim,non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,graph_type='undirected',device=device, neg_ratio=args.neg_ratio).to(device)
                optimizer = optim.Adam(model.parameters(), args.lr)  
                
                elements=(N*(N-1))*0.5
                for epoch in tqdm(range(args.epochs),desc="AICoG is Running…",ascii=False, ncols=75):
                    if epoch==args.scaling_epochs:
                        model.scaling=0
                    
                    loss=-model.LSM_likelihood_bias(epoch=epoch,euclidean=args.euclidean)/elements

                    
                
                
                    
            
                
                    optimizer.zero_grad() # clear the gradients.
                    loss.backward() # backpropagate
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping (Idea CODE-10)
                    optimizer.step() # update the weights

                

                if args.euclidean:
                    params_dict = model.state_dict()          # OrderedDict: name -> tensor
                    #torch.save(params_dict, f"./datasets/{dataset}/params_Simplex_NO_ilr_{dataset}_{latent_dim}_{run}.pt") # saves the dictionary

                else:


                    params_dict = model.state_dict()          # OrderedDict: name -> tensor
                    torch.save(params_dict, f"./datasets/{dataset}/params_AICoG_{dataset}_{latent_dim}_{run}.pt") # saves the dictionary

                    
                if args.LP:
                    roc,pr=model.link_prediction() 

                    rocs.append(roc)
                    prs.append(pr)



                if args.clas:

                    import numpy as np
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler, normalize
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import f1_score, accuracy_score
                    from sklearn.neighbors import NearestNeighbors

                    # ----------------
                    # 0) Inputs you already have
                    # ----------------
                    X = model.latent_z.detach().cpu().numpy()  # your embeddings
                    num_nodes = X.shape[0]
                    path = f"./datasets/{dataset}/labels.txt"

                    # ----------------
                    # 1) Load single labels: one int per row, row i -> node i
                    # ----------------
                    def load_single_labels_txt(path, N=None, unlabeled_values=None):
                        if unlabeled_values is None:
                            unlabeled_values = {-1}  # common convention

                        labels = []
                        with open(path, "r") as f:
                            for line in f:
                                s = line.strip()
                                if s == "":
                                    labels.append(-1)
                                else:
                                    labels.append(int(s))

                        y = np.array(labels, dtype=int)

                        # match X length if needed
                        if N is not None:
                            if len(y) < N:
                                y = np.pad(y, (0, N - len(y)), constant_values=-1)
                            elif len(y) > N:
                                y = y[:N]

                        # map unlabeled markers to -1
                        for uv in unlabeled_values:
                            y[y == uv] = -1

                        return y

                    y_all = load_single_labels_txt(path, N=num_nodes, unlabeled_values={-1})

                    # keep labeled nodes only
                    mask = (y_all != -1)
                    nodes = np.where(mask)[0]
                    X_labeled = X[nodes]
                    y = y_all[nodes]

                    if len(np.unique(y)) < 2:
                        raise ValueError("Need at least 2 classes among labeled nodes for classification.")

                    # ----------------
                    # 2) Supervised classification: tune C on val macro-F1
                    # ----------------
                    # stratify requires enough samples per class; if it errors, remove stratify=...
                    try:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X_labeled, y, test_size=0.5, random_state=42, shuffle=True, stratify=y
                        )
                        X_tr, X_va, y_tr, y_va = train_test_split(
                            X_tr, y_tr, test_size=0.2, random_state=42, shuffle=True, stratify=y_tr
                        )
                    except ValueError:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X_labeled, y, test_size=0.5, random_state=42, shuffle=True
                        )
                        X_tr, X_va, y_tr, y_va = train_test_split(
                            X_tr, y_tr, test_size=0.2, random_state=42, shuffle=True
                        )

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)
                    X_te_s = scaler.transform(X_te)

                    best = (-1.0, None)  # (val_macroF1, C)
                    for C in [0.01, 0.1, 1, 10, 100]:
                        clf = LogisticRegression(
                            solver="lbfgs",
                            max_iter=5000,
                            C=C
                        )
                        clf.fit(X_tr_s, y_tr)
                        y_hat_va = clf.predict(X_va_s)
                        macro_va = f1_score(y_va, y_hat_va, average="macro", zero_division=0)
                        if macro_va > best[0]:
                            best = (macro_va, C)

                    C_star = best[1]
                    clf = LogisticRegression(
                        solver="lbfgs",
                        max_iter=5000,
                        C=C_star
                       
                    )
                    clf.fit(np.vstack([X_tr_s, X_va_s]), np.concatenate([y_tr, y_va]))

                    y_hat_te = clf.predict(X_te_s)

                    acc = accuracy_score(y_te, y_hat_te)
                    micro = f1_score(y_te, y_hat_te, average="micro", zero_division=0)  # == acc for single-label multiclass
                    macro = f1_score(y_te, y_hat_te, average="macro", zero_division=0)

                    micros.append(micro)
                    macros.append(macro)

            if args.clas:
                print(64*'#')
                print(dataset)
                print(latent_dim)
                print('micro:', np.mean(micros))
                print(np.std(micros))
                
                print('macro:', np.mean(macros))
                print(np.std(macros))

            if args.LP:
            
                print(64*'#')
                print(dataset)
                print(latent_dim)
                print('ROC:', np.mean(rocs))
                print(np.std(rocs))
                
                print('PR:', np.mean(prs))
                print(np.std(prs))
                    # % dimensions removed -> choose Kprime from a removal fraction
                    # Works with your exact calling pattern:
                    #   Euclidean: link_prediction(out_dim=Kprime-1)
                    #   ILR:       link_prediction(S_mask=S_mask)



               

            
        plot=False
        if plot:
                            
            plt.rcParams["figure.figsize"] = (10,10)
            
            z_idx=model.latent_z_.argmax(1)
            w_idx=model.latent_z_.argmax(1)
            
            f_z=z_idx.argsort()
            f_w=w_idx.argsort()
            
            new_i=torch.cat((sparse_i,sparse_j))
            new_j=torch.cat((sparse_j,sparse_i))
            
            D=csr_matrix((np.ones(new_i.shape[0]),(new_i.cpu().numpy(),new_j.cpu().numpy())),shape=(N,N))#.todense()
        
            
            D = D[:, f_w.cpu().numpy()][f_z.cpu().numpy()]
            
            
            plt.spy(D,markersize=1)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'adjacency_{args.dataset}.pdf')
            plt.show()
        