from algorithm import *
from networkx import read_edgelist
from scipy.io import loadmat
from model import *
from utils import *
import scipy.io as io
try:
    import dgl
except ImportError:
    pass
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils import dense_to_sparse
from torch.optim import Adam
import argparse
from load_data import import_data, load_config
import time
   
def run_GADL(data, GAE_model, train_adj, train_features, args, device, test_pairs):
            
        
    optimizer = Adam(GAE_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    keys = train_adj.keys()
    A1 = train_adj[list(keys)[0]][0]
    X1 = train_features[list(keys)[0]][0]
    A2 = train_adj[list(keys)[1]][0]
    X2 = train_features[list(keys)[1]][0]
    
    X1 = X1.to(device)
    X2 = X2.to(device)   
        
    if os.path.exists(f'eigs/{data}'):
        L1 = torch.load(f'eigs/{data}/L1.pth')
        lam1 = torch.load(f'eigs/{data}/lam1.pth')
        phi1 = torch.load(f'eigs/{data}/phi1.pth')
        L2 = torch.load(f'eigs/{data}/L2.pth')
        lam2 = torch.load(f'eigs/{data}/lam2.pth')
        phi2 = torch.load(f'eigs/{data}/phi2.pth')
        print('Laplacian matrix, eigenvalues, and eigenvectros loaded succesfully!')
    else:
        # Precompute eigenvectors 
        L1, lam1, phi1 = compute_laplacian_eigenbasis(A1, args.k, norm_type='sym')
        L2, lam2, phi2 = compute_laplacian_eigenbasis(A2, args.k, norm_type='sym')
        print('Computing eigendecomposition completed!')
        os.makedirs(f'eigs/{data}', exist_ok=True)
        torch.save(L1, f'eigs/{data}/L1.pth')
        torch.save(lam1, f'eigs/{data}/lam1.pth')
        torch.save(phi1, f'eigs/{data}/phi1.pth')
        torch.save(L2, f'eigs/{data}/L2.pth')
        torch.save(lam2, f'eigs/{data}/lam2.pth')
        torch.save(phi2, f'eigs/{data}/phi2.pth')
        print('Laplacian matrix, eigenvalues, and eigenvectros saved succesfully!')
        
    
    phi1, lam1 = phi1.to(device), lam1.to(device)
    phi2, lam2 = phi2.to(device), lam2.to(device)
    
    adj1 = coo_matrix(A1.numpy())
    adj2 = coo_matrix(A2.numpy())

    pos_weight1 = float(adj1.shape[0] * adj1.shape[0] - adj1.sum()) / adj1.sum()
    pos_weight2 = float(adj2.shape[0] * adj2.shape[0] - adj2.sum()) / adj2.sum()
    norm1 = adj1.shape[0] * adj1.shape[0] / float((adj1.shape[0] * adj1.shape[0] - adj1.sum()) * 2)
    norm2 = adj2.shape[0] * adj2.shape[0] / float((adj2.shape[0] * adj2.shape[0] - adj2.sum()) * 2)
    
    adj_label1 = sparse_to_tuple(adj1)
    adj_label2 = sparse_to_tuple(adj2)
    adj_label1 = torch.sparse.FloatTensor(torch.LongTensor(adj_label1[0].T), torch.FloatTensor(adj_label1[1]),torch.Size(adj_label1[2])).to(device)
    adj_label2 = torch.sparse.FloatTensor(torch.LongTensor(adj_label2[0].T), torch.FloatTensor(adj_label2[1]), torch.Size(adj_label2[2])).to(device)


    weight_mask1 = adj_label1.to_dense().view(-1) == 1
    weight_mask2 = adj_label2.to_dense().view(-1) == 1
    weight_tensor1 = torch.ones(weight_mask1.size(0))
    weight_tensor2 = torch.ones(weight_mask2.size(0))
    weight_tensor1[weight_mask1] = pos_weight1
    weight_tensor2[weight_mask2] = pos_weight2
    weight_tensor1 = weight_tensor1.to(device)
    weight_tensor2 = weight_tensor2.to(device)
            
    with tqdm(total=args.epoch, desc='(T)') as pbar:
        for step in range(1,args.epoch+1):
            GAE_model.train()

            z1_h, z1_l, z2_h, z2_l = GAE_model(X1, adj1, X2, adj2)      

            z1 = torch.cat([z1_h, z1_l],dim=1)
            z2 = torch.cat([z2_h, z2_l],dim=1)
            
            loss_fm, loss_map = GAE_model.DF_map(z1, z2, phi1, phi2, lam1, lam2, args)
            
            adj_pred1 = torch.sigmoid(torch.matmul(z1, z1.t()))
            adj_pred2 = torch.sigmoid(torch.matmul(z2, z2.t()))
            
            rec_loss1 = norm1 * F.binary_cross_entropy(adj_pred1.view(-1), adj_label1.to_dense().view(-1), weight=weight_tensor1)
            rec_loss2 = norm2 * F.binary_cross_entropy(adj_pred2.view(-1), adj_label2.to_dense().view(-1), weight=weight_tensor2)
                         
            l_FM = max(1.0 * (1 - step / args.epoch), 0.5) if args.use_fm_decay else 1       
            
            loss = rec_loss1 + rec_loss2 + l_FM *(loss_fm + loss_map)
          

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
            'loss': f'{loss:.4f}',
            })
            pbar.update()


    GAE_model.eval()
    with torch.no_grad():
        S_emb1_h, S_emb1_l, S_emb2_h, S_emb2_l = GAE_model(X1.to(device), adj1, X2.to(device), adj2)
            
        S_emb1 = torch.cat([S_emb1_h, S_emb1_l],dim=1)
        S_emb2 = torch.cat([S_emb2_h, S_emb2_l],dim=1)    
        S_emb1, S_emb2 = F.normalize(S_emb1, p=2, dim=1).detach(), F.normalize(S_emb2, p=2, dim=1).detach()

        D = torch.cdist(S_emb1, S_emb2, 2)
        test_idx, labels = extract_test_data(test_pairs, data)
        metrics = compute_hits(D, test_idx, labels)

    print("\n" + "="*60)
    print("FINAL RESULTS ")
    print("="*60)
    print(f"Hit@1:  {metrics[1]:.4f}")
    print(f"Hit@5:  {metrics[5]:.4f}")
    print(f"Hit@10: {metrics[10]:.4f}")
    print(f"Hit@50: {metrics[50]:.4f}")
    print("="*60)
    


def main(args):
    data = args.dataset
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    train_features, train_adj, test_pairs, modals_name = import_data(data,device, args.data_path)        
       
    gconv = GAE_GADL(args.num_hidden_layers, args.input_dim, args.hidden_dim, args.output_feature_size, residual=args.residual).to(device)
        
    encoder_model = Encoder_GAE_FM(encoder1=gconv, encoder2=gconv, hidden_dim=args.hidden_dim, k = args.k).to(device)    
            
    print("Fitting model")
    run_GADL(data, encoder_model, train_adj, train_features, args, device, test_pairs)
    

def parse_args():
    parser = argparse.ArgumentParser(description="GADL for real-world graphs")
    parser.add_argument('--device',type=str, help='GPU_id')
    parser.add_argument('--dataset', type=str,  help='ACM_DBLP, Douban Online_Offline')
    parser.add_argument('--k', type=int, help='number of basis functions', default = 300)
    parser.add_argument('--use_fm_decay', action='store_true', help='Enable linear decay for FM loss weight', default = True)
    parser.add_argument('--data_path', type=str, help='Path to dataset folder')
    parser.add_argument('--alpha', type=float, default=1e-3) 
    parser.add_argument('--beta', type=float, default=1e-2) 
    parser.add_argument('--l_bij', type=float, default=1e-1) 
    parser.add_argument('--l_ort', type=float, default=1e-1) 
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--output_feature_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=5e-4) 
    parser.add_argument('--residual', type=bool, help='residual connection in GConv')
    parser.add_argument('--epoch', type=int)


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args, unknown = load_config("config.yaml", args.dataset, args)
    main(args)
