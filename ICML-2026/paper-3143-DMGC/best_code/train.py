import argparse
import os.path
import numpy as np
import torch
import torch.nn.functional as F
from utils import load_data, normalize_weight, cal_homo_ratio, add_gaussian_noise
from models import MVHGC
from evaluation import eva
from settings import get_settings
import pandas as pd
import warnings
import os
from contextlib import contextmanager
import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
os.environ['KMEANS_VERBOSE'] = '0'
os.environ['TQDM_DISABLE'] = '1'



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    null_fd = open(os.devnull, 'w')
    sys.stdout = null_fd
    sys.stderr = null_fd
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null_fd.close()

def knn(features, queries, k=20):
    dists = torch.cdist(queries, features)  
    _, indices = torch.topk(dists, k, largest=False, sorted=True)
    nearest_features = features[indices]  
    
    return nearest_features

class AdaptiveTemperature:
    def __init__(self, initial_temp=0.5):
        self.temperature = initial_temp

    def update(self, features):
        feature_std = torch.std(features).item()
        feature_mean = torch.mean(features).item()

        adaptive_temp = self.temperature * (1 + 0.1 * feature_std + 0.05 * feature_mean)

        return adaptive_temp

def batch_contrastive_loss(f_x, f_x_plus, f_x_neg, temperature=1):
    if f_x.dim() == 1:
        f_x = f_x.unsqueeze(0)
    f_x = f_x.unsqueeze(1)
    f_x_plus_transposed = f_x_plus.transpose(1, 2)
    pos_similarities = torch.bmm(f_x, f_x_plus_transposed).squeeze(1) / temperature
    f_x_transposed = f_x.transpose(1, 2)
    neg_similarities = torch.bmm(f_x_neg, f_x_transposed).squeeze(1) / temperature

    pos_exp = torch.exp(pos_similarities)
    neg_exp = torch.exp(neg_similarities)
    if pos_exp.dim() == 1:
        pos_exp = pos_exp.unsqueeze(1)
    if neg_exp.dim() == 1:
        neg_exp = neg_exp.unsqueeze(1)
    pos_exp_sum = torch.sum(pos_exp, dim=1, keepdim=True)
    neg_exp_sum = torch.sum(neg_exp, dim=1, keepdim=True)
    denominator = pos_exp_sum + neg_exp_sum
    log_probs = torch.log(pos_exp_sum / denominator)
    loss = -torch.mean(log_probs)
    return loss

_GLOBAL_CLASS_NUM_HOLDER = [None]

def run_kmeans(data, n_clusters=None, n_init=5):
    global kmeans
    if 'kmeans' not in globals() or not callable(globals()['kmeans']):
        from kmeans_pytorch import kmeans as km_pytorch_kmeans
        kmeans = km_pytorch_kmeans

    if n_clusters is None:
        if _GLOBAL_CLASS_NUM_HOLDER[0] is not None:
            n_clusters = _GLOBAL_CLASS_NUM_HOLDER[0]
        else:
            raise ValueError("n_clusters must be specified or class_num must be globally set for run_kmeans")

    with suppress_output():
        cluster_ids, centers = kmeans(
            X=data,
            num_clusters=n_clusters,
            distance='euclidean',
            device=data.device,
        )

    return cluster_ids.cpu().numpy(), centers

for matadata in ['acm']:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=matadata, help='datasets: acm, dblp, texas, chameleon, wisconsin, cornell, imdb, acm00, acm01, acm02, acm03, acm04, acm05')
    parser.add_argument('--train', type=bool, default=True, help='training mode')
    parser.add_argument('--cuda_device', type=int, default=0, help='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='')
    args = parser.parse_args()

    dataset = args.dataset
    train = args.train
    cuda_device = args.cuda_device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    settings = get_settings(dataset)
    path = settings.path
    order = settings.order
    order_h = settings.order_h
    weight_soft = settings.weight_soft
    T0 = settings.T0
    hidden_dim = settings.hidden_dim
    latent_dim = settings.latent_dim
    epoch = settings.epoch
    patience = settings.patience
    lr = settings.lr
    weight_decay = settings.weight_decay
    update_interval = settings.update_interval
    random_seed = settings.random_seed
    tao = settings.tao
    beta_up0 = settings.beta_up0
    beta_low0 = settings.beta_low0
    fusion_logit_init = settings.fusion_logit_init
    sigma = settings.sigma
    set_random_seed(random_seed)

    labels, adjs_labels, shared_feature, shared_feature_label, graph_num = load_data(dataset, path)

    for v in range(graph_num):
        r = cal_homo_ratio(adjs_labels[v], labels, self_loop=True)
        print(r)
    print('dataset: {}'.format(dataset))
    print('nodes: {}'.format(shared_feature_label.shape[0]))
    print('features: {}'.format(shared_feature_label.shape[1]))
    print('class: {}'.format(labels.max() + 1))
    print('order: {}'.format(order))
    print('order_h: {}'.format(order_h))

    feat_dim = shared_feature.shape[1]
    class_num = labels.max().item() + 1
    _GLOBAL_CLASS_NUM_HOLDER[0] = class_num
    y = labels.cpu().numpy()

    xs = []
    for v in range(graph_num):
        xs.append(shared_feature_label)

    device = torch.device(f"cuda:{cuda_device}" if use_cuda else "cpu")

    model = MVHGC(
        feat_dim,
        hidden_dim,
        latent_dim,
        order,
        order_h,
        class_num=class_num,
        num_view=graph_num,
        num_nodes=shared_feature_label.shape[0],
        fusion_logit_init=fusion_logit_init,
    ).to(device)
    if use_cuda:
        torch.cuda.set_device(cuda_device)
        torch.cuda.manual_seed(random_seed)

    adjs_labels = [a.to(device) for a in adjs_labels]
    xs = [x.to(device) for x in xs]
    xs_n = [add_gaussian_noise(x, sigma) for x in xs]
    shared_feature = shared_feature.to(device)
    shared_feature_label = shared_feature_label.to(device)
    labels = labels.to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if train:
        print('Begin trains...')

        weights_h = []
        weighh = [1e-12 for i in range(graph_num)]
        weights_h = normalize_weight(weighh, p=weight_soft)

        with torch.no_grad():
            model.eval()

            x_preds, z_norms, A_recs, A_rec_norms, S_zs, Scores_dis, hs, h_all, qgs, adj_Ss, adj_Ss_rec, adj_Ss_rec_norm, alphas = model(
                xs_n,
                adjs_labels,
                weights_h,
                dataset,
                beta_up0, beta_low0
            )

            for v in range(graph_num):
                y_pred, centers = run_kmeans(hs[v], n_clusters=class_num)
                model.cluster_layer[v].data = centers.to(device)

            y_pred, centers = run_kmeans(hs[-1], n_clusters=class_num)
            model.cluster_layer[-1].data = centers.to(device)

            h_all = model.fuse_hs(hs, weights_h)

        bad_count = 0
        best_sum_metrics_val = 0.0
        acc_at_best_sum = 0.0
        nmi_at_best_sum = 0.0
        ari_at_best_sum = 0.0
        f1_at_best_sum = 0.0
        epoch_at_best_sum = 0
        h_all_eval_best = []
        losses = []

        adaptive_temp = AdaptiveTemperature(initial_temp=tao)

        for epoch_num in range(epoch):
            model.train()

            loss_kl = 0.
            loss_re_x =0.
            loss_kl_g = 0.
            kl_step = 1.
            kl_max = 10000
            loss = 0.
            l = 0.0
            loss_Xs_recovery = 0.0
            loss_Zs_recovery = 0.0
            loss_hs_recovery = 0.0
            loss_contrastive = 0.0
            loss_Adj = 0.0

            x_preds, z_norms, A_recs, A_rec_norms, S_zs, Scores_dis, hs, h_all, qgs, adj_Ss, adj_Ss_rec, adj_Ss_rec_norm, alphas = model(
                xs_n,
                adjs_labels,
                weights_h,
                dataset,
                beta_up0, beta_low0
            )

            with torch.no_grad():
                with suppress_output():
                    cluster_ids, _ = kmeans(
                        X=h_all,
                        num_clusters=class_num,
                        distance='euclidean',
                        device=h_all.device,
                    )
                    y_prim = cluster_ids.cpu().numpy()
                    pseudo_label = y_prim

                    for v in range(graph_num):
                        cluster_ids_v, _ = kmeans(
                            X=hs[v],
                            num_clusters=class_num,
                            distance='euclidean',
                            device=hs[v].device,
                        )
                        y_pred_v = cluster_ids_v.cpu().numpy()

                        a = eva(y_prim, y_pred_v, visible=False, metrics='acc')
                        weighh[v] = a

                weights_h = normalize_weight(weighh, p=weight_soft)

                with torch.no_grad():
                    with suppress_output():
                        cluster_ids_eval, cluster_centers_eval = kmeans(
                            X=h_all,
                            num_clusters=class_num,
                            distance='euclidean',
                            device=h_all.device,
                        )
                        y_eval_all = cluster_ids_eval.cpu().numpy()

            pgh = model.target_distribution(qgs[-1])

            for v in range(graph_num):
                loss_re_x += F.binary_cross_entropy(x_preds[v], xs[v])
                loss_Adj += weights_h[v] * F.mse_loss(hs[v], h_all) / graph_num
                loss_kl_g += weights_h[v] * F.kl_div(qgs[v].log(), pgh, reduction='batchmean')

            y_eval_all_t = torch.from_numpy(y_eval_all).to(device)

            if epoch_num > T0:
                loss_contrastive = 0.0
                contrastive_total_loss = 0.0
                cluster_centers = cluster_centers_eval.to(device)
                cluster_centers = torch.tensor(cluster_centers, requires_grad=True, device=device)
                current_temperature = adaptive_temp.update(h_all)

                for i in range(class_num):
                    class_mask = (y_eval_all_t == i)
                    class_samples = h_all[class_mask].to(device)
                    nearest_features = knn(h_all, class_samples, 20)
                    neg_mask = torch.ones(class_num, dtype=torch.bool)
                    neg_mask[i] = False
                    negative_centers = cluster_centers[neg_mask]

                    for i, sample in enumerate(class_samples):
                        contrastive_total_loss = contrastive_total_loss + batch_contrastive_loss(sample.unsqueeze(0), nearest_features[i].unsqueeze(0), negative_centers.unsqueeze(0), temperature=current_temperature) / (len(h_all) * (class_num - 1))
                loss_contrastive = loss_contrastive + contrastive_total_loss

            loss += 1 * loss_re_x + 1 * loss_Adj + 1 * loss_kl_g
            if epoch_num > T0:
                loss += 1 * loss_contrastive

            losses.append(loss.item())
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            print(
                f"Epoch {epoch_num:<4} | Loss: {loss.item():.4f} || BEST ACC: {acc_at_best_sum:.4f}, NMI: {nmi_at_best_sum:.4f}, ARI: {ari_at_best_sum:.4f}, F1: {f1_at_best_sum:.4f}, Ep: {epoch_at_best_sum}")

            if epoch_num % update_interval == 0:
                model.eval()
                with torch.no_grad():
                    x_preds_eval, _, _, _, _, _, hs_eval, h_all_eval, _, _, _, _, _ = model(
                        xs_n,
                        adjs_labels,
                        weights_h,
                        dataset,
                        beta_up0, beta_low0
                    )

                with torch.no_grad():
                    cluster_ids_eval_loop, _ = run_kmeans(
                        h_all_eval,
                        n_clusters=class_num
                    )
                    y_eval = cluster_ids_eval_loop

                nmi, acc, ari, f1 = eva(y, y_eval, str(epoch_num) + 'Kz', visible=False)

                current_sum = acc + nmi + ari + f1
                if current_sum > best_sum_metrics_val:
                    best_sum_metrics_val = current_sum
                    acc_at_best_sum = acc
                    nmi_at_best_sum = nmi
                    ari_at_best_sum = ari
                    f1_at_best_sum = f1
                    h_all_eval_best = h_all_eval
                    epoch_at_best_sum = epoch_num
                    bad_count = 0
                else:
                    bad_count += 1

                if bad_count >= patience:
                    print('Early stopping. Training complete.')
                    print('Final Result: (ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}, Sum: {:.4f}) achieved at epoch: {}'.format(
                        acc_at_best_sum, nmi_at_best_sum, ari_at_best_sum, f1_at_best_sum, best_sum_metrics_val, epoch_at_best_sum))
                    print()
                    break

        print("Saving final best results to result.csv...")
        columns = ['dataset', 'acc', 'nmi', 'ari', 'f1', 'epoch', 'lr', 'weight_decay', 'order', 'order_h', 'hidden_dim', 'latent_dim', 'weight_soft', 'tao', 'sigma', 'beta_up', 'beta_low']

        acc_rounded = round(acc_at_best_sum, 4)
        nmi_rounded = round(nmi_at_best_sum, 4)
        ari_rounded = round(ari_at_best_sum, 4)
        f1_rounded = round(f1_at_best_sum, 4)

        dt = np.asarray(
            [dataset, acc_rounded, nmi_rounded, ari_rounded, f1_rounded, epoch_at_best_sum, lr, weight_decay, order, order_h, hidden_dim, latent_dim, weight_soft, tao, sigma, beta_up0, beta_low0]
        ).reshape(1, -1)
        df = pd.DataFrame(dt, columns=columns)

        file_exists = os.path.exists('./result.csv')
        if not file_exists:
            df.to_csv('./result.csv', index=False, header=True, mode='w')
            print("Created new result.csv file with header")
        else:
            df.to_csv('./result.csv', index=False, header=False, mode='a')
            print("Appended results to existing result.csv file")

        print(f"Final best results saved: ACC={acc_rounded:.4f}, NMI={nmi_rounded:.4f}, ARI={ari_rounded:.4f}, F1={f1_rounded:.4f}")

    acc_rounded = round(acc_at_best_sum, 4) if acc_at_best_sum > 1e-12 else 0.0

    if not train:
        print("Warning, not training.")
    else:
        if acc_at_best_sum > 1e-12 :
            model_name = 'NGCE_{}_acc{:.4f}'.format(dataset, acc_rounded)
        else:
            print("Warning: No best model seemed to be saved during training based on acc_at_best_sum. Loading may fail or use a default.")
            model_name = 'NGCE_{}_acc{:.4f}'.format(dataset, 0.0)

    print('Test complete...')
