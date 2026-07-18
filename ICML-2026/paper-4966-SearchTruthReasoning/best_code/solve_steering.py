import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from transformers import AutoConfig
from collections import defaultdict
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.covariance import LedoitWolf
from sklearn.manifold import TSNE

SEED = 42
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def load_data(pkl_path):
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        return pkl.load(f)
    
class SteeringAnalyzer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads

    def group_data(self, data, layer_idx, activation_key, head_idx=None):
        groups = defaultdict(lambda: {'pos': [], 'neg': []})
        for d in data:
            key = d['problem_id'], d['chunk_idx']
            activation = d[activation_key][layer_idx]
            if head_idx is not None:
                activation = activation[head_idx * self.head_dim : (head_idx + 1) * self.head_dim]
            if d['score'] == 1.0:
                groups[key]['pos'].append(activation)
            elif d['score'] == 0.0:
                groups[key]['neg'].append(activation)
            
        data_list = []
        for key, values in groups.items():
            if values['pos'] and values['neg']:
                data_list.append({
                    "steering_vector": np.mean(values['pos'], axis=0) - np.mean(values['neg'], axis=0),
                    "pos_samples": np.array(values['pos']),
                    "neg_samples": np.array(values['neg'])
                })
        if layer_idx == 0 and (head_idx is None or head_idx == 0):
            print(f"Total forks: {len(data_list)}")
            print(f"Total positive samples: {sum(len(d['pos_samples']) for d in data_list)}")
            print(f"Total negative samples: {sum(len(d['neg_samples']) for d in data_list)}")
        return data_list

    def compute_lda_direction(self, pos, neg):
        """w = Sigma^-1 * (mu_p - mu_n)"""
        mu_p, mu_n = np.mean(pos, axis=0), np.mean(neg, axis=0)

        s_w = LedoitWolf().fit(pos).covariance_ + LedoitWolf().fit(neg).covariance_

        reg_term = 1e-4 * np.trace(s_w) / s_w.shape[0]
        s_w_reg = s_w + reg_term * np.eye(s_w.shape[0])

            
        w = np.linalg.solve(s_w + reg_term * np.eye(s_w.shape[0]), (mu_p - mu_n))

        fdr_score = ((w.T @ (mu_p - mu_n)) ** 2) / (w.T @ s_w_reg @ w + 1e-8)

        return w / (np.linalg.norm(w) + 1e-8), fdr_score

    def compute_probe_accuracy(self, grouped_data):
        steering_vectors_norm = normalize(np.array([d['steering_vector'] for d in grouped_data]))
        labels = KMeans(n_clusters=self.args.n_clusters, random_state=SEED).fit(steering_vectors_norm).labels_
        probe_accs, weights = [], []
        for c in range(self.args.n_clusters):
            idx = np.where(labels == c)[0]
            p = np.concatenate([grouped_data[i]['pos_samples'] for i in idx])
            n = np.concatenate([grouped_data[i]['neg_samples'] for i in idx])

            X, y = np.vstack([p, n]), np.array([1]*len(p) + [0]*len(n))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

            clf = LogisticRegression(random_state=SEED, max_iter=2000, class_weight='balanced').fit(X_train, y_train)
            acc = clf.score(X_test, y_test)

            weights.append(len(idx))
            probe_accs.append(acc)
        return np.average(probe_accs, weights=weights)

    
    def get_unit_config(self, grouped_data, layer_idx, head_idx, act_type):
        steering_vectors_norm = normalize(np.array([d['steering_vector'] for d in grouped_data]))
        labels = KMeans(n_clusters=self.args.n_clusters, random_state=SEED).fit(steering_vectors_norm).labels_
        
        if act_type == "attention":
            prefix = f"Attn_L{layer_idx}_H{head_idx}"
        elif act_type == "layer":
            prefix = f"Res_L{layer_idx}"
        elif act_type == "mlp":
            prefix = f"MLP_L{layer_idx}"

        unit_cfg = {
            "id": prefix,
            "layer": layer_idx,
            "head": head_idx,
            "type": act_type,
            "clusters": []
        }

        probe_accs, fdr_scores, weights = [], [], []
        for c in range(self.args.n_clusters):
            idx = np.where(labels == c)[0]
            p = np.concatenate([grouped_data[i]['pos_samples'] for i in idx])
            n = np.concatenate([grouped_data[i]['neg_samples'] for i in idx])

            X, y = np.vstack([p, n]), np.array([1]*len(p) + [0]*len(n))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

            p_train = X_train[y_train == 1]
            n_train = X_train[y_train == 0]
            w_lda, _ = self.compute_lda_direction(p_train, n_train)

            p_test = X_test[y_test == 1]
            n_test = X_test[y_test == 0]
            p_test_lda = p_test @ w_lda
            n_test_lda = n_test @ w_lda
            mu_p_t, mu_n_t = np.mean(p_test_lda), np.mean(n_test_lda)
            var_p_t, var_n_t = np.var(p_test_lda), np.var(n_test_lda)
            test_fdr = ((mu_p_t - mu_n_t) ** 2) / (var_p_t + var_n_t + 1e-8)

            clf = LogisticRegression(random_state=SEED, max_iter=2000, class_weight='balanced').fit(X_train, y_train)
            acc = clf.score(X_test, y_test)

            weights.append(len(idx))
            fdr_scores.append(test_fdr)
            probe_accs.append(acc)
    
            w_lda_all = self.compute_lda_direction(p, n)[0]
            p_proj = p @ w_lda_all

            unit_cfg["clusters"].append({
                "cluster_id": c,
                "cluster_size": len(idx),
                "lda_fdr": test_fdr,
                "probe_acc": acc,
                "mu_p": p_proj.mean(),  
                "std_p": p_proj.std(),
                "w_lda": w_lda_all
            })
        unit_cfg["lda_fdr"] = np.average(fdr_scores, weights=weights)
        unit_cfg["probe_acc"] = np.average(probe_accs, weights=weights)
        return unit_cfg

    def run_analysis(self, data):
        all_exps = [("attention", "attention_activations"), 
                    ("mlp", "mlp_activations"), 
                    ("layer", "layer_activations")]
        total_steering_configs = {
            "attention": [],
            "mlp": [],
            "layer": []
        }
        for act_type, act_key in all_exps:
            print(f"\n>> Analyzing {act_type}...")
            is_attn = (act_type == "attention")
            unit_results = []
            if act_type == "mlp" or act_type == "layer":
                for_filter = []
                for layer_idx in tqdm(range(self.n_layers), desc=f"Layers ({act_type})" ):
                    grouped_data = self.group_data(data, layer_idx, act_key)
                    probe_acc = self.compute_probe_accuracy(grouped_data)
                    for_filter.append({
                        "layer": layer_idx,
                        "probe_acc": probe_acc
                    })
                top_layers = sorted(for_filter, key=lambda x: x['probe_acc'], reverse=True)[:3]
                for layer_info in tqdm(top_layers, desc=f"Top Layers ({act_type})" ):
                    layer_idx = layer_info['layer']
                    grouped_data = self.group_data(data, layer_idx, act_key)
                    cfg = self.get_unit_config(grouped_data, layer_idx, None, act_type)
                    unit_results.append(cfg)
                
            else:
                for layer_idx in tqdm(range(self.n_layers), desc=f"Layers ({act_type})" ):
                    for head_idx in (range(self.n_heads) if is_attn else [None]):
                        grouped_data = self.group_data(data, layer_idx, act_key, head_idx)
                        cfg = self.get_unit_config(grouped_data, layer_idx, head_idx, act_type)
                        unit_results.append(cfg)

            total_steering_configs[act_type] = sorted(unit_results, key=lambda x: x['lda_fdr'], reverse=True)

            if is_attn:
                h_map = np.zeros((self.n_layers, self.n_heads))
                for r in unit_results: 
                    h_map[r['layer'], r['head']] = r['probe_acc']
                plt.figure(figsize=(10, 8))
                sns.heatmap(h_map, cmap="coolwarm")
                plt.savefig(os.path.join(self.args.output_dir, f"{act_type}" + "_acc_heatmap.png"))
                plt.close()
            else:
                accs = {r['layer']: r['probe_acc'] for r in unit_results}
                plt.figure(figsize=(10, 5))
                plt.plot([accs.get(i, 0.5) for i in range(self.n_layers)], marker='o', color='#C44E52')
                plt.savefig(os.path.join(self.args.output_dir, f"{act_type}" + "_acc_line.png"))
                plt.close()
            
            if is_attn:
                h_map = np.zeros((self.n_layers, self.n_heads))
                for r in unit_results: 
                    h_map[r['layer'], r['head']] = r['lda_fdr']
                plt.figure(figsize=(10, 8))
                sns.heatmap(h_map, cmap="coolwarm")
                plt.savefig(os.path.join(self.args.output_dir, f"{act_type}" + "_fdr_heatmap.png"))
                plt.close()
            else:
                fdrs = {r['layer']: r['lda_fdr'] for r in unit_results}
                plt.figure(figsize=(10, 5))
                plt.plot([fdrs.get(i, 0.0) for i in range(self.n_layers)], marker='o', color='#C44E52')
                plt.savefig(os.path.join(self.args.output_dir, f"{act_type}" + "_fdr_line.png"))
                plt.close()
            
            
            top_units = sorted(unit_results, key=lambda x: x['probe_acc'], reverse=True)[:5]
            for unit in top_units:
                grouped_data = self.group_data(data, unit['layer'], act_key, unit['head'])
                steering_vectors_norm = normalize(np.array([d['steering_vector'] for d in grouped_data]))
                labels = KMeans(n_clusters=self.args.n_clusters, random_state=SEED).fit(steering_vectors_norm).labels_
                
                for cluster in unit['clusters']:
                    c_id = cluster['cluster_id']
                    idx = np.where(labels == c_id)[0]
                    p = np.concatenate([grouped_data[i]['pos_samples'] for i in idx])
                    n = np.concatenate([grouped_data[i]['neg_samples'] for i in idx])

                    v = p.mean(axis=0) - n.mean(axis=0)
                    v_norm = v / (np.linalg.norm(v) + 1e-8)
                    p_norm = normalize(p)
                    n_norm = normalize(n)

                    p_cos = p_norm @ v_norm
                    n_cos = n_norm @ v_norm

                    w_lda = cluster['w_lda']
                    p_lda = p @ w_lda
                    n_lda = n @ w_lda
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    sns.kdeplot(p_cos, label="Pos", fill=True, color="#55A868", ax=ax1)
                    sns.kdeplot(n_cos, label="Neg", fill=True, color="#C44E52", ax=ax1)
                    sns.kdeplot(p_lda, label="Pos", fill=True, color="#55A868", ax=ax2)
                    sns.kdeplot(n_lda, label="Neg", fill=True, color="#C44E52", ax=ax2)
                    ax1.set_title("Cosine Dist (Original)")
                    ax2.set_title("Fisher-LDA Dist (Optimized)")
                    plt.savefig(os.path.join(self.args.output_dir, f"KDE_{unit['id']}_C{cluster['cluster_id']}.png"))
                    plt.close()

                
                tsne = TSNE(n_components=2, random_state=SEED)
                tsne_results = tsne.fit_transform(steering_vectors_norm)
                plt.figure(figsize=(8, 6))
                for c in range(self.args.n_clusters):
                    idx = np.where(labels == c)[0]
                    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f"Cluster {c}, Size: {len(idx)}", alpha=0.6)
                plt.title(f"t-SNE of Steering Vectors - {unit['id']}")
                plt.xlabel("t-SNE Dimension 1")
                plt.ylabel("t-SNE Dimension 2")
                plt.legend()
                plt.savefig(os.path.join(self.args.output_dir, f"{unit['id']}_tsne.png"))
                plt.close()
        
        output_path = os.path.join(args.output_dir, f"steering_configs.pkl")
        with open(output_path, 'wb') as f:
            pkl.dump(total_steering_configs, f)
        print(f"\nSteering configurations saved to {output_path}")  

                
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to the activations pickle file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for configuration")
    parser.add_argument("--n_clusters", type=int, default=1, help="Number of clusters for kMeans")
    args = parser.parse_args()

    args.base_dir = "/".join("/".join(args.pkl_path.split("/")[:-1]).split("/")[:-1])
    args.output_dir = os.path.join(args.base_dir, "steering_configs", f"c{args.n_clusters}")
    os.makedirs(args.output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(args.model)

    analyzer = SteeringAnalyzer(args, config)
    data = load_data(args.pkl_path)
    analyzer.run_analysis(data)
