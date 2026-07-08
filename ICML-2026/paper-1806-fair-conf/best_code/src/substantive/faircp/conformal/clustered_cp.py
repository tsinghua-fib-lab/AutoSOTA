import numpy as np
import math
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json  
from pathlib import Path
from internal.util.writer import Writer

@dataclass
class ClusterConfig:
    n_clusters: int = 3
    min_points_per_key: int = 10
    clustering_ratio: float = 0.5
    random_state: int = 42
    embedding_mode: str = 'upper_percentiles'
    summary_bins: int = 50

def split_calibration_data(scores: np.ndarray, bins: np.ndarray, ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n = len(scores)
    indices = np.random.permutation(n)
    split = math.floor(n * ratio)
    return scores[indices[:split]], bins[indices[:split]], scores[indices[split:]], bins[indices[split:]]

def compute_quantile_embeddings(scores: np.ndarray, bins: np.ndarray, keys: np.ndarray, alpha: float, embedding_mode: str, summary_bins: int = 50) -> Dict[int, np.ndarray]:
    embeddings = {}
    
    if embedding_mode == 'upper_percentiles':
        # Paper specification: 0.5, 0.6, 0.7, 0.8, 0.9, and (1-alpha)
        base_quantiles = [0.5, 0.6, 0.7, 0.8, 0.9]
        target_quantile = 1 - alpha
        # Remove duplicates and sort
        quantiles = sorted(list(set(base_quantiles + [target_quantile])))
    elif embedding_mode == 'cdf_grid':
        # Use uniform grid points for empirical CDF
        quantiles = np.linspace(0.0, 1.0, summary_bins)
    else:
        raise ValueError(f"Unknown embedding_mode: {embedding_mode}")
    
    for key in keys:
        key_scores = scores[bins == key]
        if len(key_scores) > 0:
            embeddings[key] = np.quantile(key_scores, quantiles)
        else:
            embeddings[key] = np.zeros(len(quantiles))
    
    return embeddings

def weighted_kmeans(embeddings: Dict[int, np.ndarray], weights: Dict[int, float], n_clusters: int, seed: int) -> Dict[int, int]:
    keys = list(embeddings.keys())
    if len(keys) == 0:
        return {}
    X = np.array([embeddings[key] for key in keys])
    sample_weights = np.array([weights.get(key, 1.0) for key in keys])
    n_clusters_eff = min(n_clusters, len(keys))
    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=seed, n_init=10)
    labels = kmeans.fit(X, sample_weight=sample_weights).labels_
    return {keys[i]: labels[i] for i in range(len(keys))}

def compute_cluster_quantiles(scores: np.ndarray, bins: np.ndarray, key_to_cluster: Dict[int, int], alpha: float, n_clusters: int) -> Dict[int, float]:
    """Compute (1-alpha)-quantile using alpha directly."""
    quantiles = {}
    null_id = n_clusters
    
    for cid in range(n_clusters):
        keys_in_cluster = [k for k, c in key_to_cluster.items() if c == cid]
        if keys_in_cluster:
            cluster_scores = scores[np.isin(bins, keys_in_cluster)]
            quantiles[cid] = np.quantile(cluster_scores, 1-alpha, method='higher') if len(cluster_scores) > 0 else np.quantile(scores, 1-alpha, method='higher')
        else:
            quantiles[cid] = np.quantile(scores, 1-alpha, method='higher')
    
    quantiles[null_id] = np.quantile(scores, 1-alpha, method='higher')
    return quantiles

def clustered_cp_class(scores_calib: np.ndarray, scores_test: np.ndarray, labels_calib: np.ndarray, config: ClusterConfig, alpha: float, writer: Writer, label_map: Dict[int, str]) -> List[np.ndarray]:
    cluster_scores, cluster_labels, proper_scores, proper_labels = split_calibration_data(scores_calib, labels_calib, config.clustering_ratio, config.random_state)
    
    unique_classes = np.array(list(label_map.keys()))
    embeddings = compute_quantile_embeddings(cluster_scores, cluster_labels, unique_classes, alpha, config.embedding_mode, config.summary_bins)
    counts = {cls: np.sum(cluster_labels == cls) for cls in unique_classes}

    valid_classes = [cls for cls, count in counts.items() if count >= config.min_points_per_key]
    valid_embeddings = {cls: embeddings[cls] for cls in valid_classes}
    valid_weights = {cls: max(1, counts[cls]) for cls in valid_classes}

    if valid_classes:
        key_to_cluster = weighted_kmeans(valid_embeddings, valid_weights, config.n_clusters, config.random_state)
    else:
        key_to_cluster = {}
    
    for cls, count in counts.items():
        if count < config.min_points_per_key:
            key_to_cluster[cls] = config.n_clusters

    # Print clustering assignments
    print("\n" + "="*50)
    print("CLUSTERED CONFORMAL PREDICTION - LABEL CLUSTERING")
    print("="*50)
    print(f"Unique classes: {sorted(unique_classes)} -> {[label_map[cls] for cls in sorted(unique_classes)]}")
    print(f"Total classes: {len(unique_classes)}")
    print(f"Number of clusters: {config.n_clusters}")
    print(f"Clustering ratio (gamma_label): {config.clustering_ratio}")
    print(f"Null cluster ID: {config.n_clusters}")
    print(f"Min points per cluster: {config.min_points_per_key}")
    
    # Group labels by cluster
    cluster_assignments = {}
    for cls, cluster_id in key_to_cluster.items():
        if cluster_id not in cluster_assignments:
            cluster_assignments[cluster_id] = []
        cluster_assignments[cluster_id].append((cls, counts[cls]))
    
    # Print cluster assignments
    for cluster_id in sorted(cluster_assignments.keys()):
        members = cluster_assignments[cluster_id]
        cluster_name = f"Null Cluster" if cluster_id == config.n_clusters else f"Cluster {cluster_id}"
        print(f"\n{cluster_name}:")
        for cls, count in sorted(members):
            print(f"  Label {cls} ({label_map.get(cls, 'Unknown label')}): {count} samples")
        total_samples = sum(count for _, count in members)
        print(f"  Total: {total_samples} samples")
    
    
    clustering_info = {
        "type": "label_clustering",
        "unique_classes": sorted(unique_classes.tolist()),
        "class_names": {str(cls): label_map[cls] for cls in unique_classes},  
        "n_classes": len(unique_classes),
        "n_clusters": config.n_clusters,
        "null_cluster_id": config.n_clusters,
        "min_points_per_key": config.min_points_per_key,
        "cluster_assignments": {
            str(cluster_id): [
                {"label": int(cls), "label_name": label_map.get(cls, "Unknown label"), "count": int(count)} 
                for cls, count in members
            ]
            for cluster_id, members in cluster_assignments.items()
        },
        "key_to_cluster_mapping": {str(k): int(v) for k, v in key_to_cluster.items()}
    }

    writer.write_json("label_clustering_assignments", clustering_info)
        
    cluster_quantiles = compute_cluster_quantiles(proper_scores, proper_labels, key_to_cluster, alpha, config.n_clusters)
    
    pred_sets = []
    class_to_idx = {cls: i for i, cls in enumerate(sorted(unique_classes))}

    for i, test_scores in enumerate(scores_test):  
        pred_set = []
        for cls in unique_classes:
            threshold = cluster_quantiles[key_to_cluster[cls]]
            score_idx = class_to_idx[cls]
            if test_scores[score_idx] <= threshold:
                pred_set.append(cls)
        
        if len(pred_set) == 0:
            best_score_idx = np.argmin(test_scores[:len(unique_classes)])  
            best_class = sorted(unique_classes)[best_score_idx]
            pred_set = [best_class]

        pred_sets.append(np.array(pred_set))
    
    return pred_sets

def clustered_cp_group(scores_calib: np.ndarray, scores_test: np.ndarray, groups_calib: np.ndarray, groups_test: np.ndarray, config: ClusterConfig, alpha: float, writer: Writer, label_map: Dict[int, str], group_map: Dict[int, str]) -> List[np.ndarray]:
    cluster_scores, cluster_groups, proper_scores, proper_groups = split_calibration_data(scores_calib, groups_calib, config.clustering_ratio, config.random_state)
    
    unique_groups = np.array(list(group_map.keys()))
    unique_classes = np.array(list(label_map.keys()))
    embeddings = compute_quantile_embeddings(cluster_scores, cluster_groups, unique_groups, alpha, config.embedding_mode, config.summary_bins)
    counts = {grp: np.sum(cluster_groups == grp) for grp in unique_groups}
    
    valid_groups = [grp for grp, count in counts.items() if count >= config.min_points_per_key]
    valid_embeddings = {grp: embeddings[grp] for grp in valid_groups}
    valid_weights = {grp: max(1, counts[grp]) for grp in valid_groups}
    
    if valid_groups:
        group_to_cluster = weighted_kmeans(valid_embeddings, valid_weights, config.n_clusters, config.random_state)
    else:
        group_to_cluster = {}
    
    for grp, count in counts.items():
        if count < config.min_points_per_key:
            group_to_cluster[grp] = config.n_clusters

    # Print clustering assignments
    print("\n" + "="*50)
    print("CLUSTERED CONFORMAL PREDICTION - GROUP CLUSTERING")
    print("="*50)
    print(f"Unique groups: {sorted(unique_groups)} -> {[group_map.get(grp, 'Unknown group') for grp in sorted(unique_groups)]}")
    print(f"Total groups: {len(unique_groups)}")
    print(f"Number of clusters: {config.n_clusters}")
    print(f"Clustering ratio (gamma_group): {config.clustering_ratio}")
    print(f"Null cluster ID: {config.n_clusters}")
    print(f"Min points per cluster: {config.min_points_per_key}")
    
    # Group assignments by cluster
    cluster_assignments = {}
    for grp, cluster_id in group_to_cluster.items():
        if cluster_id not in cluster_assignments:
            cluster_assignments[cluster_id] = []
        cluster_assignments[cluster_id].append((grp, counts[grp]))
    
    # Print cluster assignments
    for cluster_id in sorted(cluster_assignments.keys()):
        members = cluster_assignments[cluster_id]
        cluster_name = f"Null Cluster" if cluster_id == config.n_clusters else f"Cluster {cluster_id}"
        print(f"\n{cluster_name}:")
        for grp, count in sorted(members):
            print(f"  Group {grp} ({group_map.get(grp, 'Unknown group')}): {count} samples")
        total_samples = sum(count for _, count in members)
        print(f"  Total: {total_samples} samples")
    
    clustering_info = {
        "type": "group_clustering",
        "unique_groups": sorted(unique_groups.tolist()),
        "group_names": {str(grp): group_map.get(grp, "Unknown group") for grp in unique_groups},  
        "n_groups": len(unique_groups),
        "n_clusters": config.n_clusters,
        "null_cluster_id": config.n_clusters,
        "min_points_per_key": config.min_points_per_key,
        "cluster_assignments": {
            str(cluster_id): [
                {"group": int(grp), "group_name": group_map.get(grp, "Unknown group"), "count": int(count)} 
                for grp, count in members
            ]
            for cluster_id, members in cluster_assignments.items()
        },
        "group_to_cluster_mapping": {str(k): int(v) for k, v in group_to_cluster.items()}
    }

    writer.write_json("group_clustering_assignments", clustering_info)
        
    cluster_quantiles = compute_cluster_quantiles(proper_scores, proper_groups, group_to_cluster, alpha, config.n_clusters)
    
    pred_sets = []

    class_to_idx = {cls: i for i, cls in enumerate(sorted(unique_classes))}
    for i, test_scores in enumerate(scores_test):
        test_group = groups_test[i]
        threshold = cluster_quantiles[group_to_cluster.get(test_group, config.n_clusters)]
        pred_set = []
        for cls in unique_classes:
            score_idx = class_to_idx[cls]  
            if test_scores[score_idx] <= threshold:
                pred_set.append(cls)
        
        if len(pred_set) == 0:
            best_score_idx = np.argmin(test_scores[:len(unique_classes)])
            best_class = sorted(unique_classes)[best_score_idx]
            pred_set = [best_class]
        
        pred_sets.append(np.array(pred_set))

    return pred_sets