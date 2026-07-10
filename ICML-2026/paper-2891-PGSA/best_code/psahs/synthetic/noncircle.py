import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import networkx as nx
import random
import pickle
import os
from typing import Tuple, Optional, List, Dict

# --------- 边列表转邻接矩阵和边索引 ---------
def edges_to_adj(edge_list, num_nodes):
    """将边列表转换为邻接矩阵和边索引"""
    adj = np.zeros((num_nodes, num_nodes))
    edge_index = [[], []]
    for u, v in edge_list:
        adj[u, v] = 1
        adj[v, u] = 1
        edge_index[0].append(u)
        edge_index[1].append(v)
    edge_index = torch.LongTensor(edge_index)
    return adj, edge_index

# --------- 计算同质比 ---------
def compute_global_homophily(G) -> float:
    """计算图的全局同质比：同类边数 / 总边数"""
    if G.number_of_edges() == 0:
        return 1.0
    homophilic_edges = 0
    for u, v in G.edges():
        if "y" not in G.nodes[u] or "y" not in G.nodes[v]:
            continue
        if G.nodes[u]["y"] == G.nodes[v]["y"]:
            homophilic_edges += 1
    return homophilic_edges / G.number_of_edges()

# --------- 生成三分类SBM图 ---------
def _generate_sbm_three_class(n_per_class: int, p: float, q: float, 
                             scale: Optional[float] = None, seed: int = 0) -> nx.Graph:
    """
    创建三分类的随机块模型图
    
    参数:
    - n_per_class: 每个类别的节点数
    - p: 类别内连接概率
    - q: 类别间连接概率
    - scale: 缩放因子，用于调整图密度
    - seed: 随机种子
    
    返回:
    - 三分类的NetworkX图
    """
    # 创建三分类块矩阵
    B = [
        [p, q, q],
        [q, p, q],
        [q, q, p]
    ]
    
    # 应用缩放因子
    if scale is not None:
        B = [[x * scale for x in row] for row in B]
    
    # 块大小
    sizes = [n_per_class, n_per_class, n_per_class]
    
    # 生成随机块模型图
    G = nx.stochastic_block_model(sizes, B, seed=seed)
    
    # 为节点添加类别标签
    y = {}
    for cls, part in enumerate(G.graph["partition"]):
        for u in part:
            y[u] = cls
    
    nx.set_node_attributes(G, y, "y")
    return G

# --------- 度数保持重连调整同质比（三分类版本） ---------
def degree_preserving_rewire_to_H_three_class(G: nx.Graph, H_target: float, 
                                             seed: int = 1, max_tries: int = 1_000_000) -> nx.Graph:
    """
    通过度数保持的重连操作调整三分类图的同质比
    
    参数:
    - G: 输入图
    - H_target: 目标同质比
    - seed: 随机种子
    - max_tries: 最大尝试次数
    
    返回:
    - 重连后的图
    """
    if H_target == 1.0:
        return G
    
    # 设置随机种子
    rnd = random.Random(seed)
    
    # 获取三个类别的节点
    C0 = {u for u, d in G.nodes(data=True) if "y" in d and d["y"] == 0}
    C1 = {u for u, d in G.nodes(data=True) if "y" in d and d["y"] == 1}
    C2 = {u for u, d in G.nodes(data=True) if "y" in d and d["y"] == 2}
    
    # 获取同类边
    E00 = [(u, v) for u, v in G.edges() if (u in C0 and v in C0)]
    E11 = [(u, v) for u, v in G.edges() if (u in C1 and v in C1)]
    E22 = [(u, v) for u, v in G.edges() if (u in C2 and v in C2)]
    
    # 计算当前和目标异类边数
    M = G.number_of_edges()
    inter_now = sum(
        1
        for u, v in G.edges()
        if "y" in G.nodes[u] and "y" in G.nodes[v] and G.nodes[u]["y"] != G.nodes[v]["y"]
    )
    inter_target = int(round((1 - H_target) * M))
    delta = inter_target - inter_now
    
    if delta <= 0:
        return G

    # 计算需要交换的次数
    swaps_needed = (delta + 1) // 2
    
    # 检查是否有足够的边进行交换
    intra_edges = [E00, E11, E22]
    min_intra_edges = min(len(edges) for edges in intra_edges)
    if min_intra_edges < swaps_needed:
        print(f"Warning: Not enough intra-class edges, need {swaps_needed}, available {min_intra_edges}")
        swaps_needed = min_intra_edges

    edges_set = set(G.edges())
    made, tries = 0, 0
    
    def try_swap(e0, e1):
        """尝试交换两条边"""
        a, b = e0
        c, d = e1
        cand1, cand2 = (a, d), (c, b)
        
        for x, y in (cand1, cand2):
            if x == y:  # 自环
                return False
            if "y" not in G.nodes[x] or "y" not in G.nodes[y] or G.nodes[x]["y"] == G.nodes[y]["y"]:
                return False  # 同类边
            if (x, y) in edges_set or (y, x) in edges_set:
                return False  # 已存在边
        
        # 执行交换
        G.remove_edge(*e0)
        G.remove_edge(*e1)
        G.add_edge(*cand1)
        G.add_edge(*cand2)
        
        # 更新边集合
        edges_set.discard(e0)
        edges_set.discard(e1)
        edges_set.discard((e0[1], e0[0]))
        edges_set.discard((e1[1], e1[0]))
        
        edges_set.add(cand1)
        edges_set.add(cand2)
        edges_set.add((cand1[1], cand1[0]))
        edges_set.add((cand2[1], cand2[0]))
        
        return True

    # 使用numpy的随机数生成器提高性能
    rng = np.random.default_rng(seed)
    
    # 将边列表转换为数组以提高随机选择效率
    E00_arr = np.array(E00)
    E11_arr = np.array(E11)
    E22_arr = np.array(E22)
    
    # 所有同类边数组的列表
    intra_edges_arr = [E00_arr, E11_arr, E22_arr]
    
    while made < swaps_needed and tries < max_tries:
        # 随机选择两个不同的类别
        cls_pair = rng.choice(3, size=2, replace=False)
        cls1, cls2 = cls_pair[0], cls_pair[1]
        
        # 获取这两个类别的边数组
        edges1 = intra_edges_arr[cls1]
        edges2 = intra_edges_arr[cls2]
        
        # 如果某个类别的边列表为空，跳过
        if len(edges1) == 0 or len(edges2) == 0:
            tries += 1
            continue
            
        # 随机选择边
        e0_idx = rng.integers(0, len(edges1))
        e1_idx = rng.integers(0, len(edges2))
        e0 = tuple(edges1[e0_idx])
        e1 = tuple(edges2[e1_idx])
        
        # 检查边是否仍然存在
        if e0 not in edges_set or e1 not in edges_set:
            # 更新边列表
            intra_edges_arr[cls1] = np.array([e for e in intra_edges_arr[cls1] if tuple(e) in edges_set])
            intra_edges_arr[cls2] = np.array([e for e in intra_edges_arr[cls2] if tuple(e) in edges_set])
            tries += 1
            continue
            
        if try_swap(e0, e1):
            made += 1
            # 移除已使用的边
            intra_edges_arr[cls1] = np.delete(intra_edges_arr[cls1], e0_idx, axis=0)
            intra_edges_arr[cls2] = np.delete(intra_edges_arr[cls2], e1_idx, axis=0)
        tries += 1

    if made < swaps_needed:
        print(f"Warning: Only made {made}/{swaps_needed} swaps after {tries} attempts")
    return G

# # --------- 椭圆形特征生成（三分类版本） ---------
# def generate_elliptical_features(G: nx.Graph, means, SIGMA: float, feature_dim: int = 10, 
#                                 ellipse_ratio: float = 5.0, rotation_angle: float = 0.0, seed: int = 0):
#     """
#     生成椭圆形分布的特征（非球形高斯）
    
#     参数:
#     - ellipse_ratio: 椭圆长短轴比例，越大表示分布越扁
#     - rotation_angle: 旋转角度（弧度），使椭圆方向不同
#     """
#     np.random.seed(seed)
#     num_nodes = G.number_of_nodes()
#     features = np.zeros((num_nodes, feature_dim))
#     label = np.zeros(num_nodes, dtype=np.int64)
#     means_arr = np.asarray(means, dtype=float)
    
#     # 扩展均值到高维（如果需要）
#     if means_arr.shape[1] < feature_dim:
#         extended_means = np.zeros((means_arr.shape[0], feature_dim))
#         for i in range(means_arr.shape[0]):
#             extended_means[i, :means_arr.shape[1]] = means_arr[i]
#             extended_means[i, means_arr.shape[1]:] = np.random.uniform(-0.5, 0.5, feature_dim - means_arr.shape[1])
#         means_arr = extended_means
    
#     # 创建基础对角协方差矩阵
#     base_diag = np.ones(feature_dim) * (SIGMA ** 2)
    
#     # 为不同类别创建不同的椭圆形状
#     for cls, part in enumerate(G.graph["partition"]):
#         n_cls = len(part)
#         if n_cls > 0:
#             # 为每个类别创建不同的协方差矩阵
#             cls_diag = base_diag.copy()
            
#             # 创建椭圆形：某些维度上方差大，某些维度上方差小
#             if cls == 0:
#                 # 第一个类别：前几个维度方差大，后面方差小
#                 for i in range(feature_dim):
#                     if i < feature_dim // 2:
#                         cls_diag[i] *= ellipse_ratio
#                     else:
#                         cls_diag[i] /= ellipse_ratio
#             elif cls == 1:
#                 # 第二个类别：中间维度方差大，两端方差小
#                 for i in range(feature_dim):
#                     distance_from_center = abs(i - feature_dim/2) / (feature_dim/2)
#                     cls_diag[i] *= (1 + (ellipse_ratio - 1) * (1 - distance_from_center))
#             else:
#                 # 第三个类别：交替方差大小
#                 for i in range(feature_dim):
#                     if i % 2 == 0:
#                         cls_diag[i] *= ellipse_ratio
#                     else:
#                         cls_diag[i] /= ellipse_ratio
            
#             # 创建对角协方差矩阵
#             cls_cov = np.diag(cls_diag)
            
#             # 添加旋转（如果不是对角矩阵）
#             if rotation_angle != 0 and feature_dim >= 2:
#                 # 创建旋转矩阵（仅在前两个维度上旋转）
#                 R = np.eye(feature_dim)
#                 R[0, 0] = np.cos(rotation_angle * cls)  # 不同类别不同旋转
#                 R[0, 1] = -np.sin(rotation_angle * cls)
#                 R[1, 0] = np.sin(rotation_angle * cls)
#                 R[1, 1] = np.cos(rotation_angle * cls)
                
#                 # 应用旋转：cov_rotated = R * cov * R^T
#                 cls_cov = R @ cls_cov @ R.T
            
#             cls_features = np.random.multivariate_normal(
#                 mean=means_arr[cls], cov=cls_cov, size=n_cls
#             )
#             features[list(part)] = cls_features
#             label[list(part)] = cls
            
#     features = torch.FloatTensor(features)
#     label = torch.LongTensor(label)
#     return features, label


def generate_elliptical_features(
    G: nx.Graph,
    means,
    SIGMA: float,
    feature_dim: int = 4,
    ellipse_ratio: float = 4.0,
    seed: int = 0,
):
    """
    生成 n 维椭圆分布特征（各类有不同协方差矩阵 + 随机旋转）
    
    参数:
    - feature_dim: 特征维度（n维）
    - ellipse_ratio: 控制长短轴比例
    - means: 各类的均值向量（不足 n 维时自动补零）
    """
    rng = np.random.default_rng(seed)
    num_nodes = G.number_of_nodes()
    features = np.zeros((num_nodes, feature_dim))
    label = np.zeros(num_nodes, dtype=np.int64)
    means_arr = np.asarray(means, dtype=float)

    # 如果均值维度不足 feature_dim → 扩展到 n 维（后面补零）
    if means_arr.shape[1] < feature_dim:
        ext = np.zeros((means_arr.shape[0], feature_dim))
        ext[:, :means_arr.shape[1]] = means_arr
        means_arr = ext

    for cls, part in enumerate(G.graph["partition"]):
        n_cls = len(part)
        if n_cls == 0:
            continue

        # ---- 构造对角矩阵 D（长短轴差异）----
        diag = np.ones(feature_dim) * (SIGMA ** 2)

        if cls == 0:
            # 前半大，后半小
            diag[: feature_dim // 2] *= ellipse_ratio
            diag[feature_dim // 2 :] /= ellipse_ratio
        elif cls == 1:
            # 中间大，两端小
            pos = np.arange(feature_dim)
            dist = np.abs(pos - (feature_dim - 1) / 2) / ((feature_dim - 1) / 2 + 1e-8)
            factor = 1 + (ellipse_ratio - 1) * (1 - dist)
            diag *= factor
        else:
            # 奇偶交替
            diag[::2] *= ellipse_ratio
            diag[1::2] /= ellipse_ratio

        D = np.diag(diag)

        # ---- 随机旋转矩阵 Q ----
        Q, _ = np.linalg.qr(rng.normal(size=(feature_dim, feature_dim)))
        cov = Q @ D @ Q.T

        # ---- 按类采样 ----
        cls_features = rng.multivariate_normal(
            mean=means_arr[cls],
            cov=cov,
            size=n_cls
        )
        features[list(part)] = cls_features
        label[list(part)] = cls

    return torch.from_numpy(features).float(), torch.from_numpy(label).long()


# --------- Data 补齐（三分类版本） ---------
def create_pyg_graph_with_attributes_three_class(G, features, label, c0_idx, c1_idx, c2_idx) -> Data:
    """为三分类图创建PyG数据对象并添加所有属性"""
    edge_list = list(G.edges)
    
    # 三分类的数据集划分
    idx_source_train = np.concatenate((
        c0_idx[:int(0.6 * len(c0_idx))],
        c1_idx[:int(0.6 * len(c1_idx))],
        c2_idx[:int(0.6 * len(c2_idx))]
    ))
    idx_source_valid = np.concatenate((
        c0_idx[int(0.6 * len(c0_idx)): int(0.8 * len(c0_idx))],
        c1_idx[int(0.6 * len(c1_idx)): int(0.8 * len(c1_idx))],
        c2_idx[int(0.6 * len(c2_idx)): int(0.8 * len(c2_idx))]
    ))
    idx_source_test = np.concatenate((
        c0_idx[int(0.8 * len(c0_idx)):],
        c1_idx[int(0.8 * len(c1_idx)):],
        c2_idx[int(0.8 * len(c2_idx)):]
    ))
    idx_target_valid = np.concatenate((
        c0_idx[:int(0.2 * len(c0_idx))],
        c1_idx[:int(0.2 * len(c1_idx))],
        c2_idx[:int(0.2 * len(c2_idx))]
    ))
    idx_target_test = np.concatenate((
        c0_idx[int(0.2 * len(c0_idx)):],
        c1_idx[int(0.2 * len(c1_idx)):],
        c2_idx[int(0.2 * len(c2_idx)):]
    ))

    num_nodes_total = len(label)
    adj, edge_index = edges_to_adj(edge_list, num_nodes_total)

    graph = Data(x=features, edge_index=edge_index, y=label)
    graph.source_training_mask = idx_source_train
    graph.source_validation_mask = idx_source_valid
    graph.source_testing_mask = idx_source_test
    graph.target_validation_mask = idx_target_valid
    graph.target_testing_mask = idx_target_test
    graph.source_mask = np.arange(graph.num_nodes)
    graph.target_mask = np.arange(graph.num_nodes)
    graph.adj = adj
    graph.y_hat = torch.full_like(label, -1)
    graph.num_classes = 3  # 三分类
    graph.edge_weight = torch.ones(graph.num_edges)
    graph.mod_edge_index = edge_index
    graph.mod_edge_weight = torch.ones(graph.num_edges)
    graph.homophily_ratio = compute_global_homophily(G)
    return graph

# --------- 保存/加载核心数据（三分类版本） ---------
def save_graph_core_three_class(G, features, label, c0_idx, c1_idx, c2_idx, filepath):
    """保存三分类图的核心数据"""
    graph_data = {
        "G": G, 
        "features": features, 
        "label": label, 
        "c0_idx": c0_idx, 
        "c1_idx": c1_idx,
        "c2_idx": c2_idx
    }
    with open(filepath, "wb") as f:
        pickle.dump(graph_data, f)
    print(f"Graph core saved to {filepath}")

def load_graph_core_three_class(filepath):
    """加载三分类图的核心数据"""
    with open(filepath, "rb") as f:
        graph_data = pickle.load(f)
    print(f"Graph core loaded from {filepath}")
    return (
        graph_data["G"], 
        graph_data["features"], 
        graph_data["label"], 
        graph_data["c0_idx"], 
        graph_data["c1_idx"],
        graph_data["c2_idx"]
    )

# --------- 生成三分类源图 ---------
def generate_source_graph_three_class(num_nodes, SIGMA, p, q, 
                                    means=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                                    feature_dim=10, ellipse_ratio=5.0,
                                    save_path=None, seed=0):
    """生成三分类源图（使用椭圆形分布）"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 生成三分类SBM图
    G = _generate_sbm_three_class(num_nodes, p, q, scale=0.1, seed=seed)
    
    # 生成节点特征和标签（使用椭圆形分布）
    features, label = generate_elliptical_features(
        G, means, SIGMA, feature_dim, ellipse_ratio, seed
    )
    
    # 获取节点索引
    c0_idx = np.array(list(G.graph["partition"][0]))
    c1_idx = np.array(list(G.graph["partition"][1]))
    c2_idx = np.array(list(G.graph["partition"][2]))
    
    # 打乱索引
    random.shuffle(c0_idx)
    random.shuffle(c1_idx)
    random.shuffle(c2_idx)
    
    # 创建PyG图数据对象
    graph = create_pyg_graph_with_attributes_three_class(G, features, label, c0_idx, c1_idx, c2_idx)
    
    if save_path is not None:
        save_graph_core_three_class(G, features, label, c0_idx, c1_idx, c2_idx, save_path)
        
    return G, features, label, c0_idx, c1_idx, c2_idx, graph

# --------- 从三分类源图生成目标图 ---------
def generate_target_graph_from_source_three_class(source_data, H_target, 
                                                means=((3.0, 2.0), (-3.0, -2.0), (0.0, -3.0)),
                                                SIGMA=0.5, feature_dim=10, ellipse_ratio=5.0, 
                                                seed=0, save_path=None):
    """从三分类源图生成具有目标同质比的目标图（使用椭圆形分布）"""
    G, _, _, c0_idx, c1_idx, c2_idx = source_data
    G_target = G.copy()
    
    print(f"Adjusting homophily ratio to {H_target}")
    G_target = degree_preserving_rewire_to_H_three_class(G_target, H_target, seed=seed)
    
    # 为目标图生成新的特征和标签（使用椭圆形分布）
    features, label = generate_elliptical_features(
        G_target, means, SIGMA, feature_dim, ellipse_ratio,seed*2
    )
    
    # 创建PyG图数据对象
    graph = create_pyg_graph_with_attributes_three_class(G_target, features, label, c0_idx, c1_idx, c2_idx)
    
    if save_path is not None:
        save_graph_core_three_class(G_target, features, label, c0_idx, c1_idx, c2_idx, save_path)
        
    return graph


if __name__ == "__main__":
    print("Use: python scripts/generate_synthetic_graphs.py --num_nodes 4000")
