# scino.py
"""
SciNO: Score-informed Neural Operator for Causal Discovery

性能优化版本：
- 增大默认batch_size提升GPU利用率
- DataLoader多进程加载
- 向量化Hessian计算
- 批量频域卷积

论文: "Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery"

核心创新:
1. 使用FNO替代直接求导，解决Hessian估计的数值不稳定性
2. 在频域学习平滑算子，输出天然光滑
3. 相比DiffAN误差降低约40%
"""

import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from collections import Counter

from .models.score_unet import ScoreUNet, DiffMLP
from .models.fno_operator import FNOOperator, LightFNOOperator
from .trainers.train_score import train_score
from .trainers.train_operator import train_operator
from .data.dataset import TensorDataset


# 尝试导入DiffAN的剪枝模块
try:
    import sys
    sys.path.insert(0, 'DiffAN')
    from diffan.pruning import cam_pruning
    from diffan.utils import full_DAG
    HAS_DIFFAN_PRUNING = True
except ImportError:
    HAS_DIFFAN_PRUNING = False


def simple_pruning(order, X, cutoff=0.001):
    """简单线性回归剪枝（CPU版本，兼容旧代码）"""
    from sklearn.linear_model import LinearRegression

    n = len(order)
    W = np.zeros((n, n))

    for i, node in enumerate(order):
        if i == 0:
            continue
        parents = order[:i]
        y = X[:, node]
        X_parents = X[:, parents]

        reg = LinearRegression().fit(X_parents, y)
        for j, p in enumerate(parents):
            if abs(reg.coef_[j]) > cutoff:
                W[p, node] = reg.coef_[j]

    return W


def simple_pruning_gpu(order, X_tensor, cutoff=0.001, device='cuda'):
    """
    GPU加速的线性回归剪枝

    使用 torch.linalg.lstsq 替代 sklearn.LinearRegression
    所有计算在GPU上完成，最后才传回CPU

    Args:
        order: 拓扑排序
        X_tensor: torch.Tensor (n_samples, n_nodes) 在GPU上
        cutoff: 系数阈值
        device: 设备

    Returns:
        W: np.ndarray 邻接矩阵
    """
    n = len(order)
    W = torch.zeros((n, n), device=device, dtype=torch.float32)

    # 确保 X 在正确的设备上
    if not isinstance(X_tensor, torch.Tensor):
        X_tensor = torch.from_numpy(X_tensor).float().to(device)
    elif X_tensor.device.type != device:
        X_tensor = X_tensor.to(device)

    for i, node in enumerate(order):
        if i == 0:
            continue
        parents = order[:i]
        y = X_tensor[:, node].unsqueeze(1)  # (n_samples, 1)
        X_parents = X_tensor[:, parents]     # (n_samples, n_parents)

        # 使用最小二乘法: (X^T X)^-1 X^T y
        # torch.linalg.lstsq 更数值稳定
        n_parents = len(parents)
        try:
            solution = torch.linalg.lstsq(X_parents, y)
            coef = solution.solution.flatten()  # 始终保持1D，避免0维张量
        except Exception:
            # 回退到正规方程
            XtX = X_parents.T @ X_parents
            Xty = X_parents.T @ y
            # 添加小的正则化避免奇异矩阵
            XtX += 1e-6 * torch.eye(XtX.shape[0], device=device)
            coef = torch.linalg.solve(XtX, Xty).flatten()  # 始终保持1D

        # 确保 coef 是1D张量
        if coef.dim() == 0:
            coef = coef.unsqueeze(0)

        # 应用 cutoff 阈值 - 向量化操作
        mask = torch.abs(coef) > cutoff
        for j in range(n_parents):
            if mask[j].item():
                W[parents[j], node] = coef[j].item()

    # 只在最后传回 CPU
    return W.cpu().numpy()


class SciNO:
    """
    SciNO: 基于神经算子的因果发现（性能优化版）

    相比DiffAN的改进:
    - 使用FNO算子直接预测Hessian对角线，避免数值不稳定
    - 训练更稳定，适合高维数据
    - 在合成和真实数据上精度更高

    性能优化:
    - 增大batch_size充分利用GPU
    - DataLoader多进程加载
    - 向量化计算减少Python循环

    用法:
        scino = SciNO(n_nodes)
        adj_matrix, order = scino.fit(X)
    """

    def __init__(
        self,
        n_nodes=None,
        # 分数网络参数
        score_hidden=256,
        score_epochs=100,
        score_lr=1e-4,
        score_batch_size=256,  # 增大默认值
        # 噪声调度
        sigma_min=0.01,
        sigma_max=50.0,
        # FNO算子参数
        op_width=128,
        op_modes=(32, 16, 8),
        op_depth=4,
        op_epochs=60,
        op_lr=3e-4,
        op_batch_size=256,  # 增大默认值
        op_hutchinson_probes=4,
        op_use_hutchinson=True,  # 按论文思路默认用Hutchinson估计监督算子
        op_use_batched_hutchinson=True,
        small_dim_for_supervision=16,
        # 拓扑排序
        masking=True,
        residue=False,
        n_votes=3,
        # 剪枝
        cutoff=0.001,
        pruning_method='auto',
        pruning_verbose=False,
        # 高维优化
        high_dim_threshold=100,
        high_dim_score_epochs=60,
        high_dim_op_epochs=40,
        high_dim_score_batch_size=128,
        high_dim_op_batch_size=128,
        high_dim_subset_dim=32,
        use_light_operator=None,  # None=自动检测
        # 数据处理
        standardize_input=True,
        # 通用
        use_amp=True,
        grad_clip=1.0,
        device=None,
        verbose=True,
        # 新增优化参数
        num_workers=0,  # DataLoader多进程，小数据集建议设为0
        pin_memory=False,  # 单进程时不需要锁页内存
        compile_model=False  # torch.compile (PyTorch 2.0+)
    ):
        self.n_nodes = n_nodes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 分数网络配置
        self.score_hidden = score_hidden
        self.score_epochs = score_epochs
        self.score_lr = score_lr
        self.score_batch_size = score_batch_size

        # 噪声调度
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # FNO算子配置
        self.op_width = op_width
        self.op_modes = op_modes
        self.op_depth = op_depth
        self.op_epochs = op_epochs
        self.op_lr = op_lr
        self.op_batch_size = op_batch_size
        self.op_hutchinson_probes = op_hutchinson_probes
        self.op_use_hutchinson = op_use_hutchinson
        self.op_use_batched_hutchinson = op_use_batched_hutchinson
        self.small_dim_for_supervision = small_dim_for_supervision

        # 拓扑排序
        self.masking = masking
        self.residue = residue
        self.n_votes = n_votes

        # 剪枝
        self.cutoff = cutoff
        self.pruning_method = pruning_method
        self.pruning_verbose = pruning_verbose

        # 高维优化
        self.high_dim_threshold = high_dim_threshold
        self.high_dim_score_epochs = high_dim_score_epochs
        self.high_dim_op_epochs = high_dim_op_epochs
        self.high_dim_score_batch_size = high_dim_score_batch_size
        self.high_dim_op_batch_size = high_dim_op_batch_size
        self.high_dim_subset_dim = high_dim_subset_dim
        self.use_light_operator = use_light_operator
        self.is_high_dim = False

        # 训练配置
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.verbose = verbose

        # 性能优化参数
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.compile_model = compile_model

        # 数据标准化
        self.standardize_input = standardize_input
        self.data_mean_ = None
        self.data_std_ = None
        self.training_config_ = None

        # 模型 (延迟初始化)
        self.score_model = None
        self.operator = None

        # 结果
        self.graph_ = None
        self.order_ = None

    def _prepare_data(self, X, fit=False):
        """根据配置对输入数据做（可选）标准化"""
        X = np.asarray(X, dtype=np.float32)
        if not self.standardize_input:
            return X

        if fit or self.data_mean_ is None or self.data_std_ is None:
            self.data_mean_ = X.mean(0, keepdims=True)
            self.data_std_ = X.std(0, keepdims=True) + 1e-8

        return (X - self.data_mean_) / self.data_std_

    def _resolve_training_params(self, n_samples, n_nodes):
        """根据维度规模自动调整训练超参数和训练策略"""
        if self.is_high_dim:
            cfg = {
                'score_epochs': min(self.score_epochs, self.high_dim_score_epochs),
                'score_batch': max(1, min(self.high_dim_score_batch_size, n_samples)),
                'op_epochs': min(self.op_epochs, self.high_dim_op_epochs),
                'op_batch': max(1, min(self.high_dim_op_batch_size, n_samples)),
                'small_dim': max(1, min(self.high_dim_subset_dim, n_nodes)),
                'use_subset': True,  # 高维使用子采样
                'use_hutchinson': self.op_use_hutchinson,
                'use_batched_hutchinson': self.op_use_batched_hutchinson,
                'hutchinson_probes': self.op_hutchinson_probes
            }
        elif n_nodes >= 100:
            # 中等维度: 使用Hutchinson估计器(比子采样更准确)
            cfg = {
                'score_epochs': self.score_epochs,
                'score_batch': max(1, min(self.score_batch_size, n_samples)),
                'op_epochs': self.op_epochs,
                'op_batch': max(1, min(self.op_batch_size, n_samples)),
                'small_dim': n_nodes,  # 不需要子采样
                'use_subset': False,
                'use_hutchinson': True,
                'use_batched_hutchinson': True,
                'hutchinson_probes': max(2, self.op_hutchinson_probes)
            }
        else:
            # 低维: 精确计算
            cfg = {
                'score_epochs': self.score_epochs,
                'score_batch': max(1, min(self.score_batch_size, n_samples)),
                'op_epochs': self.op_epochs,
                'op_batch': max(1, min(self.op_batch_size, n_samples)),
                'small_dim': n_nodes,
                'use_subset': False,
                'use_hutchinson': self.op_use_hutchinson,
                'use_batched_hutchinson': self.op_use_batched_hutchinson,
                'hutchinson_probes': self.op_hutchinson_probes
            }
        self.training_config_ = cfg
        return cfg

    def _init_models(self, n_nodes):
        """初始化模型"""
        self.n_nodes = n_nodes
        self.is_high_dim = n_nodes >= self.high_dim_threshold
        is_high_dim = self.is_high_dim

        # 自动决定是否使用轻量级算子
        use_light = self.use_light_operator
        if use_light is None:
            use_light = is_high_dim

        if self.verbose:
            print(f"[SciNO] 初始化: {n_nodes}节点, 高维阈值={self.high_dim_threshold}, "
                  f"高维模式={is_high_dim}, 轻量算子={use_light}")

        # 分数网络
        hidden_dim = self.score_hidden if not is_high_dim else min(self.score_hidden, 128)
        self.score_model = ScoreUNet(
            dim=n_nodes,
            hidden=hidden_dim
        ).to(self.device)

        # FNO算子
        if use_light:
            self.operator = LightFNOOperator(
                in_dim=n_nodes,
                width=min(64, self.op_width),
                modes=min(16, self.op_modes[0] if isinstance(self.op_modes, tuple) else self.op_modes),
                depth=2,
                time_emb_dim=32
            ).to(self.device)
        else:
            self.operator = FNOOperator(
                in_dim=n_nodes,
                width=self.op_width,
                modes_list=self.op_modes,
                depth=self.op_depth
            ).to(self.device)

    def fit(self, X):
        """
        训练SciNO并发现因果结构

        Args:
            X: (n_samples, n_nodes) 观测数据

        Returns:
            adj_matrix: 邻接矩阵
            order: 拓扑排序
        """
        # 数据预处理
        X = self._prepare_data(X, fit=True)
        n_samples, n_nodes = X.shape

        # 初始化模型
        self._init_models(n_nodes)
        train_cfg = self._resolve_training_params(n_samples, n_nodes)

        if self.verbose:
            if train_cfg.get('use_hutchinson', False):
                mode = "Hutchinson监督"
            elif self.is_high_dim:
                mode = "高维(子采样)"
            else:
                mode = "低维(精确)"
            print(f"[SciNO] 训练配置({mode}): "
                  f"score_epochs={train_cfg['score_epochs']}, "
                  f"score_batch={train_cfg['score_batch']}, "
                  f"op_epochs={train_cfg['op_epochs']}, "
                  f"op_batch={train_cfg['op_batch']}, "
                  f"small_dim={train_cfg['small_dim']}, "
                  f"hutchinson={train_cfg.get('use_hutchinson', False)}, "
                  f"probes={train_cfg.get('hutchinson_probes', self.op_hutchinson_probes)}")

        # 创建数据集
        dataset = TensorDataset(X)

        # 阶段1: 训练分数网络
        if self.verbose:
            print("\n[SciNO] 阶段1: 训练分数网络")

        train_score(
            self.score_model, dataset,
            n_epochs=train_cfg['score_epochs'],
            batch_size=train_cfg['score_batch'],
            lr=self.score_lr,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            use_amp=self.use_amp,
            grad_clip=self.grad_clip,
            device=self.device,
            verbose=self.verbose,
            # 性能优化参数
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            compile_model=self.compile_model
        )

        # 阶段2: 训练FNO算子
        if self.verbose:
            print("\n[SciNO] 阶段2: 训练FNO算子")

        train_operator(
            self.operator, self.score_model, dataset,
            n_epochs=train_cfg['op_epochs'],
            batch_size=train_cfg['op_batch'],
            lr=self.op_lr,
            small_dim_for_supervision=train_cfg['small_dim'],
            device=self.device,
            verbose=self.verbose,
            # Hutchinson标签（默认开启以避免显式二阶求导）
            use_hutchinson_labels=train_cfg.get('use_hutchinson', True),
            hutchinson_probes=train_cfg.get('hutchinson_probes', self.op_hutchinson_probes),
            use_batched_hessian=train_cfg.get('use_batched_hutchinson', True),
            # 性能优化参数
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            compile_model=self.compile_model
        )

        # 阶段3: 拓扑排序
        if self.verbose:
            print("\n[SciNO] 阶段3: 拓扑排序")

        X_tensor = torch.FloatTensor(X).to(self.device)
        order = self._topological_ordering(X_tensor)
        self.order_ = order

        # 阶段4: 剪枝
        if self.verbose:
            print("\n[SciNO] 阶段4: 剪枝")

        self.graph_ = self._pruning(order, X)

        return self.graph_, order

    def _topological_ordering(self, X):
        """
        使用FNO算子进行拓扑排序（GPU优化版）

        核心思想: 叶节点的Hessian对角线方差最小

        优化: 使用纯PyTorch操作，避免每轮GPU-CPU传输
        """
        self.score_model.eval()
        self.operator.eval()

        n_samples, n_nodes = X.shape
        order = []

        # 使用 tensor 存储活跃节点索引，保持在 GPU 上
        active_mask = torch.ones(n_nodes, dtype=torch.bool, device=self.device)

        # 使用少量样本加速
        max_samples = 64 if self.is_high_dim else 128
        n_use = min(max_samples, n_samples)
        X_sub = X[:n_use]

        pbar = tqdm(range(n_nodes - 1), desc="拓扑排序", disable=not self.verbose)
        for step in pbar:
            with torch.no_grad():
                # 计算分数
                t = torch.zeros(X_sub.shape[0], device=self.device)
                scores = self.score_model(X_sub, t)

                # FNO算子预测Hessian对角线
                hess_diag = self.operator(scores, t)

                # 计算方差，只考虑活跃节点
                # 使用 masked 操作避免索引
                var = hess_diag.var(dim=0)  # (n_nodes,)

                # 将非活跃节点的方差设为无穷大
                var = torch.where(active_mask, var, torch.tensor(float('inf'), device=self.device))

                # 在 GPU 上找最小值（纯 PyTorch，无 CPU 传输）
                leaf_idx = torch.argmin(var).item()

                order.append(leaf_idx)
                active_mask[leaf_idx] = False

        # 添加最后一个节点
        last_node = torch.where(active_mask)[0][0].item()
        order.append(last_node)
        order.reverse()

        return order

    def _pruning(self, order, X):
        """
        对排序后的图进行剪枝

        优化: 优先使用 GPU 版本的线性回归剪枝

        剪枝策略:
        - 'none': 不剪枝，仅返回基于拓扑排序的全连接 DAG（用于后续IAMB等优化）
        - 'linear'/'auto'/None: 使用线性回归剪枝（GPU加速优先）
        - 其他: 尝试使用 CAM/GAM 剪枝
        """
        import traceback
        n_nodes = len(order)

        if self.verbose:
            print(f"[SciNO] 剪枝配置: method={self.pruning_method}, cutoff={self.cutoff}, n_nodes={n_nodes}")

        # Case 1: 不剪枝 - 用于 SciNO-IAMB 等需要自己做后处理的场景
        if self.pruning_method == 'none':
            if self.verbose:
                print(f"[SciNO] pruning_method='none', 跳过剪枝，返回基于拓扑排序的父节点评分")
            # 返回一个全零矩阵，让后续处理（如IAMB）自己决定边
            # 注意：这里不使用 full_DAG，因为那样边太多会导致后续处理慢
            W = np.zeros((n_nodes, n_nodes))
            # 但我们需要返回一些信息给调用者，用简单的线性回归获取权重但不做阈值截断
            try:
                if self.verbose:
                    print(f"[SciNO] 计算父节点评分（无阈值截断）...")
                W = simple_pruning_gpu(order, X, cutoff=0.0, device=self.device) if self.device != 'cpu' else simple_pruning(order, X, cutoff=0.0)
                if self.verbose:
                    n_edges = int(np.sum(np.abs(W) > 0))
                    print(f"[SciNO] 父节点评分计算完成: {n_edges} 条非零边")
            except Exception as e:
                if self.verbose:
                    print(f"[SciNO] 父节点评分计算失败: {e}")
                    traceback.print_exc()
            return W

        # Case 2: GPU 加速线性回归剪枝（推荐）
        if self.device != 'cpu' and self.pruning_method in ['linear', 'auto', None]:
            try:
                if self.verbose:
                    print(f"[SciNO] 使用GPU加速剪枝 (device={self.device}, cutoff={self.cutoff})")
                W = simple_pruning_gpu(order, X, self.cutoff, device=self.device)
                if self.verbose:
                    n_edges = int(np.sum(np.abs(W) > 0))
                    print(f"[SciNO] GPU剪枝完成: {n_edges} 条边")
                return W
            except Exception as e:
                if self.verbose:
                    print(f"[SciNO] GPU剪枝失败，回退到CPU: {e}")
                    traceback.print_exc()

        # Case 3: CAM/GAM 剪枝（需要 DiffAN 剪枝模块）
        # 注意：对于高维数据，CAM剪枝非常慢，不推荐使用
        if HAS_DIFFAN_PRUNING and self.pruning_method not in ['linear', 'none', None]:
            if self.verbose:
                print(f"[SciNO] 尝试CAM剪枝 (method={self.pruning_method})")
                if n_nodes > 50:
                    print(f"[SciNO] ⚠ 警告: 高维数据({n_nodes}节点)使用CAM剪枝可能非常慢!")
            try:
                if self.verbose:
                    print(f"[SciNO] 构建完全DAG...")
                full_dag = full_DAG(order)
                n_full_edges = int(np.sum(full_dag))
                if self.verbose:
                    print(f"[SciNO] 完全DAG: {n_full_edges} 条边，开始CAM剪枝...")

                W = cam_pruning(
                    full_dag, X, self.cutoff,
                    method=self.pruning_method,
                    verbose=self.pruning_verbose
                )
                if self.verbose:
                    n_edges = int(np.sum(np.abs(W) > 0))
                    print(f"[SciNO] CAM剪枝完成: {n_full_edges} -> {n_edges} 条边")
                return W
            except Exception as e:
                if self.verbose:
                    print(f"[SciNO] CAM剪枝失败: {e}")
                    traceback.print_exc()
                    print(f"[SciNO] 回退到简单线性回归剪枝...")
                return simple_pruning(order, X, self.cutoff)

        # Case 4: 默认使用 CPU 简单线性回归剪枝
        if self.verbose:
            print(f"[SciNO] 使用CPU简单线性回归剪枝 (cutoff={self.cutoff})")
        try:
            W = simple_pruning(order, X, self.cutoff)
            if self.verbose:
                n_edges = int(np.sum(np.abs(W) > 0))
                print(f"[SciNO] CPU剪枝完成: {n_edges} 条边")
            return W
        except Exception as e:
            if self.verbose:
                print(f"[SciNO] CPU剪枝失败: {e}")
                traceback.print_exc()
            # 最后的兜底：返回空矩阵
            return np.zeros((n_nodes, n_nodes))

    def get_hessian_diag(self, X):
        """
        获取Hessian对角线估计

        这是SciNO的核心输出，可用于下游任务

        Args:
            X: (n_samples, n_nodes) 输入数据

        Returns:
            hess_diag: (n_samples, n_nodes) Hessian对角线
        """
        if self.score_model is None or self.operator is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        self.score_model.eval()
        self.operator.eval()

        with torch.no_grad():
            X_proc = self._prepare_data(X, fit=False)
            X_tensor = torch.from_numpy(X_proc).to(self.device)
            t = torch.zeros(X_tensor.shape[0], device=self.device)
            scores = self.score_model(X_tensor, t)
            hess_diag = self.operator(scores, t)

        return hess_diag.cpu().numpy()


# 简便接口
def run_scino(X, **kwargs):
    """
    SciNO快捷调用接口

    Args:
        X: (n_samples, n_nodes) 观测数据
        **kwargs: SciNO参数

    Returns:
        adj_matrix: 邻接矩阵
        order: 拓扑排序
    """
    scino = SciNO(**kwargs)
    return scino.fit(X)
