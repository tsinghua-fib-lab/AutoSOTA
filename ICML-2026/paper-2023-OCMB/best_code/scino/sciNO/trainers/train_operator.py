# trainers/train_operator.py
"""神经算子训练：学习分数到Hessian对角线的映射

性能优化版本：
- DataLoader多进程加载 + pin_memory
- 更大的默认batch_size
- 优化的Hutchinson估计器
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.math_utils import (
    compute_true_hessian_diag,
    compute_true_hessian_diag_subset,
    compute_hessian_diag_hutchinson,
    compute_hessian_diag_hutchinson_batched
)


def train_operator(
    op_model,
    score_model,
    dataset,
    n_epochs=60,
    batch_size=256,  # 增大默认batch_size
    lr=3e-4,
    small_dim_for_supervision=16,
    device='cuda',
    verbose=True,
    # 新增优化参数
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    use_batched_hessian=True,  # 使用批量Hutchinson（内存换速度）
    use_hutchinson_labels=True,  # 默认按照论文“避免显式Hessian”思路
    hutchinson_probes=4,
    compile_model=False
):
    """
    训练FNO算子（优化版）

    使用完整维度训练：
    1. 计算完整维度的分数
    2. 使用子集计算真实Hessian对角线作为监督
    3. 训练算子预测完整维度的Hessian对角线

    Args:
        op_model: FNO算子模型
        score_model: 预训练的分数网络
        dataset: 训练数据集
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        small_dim_for_supervision: 监督用的维度数量（用于计算真实Hessian）
        device: 设备
        verbose: 是否显示进度
        num_workers: DataLoader工作进程数
        pin_memory: 是否使用锁页内存
        prefetch_factor: 预取因子
        use_batched_hessian: 是否使用批量Hutchinson计算
        use_hutchinson_labels: 是否用Hutchinson估计作为监督
        hutchinson_probes: Hutchinson探测向量数量
        compile_model: 是否编译模型

    Returns:
        loss_history: 训练损失历史
    """
    # 检测设备类型
    is_cuda = device == 'cuda' or (isinstance(device, str) and device.startswith('cuda'))

    # 配置DataLoader
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
    }

    if is_cuda and torch.cuda.is_available():
        loader_kwargs['num_workers'] = num_workers
        loader_kwargs['pin_memory'] = pin_memory
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['persistent_workers'] = True

    loader = DataLoader(dataset, **loader_kwargs)

    # 优化器
    optimizer = torch.optim.AdamW(op_model.parameters(), lr=lr, weight_decay=0.01)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    op_model.train()
    op_model.to(device)
    score_model.eval()
    score_model.to(device)

    # 可选：编译模型
    if compile_model and hasattr(torch, 'compile'):
        try:
            op_model = torch.compile(op_model, mode='reduce-overhead')
            if verbose:
                print("[train_operator] Operator model compiled with torch.compile")
        except Exception as e:
            if verbose:
                print(f"[train_operator] torch.compile failed: {e}")

    loss_history = []

    pbar = tqdm(range(n_epochs), desc="Operator Training", disable=not verbose)
    for epoch in pbar:
        epoch_losses = []

        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device, non_blocking=True).float()

            D = xb.shape[1]
            t_zero = torch.zeros(xb.shape[0], device=device)

            # 计算完整维度的分数
            with torch.no_grad():
                scores_full = score_model(xb, t_zero)

            # 随机选择维度子集计算真实Hessian
            subset_dim = min(small_dim_for_supervision, D)
            idx = torch.randperm(D, device=xb.device)[:subset_dim]

            # 计算子集上的真实Hessian对角线（默认使用Hutchinson以避免重复反向）
            if use_hutchinson_labels:
                hutch_fn = compute_hessian_diag_hutchinson_batched if use_batched_hessian else compute_hessian_diag_hutchinson
                try:
                    Htrue_full = hutch_fn(
                        lambda x, t: score_model(x, t),
                        xb, t_zero, n_probes=hutchinson_probes
                    )
                except RuntimeError as e:
                    # 防止显存溢出时自动回退到非批量版本
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        Htrue_full = compute_hessian_diag_hutchinson(
                            lambda x, t: score_model(x, t),
                            xb, t_zero, n_probes=hutchinson_probes
                        )
                    else:
                        raise
                Htrue_subset = Htrue_full[:, idx]
            else:
                if subset_dim == D:
                    Htrue_subset = compute_true_hessian_diag(
                        lambda x, t: score_model(x, t),
                        xb, t_zero
                    )[:, idx]
                else:
                    Htrue_subset = compute_true_hessian_diag_subset(
                        lambda x, t: score_model(x, t),
                        xb, t_zero, idx
                    )

            # 算子预测完整Hessian，取子集进行监督
            Hpred_full = op_model(scores_full, t_zero)
            Hpred_subset = Hpred_full[:, idx]

            # 损失
            loss = ((Hpred_subset - Htrue_subset) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op_model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # 更新学习率
        scheduler.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})

    return loss_history


def train_operator_hutchinson(
    op_model,
    score_model,
    dataset,
    n_epochs=60,
    batch_size=128,  # 稍小的batch因为Hutchinson需要更多内存
    lr=3e-4,
    n_probes=6,
    device='cuda',
    verbose=True,
    # 新增优化参数
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    use_batched_hutchinson=True,  # 默认使用完全批量化版本，一次反向
    compile_model=False
):
    """
    使用Hutchinson估计器训练算子(适合中等维度)（优化版）

    相比子采样方法,Hutchinson提供更准确的全维度Hessian估计,
    但计算成本略高,适合100-1000维的数据

    Args:
        op_model: FNO算子模型
        score_model: 预训练的分数网络
        dataset: 训练数据集
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        n_probes: Hutchinson探测向量数量
        device: 设备
        verbose: 是否显示进度
        num_workers: DataLoader工作进程数
        pin_memory: 是否使用锁页内存
        prefetch_factor: 预取因子
        use_batched_hutchinson: 是否使用完全批量化的Hutchinson估计器
        compile_model: 是否编译模型

    Returns:
        loss_history: 训练损失历史
    """
    # 检测设备类型
    is_cuda = device == 'cuda' or (isinstance(device, str) and device.startswith('cuda'))

    # 配置DataLoader
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
    }

    if is_cuda and torch.cuda.is_available():
        loader_kwargs['num_workers'] = num_workers
        loader_kwargs['pin_memory'] = pin_memory
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['persistent_workers'] = True

    loader = DataLoader(dataset, **loader_kwargs)

    # 优化器
    optimizer = torch.optim.AdamW(op_model.parameters(), lr=lr, weight_decay=0.01)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    op_model.train()
    op_model.to(device)
    score_model.eval()
    score_model.to(device)

    # 选择Hutchinson计算函数
    hutchinson_fn = compute_hessian_diag_hutchinson_batched if use_batched_hutchinson else compute_hessian_diag_hutchinson

    # 可选：编译模型
    if compile_model and hasattr(torch, 'compile'):
        try:
            op_model = torch.compile(op_model, mode='reduce-overhead')
            if verbose:
                print("[train_operator_hutchinson] Operator model compiled")
        except Exception as e:
            if verbose:
                print(f"[train_operator_hutchinson] torch.compile failed: {e}")

    loss_history = []

    pbar = tqdm(range(n_epochs), desc="Operator Training (Hutchinson)", disable=not verbose)
    for epoch in pbar:
        epoch_losses = []

        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device, non_blocking=True).float()

            t_zero = torch.zeros(xb.shape[0], device=device)

            with torch.no_grad():
                scores = score_model(xb, t_zero)

            # 使用Hutchinson估计器计算真实Hessian对角线
            Htrue = hutchinson_fn(
                lambda x, t: score_model(x, t),
                xb, t_zero, n_probes=n_probes
            )

            # 算子预测
            Hpred = op_model(scores, t_zero)

            loss = ((Hpred - Htrue) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op_model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # 更新学习率
        scheduler.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})

    return loss_history
