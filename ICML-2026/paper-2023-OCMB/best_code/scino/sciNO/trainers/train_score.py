# trainers/train_score.py
"""分数网络训练：去噪分数匹配 (DSM)

性能优化版本：
- DataLoader多进程加载 + pin_memory
- 更大的默认batch_size
- 可选的梯度累积
"""

import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def perturb(x, sigma_min, sigma_max):
    """
    对数据施加随机噪声扰动

    Args:
        x: (B, D) 原始数据
        sigma_min: 最小噪声尺度
        sigma_max: 最大噪声尺度

    Returns:
        x_perturbed: 扰动后的数据
        sigma: 噪声尺度
    """
    # 对数均匀采样噪声尺度
    sigma = torch.exp(
        torch.rand(x.shape[0], device=x.device) *
        (math.log(sigma_max) - math.log(sigma_min)) +
        math.log(sigma_min)
    )
    noise = torch.randn_like(x) * sigma.view(-1, 1)
    return x + noise, sigma


def train_score(
    score_model,
    dataset,
    n_epochs=100,
    batch_size=256,  # 增大默认batch_size
    lr=1e-4,
    sigma_min=0.01,
    sigma_max=50.0,
    use_amp=True,
    grad_clip=1.0,
    device='cuda',
    verbose=True,
    # 新增优化参数
    num_workers=4,  # DataLoader多进程
    pin_memory=True,  # 锁页内存加速传输
    prefetch_factor=2,  # 预取批次数
    accumulation_steps=1,  # 梯度累积步数
    compile_model=False  # 是否使用torch.compile (PyTorch 2.0+)
):
    """
    训练分数网络（优化版）

    使用去噪分数匹配 (DSM) 损失：
        L = E_σ E_x E_ε [ ||s_θ(x+σε, σ) - (-ε/σ)||² ]

    Args:
        score_model: 分数网络模型
        dataset: 训练数据集
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        sigma_min: 最小噪声尺度
        sigma_max: 最大噪声尺度
        use_amp: 是否使用混合精度
        grad_clip: 梯度裁剪阈值
        device: 设备
        verbose: 是否显示进度
        num_workers: DataLoader工作进程数
        pin_memory: 是否使用锁页内存
        prefetch_factor: 预取因子
        accumulation_steps: 梯度累积步数
        compile_model: 是否编译模型（PyTorch 2.0+）

    Returns:
        loss_history: 训练损失历史
    """
    # 检测设备类型
    is_cuda = device == 'cuda' or (isinstance(device, str) and device.startswith('cuda'))

    # 配置DataLoader - 多进程加载显著提升吞吐量
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
    }

    # 只有在CUDA设备上才启用多进程和pin_memory
    if is_cuda and torch.cuda.is_available():
        loader_kwargs['num_workers'] = num_workers
        loader_kwargs['pin_memory'] = pin_memory
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['persistent_workers'] = True

    loader = DataLoader(dataset, **loader_kwargs)

    # 优化器
    optimizer = torch.optim.AdamW(score_model.parameters(), lr=lr, weight_decay=0.01)

    # 学习率调度器 - Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if use_amp and is_cuda else None

    score_model.train()
    score_model.to(device)

    # 可选：编译模型（PyTorch 2.0+）
    if compile_model and hasattr(torch, 'compile'):
        try:
            score_model = torch.compile(score_model, mode='reduce-overhead')
            if verbose:
                print("[train_score] Model compiled with torch.compile")
        except Exception as e:
            if verbose:
                print(f"[train_score] torch.compile failed: {e}")

    loss_history = []

    pbar = tqdm(range(n_epochs), desc="Score Training", disable=not verbose)
    for epoch in pbar:
        epoch_losses = []
        optimizer.zero_grad()

        for step, xb in enumerate(loader):
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device, non_blocking=True).float()

            # 扰动数据
            x_pert, sigma = perturb(xb, sigma_min, sigma_max)
            t = torch.log(sigma)  # 时间 = log(sigma)

            # 前向传播
            with torch.cuda.amp.autocast(enabled=use_amp and is_cuda):
                pred = score_model(x_pert, t)
                # 目标分数: -ε/σ² = -(x_pert - x)/σ²
                target = -(x_pert - xb) / (sigma.view(-1, 1) ** 2)
                loss = ((pred - target) ** 2).mean()

                # 梯度累积
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(score_model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(score_model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_losses.append(loss.item() * (accumulation_steps if accumulation_steps > 1 else 1))

        # 更新学习率
        scheduler.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})

    return loss_history
