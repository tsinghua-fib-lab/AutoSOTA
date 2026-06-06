import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientFedFewS(Client):
    """
    FedFewS Client: Few-for-Many via STCH-Set

    Design:
    - Computes gradients for all K models on local data
    - Returns gradients to server for dual-layer weighted aggregation
    - vs FedFew: No automatic differentiation, manual gradient computation

    Training:
    - For each model k: compute ∇_{θ_k} L_i(θ_k) on local data
    - Server aggregates: ∇_{θ_k} g = Σ_i α_i · w_{ik} · ∇_{θ_k} L_i(θ_k)
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # FedFewS parameters
        self.K = args.num_server_models
        self.mu = args.smooth_mu
        self.use_rep_mode = args.use_rep_mode  # Base-head 分离模式

        self.model_set = [copy.deepcopy(self.model) for _ in range(self.K)]

        # 根据是否使用 rep 模式创建不同的优化器
        if self.use_rep_mode:
            # Rep 模式：每个模型需要两个优化器（base + head）
            self.optimizer_set = [
                torch.optim.SGD(model.base.parameters(), lr=self.learning_rate)
                for model in self.model_set
            ]
            self.optimizer_per_set = [
                torch.optim.SGD(model.head.parameters(), lr=self.learning_rate)
                for model in self.model_set
            ]
            # Optimizer Iter-1: CosineAnnealingLR replaces ExponentialLR
            # Cosine decay from lr to 0.0001 over global_rounds steps
            self.learning_rate_scheduler_set = [
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.global_rounds, eta_min=0.0001)
                for optimizer in self.optimizer_set
            ]
            self.learning_rate_scheduler_per_set = [
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.global_rounds, eta_min=0.0001)
                for optimizer in self.optimizer_per_set
            ]
            self.plocal_epochs = getattr(args, 'plocal_epochs', 1)  # Head 训练轮次
        else:
            # 标准模式：每个模型一个优化器
            self.optimizer_set = [
                torch.optim.SGD(model.parameters(), lr=self.learning_rate)
                for model in self.model_set
            ]
            # Optimizer Iter-1: CosineAnnealingLR replaces ExponentialLR
            # Cosine decay from lr to eta_min=0.0001 over global_rounds steps
            self.learning_rate_scheduler_set = [
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.global_rounds, eta_min=0.0001)
                for optimizer in self.optimizer_set
            ]

        self.learning_rate_decay = args.learning_rate_decay

        # 部署时选择的最优模型
        self.best_model_idx = 0
        self.deployed_model = None

        # 训练时记录的K个模型loss
        self.train_model_losses = None  # shape: (K,)

    def set_parameters(self, server_model_set):
        """
        从服务器接收参数（参数拷贝，模拟真实通信）

        Args:
            server_model_set:
        """
        for model, server_model in zip(server_model_set, self.model_set):
            for new_param, old_param in zip(model.parameters(), server_model.parameters()):
                old_param.data = new_param.data.clone()

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # 累积 K 个模型的 loss（用于记录和计算软选择权重）
        total_losses = torch.zeros(self.K, device=self.device)
        total_samples = self.train_samples

        if self.use_rep_mode:
            # Rep 模式：分阶段训练（head → base）
            for k in range(self.K):
                model_k = self.model_set[k]
                model_k.train()

                # 阶段1：训练 head（个性化层），冻结 base
                for param in model_k.base.parameters():
                    param.requires_grad = False
                for param in model_k.head.parameters():
                    param.requires_grad = True

                optimizer_per_k = self.optimizer_per_set[k]
                for epoch in range(self.plocal_epochs):
                    for x, y in trainloader:
                        if type(x) == type([]):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)

                        optimizer_per_k.zero_grad()
                        loss_k = self.loss(model_k(x), y)
                        loss_k.backward()
                        optimizer_per_k.step()

                # 阶段2：训练 base（共享层），冻结 head
                for param in model_k.base.parameters():
                    param.requires_grad = True
                for param in model_k.head.parameters():
                    param.requires_grad = False

                optimizer_k = self.optimizer_set[k]
                for epoch in range(self.local_epochs):
                    total_losses[k] = 0.0  # 重置损失累积
                    for x, y in trainloader:
                        if type(x) == type([]):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)

                        batch_size = y.shape[0]

                        optimizer_k.zero_grad()
                        loss_k = self.loss(model_k(x), y)
                        loss_k.backward()
                        optimizer_k.step()

                        total_losses[k] += loss_k.item() * batch_size
        else:
            # 标准模式：直接训练整个模型 (with Mixup augmentation from Optimizer Iter-3)
            self.mixup_alpha = 0.2  # Beta distribution parameter for Mixup
            for k in range(self.K):
                model_k = self.model_set[k]
                model_k.train()
                optimizer_k = self.optimizer_set[k]
                for epoch in range(self.local_epochs):
                    total_losses[k] = 0.0  # 重置损失累积
                    for x, y in trainloader:
                        if type(x) == type([]):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)

                        batch_size = y.shape[0]

                        # Optimizer Iter-3: Mixup augmentation (α=0.2)
                        # λ ~ Beta(α, α), mix samples within the batch
                        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                        lam = max(lam, 1.0 - lam)  # enforce λ ≥ 0.5 (convex combination toward first ordering)
                        index = torch.randperm(batch_size, device=self.device)
                        mixed_x = lam * x + (1 - lam) * x[index]
                        y_a, y_b = y, y[index]

                        optimizer_k.zero_grad()
                        # Mixup loss: λ * CE(pred, y_a) + (1-λ) * CE(pred, y_b)
                        logits = model_k(mixed_x)
                        loss_k = lam * self.loss(logits, y_a) + (1 - lam) * self.loss(logits, y_b)
                        loss_k.backward()
                        optimizer_k.step()

                        total_losses[k] += loss_k.item() * batch_size

        # 使用 eval 模式的 loss 作为 train_model_losses
        avg_losses = (total_losses / total_samples).cpu().numpy()

        # 保存结果（用于记录和服务器聚合）
        self.train_model_losses = avg_losses

        # 记录训练时间
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # 学习率衰减
        if self.learning_rate_decay:
            for scheduler in self.learning_rate_scheduler_set:
                scheduler.step()
            if self.use_rep_mode:
                for scheduler in self.learning_rate_scheduler_per_set:
                    scheduler.step()

    def select_best_model(self):
        """
        ⚠️ CRITICAL: 基于训练损失选择最优模型

        理论依据（methodology_stchset.md）：
        - k* = argmin_k L_i^train(θ_k)
        - 选择在客户端本地数据分布上表现最好的模型
        - 本地数据分布 = 训练数据（不是测试数据）
        """
        trainloader = self.load_train_data()
        for model_k in self.model_set:
            model_k.eval()

        total_losses = torch.zeros(self.K, device=self.device)
        total_samples = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                batch_size = y.shape[0]

                # 计算K个模型在训练数据上的 loss
                for k in range(self.K):
                    model = self.model_set[k]
                    output = model(x)
                    loss = self.loss(output, y)
                    total_losses[k] += loss.item() * batch_size

                total_samples += batch_size

            train_losses = (total_losses / total_samples).cpu().numpy()

        # 选择训练损失最小的模型
        best_idx = int(np.argmin(train_losses))

        self.best_model_idx = best_idx
        self.deployed_model = self.model_set[best_idx]

        return best_idx, train_losses  # 返回所有K个模型的损失

    def train_metrics(self):
        """
        计算训练指标（使用最优模型）
        """
        if self.deployed_model is None:
            self.select_best_model()

        trainloader = self.load_train_data(batch_size=self.test_batch_size)
        self.deployed_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.deployed_model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics(self):
        if self.deployed_model is None:
            self.select_best_model()

        testloaderfull = self.load_test_data()
        self.deployed_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                batch_size = y.shape[0]

                # 使用单个最优模型推理
                output = self.deployed_model(x)

                # 计算准确率
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += batch_size

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = 0

        return test_acc, test_num, auc