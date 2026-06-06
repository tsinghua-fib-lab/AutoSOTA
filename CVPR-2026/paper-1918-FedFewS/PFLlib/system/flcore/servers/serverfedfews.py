import copy
import torch
import random
import numpy as np
import time
from scipy.special import logsumexp
from flcore.servers.serverbase import Server
from flcore.clients.clientfedfews import clientFedFewS


class FedFewS(Server):
    """
    FedFewS: Few-for-Many Federated Learning via Smooth Tchebycheff Set Scalarization

    Theory: Approximates the Pareto front via STCH-Set with dual-layer weighting.

    Objective (methodology_stchset.md Section 4.1):
    g^{STCH-Set}(Θ) = μ log Σ_{i=1}^M (Σ_{k=1}^K exp(-L_i(θ_k)/μ))^{-1}

    Key Features:
    - Dual-layer weights: α_i (client importance) and w_{ik} (model-client matching)
    - Manual gradient aggregation: ∇_{θ_k} g = Σ_i α_i · w_{ik} · ∇_{θ_k} L_i(θ_k)
    - Client importance: α_i = S_i^{-1} / Σ_j S_j^{-1}, where S_i = Σ_k exp(-L_i(θ_k)/μ)

    Physical Meaning:
    - α_i: Clients with high loss on all models get higher weight (harder to train)
    - w_{ik}: Softly selects best model for each client
    - vs FedFew: FedFewS explicitly optimizes Pareto front, FedFew optimizes convex hull

    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # FedFewS parameters
        self.K = args.num_server_models  # 默认5
        self.mu = args.smooth_mu  # 平滑参数，默认0.001
        self.use_rep_mode = args.use_rep_mode  # Base-head 分离模式

        self.global_model_set = [copy.deepcopy(self.global_model) for _ in range(self.K)]

        # Optimizer Iter-4: Distinct Model Initialization (IDEA-005)
        # Reinitialize models k=1..K-1 with fresh random weights so K models
        # start from different points in parameter space, accelerating specialization.
        # The STCH-Set w_{ik} weights become non-uniform earlier, improving client-model matching.
        def _reset_parameters(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for k in range(1, self.K):
            self.global_model_set[k].apply(_reset_parameters)

        # 初始化客户端
        self.set_slow_clients()
        self.set_clients(clientFedFewS)

        # 打印配置信息
        self._print_configuration(args)

        self.Budget = []

        self.rs_model_losses = []  # 记录每个模型在每个客户端上的loss
        self.rs_alpha_weights = []  # 外层权重 α_i
        self.rs_soft_weights = []   # 内层权重 w_{ik}
        self.rs_global_model_losses = []  # K个模型在所有客户端混合数据上的平均loss
        self.rs_phase1_losses = []  # 每轮记录 K 个模型的平均训练损失

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model_set)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model_set)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """
        理论公式（methodology_stchset.md 第4.1节）：
        ∇_{θ_k} g^{STCH-Set} = Σ_{i=1}^M α_i · w_{ik} · ∇_{θ_k} L_i(θ_k)

        其中：
        - S_i = Σ_{k=1}^K exp(-L_i(θ_k)/μ)  (客户端i的归一化因子)
        - α_i = S_i^{-1} / Σ_{j∈B_t} S_j^{-1}  (客户端权重)
        - w_{ik} = exp(-L_i(θ_k)/μ) / S_i  (客户端-模型匹配)
        """
        assert len(self.selected_clients) > 0

        len_active = len(self.uploaded_ids)
        all_loss_values = np.zeros((len_active, self.K))
        for idx, ids in enumerate(self.uploaded_ids):
            all_loss_values[idx] = self.clients[ids].train_model_losses
        weights = np.array(self.uploaded_weights)

        all_loss_values = all_loss_values * weights[:, np.newaxis]

        alpha_weights, w_ik = self._compute_alpha_weights(all_loss_values)

        print(f"\n🔄 Aggregating models with STCH-Set weights:")
        # 计算模型差分
        param_diff_model_list = []
        for upload_model_set in self.uploaded_models:
            param_diff_model_k_list = []
            for k in range(self.K):
                model_k = upload_model_set[k]
                diff_param_model = copy.deepcopy(self.global_model_set[k])

                if self.use_rep_mode:
                    # Rep 模式：只计算 base 的差分
                    for diff_param, orig_param in zip(diff_param_model.base.parameters(), model_k.base.parameters()):
                        diff_param.data = orig_param.data - diff_param.data
                else:
                    # 标准模式：计算整个模型的差分
                    for diff_param, orig_param in zip(diff_param_model.parameters(), model_k.parameters()):
                        diff_param.data = orig_param.data - diff_param.data

                param_diff_model_k_list.append(diff_param_model)
            param_diff_model_list.append(param_diff_model_k_list)


        # 保存权重（用于统计显示）
        client_weights_map = {}  # {client_id: (alpha_i, w_ik)}

        for client_i in range(len(self.uploaded_ids)):
            for k in range(self.K):
                weight = alpha_weights[client_i] * w_ik[client_i][k]
                weight = weight[0]
                client_id = self.uploaded_ids[client_i]
                client_weights_map[client_id] = (alpha_weights[client_i], w_ik[client_i])

                if self.use_rep_mode:
                    # Rep 模式：只聚合 base 部分
                    for server_param, diff_param in zip(
                        self.global_model_set[k].base.parameters(),
                        param_diff_model_list[client_i][k].base.parameters()
                    ):
                        server_param.data.add_(diff_param.data, alpha=weight)
                else:
                    # 标准模式：聚合整个模型
                    for server_param, diff_param in zip(
                        self.global_model_set[k].parameters(),
                        param_diff_model_list[client_i][k].parameters()
                    ):
                        server_param.data.add_(diff_param.data, alpha=weight)
        # 收集统计信息（用于记录）- 直接使用已计算的权重
        round_model_losses = []
        round_alpha_weights = []
        round_soft_weights = []

        for client in self.clients:
            if hasattr(client, 'train_model_losses') and client.train_model_losses is not None:
                round_model_losses.append(client.train_model_losses.tolist())

                # ✅ 直接使用上面已计算好的权重
                if client.id in client_weights_map:
                    alpha_i, w_ik = client_weights_map[client.id]
                    round_alpha_weights.append(alpha_i)
                    round_soft_weights.append(w_ik.tolist())
                else:
                    # 未参与训练的客户端：使用默认值
                    round_alpha_weights.append(0.0)
                    round_soft_weights.append([1.0 / self.K] * self.K)
            else:
                round_model_losses.append([0.0] * self.K)
                round_alpha_weights.append(0.0)
                round_soft_weights.append([1.0 / self.K] * self.K)

        return round_model_losses, round_alpha_weights, round_soft_weights


    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n{'=' * 50}")
            print(f"Round {i}/{self.global_rounds}")
            print(f"{'=' * 50}")

            # 客户端训练
            print(f"\n🔄 Training clients:")
            for client in self.selected_clients:
                client.train()
                # 训练完成后立即打印该客户端的结果
                if client.train_model_losses is not None:
                    losses = client.train_model_losses
                    best_model = np.argmin(losses)
                    n_samples = client.train_samples

                    losses_str = "[" + ", ".join([f"{l:.3f}" for l in losses]) + "]"

                    print(f"  Client {client.id:2d} (n={n_samples:4d}): "
                          f"loss={losses_str} → Model {best_model}")

            # 评估条件：按 eval_gap 间隔评估，或者最后一轮强制评估
            should_evaluate = (i % self.eval_gap == 0) or (i == self.global_rounds)
            if should_evaluate:
                # 评估前先让所有客户端选择最优模型
                print(f"\n🔍 Model Selection (based on train loss):")
                model_counts = [0] * self.K

                for client in self.clients:
                    best_idx, train_losses = client.select_best_model()
                    model_counts[best_idx] += 1
                    losses_str = ", ".join([f"M{k}={train_losses[k]:.4f}" for k in range(self.K)])
                    print(f"  Client {client.id}: [{losses_str}] → selected Model {best_idx}")

                self.evaluate()
            self.receive_models()
            # 聚合模型
            round_losses, round_alphas, round_weights = self.aggregate_parameters()
            self.rs_model_losses.append(round_losses)
            self.rs_alpha_weights.append(round_alphas)
            self.rs_soft_weights.append(round_weights)


            # 外层权重 α_i（归一化）+ α_global（并排打印）
            alphas_array = np.array(round_alphas).squeeze()
            alpha_str = f"[{', '.join([f'{a:.4f}' for a in alphas_array])}]"
            print(f"  α_i (client importance): {alpha_str}")

            # 内层权重 w_{ik}（逐模型打印 - 转置视图）
            weights_array = np.array(round_weights)  # shape: (M, K)

            print(f"  w_{{ik}} (client-model matching, transposed):")
            num_models = weights_array.shape[1]  # K
            for k in range(num_models):
                # 所有客户端在模型 k 上的权重
                weights_per_model = [f"C{i}={weights_array[i, k]:.4f}" for i in range(len(weights_array))]

                weights_str = ", ".join(weights_per_model)
                print(f"    Model {k}: {weights_str}")

            # 5. 记录时间
            self.Budget.append(time.time() - s_t)
            print(f"Round {i} time cost: {self.Budget[-1]:.2f}s")

            # 6. 早停检查
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 训练结束
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best test accuracy: {max(self.rs_test_acc):.4f}")
        print(f"Average time per round: {sum(self.Budget[1:]) / len(self.Budget[1:]):.2f}s")
        print("=" * 50)

        self.save_results()
        self.save_global_model()

        # 新客户端评估（如果有）
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFedFewS)
            print("\n=== Fine-tuning new clients ===")
            self.evaluate()

    def evaluate(self, acc=None, loss=None):
        """
        评估所有客户端（每个客户端选择最优模型）
        """
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        losses = [l / n for l, n in zip(stats_train[2], stats_train[1])]

        print(f"  Averaged Train Loss: {train_loss:.4f}")
        print(f"  Averaged Test Acc: {test_acc:.4f}")
        print(f"  Averaged Test AUC: {test_auc:.4f}")
        print(f"  Std Test Acc: {np.std(accs):.4f}")

        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_client_test_acc.append(accs)
            self.rs_client_test_auc.append(aucs)
            self.rs_client_ids.append(stats[0])
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
            self.rs_client_train_loss.append(losses)
        else:
            loss.append(train_loss)

    def save_results(self):
        """
        保存结果（继承父类并添加FedFewS特有的双层权重）
        """
        import os
        import h5py

        # 解析 dataset：如果包含 /，则分为数据集名和数据划分配置
        # 例如：AGNews/noniid_dir_20_a0p1 -> 数据集 AGNews，配置 noniid_dir_20_a0p1
        if "/" in self.dataset:
            dataset_name, config_name = self.dataset.split("/", 1)
            # 结果保存在：result_dir/dataset_name/config_name/
            result_path = os.path.join(self.result_dir, dataset_name, config_name)
        else:
            # 兼容没有配置的情况（如老版本数据）
            dataset_name = self.dataset
            result_path = os.path.join(self.result_dir, dataset_name)

        # 确保目录存在
        os.makedirs(result_path, exist_ok=True)

        if len(self.rs_test_acc):
            # 文件名：algorithm_goal_times.h5（不包含 dataset 信息，因为已在目录中）
            algo = self.algorithm + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                # 保存全局平均指标
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

                # 保存每个客户端的详细指标
                if len(self.rs_client_test_acc) > 0:
                    hf.create_dataset('rs_client_test_acc', data=np.array(self.rs_client_test_acc))
                if len(self.rs_client_test_auc) > 0:
                    hf.create_dataset('rs_client_test_auc', data=np.array(self.rs_client_test_auc))
                if len(self.rs_client_train_loss) > 0:
                    hf.create_dataset('rs_client_train_loss', data=np.array(self.rs_client_train_loss))
                if len(self.rs_client_ids) > 0:
                    hf.create_dataset('rs_client_ids', data=np.array(self.rs_client_ids))

                # 保存最后一轮每个客户端的测试准确率
                if len(self.rs_client_test_acc) > 0:
                    final_client_acc = np.array(self.rs_client_test_acc[-1])
                    hf.create_dataset('final_client_test_acc', data=final_client_acc)
                    hf.create_dataset('final_client_acc_mean', data=np.mean(final_client_acc))
                    hf.create_dataset('final_client_acc_std', data=np.std(final_client_acc))
                    hf.create_dataset('final_client_acc_max', data=np.max(final_client_acc))
                    hf.create_dataset('final_client_acc_min', data=np.min(final_client_acc))

                    print("\n=== 最终轮次每个客户端的测试准确率 ===")
                    for i, acc in enumerate(final_client_acc):
                        print(f"  Client {i}: {acc:.4f}")
                    print(f"\n统计信息:")
                    print(f"  平均值: {np.mean(final_client_acc):.4f}")
                    print(f"  标准差: {np.std(final_client_acc):.4f}")
                    print(f"  最大值: {np.max(final_client_acc):.4f} (Client {np.argmax(final_client_acc)})")
                    print(f"  最小值: {np.min(final_client_acc):.4f} (Client {np.argmin(final_client_acc)})")
                    print("=" * 50)

                # 保存FedFewS特有的K个模型loss
                if len(self.rs_model_losses) > 0:
                    model_losses_array = np.array(self.rs_model_losses)
                    hf.create_dataset('rs_model_losses', data=model_losses_array)
                    print(f"Saved model losses with shape: {model_losses_array.shape}")

                # 保存双层权重
                if len(self.rs_alpha_weights) > 0:
                    alpha_weights_array = np.array(self.rs_alpha_weights)
                    hf.create_dataset('rs_alpha_weights', data=alpha_weights_array)
                    print(f"Saved alpha weights (α_i) with shape: {alpha_weights_array.shape}")

                if len(self.rs_soft_weights) > 0:
                    soft_weights_array = np.array(self.rs_soft_weights)
                    hf.create_dataset('rs_soft_weights', data=soft_weights_array)
                    print(f"Saved soft selection weights (w_{{ik}}) with shape: {soft_weights_array.shape}")

                # 保存混合分布loss
                if len(self.rs_global_model_losses) > 0:
                    global_losses_array = np.array(self.rs_global_model_losses)
                    hf.create_dataset('rs_global_model_losses', data=global_losses_array)
                    print(f"Saved global model losses (mixed distribution) with shape: {global_losses_array.shape}")

                # 保存 Phase 1 训练损失
                if len(self.rs_phase1_losses) > 0:
                    phase1_losses_array = np.array(self.rs_phase1_losses)
                    hf.create_dataset('rs_phase1_losses', data=phase1_losses_array)
                    print(f"Saved Phase 1 losses (K models) with shape: {phase1_losses_array.shape}")

                # 保存元数据
                hf.attrs['K'] = self.K
                hf.attrs['num_clients'] = self.num_clients
                hf.attrs['mu'] = self.mu

    def _compute_alpha_weights(self, all_loss_values):
        """
        ⚠️ CRITICAL: 数值稳定的外层权重计算（不存储配分函数 S）

        理论依据: docs/stable_todo.md - 对数空间归一化

        数学推导：
            α_i = S_i^{-1} / Σ_j S_j^{-1}
                = exp(-log(S_i)) / Σ_j exp(-log(S_j))
                = exp(-log(S_i) - logsumexp(-log(S_j)))

        其中配分函数：
            S_i = Σ_k exp(-L_i(θ_k)/μ)
            log(S_i) = logsumexp(-L_i(θ_k)/μ)

        数值稳定性保证：
            - 全程在对数空间操作，避免 exp 溢出/下溢
            - 避免显式计算 S_i 和 1/S_i，消除除零风险
            - 使用 scipy.special.logsumexp 保证 max-shift 技巧

        数学推导：
            w_{ik} = exp(-L_i(θ_k)/μ) / S_i
                   = exp(-L_i(θ_k)/μ) / Σ_k exp(-L_i(θ_k)/μ)
                   = softmax(-L_i/μ)_k
                   = exp(-L_i(θ_k)/μ - logsumexp(-L_i/μ))

        数值稳定性保证：
            - 直接使用 softmax 公式，避免显式计算 S_i
            - scipy.special.logsumexp 保证数值稳定
            - 结果严格满足 Σ_k w_{ik} = 1 和 w_{ik} ∈ [0, 1]

        Args:
            all_loss_values: M or M + 1 个客户端的损失值 (M x K)，可以是加权或原始损失

        Returns:
            alpha_weights: M or M + 1 个客户端的外层权重 (list of M floats)
        """

        # 步骤 1：计算所有 -log(S_i)
        logits = -all_loss_values / self.mu
        log_S_i = logsumexp(logits, axis=1, keepdims=True)
        log_w_ik = logits - log_S_i
        w_ik = np.exp(log_w_ik)

        # 步骤 2：logsumexp(-log(S_j)) for all j (M 或 M+1 个目标)
        log_sum_inv_S = logsumexp(-log_S_i)

        # 步骤 3：α_i = exp(-log(S_i) - log_sum_inv_S)
        alpha_weights = np.exp( -log_S_i - log_sum_inv_S)

        return alpha_weights, w_ik

    def _print_configuration(self, args):
        """
        打印 FedFewS 配置信息（私有方法）
        """
        print(f"\n{'='*60}")
        print(f"FedFewS Configuration (STCH-Set)")
        print(f"{'='*60}")
        print(f"Number of server models (K): {self.K}")
        print(f"Number of clients (M): {self.num_clients}")
        print(f"Join ratio: {self.join_ratio}")
        print(f"Local epochs: {self.local_epochs}")
        print(f"Smoothing parameter (μ): {self.mu}")
        print(f"Learning rate: {self.learning_rate}")
        if args.learning_rate_decay:
            print(f"Learning rate decay: Enabled (gamma={args.learning_rate_decay_gamma})")
        else:
            print(f"Learning rate decay: Disabled")
        if self.use_rep_mode:
            print(f"Training mode: Rep mode (base-head split)")
            print(f"  - Base: shared across clients (aggregated)")
            print(f"  - Head: personalized (not aggregated)")
        else:
            print(f"Training mode: Standard (full model aggregation)")
        print(f"\n📊 Optimization Method:")
        print(f"  🎯 M Objectives: {self.num_clients} clients only")
        print(f"  Objective: g = μ log Σ_i (Σ_k exp(-L_i(θ_k)/μ))^{{-1}}")
        print(f"{'='*60}\n")
