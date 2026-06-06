import copy
import torch
import random
import numpy as np
import time
from flcore.servers.serverbase import Server
from flcore.clients.clientifca import clientIFCA


class IFCA(Server):
    """
    IFCA: Iterative Federated Clustering Algorithm

    Reference: Ghosh et al., "An Efficient Framework for Clustered Federated Learning", NeurIPS 2020

    Theory: Hard clustering with K independent models

    Algorithm:
        1. Client selects cluster: c_i = argmin_k L_i(θ_k)
        2. Client trains only the selected model
        3. Server aggregates each cluster independently (FedAvg)

    vs FedFewS:
        - IFCA: K independent models, hard clustering, per-cluster FedAvg
        - FedFewS: K coupled models, soft selection, dual-layer weighted aggregation

    Key Features:
        - Hard clustering: each client belongs to exactly one cluster
        - Independent aggregation: each cluster is a separate FedAvg
        - Empty cluster handling: model stays unchanged if no clients select it
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # IFCA parameters
        self.K = args.num_server_models  # Number of clusters
        self.global_model_set = [copy.deepcopy(self.global_model) for _ in range(self.K)]

        # Initialize clients
        self.set_slow_clients()
        self.set_clients(clientIFCA)

        # Print configuration
        self._print_configuration(args)

        self.Budget = []

        # Cluster statistics (for logging and analysis)
        self.rs_cluster_sizes = []  # Per-round cluster sizes: (T, K)
        self.rs_client_clusters = []  # Per-round client cluster assignments: (T, M)
        self.rs_model_losses = []  # Per-round K model losses per client: (T, M, K)

    def send_models(self):
        """
        Send K global models to all clients (for cluster selection)
        """
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            client.set_global_models(self.global_model_set)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """
        Receive models from active clients (only the selected cluster's model)

        Note: Unlike FedFewS which receives K models per client,
              IFCA receives only 1 model per client.
        """
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_cluster_ids = []  # Track which cluster each model belongs to

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                self.uploaded_cluster_ids.append(client.cluster_id)

        # Note: No normalization here - will be done per-cluster

    def aggregate_parameters(self):
        """
        ⚠️ CRITICAL: Per-cluster FedAvg aggregation

        Theory:
            - Each cluster k aggregates independently
            - Within cluster k: θ_k = Σ_{i∈C_k} (n_i / Σ_{j∈C_k} n_j) · θ_i
            - Empty clusters: model stays unchanged

        vs FedFewS:
            - FedFewS: Dual-layer weights (α_i, w_ik), all clients contribute to all models
            - IFCA: Per-cluster FedAvg, clients only contribute to their selected cluster
        """
        assert len(self.uploaded_ids) > 0

        # Group clients by cluster
        cluster_clients = {k: [] for k in range(self.K)}
        cluster_weights = {k: [] for k in range(self.K)}
        cluster_models = {k: [] for k in range(self.K)}

        for client_id, weight, model, cluster_id in zip(
            self.uploaded_ids, self.uploaded_weights, self.uploaded_models, self.uploaded_cluster_ids
        ):
            cluster_clients[cluster_id].append(client_id)
            cluster_weights[cluster_id].append(weight)
            cluster_models[cluster_id].append(model)

        print(f"\n🔄 Aggregating models per cluster:")

        # Aggregate each cluster independently
        cluster_sizes = [0] * self.K

        for k in range(self.K):
            if len(cluster_clients[k]) > 0:
                # Normalize weights within cluster
                total_samples = sum(cluster_weights[k])
                weights = [w / total_samples for w in cluster_weights[k]]

                # FedAvg aggregation
                self.global_model_set[k] = copy.deepcopy(cluster_models[k][0])
                for param in self.global_model_set[k].parameters():
                    param.data.zero_()

                for weight, model in zip(weights, cluster_models[k]):
                    for server_param, client_param in zip(
                        self.global_model_set[k].parameters(),
                        model.parameters()
                    ):
                        server_param.data += client_param.data * weight

                cluster_sizes[k] = len(cluster_clients[k])
                clients_str = ", ".join([f"C{cid}" for cid in cluster_clients[k]])
                print(f"  Cluster {k}: {cluster_sizes[k]} clients [{clients_str}]")
            else:
                # Empty cluster: model unchanged
                cluster_sizes[k] = 0
                print(f"  Cluster {k}: No clients (model unchanged)")

        return cluster_sizes

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n{'=' * 50}")
            print(f"Round {i}/{self.global_rounds}")
            print(f"{'=' * 50}")

            # Phase 1: Cluster selection
            print(f"\n🔍 Cluster Selection:")
            for client in self.selected_clients:
                cluster_id, losses = client.select_cluster()
                losses_str = "[" + ", ".join([f"{l:.3f}" for l in losses]) + "]"
                print(f"  Client {client.id:2d} → Cluster {cluster_id} (losses: {losses_str})")

            # Phase 2: Client training (only selected cluster's model)
            print(f"\n🔄 Training clients:")
            for client in self.selected_clients:
                # Set client model to the selected cluster's model
                client.set_parameters(self.global_model_set[client.cluster_id])
                client.train()
                print(f"  Client {client.id:2d} trained Cluster {client.cluster_id} model")

            # Evaluation (every eval_gap rounds or last round)
            should_evaluate = (i % self.eval_gap == 0) or (i == self.global_rounds)
            if should_evaluate:
                # Select best model for each client before evaluation
                print(f"\n🔍 Model Selection (based on train loss):")
                for client in self.clients:
                    # Set global models for evaluation
                    client.set_global_models(self.global_model_set)
                    best_idx, train_losses = client.select_best_model()
                    losses_str = ", ".join([f"M{k}={train_losses[k]:.4f}" for k in range(self.K)])
                    print(f"  Client {client.id}: [{losses_str}] → selected Model {best_idx}")

                self.evaluate()

            # Phase 3: Receive and aggregate
            self.receive_models()
            cluster_sizes = self.aggregate_parameters()

            # Collect statistics
            self._collect_statistics(cluster_sizes)

            # Record time
            self.Budget.append(time.time() - s_t)
            print(f"\nRound {i} time cost: {self.Budget[-1]:.2f}s")

            # Early stopping
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Training completed
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best test accuracy: {max(self.rs_test_acc):.4f}")
        print(f"Average time per round: {sum(self.Budget[1:]) / len(self.Budget[1:]):.2f}s")
        print("=" * 50)

        self.save_results()
        self.save_global_model()

        # New clients evaluation (if any)
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientIFCA)
            print("\n=== Fine-tuning new clients ===")
            self.evaluate()

    def evaluate(self, acc=None, loss=None):
        """
        Evaluate all clients (each using its best model)
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

    def _collect_statistics(self, cluster_sizes):
        """
        Collect cluster statistics for analysis

        Args:
            cluster_sizes: List of cluster sizes (K,)
        """
        self.rs_cluster_sizes.append(cluster_sizes)

        # Collect client cluster assignments
        client_clusters = [-1] * self.num_clients  # -1 for clients not selected
        for client in self.clients:
            if hasattr(client, 'cluster_id') and client.cluster_id is not None:
                client_clusters[client.id] = client.cluster_id
        self.rs_client_clusters.append(client_clusters)

        # Collect model losses per client
        round_model_losses = []
        for client in self.clients:
            if hasattr(client, 'train_model_losses') and client.train_model_losses is not None:
                round_model_losses.append(client.train_model_losses.tolist())
            else:
                round_model_losses.append([0.0] * self.K)
        self.rs_model_losses.append(round_model_losses)

    def save_results(self):
        """
        Save results (including IFCA-specific cluster statistics)
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
                # Global metrics
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

                # Per-client metrics
                if len(self.rs_client_test_acc) > 0:
                    hf.create_dataset('rs_client_test_acc', data=np.array(self.rs_client_test_acc))
                if len(self.rs_client_test_auc) > 0:
                    hf.create_dataset('rs_client_test_auc', data=np.array(self.rs_client_test_auc))
                if len(self.rs_client_train_loss) > 0:
                    hf.create_dataset('rs_client_train_loss', data=np.array(self.rs_client_train_loss))
                if len(self.rs_client_ids) > 0:
                    hf.create_dataset('rs_client_ids', data=np.array(self.rs_client_ids))

                # Final client accuracy statistics
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

                # IFCA-specific: cluster statistics
                if len(self.rs_cluster_sizes) > 0:
                    cluster_sizes_array = np.array(self.rs_cluster_sizes)
                    hf.create_dataset('rs_cluster_sizes', data=cluster_sizes_array)
                    print(f"Saved cluster sizes with shape: {cluster_sizes_array.shape}")

                if len(self.rs_client_clusters) > 0:
                    client_clusters_array = np.array(self.rs_client_clusters)
                    hf.create_dataset('rs_client_clusters', data=client_clusters_array)
                    print(f"Saved client cluster assignments with shape: {client_clusters_array.shape}")

                if len(self.rs_model_losses) > 0:
                    model_losses_array = np.array(self.rs_model_losses)
                    hf.create_dataset('rs_model_losses', data=model_losses_array)
                    print(f"Saved model losses with shape: {model_losses_array.shape}")

                # Metadata
                hf.attrs['K'] = self.K
                hf.attrs['num_clients'] = self.num_clients

    def _print_configuration(self, args):
        """
        Print IFCA configuration
        """
        print(f"\n{'='*60}")
        print(f"IFCA Configuration")
        print(f"{'='*60}")
        print(f"Number of clusters (K): {self.K}")
        print(f"Number of clients (M): {self.num_clients}")
        print(f"Join ratio: {self.join_ratio}")
        print(f"Local epochs: {self.local_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        if args.learning_rate_decay:
            print(f"Learning rate decay: Enabled (gamma={args.learning_rate_decay_gamma})")
        else:
            print(f"Learning rate decay: Disabled")
        print(f"\n📊 Clustering Method:")
        print(f"  🎯 Hard Clustering: c_i = argmin_k L_i(θ_k)")
        print(f"  Each client belongs to exactly one cluster")
        print(f"  Each cluster aggregates independently (FedAvg)")
        print(f"{'='*60}\n")
