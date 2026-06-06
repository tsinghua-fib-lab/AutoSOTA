import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientIFCA(Client):
    """
    IFCA Client: Iterative Federated Clustering Algorithm

    Reference: Ghosh et al., "An Efficient Framework for Clustered Federated Learning", NeurIPS 2020

    Design:
    - Hard clustering: each client belongs to exactly one cluster
    - Two-phase per round:
        1. Evaluate K models on local data
        2. Select cluster with minimum loss (argmin)
        3. Train only the selected cluster's model

    vs FedFewS:
    - IFCA: Hard clustering (argmin), trains 1 model
    - FedFewS: Soft selection (softmax), trains K models with dual-layer weights
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # IFCA parameters
        self.K = args.num_server_models  # Number of clusters

        # Current cluster assignment (updated each round)
        self.cluster_id = None

        # Loss of K models on local data (for cluster selection)
        self.train_model_losses = None  # shape: (K,)

        # Best model for deployment (selected after training)
        self.best_model_idx = 0
        self.deployed_model = None

    def set_global_models(self, global_model_set):
        """
        Receive K global models from server (for cluster selection)

        Args:
            global_model_set: List of K global models
        """
        self.global_model_set = [copy.deepcopy(model) for model in global_model_set]

    def select_cluster(self):
        """
        ⚠️ CRITICAL: Hard clustering - select the cluster with minimum loss

        Algorithm:
            1. Evaluate K models on local training data
            2. c_i = argmin_k L_i(θ_k)  # Hard selection

        vs FedFewS:
            - FedFewS: Soft selection via w_ik = softmax(-L_i/μ)
            - IFCA: Hard selection via c_i = argmin_k L_i(θ_k)

        Returns:
            cluster_id: Selected cluster ID (0 to K-1)
            losses: Loss of K models on local data (for logging)
        """
        trainloader = self.load_train_data()

        losses = np.zeros(self.K)

        with torch.no_grad():
            for k, model in enumerate(self.global_model_set):
                model.eval()
                total_loss = 0.0
                total_samples = 0

                for x, y in trainloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output = model(x)
                    loss = self.loss(output, y)
                    total_loss += loss.item() * y.shape[0]
                    total_samples += y.shape[0]

                losses[k] = total_loss / total_samples

        # Hard selection: argmin
        self.cluster_id = int(np.argmin(losses))
        self.train_model_losses = losses

        return self.cluster_id, losses

    def set_parameters(self, model):
        """
        Receive the selected cluster's model from server (for training)

        Args:
            model: The global model of the selected cluster
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self):
        """
        ⚠️ CRITICAL: Train only the selected cluster's model

        vs FedFewS:
            - FedFewS: Trains all K models jointly
            - IFCA: Trains only 1 model (the selected cluster)
        """
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # Learning rate decay
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # Record training time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def select_best_model(self):
        """
        ⚠️ CRITICAL: Select best model for deployment based on training loss

        Same as FedFewS: k* = argmin_k L_i^train(θ_k)
        """
        trainloader = self.load_train_data()

        losses = np.zeros(self.K)

        with torch.no_grad():
            for k, model in enumerate(self.global_model_set):
                model.eval()
                total_loss = 0.0
                total_samples = 0

                for x, y in trainloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output = model(x)
                    loss = self.loss(output, y)
                    total_loss += loss.item() * y.shape[0]
                    total_samples += y.shape[0]

                losses[k] = total_loss / total_samples

        # Select best model
        best_idx = int(np.argmin(losses))

        self.best_model_idx = best_idx
        self.deployed_model = self.global_model_set[best_idx]

        return best_idx, losses

    def train_metrics(self):
        """
        Compute training metrics using the deployed model
        """
        if self.deployed_model is None:
            self.select_best_model()

        trainloader = self.load_train_data()
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
        """
        Compute test metrics using the deployed model
        """
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

                # Use deployed model for inference
                output = self.deployed_model(x)

                # Calculate accuracy
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += batch_size

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = 0

        return test_acc, test_num, auc
