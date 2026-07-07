import torch
from torch import nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from model_utils import calculate_burden_to_AOP, calculate_utility, calculate_recall, calculate_burden_to_w_chosen, calculate_moving_ratio
import numpy as np
import traceback
from joblib import Parallel, delayed

class StrategicClassifier(nn.Module):
    def __init__(self, d):
        self.d = d
        super().__init__()
    
    def _create_proj_layer(self):
        pass
    
    def _get_w_chosen(self):
        pass

    def product(self, X):
        pass

    def w_norm(self):
        pass

    def forward(self, X, y_true):
        pass

    def evaluate(self, X_test, y_true):
        pass

class StrategicClassifierForWarmup(StrategicClassifier):
    def __init__(self, d, cost_scaling=1.0, strategic_aware=True):
        super().__init__(d)

        self.w = nn.Parameter(torch.randn(d))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.proj_layer = self._create_proj_layer()
        self.cost_scaling = cost_scaling
        self.strategic_aware = strategic_aware

    def product(self, X):
        return X @ self.w + self.b

    def w_norm(self):
        return torch.norm(self.w)
    
    def get_w_chosen(self):
        return self.w
    
    def get_b_chosen(self):
        return self.b
    
    def forward(self, X, y_true):
        if not self.strategic_aware:
            raw_movement_X = X
        else:
            raw_movement_X = self.calc_raw_movement(X, self.w, self.b)
        eps = 1e-3
        logits = self.product(raw_movement_X) + eps
        preds = torch.sign(logits).detach()
        accuracy = (preds.squeeze() == y_true.squeeze()).float().mean()
        total_burden_to_AOP, avg_burden_to_AOP = calculate_burden_to_AOP(self, raw_movement_X, y_true, logits, torch.norm(X - raw_movement_X, dim=1))
        pos_recall, neg_recall = calculate_recall(self, raw_movement_X, y_true, logits)
        total_burden_to_classifier, avg_burden_to_classifier = calculate_burden_to_w_chosen(self, X, y_true)
        total_utility, avg_utility = calculate_utility(preds, torch.norm(X - raw_movement_X, dim=1) * self.cost_scaling)
        pos_moving_ratio, neg_moving_ratio = calculate_moving_ratio(X, raw_movement_X, y_true)
        
        metrics_to_return ={
            "accuracy": accuracy,
            "total_burden_to_AOP": total_burden_to_AOP,
            "avg_burden_to_AOP": avg_burden_to_AOP,
            "total_burden_to_classifier": total_burden_to_classifier,
            "avg_burden_to_classifier": avg_burden_to_classifier,
            "total_utility": total_utility,
            "avg_utility": avg_utility,
            "pos_recall": pos_recall,
            "neg_recall": neg_recall,
            "pos_moving_ratio": pos_moving_ratio,
            "neg_moving_ratio": neg_moving_ratio 
        }
        return metrics_to_return

    def evaluate(self, X_test, y_true):
        metrics_to_return = self.forward(X_test, y_true)
        return metrics_to_return

    def calc_raw_movement(self, X, W, b):
        moved_X = []
        device = X.device
        
        for x in X:
            try:
                x_cpu = x.detach().cpu()
                W_cpu = W.detach().cpu()
                b_cpu = b.detach().cpu()
                
                x_proj, = self.proj_layer(x_cpu, W_cpu, b_cpu)
                
                x_proj = x_proj.to(device)
                
                if torch.norm(x - x_proj) > (2 / self.cost_scaling):
                    x_proj = x
            except:
                x_proj = x
            moved_X.append(x_proj)
        
        moved_X = torch.stack(moved_X)
        return moved_X

    def _create_proj_layer(self):
        x_proj = cp.Variable(self.d)
        x_input = cp.Parameter(self.d)

        W_param = cp.Parameter(self.d)
        b_param = cp.Parameter()

        objective = cp.Minimize(cp.sum_squares(x_proj - x_input))
        constraints = [W_param @ x_proj + b_param >= 0]

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        return CvxpyLayer(problem, parameters=[x_input, W_param, b_param], variables=[x_proj])
    
    def get_regularization_loss(self):
        reg_loss = torch.norm(self.w)
        return (reg_loss)

class StrategicClassifierFiniteSet(StrategicClassifier):
    def __init__(self, d, num_classifiers, tau=0.15, beta=2, dev=3, cost_scaling=1.0):
        super().__init__(d)
        self.num_classifiers = num_classifiers
        self.tau = tau
        self.beta = beta
        self.dev = dev
        self.cost_scaling = cost_scaling

        self.w_chosen = nn.Parameter(F.normalize(torch.randn(d), dim=0))

        self.classifiers_disguise = nn.ParameterList([
            nn.Parameter(torch.randn(d)) for _ in range(num_classifiers - 1)
        ])

        if d <= 1:
            self.b_disguise = nn.ParameterList([
            nn.Parameter(torch.tensor(-0.1)) for _ in range(num_classifiers - 1)
            ])
            self.b_chosen = nn.Parameter(torch.tensor(-0.1))
        else:
            self.b_disguise = nn.ParameterList([
                nn.Parameter(torch.tensor(-1.0)) for _ in range(num_classifiers - 1)
            ])
            self.b_chosen = nn.Parameter(torch.tensor(-1.0))

            
        with torch.no_grad():
            w_init = self.w_chosen.data
            w_norm = w_init.norm()

            for i, w in enumerate(self.classifiers_disguise):
                w_noisy = w_init + self.dev * torch.randn_like(w_init)
                w.copy_((w_noisy / w_noisy.norm()) * w_norm)

        self.proj_layer = self._create_proj_layer()

    def _create_proj_layer(self):
        x_proj = cp.Variable(self.d)
        x_input = cp.Parameter(self.d)

        num_total_classifiers = self.num_classifiers
        W_param = cp.Parameter((num_total_classifiers, self.d))
        b_param = cp.Parameter(num_total_classifiers)

        objective = cp.Minimize(cp.sum_squares(x_proj - x_input))

        x_aug = cp.hstack([x_proj, 1.0])
        W_aug = cp.hstack([W_param, cp.reshape(b_param, (num_total_classifiers, 1))])
        constraints = [W_aug @ x_aug >= 0]

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        return CvxpyLayer(problem, parameters=[x_input, W_param, b_param], variables=[x_proj])

    def product(self, X):
        return X @ self.w_chosen + self.b_chosen

    def w_norm(self):
        return torch.norm(self.w_chosen)
    
    def get_b_chosen(self):
        return self.b_chosen
    
    def get_w_chosen(self):
        return self.w_chosen
    
    def get_classifiers_with_bias(self):
        if self.num_classifiers < 2:
            raise ValueError("Number of classifiers must be at least 2 to include disguise classifiers.")
        W = torch.vstack(list(self.classifiers_disguise))
        b = torch.hstack(list(self.b_disguise))
        return W, b
    
    def get_all_classifiers_with_bias(self):
        W = torch.vstack([self.w_chosen] + list(self.classifiers_disguise))
        b = torch.hstack([self.b_chosen] + list(self.b_disguise))
        return W, b

    def get_regularization_loss(self, beta=2.0, eps=1e-8):
        """
        Compute the total regularization loss:
        - ||w_chosen|| (for weight decay)
        - Angle-based penalty for disguise classifiers
        """
        # Standard norm regularization
        reg_loss = torch.norm(self.w_chosen)
        reg_direction_loss = 0.0
        W, _ = self.get_all_classifiers_with_bias() 

        if W.shape[0] > 1:
            W_disguise = W[1:] 
            w0 = self.w_chosen
            w0_norm = w0.norm() + eps
            W_norms = W_disguise.norm(dim=1) + eps

            cos_alpha = (W_disguise @ w0) / (W_norms * w0_norm)
            cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)

            angle_penalty = torch.exp(self.beta * (1.0 - cos_alpha)) - 1.0

            reg_direction_loss = angle_penalty.mean()

        return reg_loss, reg_direction_loss
    
    def forward(self, X, y_true):
        W = torch.vstack([self.w_chosen] + list(self.classifiers_disguise))
        b = torch.hstack([self.b_chosen] + list(self.b_disguise))
        
        raw_moved_X = self.calc_raw_movement(X, W, b)
        margins = X @ W.T + b.unsqueeze(0) 
        two_norm = (2.0 / self.cost_scaling) * W.norm(dim=1)
        numerator = margins + two_norm  
        denominator = W @ self.w_chosen
        denominator = torch.where(
            denominator == 0,
            torch.tensor(1e-8, device=denominator.device),
            denominator
        )
        t_all = - numerator / denominator.unsqueeze(0) 
        t_max_soft = self.tau * torch.logsumexp(t_all / self.tau, dim=1) 
        w_norm_sq = torch.dot(self.w_chosen, self.w_chosen)
        w_norm_sq = torch.clamp(w_norm_sq, min=1e-8)

        t_chosen = - (X @ self.w_chosen + self.b_chosen) / w_norm_sq 

        t_per_point = -self.tau * torch.logsumexp(
            torch.stack([-t_chosen / self.tau, -t_max_soft / self.tau], dim=1),
            dim=1
        )
        t_per_point = torch.where(
            torch.isfinite(t_per_point),
            t_per_point,
            torch.zeros_like(t_per_point)
        )

        movement = t_per_point.unsqueeze(1) * self.w_chosen
        x_new = X + movement
        scores = x_new @ self.w_chosen + self.b_chosen

        values = torch.where(
            scores >= 0,
            torch.zeros_like(scores),
            -scores
        )
        eps = 1e-3
        logits = self.product(raw_moved_X) + eps
        preds = torch.sign(logits).detach()
        accuracy = self.calc_accuracy(raw_moved_X, X, y_true, W, b, logits)
        total_burden_to_AOP, avg_burden_to_AOP = calculate_burden_to_AOP(self, raw_moved_X, y_true, logits, torch.norm(X - raw_moved_X, dim=1))
        total_burden_to_classifier, avg_burden_to_classifier = calculate_burden_to_w_chosen(self, X, y_true)
        total_utility, avg_utility = calculate_utility(preds, torch.norm(X - raw_moved_X, dim=1) * self.cost_scaling)
        pos_recall, neg_recall = calculate_recall(self, raw_moved_X, y_true, logits)
        pos_moving_ratio, neg_moving_ratio = calculate_moving_ratio(X, raw_moved_X, y_true)

        metrics_to_return ={
            "values_of_proj": values,
            "accuracy": accuracy,
            "total_burden_to_AOP": total_burden_to_AOP,
            "avg_burden_to_AOP": avg_burden_to_AOP,
            "total_burden_to_classifier": total_burden_to_classifier,
            "avg_burden_to_classifier": avg_burden_to_classifier,
            "total_utility": total_utility,
            "avg_utility": avg_utility,
            "pos_recall": pos_recall,
            "neg_recall": neg_recall,
            "pos_moving_ratio": pos_moving_ratio,
            "neg_moving_ratio": neg_moving_ratio
        }
        return metrics_to_return

    def evaluate(self, X_test, y_true):
        metrics_to_return = self.forward(X_test, y_true)
        return metrics_to_return

    def calc_raw_movement(self, X, W, b):
        moved_X = []
        device = X.device
        
        for x in X:
            try:
                x_cpu = x.detach().cpu()
                W_cpu = W.detach().cpu()
                b_cpu = b.detach().cpu()
                
                x_proj, = self.proj_layer(x_cpu, W_cpu, b_cpu)
                
                x_proj = x_proj.to(device)
            except Exception as e:
                print(f"Projection failed for x={x}: {e}")
                traceback.print_exc()
                x_proj = x
            dist = torch.norm(x - x_proj)
            if dist <= 2 / self.cost_scaling:
                moved_X.append(x_proj.detach())
            else:
                moved_X.append(x.detach())
        
        moved_X = torch.stack(moved_X)
        return moved_X

    def calc_accuracy(self, raw_moved_X, X, y_true, W, b, logits=None):
        if logits is None:
            logits = raw_moved_X @ self.w_chosen + self.b_chosen
            eps = 1e-3
            logits = logits + eps
        preds = torch.sign(logits).detach()
        accuracy = (preds.squeeze() == y_true.squeeze()).float().mean()

        return accuracy

class StrategicClassifierInfiniteSet(StrategicClassifier):
    def __init__(self, d, cost_scaling=1.0, tau_max=0.1, tau_min=0.1, norm_constraint=5.0, norm_limit_type='l2', alpha=10):
        super().__init__(d)

        self.w_min = nn.Parameter(F.normalize(torch.randn(d), dim=0))
        dir = F.normalize(torch.randn(d), dim=0)
        center = dir * 1.0
        epsilon = 0.3 * torch.abs(center)
        self.w_min = nn.Parameter(center - epsilon)
        self.w_max = nn.Parameter(center + epsilon)
        with torch.no_grad():
            beta = torch.rand(d)
            self.w_chosen = nn.Parameter(
                (1 - beta) * self.w_min + beta * self.w_max
            )

        self.b = nn.Parameter(torch.tensor(-0.1))
        self.cost_scaling = cost_scaling
        self.norm_constraint = norm_constraint
        self.norm_limit_type = norm_limit_type
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.alpha = alpha

    def get_w_chosen(self):
        return self.w_chosen
    
    def get_b_chosen(self):
        return self.b

    def product(self, X):
        return X @ self.w_chosen + self.b

    def w_norm(self):
        return torch.norm(self.w_chosen)

    def get_w_max(self):
        return self.w_max

    def get_w_min(self):
        return self.w_min
    
    def get_regularization_loss(self):
        reg_loss = torch.norm(self.w_chosen) + torch.norm(self.w_min) + torch.norm(self.w_max)
        products = self.w_max * self.w_min
        smooth_max = (1/self.alpha) * torch.logsumexp(self.alpha * products, dim=0)
        aop_reg = F.softplus(-smooth_max)
        alignment_reg = self.alignment_reg()
        return reg_loss, aop_reg, alignment_reg

    def alignment_reg(self):
        w_chosen = self.get_w_chosen()
        w_min = self.get_w_min()
        w_max = self.get_w_max()
        
        w_chosen_dir = F.normalize(w_chosen, p=2, dim=0)
        
        prod_min = w_min * w_chosen_dir
        prod_max = w_max * w_chosen_dir
        temp = 50.0
        smooth_worst_features = -torch.logaddexp(-temp * prod_min, -temp * prod_max) / temp
        smooth_dot_product = torch.sum(smooth_worst_features)
        
        with torch.no_grad():
            w_worst_hard = torch.where(w_chosen_dir > 0, w_min, w_max)
            worst_norm = torch.norm(w_worst_hard, p=2) + 1e-8

        worst_case_cosine = smooth_dot_product / worst_norm
        
        margin = 0.1 
        alignment_loss = F.softplus(margin - worst_case_cosine, beta=5)
        
        return alignment_loss
    
    def forward(self, X, y_true):
        B, d = X.shape
        device = X.device
        w_chosen = self.get_w_chosen()

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(StrategicClassifierInfiniteSet.solve_single_sample)(
                X[i], self.w_min, self.w_max, self.b, d, self.cost_scaling
            ) for i in range(B)
        )
        
        X_proj_list = [r[0] for r in results]
        raw_x_proj = [r[1] for r in results]
        all_masks = [r[2] for r in results]
        num_iterations = [r[3] for r in results]
        max_num_iterations = max(num_iterations)

        t_responsible_list = []
        
        for i in range(B):
            x = X[i]
            masks = all_masks[i]

            if len(masks) == 0:
                t_responsible_list.append(torch.tensor(0.0, device=device, dtype=X.dtype))
                continue
        
            W_list = [torch.where(m.to(device).bool(), self.w_max, self.w_min) for m in masks]
            W = torch.stack(W_list, dim=0)

            margins = W @ x + self.b
            denominator = W @ w_chosen
            denominator = torch.where(denominator == 0, torch.tensor(1e-8, device=device), denominator)

            two_norm = (2.0 / self.cost_scaling) * W.norm(dim=1)
            numerator = margins + two_norm
            t_all = -numerator / denominator
            t_responsible = self.tau_max * torch.logsumexp(t_all / self.tau_max, dim=0)
            t_responsible_list.append(t_responsible)

        t_responsible_batch = torch.stack(t_responsible_list, dim=0)

        w_norm_sq = torch.dot(w_chosen, w_chosen).clamp(min=1e-8)
        t_chosen = -(X @ w_chosen + self.b) / w_norm_sq 
        
        t_per_point = -self.tau_min * torch.logsumexp(
            torch.stack([-t_chosen / self.tau_min, -t_responsible_batch / self.tau_min], dim=1),
            dim=1
        )
        
        movement = t_per_point.unsqueeze(1) * w_chosen
        x_new = X + movement

        scores = x_new @ w_chosen + self.b
        values = torch.where(scores >= 0, 0.0, -scores)
        
        eps = 1e-3
        X_proj_for_accuracy = torch.stack(raw_x_proj)
        
        logits = X_proj_for_accuracy @ w_chosen + self.b + eps
        preds = torch.sign(logits).detach()
        accuracy = self.calc_accuracy(X_proj_for_accuracy, y_true, logits)
        total_burden_to_AOP, avg_burden_to_AOP = calculate_burden_to_AOP(self, X_proj_for_accuracy, y_true, logits, torch.norm(X - X_proj_for_accuracy, dim=1))
        total_burden_to_classifier, avg_burden_to_classifier = calculate_burden_to_w_chosen(self, X, y_true)
        total_utility, avg_utility = calculate_utility(preds, torch.norm(X - X_proj_for_accuracy, dim=1)* self.cost_scaling)
        pos_recall, neg_recall = calculate_recall(self, X_proj_for_accuracy, y_true, logits)
        pos_moving_ratio, neg_moving_ratio = calculate_moving_ratio(X, X_proj_for_accuracy, y_true)

        metrics_to_return ={
            "values_of_proj": values,
            "accuracy": accuracy,
            "total_burden_to_AOP": total_burden_to_AOP,
            "avg_burden_to_AOP": avg_burden_to_AOP,
            "total_burden_to_classifier": total_burden_to_classifier,
            "avg_burden_to_classifier": avg_burden_to_classifier,
            "total_utility": total_utility,
            "avg_utility": avg_utility,
            "pos_recall": pos_recall,
            "neg_recall": neg_recall,
            "max_num_iterations": max_num_iterations,
            "pos_moving_ratio": pos_moving_ratio,
            "neg_moving_ratio": neg_moving_ratio
        }
        return metrics_to_return
    
    def calc_accuracy(self, raw_moved_X, y_true, logits=None):
        w_chosen = self.get_w_chosen()
        if logits is None:
            logits = raw_moved_X @ w_chosen + self.b
            eps = 1e-3
            logits = logits + eps
        preds = torch.sign(logits).detach()
        accuracy = (preds.squeeze() == y_true.squeeze()).float().mean()

        return accuracy

    def evaluate(self, X_test, y_true):
        metrics_to_return = self.forward(X_test, y_true)
        return metrics_to_return
    
    def validate_weights_in_bounds_and_fix(self):
        w_chosen = self.get_w_chosen()
        w_min = self.get_w_min()
        w_max = self.get_w_max()
        in_bounds = torch.all(w_chosen >= w_min) and torch.all(w_chosen <= w_max)
        if self.norm_limit_type == 'l2':
            norm_const = torch.norm(w_max - w_min, 2) <= self.norm_constraint
        elif self.norm_limit_type == 'l1':
            norm_const = torch.norm(w_max - w_min, 1) <= self.norm_constraint
        elif self.norm_limit_type == 'inf':
            norm_const = torch.norm(w_max - w_min, float('inf')) <= self.norm_constraint
        
        if not in_bounds or not norm_const:
            print("Weights out of bounds or norm constraint violated. Fixing weights.")
            self.fix_weights_to_bounds()
    
    def fix_weights_to_bounds(self):
        w_min_np = self.get_w_min().detach().cpu().numpy()
        w_chosen_np = self.get_w_chosen().detach().cpu().numpy()
        w_max_np = self.get_w_max().detach().cpu().numpy()

        d = w_min_np.shape[0]
        C = self.norm_constraint

        w_min_var = cp.Variable(d)
        w_chosen_var = cp.Variable(d)
        w_max_var = cp.Variable(d)

        objective = cp.Minimize(
            cp.sum_squares(w_min_var - w_min_np)
            + cp.sum_squares(w_chosen_var - w_chosen_np)
            + cp.sum_squares(w_max_var - w_max_np)
        )

        constraints = [
            w_min_var <= w_chosen_var,
            w_chosen_var <= w_max_var,
        ]
        if self.norm_limit_type == 'l1':
            constraints.append(cp.norm(w_max_var - w_min_var, 1) <= C)
        elif self.norm_limit_type == 'inf':
            constraints.append(cp.norm(w_max_var - w_min_var, 'inf') <= C)
        else:
            constraints.append(cp.norm(w_max_var - w_min_var, 2) <= C)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"CVXPY projection failed: {problem.status}")

        with torch.no_grad():
            self.w_min.copy_(torch.from_numpy(w_min_var.value).to(self.w_min.device))
            self.w_chosen.copy_(torch.from_numpy(w_chosen_var.value).to(self.w_chosen.device))
            self.w_max.copy_(torch.from_numpy(w_max_var.value).to(self.w_max.device))

    @staticmethod
    def solve_single_sample(x_tensor, w_min, w_max, b, d, cost_scaling):
        """
        Solves the iterative projection for a SINGLE sample.
        Returns: (x_proj, raw_moved_x, active_masks, iterations)
        """
        target_dtype = x_tensor.dtype
        x = x_tensor.detach().to(dtype=target_dtype)
        w_min = w_min.detach().to(dtype=target_dtype)
        w_max = w_max.detach().to(dtype=target_dtype)
        
        constraints_values = []
        constraints_masks = []
        max_iter = 2 ** d 
        
        iterations = 0
        
        x_proj = x.clone()

        for i in range(max_iter):
            iterations = i + 1 
            
            if len(constraints_values) == 0:
                x_proj = x.clone()
            else:
                x_np = x.cpu().numpy()
                W_np = np.stack([w.cpu().numpy().flatten() for w in constraints_values])
                b_val = b.item()
                
                x_var = cp.Variable(d)
                objective = cp.Minimize(cp.sum_squares(x_var - x_np))
                cons = [W_np @ x_var + b_val >= 0]
                problem = cp.Problem(objective, cons)
                
                try:
                    problem.solve(solver=cp.OSQP, verbose=False)
                except:
                    pass 
                
                if x_var.value is None or problem.status in ["infeasible", "unbounded"]:
                    x_proj = x.clone()
                else:
                    x_proj = torch.tensor(x_var.value, device=x.device, dtype=x.dtype)

            mask = (x_proj < 0)
            w_worst = torch.where(mask, w_max, w_min)

            score = x_proj @ w_worst + b
            if score.item() >= -1e-5:
                break

            is_duplicate = False
            for existing_mask in constraints_masks:
                if torch.equal(mask, existing_mask):
                    is_duplicate = True
                    break
            if is_duplicate:
                return x.clone(), x.clone(), [], iterations 

            constraints_values.append(w_worst)
            constraints_masks.append(mask)

        dist = torch.norm(x_proj - x, p=2)
        threshold = 2.0 / cost_scaling

        if dist <= threshold:
            raw_moved_x = x_proj.clone()
        else:
            raw_moved_x = x.clone()
                
        return x_proj, raw_moved_x, constraints_masks, iterations
