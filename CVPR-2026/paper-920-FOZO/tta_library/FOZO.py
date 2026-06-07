import os
import torch
import torch.nn as nn
import numpy as np
from models.vpt import PromptViT 
from quant_library.quant_layers.matmul import * 

# Initialize MSE loss for computing deep-shallow feature alignment error (Eq. 13)
criterion_mse = nn.MSELoss(reduction='none').cuda()

@torch.jit.script 
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the prediction entropy loss L_ent (Eq. 14 in the paper):
    L_ent = - \sum_{i=1}^B \sum_{a=1}^K p_{i,a} \log(p_{i,a})
    """
    temperature = 1 
    x = x / temperature  
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1) 
    return x

def forward_and_get_loss(images, model:PromptViT, fitness_lambda, train_info, shift_vector):
    """
    Performs forward pass and computes the total objective L (Eq. 15 in the paper):
    L = \lambda * L_stats + L_ent
    
    Args:
        fitness_lambda: The trade-off parameter \lambda (Eq. 15) balancing alignment and entropy.
        train_info: Pre-computed source domain statistics \mu^S and \sigma^S (Section 4.4.1).
        shift_vector: Feature shift vector used to mitigate distribution shifts.
    """
    # Corresponds to Algorithm 1, Step 8: Propagate Bt to obtain [CLS] token features {e_N^0}+/-
    features_per_layer = model.layers_cls_features_with_prompts(images) 
    
    num_layers = len(features_per_layer)
    if num_layers < 12:
        raise ValueError(f"Expected at least 12 layers, got {num_layers}")

    # Section 4.4.1: Grouping layers into Shallow (1...N/2) and Deep (N/2+1...N) sets
    first_half_features = torch.cat(features_per_layer[:num_layers//2], dim=1)
    second_half_features = torch.cat(features_per_layer[num_layers//2:], dim=1)

    # Calculate current batch mean \mu^T and standard deviation \sigma^T (Eq. 13)
    batch_std_first_half, batch_mean_first_half = torch.std_mean(first_half_features, dim=0)
    batch_std_second_half, batch_mean_second_half = torch.std_mean(second_half_features, dim=0)
    
    # Compute L_stats (Eq. 13): Aligning mean and variance for shallow and deep layer groups
    std_mse_first = criterion_mse(batch_std_first_half, train_info['first_half'][0])
    mean_mse_first = criterion_mse(batch_mean_first_half, train_info['first_half'][1])
    std_mse_second = criterion_mse(batch_std_second_half, train_info['second_half'][0])
    mean_mse_second = criterion_mse(batch_mean_second_half, train_info['second_half'][1])
    
    # IDEA-004: Weighted statistical alignment loss with deep layer emphasis
    deep_weight = 2.0  # Deep layers weighted more for semantic alignment
    discrepancy_loss = fitness_lambda * (std_mse_first.sum() + mean_mse_first.sum() + \
                                         deep_weight * (std_mse_second.sum() + mean_mse_second.sum())) * images.shape[0] / 64
    
    # Extract the last layer features for head prediction
    cls_features = features_per_layer[num_layers - 1] 
    
    if shift_vector is not None:
        cls_features = cls_features + 1.0 * shift_vector 
    
    # Corresponds to Algorithm 1, Step 9: Predict output \hat{Y}
    output = model.vit.head(cls_features) 
    
    # Compute Entropy Minimization Loss L_ent (Eq. 14)
    entropy_loss = softmax_entropy(output).sum()
    
    # IDEA-021: Conservative Entropy Minimization (COME) — prevent collapse
    # Add a penalty when batch entropy drops too low
    num_classes = output.shape[1]
    max_possible_entropy = torch.log(torch.tensor(num_classes, dtype=output.dtype, device=output.device))
    target_min_entropy = 0.25 * max_possible_entropy  # At least 25% of max entropy
    per_sample_ent = -(output.softmax(1) * output.log_softmax(1)).sum(1)
    batch_avg_entropy = per_sample_ent.mean()
    # Only penalize if average entropy is below target
    come_penalty = torch.clamp(target_min_entropy - batch_avg_entropy, min=0)
    come_weight = 1.0
    
    # Final Objective Eq. (15): L = \lambda * L_stats + L_ent
    # Corresponds to Algorithm 1, Step 10: Calculate l+ or l-
    loss = discrepancy_loss + entropy_loss + come_weight * come_penalty 
    
    batch_mean_combined = torch.cat((batch_mean_first_half, batch_mean_second_half), dim=0)
    return output, loss, batch_mean_combined


class FOZO(nn.Module):
    """
    Forward-Only Zeroth-Order (FOZO) Optimizer.
    Implements a backpropagation-free Test-Time Adaptation (TTA) paradigm.
    """
    def __init__(self, model:PromptViT, zo_eps=0.5, lr=0.08, fitness_lambda=0.4, n_spsa=1, log_dir='/root/FOA-main'):
        super().__init__()
        self.model = model
        self.zo_eps_initial = zo_eps 
        self.zo_eps = zo_eps # Perturbation scale \epsilon_t (Algorithm 1)
        self.lr_initial = lr 
        self.lr = lr # Learning rate \eta (Eq. 7)
        self.lr_min = 0.1 
        self.lr_max = 3.0
        self.lr_step_size = 0.1 
        self.fitness_lambda = fitness_lambda
        self.n_spsa = max(2, n_spsa) # Number of SPSA samples n (Eq. 6) — internally use >=2 for variance reduction
        self.zo_random_seed = [0]*self.n_spsa 
        self.projected_grad = [0]*self.n_spsa # Estimated scalar gradient values (Eq. 5)
        
        self.named_parameters_to_optim = []
        if model.num_prompts > 0:
            # Algorithm 1, Step 1: Initialize learnable prompts P
            self.named_parameters_to_optim.append(('input_prompts', model.prompts)) 
        
        self.hist_stat = None 
        self.batch_count = 0
        # EMA of prompt parameters for stable prediction (IDEA-001)
        self.prompt_ema = None
        self.ema_beta = 0.9
 

        # Dynamic Adjustment Parameters (Section 4.3 & Eq. 12)
        self.avg_loss = None # Historical average loss \bar{L}
        self.loss_alpha = 0.9 # Smoothing factor \beta (Eq. 12)
        self.loss_increase_threshold_factor = 1.05 # Threshold factor \tau (Eq. 12)
        self.eps_min = 0.2 
        self.eps_max = 0.5 
        self.eps_decay_factor = 0.9 # Decay rate \alpha (Eq. 12)

        self.metrics_log_path = None
        self._log_header_written = False
        if log_dir:
            self.metrics_log_path = os.path.join(log_dir, "fozo_dynamic_metrics.csv")
            os.makedirs(log_dir, exist_ok=True) 

    def _update_hist(self, batch_mean_tensor):
        """Updates historical feature statistics for calculating the shift vector.
        IDEA-010: Adaptive EMA based on feature drift magnitude.
        Larger drift -> faster adaptation (lower alpha)."""
        if self.hist_stat is None:
            self.hist_stat = batch_mean_tensor.clone() 
        else:
            # Compute feature drift magnitude
            drift = (batch_mean_tensor - self.hist_stat).norm().item()
            # Adaptive alpha: high drift -> lower smoothing (faster update)
            # Normal alpha=0.9 when drift is small, drops to 0.7 during large shifts
            base_alpha = 0.9
            min_alpha = 0.7
            # Scale alpha inversely with drift (clipped to [min_alpha, base_alpha])
            drift_scale = min(1.0, drift / 1.0)  # Normalize drift
            alpha = base_alpha - drift_scale * (base_alpha - min_alpha)
            self.hist_stat.mul_(alpha).add_(batch_mean_tensor, alpha=1-alpha)
            
    def _get_shift_vector(self):
        """Calculates difference between source mean and target historical mean to aid alignment."""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info['combined_stats'][1][-768:] - self.hist_stat[-768:]
    
    def zo_perturb_parameters(self, scaling_factor=1, seed=None):
        """
        Parameter perturbation (Algorithm 1, Steps 6-7):
        P+ = P + \epsilon_t * Z
        P- = P - \epsilon_t * Z
        where Z ~ N(0, I)
        """
        torch.manual_seed(seed)
        for _, param in self.named_parameters_to_optim:
            # Sample random direction Z using Rademacher (±1) for lower variance (Spall 1998)
            z = (torch.randint(0, 2, size=param.shape, device=param.device).float() * 2 - 1) 
            param.data.add_(z, alpha=scaling_factor * self.zo_eps) 
    
    def zo_forward(self, images, model:PromptViT, fitness_lambda, train_info, shift_vector):
        output, loss, batch_mean_combined = forward_and_get_loss(images, model, fitness_lambda, train_info, shift_vector)
        return output, loss, batch_mean_combined

    def zo_step(self, inputs):
        """
        SPSA-based gradient estimation (Algorithm 1, Steps 5-13):
        Estimates the gradient of the loss w.r.t prompts P via two forward passes (P+ and P-).
        """
        best_outputs = None
        best_loss = torch.tensor(np.inf)
        batch_mean1_list = [None]*self.n_spsa
        batch_mean2_list = [None]*self.n_spsa
        shift_vector = self._get_shift_vector()

        for j in range(self.n_spsa): 
            # Step 6: Record random seed for regenerating Z later
            self.zo_random_seed[j] = np.random.randint(1000000000) 
            
            # Steps 7 & 8: Positive perturbation P+ and loss l+
            self.zo_perturb_parameters(scaling_factor=1, seed=self.zo_random_seed[j]) 
            temp_output1, temp_loss1, temp_mean1 = self.zo_forward(
                inputs, self.model, self.fitness_lambda, self.train_info, shift_vector
            ) 
            batch_mean1_list[j] = temp_mean1
            if temp_loss1 < best_loss:
                best_loss = temp_loss1
                best_outputs = temp_output1
            
            # Steps 7 & 8: Negative perturbation P- and loss l-
            # scaling_factor=-2 because parameter is currently P+, we subtract 2*eps*Z to reach P-
            self.zo_perturb_parameters(scaling_factor=-2, seed=self.zo_random_seed[j]) 
            temp_output2, temp_loss2, temp_mean2 = self.zo_forward(
                inputs, self.model, self.fitness_lambda, self.train_info, shift_vector
            ) 
            batch_mean2_list[j] = temp_mean2
            if temp_loss2 < best_loss:
                best_loss = temp_loss2
                best_outputs = temp_output2

            # Core Zeroth-Order Gradient Estimation (Eq. 5): 
            # g(Z) = (l+ - l-) / (2 * \epsilon_t)
            self.projected_grad[j] = (temp_loss1 - temp_loss2) / (2 * self.zo_eps) 

            # Step 14: Restore parameters to original state P
            self.zo_perturb_parameters(scaling_factor=1, seed=self.zo_random_seed[j])
            
        avg_batch_mean1 = torch.stack(batch_mean1_list).mean(dim=0) if self.n_spsa > 0 else None
        avg_batch_mean2 = torch.stack(batch_mean2_list).mean(dim=0) if self.n_spsa > 0 else None

        return best_outputs, best_loss, avg_batch_mean1, avg_batch_mean2

    def zo_update(self):
        """
        Zeroth-order gradient update (Eq. 7 & Algorithm 1, Steps 15-17):
        P_{t+1} = P_t - (\eta / n) * \sum (g(Z_j) * Z_j)
        """
        for j in range(self.n_spsa): 
            torch.manual_seed(self.zo_random_seed[j]) # Regenerate the same Z_j
            for _, param in self.named_parameters_to_optim:
                z = (torch.randint(0, 2, size=param.shape, device=param.data.device).float() * 2 - 1)  # Rademacher
                # Perform gradient update (Eq. 7)
                param.data.addcmul_(z, self.projected_grad[j], value=-self.lr/self.n_spsa) 
                
    def forward(self, x):
        """
        Main FOZO Adaptation Pipeline (Algorithm 1, Steps 2-20)
        """
        self.batch_count += 1 

        # Execute n-SPSA gradient estimation
        best_outputs, current_loss, batch_mean1, batch_mean2 = self.zo_step(x) 

        # Update Exponential Moving Average (EMA) of historical loss \bar{L} (Eq. 12)
        if self.avg_loss is None:
            self.avg_loss = current_loss.item()
        else:
            self.avg_loss = self.loss_alpha * self.avg_loss + (1 - self.loss_alpha) * current_loss.item()

        # Dynamic Perturbation Strategy (Eq. 12)
        # Default: decay perturbation scale and learning rate to ensure convergence
        self.zo_eps *= self.eps_decay_factor
        self.lr *= 0.7 
        
        # Check if loss is increasing significantly (Lt > \tau * \bar{L})
        loss_is_increasing = current_loss.item() > self.avg_loss * self.loss_increase_threshold_factor

        if loss_is_increasing:
            # Section 4.3: Reset \epsilon upon domain shift/optimization stall to enhance exploration
            self.zo_eps = self.eps_max
            self.lr = self.lr_min 
        else:
            # Otherwise, increase learning rate to accelerate convergence
            self.lr = min(self.lr_max, self.lr + self.lr_step_size)

        # Clip parameters to valid ranges
        self.zo_eps = max(self.eps_min, min(self.eps_max, self.zo_eps))
        self.lr = max(self.lr_min, min(self.lr_max, self.lr))

        if self.metrics_log_path:
            with open(self.metrics_log_path, 'a') as f:
                if not self._log_header_written:
                    f.write("batch_count,current_loss,avg_loss,lr,zo_eps\n")
                    self._log_header_written = True
                f.write(f"{self.batch_count},{current_loss.item():.6f},{self.avg_loss:.6f},{self.lr:.6f},{self.zo_eps:.6f}\n")

        # Algorithm 1, Step 17: Update prompt parameters
        self.zo_update() 
        
        # IDEA-001: Apply EMA smoothing to model prompts for stable prediction
        if self.prompt_ema is None:
            self.prompt_ema = self.model.prompts.data.clone()
        else:
            # EMA: new_ema = beta * old_ema + (1-beta) * new_prompts
            self.prompt_ema.mul_(self.ema_beta).add_(self.model.prompts.data, alpha=1 - self.ema_beta)
            # Set model prompts to EMA-smoothed version
            self.model.prompts.data.copy_(self.prompt_ema) 
        
        if batch_mean1 is not None and batch_mean2 is not None:
            combined_batch_mean = (batch_mean1 + batch_mean2)
            self._update_hist(combined_batch_mean)
        
        # Step 19: Return prediction results from the best perturbation
        return best_outputs 
        
    def obtain_origin_stat(self, train_loader, stats_file="train_stats_2.pt"):
        """
        Pre-computation step: Extract source domain feature statistics (Section 4.4.1).
        Used for L_stats feature alignment during Test-Time Adaptation.
        """
        if os.path.exists(stats_file):
            try:
                print(f'===> Loading source statistics from {stats_file}')
                self.train_info = torch.load(stats_file)
                return
            except Exception:
                print('Cache mismatch, recalculating...')

        print('===> Pre-computing source feature statistics (\mu^S, \sigma^S)')
        all_layer_features_across_batches = [] 

        with torch.no_grad():
            for idx, dl in enumerate(train_loader):
                images = dl[0].cuda()
                # Collect [CLS] token activations for every layer
                features_per_layer_current_batch = self.model.layers_cls_features(images)
                
                if not all_layer_features_across_batches: 
                    all_layer_features_across_batches = [[] for _ in range(len(features_per_layer_current_batch))]
                
                for i, layer_feat in enumerate(features_per_layer_current_batch):
                    all_layer_features_across_batches[i].append(layer_feat)
            
        concatenated_features_per_layer = [torch.cat(feats_list, dim=0) for feats_list in all_layer_features_across_batches]
        num_layers = len(concatenated_features_per_layer)

        # Combine shallow and deep features to compute global source statistics
        first_half_features_combined = torch.cat(concatenated_features_per_layer[:num_layers//2], dim=1)
        second_half_features_combined = torch.cat(concatenated_features_per_layer[num_layers//2:], dim=1)
        all_features_combined = torch.cat(concatenated_features_per_layer, dim=1)

        self.train_info = {
            'first_half': torch.std_mean(first_half_features_combined, dim=0),
            'second_half': torch.std_mean(second_half_features_combined, dim=0),
            'combined_stats': torch.std_mean(all_features_combined, dim=0) 
        }
        
        torch.save(self.train_info, stats_file)
        
        # Prepare padding parameters for quantized models (INT8) (Section 5.3)
        for _, m in self.model.vit.named_modules():
            if type(m) == PTQSLBatchingQuantMatMul:
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,64)).cuda(), torch.zeros((1,12,64,197+self.model.num_prompts)).cuda())
            elif type(m) == SoSPTQSLBatchingQuantMatMul:
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,197+self.model.num_prompts)).cuda(), torch.zeros((1,12,197+self.model.num_prompts,64)).cuda())
        print('===> Source statistics computation complete')

    def reset(self):
        """Resets FOZO internal states for a new adaptation session."""
        self.hist_stat = None 
        self.model.reset() 
        self.prompt_ema = None
        self.batch_count = 0 
        self.avg_loss = None 
        self.zo_eps = self.zo_eps_initial 
        self.lr = self.lr_initial
        self._log_header_written = False