import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from hide_and_seek.perturbation_methods import (
    begin_knockoff_run,
    end_knockoff_run,
    fit_rf_samplers,
    perturb_X,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemperatureScaledSigmoid(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature  # Lower T → sharper push to 0/1
    
    def forward(self, x):
        return torch.sigmoid(x / self.temperature)

class net_hide(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 num_hidden_layers=1,
                 batchnorm=False):
        super().__init__()

        layers = []

        #fist layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batchnorm == True:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        
        #hidden layers: hidden -> hidden
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer: hidden → input
        layers.append(nn.Linear(hidden_dim, input_dim))
        # layers.append(nn.Sigmoid())
        layers.append(TemperatureScaledSigmoid(temperature=1))  # Optional: temperature scaling for sharper outputs
        self.net = nn.Sequential(*layers)
        
    def forward(self, true_x):
        mask = self.net(true_x)
        return mask
    
class net_seek(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 num_hidden_layers,
                 lmbda,
                 task='regression',
                 batchnorm=False,
                 num_classes=2
                 ):
        super().__init__()
        self.lmbda = lmbda
        self.last_loss = None #used for baseline model
        layers = []

        #fist layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batchnorm == True:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        
        #hidden layers: hidden -> hidden
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer: hidden → input
        if task == 'regression':
            layers.append(nn.Linear(hidden_dim, 1))

        elif task in ('classification', 'multiclass', 'multilabel'):
            layers.append(nn.Linear(hidden_dim, num_classes)) #note - loss functions expect logits, not probabilities

        self.net = nn.Sequential(*layers)
        
    def forward(self, true_x, perturbed_x,
                mask):

        x = mask * (true_x) + (1 - mask) * perturbed_x

        y_pred = self.net(x)
        return y_pred
    
    def baseline_forward(self, x):
        
        y_pred = self.net(x)
        return y_pred
    
    def predict_proba(self, x, clip_eps=None):
        """
        Only for binary classification
        method used for LIME - no masking

        notes: 
        - neural net needs x to be a tensor, LIME needs it to be numpy
        - LIME expects the output to be probabilities, not logits
        """
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            y_pred = self.baseline_forward(x)
            probs = F.softmax(y_pred, dim=1)
            probs = probs.numpy()

        if clip_eps is not None:
            probs = np.clip(probs, clip_eps, 1 - clip_eps)

        return probs

class hide_and_seek(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hide_hidden_dim,
                 seek_hidden_dim,
                 hide_num_hidden_layers,
                 seek_num_hidden_layers,
                 lmbda,
                 task='regression',
                 batchnorm=False,
                 num_classes=2
                 ):
        super().__init__()
        self.task = task
        self.lmbda = lmbda
        self.hide_hidden_dim = hide_hidden_dim
        self.seek_hidden_dim = seek_hidden_dim
        self.hide_num_hidden_layers = hide_num_hidden_layers
        self.seek_num_hidden_layers = seek_num_hidden_layers
        self.batchnorm = batchnorm
        self.num_classes = num_classes

        self.net_hide = net_hide(input_dim=input_dim,
                                 hidden_dim=hide_hidden_dim,
                                 num_hidden_layers=hide_num_hidden_layers,
                                 batchnorm=batchnorm)
        
        self.net_seek = net_seek(input_dim=input_dim,
                                 hidden_dim=seek_hidden_dim,
                                    num_hidden_layers=seek_num_hidden_layers,
                                 lmbda=lmbda,
                                task=task,
                                batchnorm=batchnorm,
                                num_classes=num_classes
                                 )
        
    def forward(self, true_x, perturbed_x):
        mask = self.net_hide(true_x)
        y_pred = self.net_seek(true_x, perturbed_x, mask)
        return y_pred, mask
        
def loss_mse(y_pred, y_true):
    """Calculate mean squared error between predictions and true values."""
    
    mse_loss = F.mse_loss(y_pred, y_true)
    
    return mse_loss

def custom_mse(y_pred, 
                y_true, 
                mask, 
                lmbda, 
                epoch=None,
                n_epochs=None,
                lmbda_exponent=2,
                return_separate_losses=False
                               ):

    mse_loss = F.mse_loss(y_pred, y_true)
    
    mask_mean_size = mask.mean(dim=1).mean()

    if epoch is not None and n_epochs is not None:
        # Adjust lambda dynamically based on epoch
        lmbda = lmbda * (epoch / n_epochs)**(lmbda_exponent)

    if return_separate_losses == False:
        return mse_loss + lmbda * mask_mean_size
    elif return_separate_losses == True:
        #this is used for analysis of the validation dataset
        return mse_loss, mask_mean_size

def cross_entropy(y_pred, 
                    y_true, 
                ):
    
    ce_loss = F.cross_entropy(y_pred, y_true)
    return ce_loss

# def compute_normalized_class_weight(y_train: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
#     """
#     w'_c = (1/pi_c)^alpha
#     w_c  = w'_c / (pi0*w'_0 + pi1*w'_1)  so that pi0*w0 + pi1*w1 = 1
#     Returns [w0, w1].
#     """
#     y = y_train.view(-1)
#     n0 = (y == 0).sum().item()
#     n1 = (y == 1).sum().item()
#     if n0 == 0 or n1 == 0:
#         raise ValueError("Both classes must be present.")
#     N = n0 + n1
#     pi0, pi1 = n0 / N, n1 / N
#
#     w0_raw = (1.0 / pi0) ** alpha
#     w1_raw = (1.0 / pi1) ** alpha
#
#     norm = pi0 * w0_raw + pi1 * w1_raw
#     w0 = w0_raw / norm
#     w1 = w1_raw / norm
#     return torch.tensor([w0, w1], dtype=torch.float32)

def custom_cross_entropy(y_pred, 
                         y_true, 
                         mask, 
                         lmbda,
                         epoch=None,
                         n_epochs=None,
                         lmbda_exponent=2,
                         return_separate_losses=False,
                         class_weights=None):
    """
    Args:
        y_pred (Tensor): Raw logits, shape (batch_size, num_classes)
        y_true (Tensor): one-hot encoded, shape (batch_size, num_classes)
                         
    Returns:
        Tensor: scalar loss
    """
    # if class_weights is None:
    ce_loss = F.cross_entropy(y_pred, y_true)
    # else:
    #     ce_loss = F.cross_entropy(y_pred, y_true, weight=class_weights)
        
    mask_mean_size = mask.mean(dim=1).mean()

    if epoch is not None and n_epochs is not None:
        # Adjust lambda dynamically based on epoch
        lmbda = lmbda * (epoch / n_epochs)**(lmbda_exponent)
    
    if return_separate_losses == False:
        return ce_loss + lmbda * mask_mean_size
    elif return_separate_losses == True:
        #this is used for analysis of the validation dataset
        return ce_loss, mask_mean_size

def custom_bce(y_pred,
               y_true,
               mask,
               lmbda,
               epoch=None,
               n_epochs=None,
               lmbda_exponent=2,
               return_separate_losses=False):
    """
    Binary cross-entropy + mask regularization for multilabel task.
    y_pred: logits (batch_size, num_labels)
    y_true: float binary targets (batch_size, num_labels)
    """
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    mask_mean_size = mask.mean(dim=1).mean()

    if epoch is not None and n_epochs is not None:
        lmbda = lmbda * (epoch / n_epochs) ** lmbda_exponent

    if not return_separate_losses:
        return bce_loss + lmbda * mask_mean_size
    return bce_loss, mask_mean_size


def train_nn(X_train,
             y_train,
             lmbda,
             n_epochs,
             task='classification',
                hide_hidden_dim=100,
                seek_hidden_dim=200,
                hide_num_hidden_layers=1,
                seek_num_hidden_layers=1,
                batch_size=None,
             seed=42,
             train_baseline=False,
             print_description='',
             batchnorm=False,
             num_classes=2,
             lmbda_exponent=2,
             return_losses_on_val=False,
             class_weight_alpha=None,
             perturbation_method='draw_marginal',
             warmup_epochs=0):
    if class_weight_alpha is not None:
        raise NotImplementedError("Class weighting is not used in our experiments. Set class_weight_alpha=None.")

    if task == 'regression':
        raise NotImplementedError(
            "Regression support is still in progress: it has not been tested against ground truth data. "
            "Only enable after testing results with ground truth knowledge and reviewing the code."
        )

    knockoff_run_key = None
    if perturbation_method == 'knock_off' and train_baseline is False:
        knockoff_run_key = begin_knockoff_run(seed=seed)

    y_scaler = None
    
    if train_baseline == True:
        print('training baseline model')
    else:
        print('training model')
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

    if task == 'regression':
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
    elif task in ('classification', 'multiclass'):
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    elif task == 'multilabel':
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    else:
        raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")

    torch.manual_seed(seed)

    input_dim = X_train_tensor.shape[1]
    
    if train_baseline == False or warmup_epochs > 0:
        model = hide_and_seek(input_dim=input_dim,
                            hide_hidden_dim=hide_hidden_dim,
                            seek_hidden_dim=seek_hidden_dim,
                                hide_num_hidden_layers=hide_num_hidden_layers,
                                seek_num_hidden_layers=seek_num_hidden_layers,
                        lmbda=lmbda,
                        task=task,
                        batchnorm=batchnorm,
                        num_classes=num_classes
                        )
    else:
        model = net_seek(input_dim=input_dim,
                         hidden_dim=seek_hidden_dim,
                         num_hidden_layers=seek_num_hidden_layers,
                         lmbda=None,
                         task=task,
                         batchnorm=batchnorm,
                         num_classes=num_classes)
    
    model = model.to(DEVICE)

    rf_samplers = None
    if perturbation_method == 'conditional_rf' and train_baseline is False:
        rf_samplers = fit_rf_samplers(
            X_train_tensor,
            n_estimators=100,
            min_samples_leaf=30,
            random_state=seed,
        )

    # ==== 3. Set up loss and optimizer ====
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==== 4. Train the model ====
    
    if batch_size is None: #this is the setting used in experiments. 
        if return_losses_on_val == True:
            n_val = int(0.1 * X_train_tensor.size(0))
            X_val_tensor = X_train_tensor[:n_val].clone()
            y_val_tensor = y_train_tensor[:n_val].clone()

            # Use the remaining 90% for actual training
            X_train_tensor_actual = X_train_tensor[n_val:].clone()
            y_train_tensor_actual = y_train_tensor[n_val:].clone()

            if task == 'regression':
                y_mean = y_train_tensor_actual.mean()
                y_std = y_train_tensor_actual.std(unbiased=False)
                y_std = torch.clamp(y_std, min=1e-8)
                y_train_tensor_actual = (y_train_tensor_actual - y_mean) / y_std
                y_val_tensor = (y_val_tensor - y_mean) / y_std
                y_scaler = {'mean': y_mean.item(), 'std': y_std.item()}

            losses_on_val = {}
        else:
            X_train_tensor_actual = X_train_tensor.clone()
            y_train_tensor_actual = y_train_tensor.clone()

            if task == 'regression':
                y_mean = y_train_tensor_actual.mean()
                y_std = y_train_tensor_actual.std(unbiased=False)
                y_std = torch.clamp(y_std, min=1e-8)
                y_train_tensor_actual = (y_train_tensor_actual - y_mean) / y_std
                y_scaler = {'mean': y_mean.item(), 'std': y_std.item()}

        # if task in ('classification', 'multiclass'):
        #     if class_weight_alpha is not None:
        #         class_weights = compute_normalized_class_weight(y_train_tensor_actual, alpha=class_weight_alpha)
        #         class_weights = class_weights.to(DEVICE)
        #     else:
        #         class_weights = None
        class_weights = None

        for epoch in range(n_epochs):
            model.train()

            optimizer.zero_grad()
            if train_baseline == True or (warmup_epochs > 0 and epoch < warmup_epochs):

                if train_baseline == True:
                    y_pred_train = model.baseline_forward(x=X_train_tensor_actual)
                else:
                    # Warmup phase: train only seek_net (no masking)
                    y_pred_train = model.net_seek.baseline_forward(x=X_train_tensor_actual)
                
                if task == 'regression':

                        loss = loss_mse(y_pred=y_pred_train,
                                        y_true=y_train_tensor_actual
                                        )
                elif task in ('classification', 'multiclass'):
                    loss = cross_entropy(y_pred=y_pred_train,
                                        y_true=y_train_tensor_actual,
                                      )
                elif task == 'multilabel':
                    loss = F.binary_cross_entropy_with_logits(y_pred_train, y_train_tensor_actual)
                else:
                    raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")

            else:
                X_train_tensor_shuffled = perturb_X(
                    X_train_tensor,
                    method=perturbation_method,
                    random_state=seed + epoch,
                    replace=True,
                    knockoff_run_cache_key=knockoff_run_key,
                    rf_samplers=rf_samplers,
                )

                if return_losses_on_val == True:
                    X_val_tensor_shuffled = X_train_tensor_shuffled[:n_val].clone()
                    X_train_tensor_shuffled_actual = X_train_tensor_shuffled[n_val:].clone()
                else:
                    X_train_tensor_shuffled_actual = X_train_tensor_shuffled.clone()

                y_pred_train, mask_train = model(true_x=X_train_tensor_actual,
                                            perturbed_x=X_train_tensor_shuffled_actual)
                
                if task == 'regression':

                        loss = custom_mse(y_pred=y_pred_train,
                                    y_true=y_train_tensor_actual,
                                                    mask=mask_train,
                                                    lmbda=lmbda,
                                                    epoch=epoch,
                                                    n_epochs=n_epochs,
                                                    lmbda_exponent=lmbda_exponent
                                                    )

                elif task in ('classification', 'multiclass'):

                    loss = custom_cross_entropy(y_pred=y_pred_train,
                                                y_true=y_train_tensor_actual,
                                                mask=mask_train,
                                                lmbda=lmbda,
                                                epoch=epoch,
                                                n_epochs=n_epochs,
                                                lmbda_exponent=lmbda_exponent,
                                                class_weights=class_weights
                                                )

                elif task == 'multilabel':

                    loss = custom_bce(y_pred=y_pred_train,
                                      y_true=y_train_tensor_actual,
                                      mask=mask_train,
                                      lmbda=lmbda,
                                      epoch=epoch,
                                      n_epochs=n_epochs,
                                      lmbda_exponent=lmbda_exponent
                                      )

                else:
                    raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")
            
            loss.backward()
            optimizer.step()

            model.eval()
            if return_losses_on_val == True:
                with torch.no_grad():
                    y_pred_val, mask_val = model(true_x=X_val_tensor,
                                                perturbed_x=X_val_tensor_shuffled)

                    if task == 'regression':
                        val_mse_loss, val_mask_mean_size = custom_mse(y_pred=y_pred_val,
                                                        y_true=y_val_tensor,
                                                        mask=mask_val,
                                                        lmbda=lmbda,
                                                        epoch=epoch,
                                                        n_epochs=n_epochs,
                                                        lmbda_exponent=lmbda_exponent,
                                                        return_separate_losses=True
                                                        )
                    elif task in ('classification', 'multiclass'):
                        val_ce_loss, val_mask_mean_size = custom_cross_entropy(y_pred=y_pred_val,
                                                        y_true=y_val_tensor,
                                                        mask=mask_val,
                                                        lmbda=lmbda,
                                                        epoch=epoch,
                                                        n_epochs=n_epochs,
                                                        lmbda_exponent=lmbda_exponent,
                                                        return_separate_losses=True,
                                                        class_weights=class_weights
                                                        )
                    elif task == 'multilabel':
                        val_ce_loss, val_mask_mean_size = custom_bce(y_pred=y_pred_val,
                                                        y_true=y_val_tensor,
                                                        mask=mask_val,
                                                        lmbda=lmbda,
                                                        epoch=epoch,
                                                        n_epochs=n_epochs,
                                                        lmbda_exponent=lmbda_exponent,
                                                        return_separate_losses=True
                                                        )
                    else:
                        raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")

                    epoch_losses_on_val = {}
                    if task == 'regression':
                        epoch_losses_on_val['val_mse_loss'] = val_mse_loss.item()
                    else:
                        epoch_losses_on_val['val_ce_loss'] = val_ce_loss.item()
                    epoch_losses_on_val['val_mask_mean_size'] = val_mask_mean_size.item()
                    epoch_losses_on_val['lmbda'] = lmbda
                    epoch_losses_on_val['lmbda_exponent'] = lmbda_exponent

                    losses_on_val[epoch] = epoch_losses_on_val

            if (epoch % (n_epochs/5) == 0) or (epoch == n_epochs - 1):
                if return_losses_on_val == True:
                    if task == 'regression':
                        print(f"""{print_description} 
                        Epoch: {epoch},
                        Loss: {loss.item():.4f}, 
                        val_mse_loss: {val_mse_loss.item():.4f}, 
                        val_mask_mean_size: {val_mask_mean_size.item():.4f}
                        """)
                    else:
                        print(f"""{print_description} 
                        Epoch: {epoch},
                        Loss: {loss.item():.4f}, 
                        val_ce_loss: {val_ce_loss.item():.4f}, 
                        val_mask_mean_size: {val_mask_mean_size.item():.4f}
                        """)
                else:
                    print(f"{print_description} | Epoch: {epoch} | Loss: {loss.item():.4f}")
            
    else: #batching - not used in our experiments.
        raise NotImplementedError("Batch training is not used in our experiments. Set batch_size=None to use full-batch training.")
        # if task == 'regression':
        #     y_mean = y_train_tensor.mean()
        #     y_std = y_train_tensor.std(unbiased=False)
        #     y_std = torch.clamp(y_std, min=1e-8)
        #     y_train_tensor = (y_train_tensor - y_mean) / y_std
        #     y_scaler = {'mean': y_mean.item(), 'std': y_std.item()}
        #
        # #not yet ready for regression
        # for epoch in range(n_epochs):
        #     model.train()
        #     total_loss = 0
        #
        #     if train_baseline == True:
        #         #might need to update this after having changed batch shuffle set up above and below
        #         for X_batch, idxs_batch, y_batch in dataloader:
        #             X_batch = X_batch.to(DEVICE)
        #             y_batch = y_batch.to(DEVICE)
        #
        #             optimizer.zero_grad()
        #
        #             y_pred_batch = model.baseline_forward(x=X_batch)
        #
        #             if task == 'regression':
        #                 loss = loss_mse(y_pred=y_pred_batch,
        #                                 y_true=y_batch
        #                                 )
        #             elif task in ('classification', 'multiclass'):
        #                 loss = cross_entropy(y_pred=y_pred_batch,
        #                                     y_true=y_batch
        #                                     )
        #             elif task == 'multilabel':
        #                 loss = F.binary_cross_entropy_with_logits(y_pred_batch, y_batch)
        #             else:
        #                 raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")
        #
        #             loss.backward()
        #             optimizer.step()
        #             total_loss += loss.item()
        #
        #     else:
        #         X_train_tensor_shuffled = perturb_X(
        #             X_train_tensor,
        #             method=perturbation_method,
        #             random_state=seed + epoch,
        #             replace=True,
        #             knockoff_run_cache_key=knockoff_run_key,
        #             rf_samplers=rf_samplers,
        #         )
        #         dataset = TensorDataset(X_train_tensor, X_train_tensor_shuffled, y_train_tensor)
        #         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        #
        #         if class_weight_alpha is not None:
        #             class_weights = compute_normalized_class_weight(y_train_tensor, alpha=class_weight_alpha)
        #             class_weights = class_weights.to(DEVICE)
        #         else:
        #             class_weights = None
        #
        #         for X_batch, X_batch_shuffled, y_batch in dataloader: #batching gave worse results. Perhaps this can be improved.
        #
        #             X_batch = X_batch.to(DEVICE)
        #             X_batch_shuffled = X_batch_shuffled.to(DEVICE)
        #             y_batch = y_batch.to(DEVICE)
        #
        #             optimizer.zero_grad()
        #
        #             y_pred_batch, mask_batch = model(true_x=X_batch,
        #                                             perturbed_x=X_batch_shuffled)
        #
        #             if task == 'regression':
        #                 loss = custom_mse(y_pred=y_pred_batch,
        #                                             y_true=y_batch,
        #                                             mask=mask_batch,
        #                                             lmbda=lmbda,
        #                                             epoch=epoch,
        #                                             n_epochs=n_epochs,
        #                                             lmbda_exponent=lmbda_exponent
        #                                             )
        #
        #             elif task in ('classification', 'multiclass'):
        #
        #                 loss = custom_cross_entropy(y_pred=y_pred_batch,
        #                                         y_true=y_batch,
        #                                         mask=mask_batch,
        #                                         lmbda=lmbda,
        #                                         epoch=epoch,
        #                                         n_epochs=n_epochs,
        #                                         lmbda_exponent=lmbda_exponent,
        #                                         class_weights=class_weights
        #                                         )
        #
        #             elif task == 'multilabel':
        #
        #                 loss = custom_bce(y_pred=y_pred_batch,
        #                                   y_true=y_batch,
        #                                   mask=mask_batch,
        #                                   lmbda=lmbda,
        #                                   epoch=epoch,
        #                                   n_epochs=n_epochs,
        #                                   lmbda_exponent=lmbda_exponent
        #                                   )
        #
        #             else:
        #                 raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")
        #
        #             loss.backward()
        #             optimizer.step()
        #             total_loss += loss.item()
        #
        #     if (epoch % (n_epochs // 5) == 0) or (epoch == n_epochs - 1):
        #         avg_loss = total_loss / len(dataloader)
        #         print(f"{print_description} Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
            
    output = {}
    output['model'] = model
    if task == 'regression' and y_scaler is not None:
        output['y_scaler'] = y_scaler
        model.y_scaler_mean = torch.tensor(y_scaler['mean'], dtype=torch.float32)
        model.y_scaler_std = torch.tensor(y_scaler['std'], dtype=torch.float32)
    if return_losses_on_val == True:
        output['losses_on_val'] = losses_on_val

    if knockoff_run_key is not None:
        # Keep the trained knockoff machine alive for pred_nn in the same run.
        model.__dict__['knockoff_run_key'] = knockoff_run_key

    if rf_samplers is not None:
        # Keep the fitted conditional-RF samplers alive for pred_nn in the same run.
        model.__dict__['rf_samplers'] = rf_samplers

    print('training finished')
    return output

def pred_nn(model,
            X_test,
            X_train,
            return_masks=True,
            seed=42,
            task='classification',
            perturbation_method='draw_marginal'
           ):
    
    release_knockoff_key = None

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # perturbed samples come from training data. first break dependencies, then draw len(X_test) samples
    run_cache_key = None
    if perturbation_method == 'knock_off':
        run_cache_key = getattr(model, 'knockoff_run_key', None)
        if run_cache_key is not None:
            release_knockoff_key = run_cache_key

    X_train_tensor_shuffled = perturb_X(
        X_train_tensor,
        method=perturbation_method,
        random_state=seed,
        replace=False,
        knockoff_run_cache_key=run_cache_key,
        rf_samplers=getattr(model, 'rf_samplers', None),
    )
    try:
        torch.manual_seed(seed)
        indices = torch.randint(low=0, high=len(X_train_tensor_shuffled), 
                                size=(len(X_test_tensor),), dtype=torch.long) #this happens with replacement
        X_train_tensor_shuffled = X_train_tensor_shuffled[indices] 

        X_train_tensor_shuffled = X_train_tensor_shuffled.to(DEVICE)
        X_test_tensor = X_test_tensor.to(DEVICE)
        
        # ==== 5. Evaluate on test data ====
        model.eval()
        with torch.no_grad():
            logit, mask_test = model(true_x=X_test_tensor,
                                          perturbed_x=X_train_tensor_shuffled)
            logit = logit.cpu()

            if task in ('classification', 'multiclass'):
                y_pred_test = torch.softmax(logit, dim=1).numpy() #probabilities
            elif task == 'regression':
                y_pred_test = logit.numpy()
                y_scaler_mean = getattr(model, 'y_scaler_mean', None)
                y_scaler_std = getattr(model, 'y_scaler_std', None)
                if y_scaler_mean is not None and y_scaler_std is not None:
                    y_pred_test = y_pred_test * y_scaler_std.item() + y_scaler_mean.item()
            elif task == 'multilabel':
                y_pred_test = torch.sigmoid(logit).numpy()
            else:
                raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")

            if return_masks == True:
                return  y_pred_test, mask_test.cpu().detach().numpy()
            else:
                return  y_pred_test
    finally:
        if release_knockoff_key is not None:
            end_knockoff_run(release_knockoff_key)
            if hasattr(model, 'knockoff_run_key'):
                delattr(model, 'knockoff_run_key')

