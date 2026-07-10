import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from .network import IMPACTNet
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
import time
from torch.autograd import grad
from operator import itemgetter
from tqdm import tqdm


class IMPACT:
    def __init__(self, epochs=10, batch_size=64, lr=1e-3, epoch_steps=-1, prt_steps=1,
                 rep_dim=64, hidden_dims='64', act='ReLU', bias=True,
                 kernel_size=2, dropout=0.0, maxpool_out_channels=8, normalize_embedding=True,
                 score_dim=3, weight=3, max_con_num=5, k=5,
                 total=1, val_weight=3, val_total=1, val_batch_size=64, val_ratio=0.8, lambd=1.0,
                 device='cuda', random_state=42, **kwargs):
        """
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch_steps = epoch_steps
        self.weight = weight
        self.total = total
        self.val_weight = val_weight
        self.val_total = val_total
        self.val_batch_size = val_batch_size
        self.val_ratio = val_ratio
        self.prt_steps = prt_steps
        self.rep_dim = rep_dim
        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.maxpool_out_channels = maxpool_out_channels
        self.normalize_embedding = normalize_embedding
        self.score_dim = score_dim
        self.max_con_num = max_con_num
        self.max_per_num = k
        self.max_ref_num = k
        self.lambd = lambd

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.n_features = None

        self.ref_x = []
        self.random_state = random_state
        self.set_seed(random_state)
        return

    def fit(self, x, y=None):
        self.train_data = x
        if y is None:
            y = np.zeros(len(x))
        self.train_label = y
        self.n_samples, self.len, self.n_features = x.shape
        n_anom = np.where(y == 1)[0].shape[0]
        n_norm = self.n_samples - n_anom
        if n_anom == 0:
            weight_map = {0: 1. / n_norm}
        else:
            weight_map = {0: 1. / n_norm, 1: 1. / (self.weight*n_anom)}

        dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
        train_size = int(self.val_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data, label in train_dataset],
                                        num_samples=len(train_dataset)//self.total, replacement=True)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, drop_last=True)
        if n_anom == 0:
            val_weight_map = {0: 1. / n_norm}
        else:
            val_weight_map = {0: 1. / n_norm, 1: 1. / (self.val_weight*n_anom)}

        val_sampler = WeightedRandomSampler(weights=[val_weight_map[label.item()] for data, label in val_dataset],
                                              num_samples=len(val_dataset)//self.val_total, replacement=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size, sampler=val_sampler, drop_last=True)
        self.val = iter(self.val_loader)

        self.model = IMPACTNet(
            n_features=self.n_features,
            n_hidden=self.hidden_dims,
            n_output=self.rep_dim,
            score_dim=self.score_dim,
            kernel_size=self.kernel_size,
            bias=self.bias,
            ref_size=self.max_ref_num,
            dropout=self.dropout,
            activation=self.act,
            maxpool_out_channels=self.maxpool_out_channels,
            normalize_embedding=self.normalize_embedding
        ).to(self.device)
        self.criterion = Multi_DeviationLoss(self.score_dim)
        print('Start training...')
        self._training()
        print('Training is done')

        return self

    def _training(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.model.train()
        for epoch in range(self.epochs):
            t1 = time.time()
            total_loss = 0
            cnt = 0
            for batch_x in self.train_loader:
                loss = self.training_forward(batch_x, epoch)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                cnt += 1
                # terminate this epoch when reaching assigned maximum steps per epoch
                if cnt > self.epoch_steps != -1:
                    break
            t = time.time() - t1
            if epoch == 0 or (epoch + 1) % self.prt_steps == 0:
                print(f'epoch{epoch + 1:3d}, '
                      f'training loss: {total_loss / cnt:.6f}, '
                      f'time: {t:.1f}s')

            if epoch == 0:
                self.epoch_time = t
            self.scheduler.step()
        return

    def decision_function(self, x, y=None):
        dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
        self.test_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            z_lst = []
            score_lst = []
            _iter_ = tqdm(self.test_loader, desc='testing: ')

            for batch_x in _iter_:
                batch_z, s = self.inference_forward(batch_x)
                z_lst.append(batch_z)
                score_lst.append(s)

        z = torch.cat(z_lst).data.cpu().numpy()
        scores = torch.cat(score_lst).data.cpu().numpy()

        return scores

    def training_forward(self, batch_x, epoch):
        x0, y0 = batch_x
        x0 = x0.float().to(self.device)
        y0 = y0.float().to(self.device)
        feature = self.model.feature_extractor(x0)
        abnormal_scores = self.model.seen_head(feature[y0 != 2])
        targets = y0 != 0
        loss = self.criterion(abnormal_scores, targets.float())
        # Reconstruction auxiliary loss for TCN stability (weight=0.05)
        pooled = self.model.feature_extractor.get_pooled(x0)
        recon = self.model.recon_decoder(feature)
        recon_loss = F.mse_loss(recon, pooled.detach())
        loss = loss + 0.02 * recon_loss

        if epoch >= 5:
            params = []
            for idx, (name, p) in enumerate(self.model.named_parameters()):
                if p.requires_grad and 'seen_head' in name:
                    params.append(p)
            normal_index = np.where(y0.data.cpu().numpy() == 0)[0]
            try:
                val_x, val_y = next(self.val)
            except StopIteration:
                self.val = iter(self.val_loader)
                val_x, val_y = next(self.val)
            val_x = val_x.float().to(self.device)
            val_y = val_y.float().to(self.device)
            test_feature, test_v = self.grad_z(val_x, val_y)
            test_h_estimate = test_v.copy()
            hv = hvp(loss, params, test_h_estimate)
            damp = 0.01
            scale = 25.0
            test_v = [x for x in test_v if x is not None]
            test_h_estimate = [x for x in test_h_estimate if x is not None]
            hv = [x for x in hv if x is not None]
            test_h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(test_v, test_h_estimate, hv)]
            influences = []
            z_vec_lst = []
            single_feature_lst = []
            z_train_lst = []
            for i in range(len(batch_x[0])):
                z_train, z_label = batch_x[0][i], batch_x[1][i]
                z_train, z_label = z_train.float().to(self.device).requires_grad_(True), z_label.float().to(self.device)
                z_train = self.train_loader.collate_fn([z_train])
                z_label = self.train_loader.collate_fn([z_label])
                z_train_lst.append(z_train)

                single_feature, grad_z_vec = self.grad_z(z_train, z_label)
                single_feature_lst.append(single_feature)

                tmp_influence = -sum(
                    [
                        torch.sum(k * j).data
                        for k, j in zip(grad_z_vec, test_h_estimate)
                    ]) / len(batch_x[0])
                influences.append(tmp_influence)
                z_vec_lst.append(grad_z_vec)

            normal_getter = itemgetter(*normal_index)
            normal_influence = normal_getter(influences)
            normal_helpful = np.argsort(torch.stack(normal_influence).data.cpu().numpy())
            normal_harmful = normal_helpful[::-1]

            new_feature = [f for f in feature]
            new_y = list(y0)
            con_num = np.sum(torch.stack(normal_influence).data.cpu().numpy() > 0)
            if con_num > 0:
                con_index = normal_index[normal_harmful[:min(con_num, self.max_con_num)]]
                for i in con_index:
                    new_y[i] = torch.tensor(1.0, device=self.device)

            per_index = normal_index[normal_harmful[min(con_num, self.max_con_num):min(con_num, self.max_con_num) + self.max_per_num]]
            ref_index = normal_index[normal_helpful[:self.max_ref_num]]

            for i in per_index:
                dot_v_h = sum([torch.sum(v_i * h_i) for v_i, h_i in
                               zip(z_vec_lst[i], test_h_estimate)])

                grad_wrt_input = torch.autograd.grad(dot_v_h, single_feature_lst[i], allow_unused=True)[0]
                new_feature[i] = (single_feature_lst[i].detach() + 0.02 * torch.sign(grad_wrt_input)).squeeze(0)
                new_y[i] = torch.tensor(2.0, device=self.device)

            new_feature = torch.stack(new_feature)
            new_y = torch.stack(new_y)
            self.ref_x.append(x0[ref_index].data.cpu())
            abnormal_scores = self.model.seen_head(new_feature[new_y != 2])
            dummy_scores = self.model.pseudo_head(new_feature[new_y != 1])

            targets = self.generate_target(new_y)
            outputs = list()
            for i in range(2):
                outputs.append(list())
            for i, scores in enumerate([abnormal_scores, dummy_scores]):
                outputs[i].append(scores)
            for i in range(2):
                outputs[i] = torch.cat(outputs[i], dim=1)

            losses = list()
            for i in range(2):
                losses.append(self.criterion(outputs[i], targets[i].float(), confidence_margin=5.).view(-1, 1))
            all_loss = torch.cat(losses)
            weights = torch.tensor([1.0, self.lambd], device=loss.device).view(-1, 1)
            all_loss = torch.sum(all_loss * weights)
            return all_loss
        return loss

    def inference_forward(self, batch_x):
        batch_x, batch_y = batch_x
        batch_z = batch_x.float().to(self.device)
        all_ref = torch.cat(self.ref_x, dim=0)
        x0 = batch_x.float().to(self.device)
        all_ref = all_ref.float().to(self.device)
        z_ = self.model.feature_extractor(all_ref)
        c = torch.mean(z_, dim=0)
        eps = 0.1

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        feature = self.model.feature_extractor(x0)
        abnormal_scores = self.model.seen_head(feature)
        dummy_scores = self.model.pseudo_head(feature)

        occ = DSADLoss(c=c)
        s = occ(feature, reduction='none')

        scores = abnormal_scores + dummy_scores
        scores = torch.max(scores, dim=1)[0] + s

        return batch_z, scores

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def grad_z(self, z, t):
        self.model.eval()
        feature = self.model.feature_extractor(z)
        abnormal_scores = self.model.seen_head(feature)
        loss = self.criterion(abnormal_scores, t, confidence_margin=15.)
        params = [p for name, p in self.model.named_parameters() if 'seen_head' in name]
        return feature, list(grad(loss, params, create_graph=True, allow_unused=True))

    def generate_target(self, target, eval=False):
        targets = list()
        if eval:
            targets.append(target)
            targets.append(target)
            return targets
        else:
            temp_t = target != 0
            targets.append(temp_t[target != 2])
            targets.append(temp_t[target != 1])
        return targets

def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        if grad_elem is None or v_elem is None:
            continue
        elemwise_products += torch.sum(grad_elem * v_elem)
    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True, allow_unused=True)

    return return_grads

class Multi_DeviationLoss(nn.Module):

    def __init__(self, score_dim):
        super().__init__()
        self.score_dim = score_dim

    def forward(self, y_pred, y_true, confidence_margin=5.):
        mean = torch.tensor([0.]*self.score_dim).cuda()
        std = torch.tensor([1.]*self.score_dim).cuda()
        y_true = y_true.unsqueeze(1).repeat(1, self.score_dim)
        ref = torch.normal(mean=mean.expand(5000, -1), std=std.expand(5000, -1))
        dev = (y_pred - torch.mean(ref, dim=0)) / torch.std(ref, dim=0)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        dev_loss = (1 - y_true) * inlier_loss + y_true * outlier_loss
        return torch.mean(dev_loss)

class DSADLoss(torch.nn.Module):
    def __init__(self, c, eta=1.0, eps=1e-6, reduction='mean'):
        super(DSADLoss, self).__init__()
        self.c = c
        self.reduction = reduction
        self.eta = eta
        self.eps = eps

    def forward(self, rep, semi_targets=None, reduction=None):
        dist = torch.sum((rep - self.c) ** 2, dim=1)

        if semi_targets is not None:
            loss = torch.where(semi_targets == 0, dist,
                               self.eta * ((dist+self.eps) ** semi_targets.float()))
        else:
            loss = dist

        if reduction is None:
            reduction = self.reduction
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss