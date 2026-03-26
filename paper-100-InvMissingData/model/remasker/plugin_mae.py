# stdlib
from typing import Any

# third party
import numpy as np
import math, sys
import json
import random
import pandas as pd
import torch, time
from torch import nn
from functools import partial
from utils import NativeScaler, MAEDataset, adjust_learning_rate, get_dataset
import model_mae 
from torch.utils.data import DataLoader, RandomSampler
import sys
from CKA import linear_CKA, kernel_CKA

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin
from hyperimpute.utils.benchmarks import compare_models

# configurations
from configs import get_configs


eps = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MAEPlugin(ImputerPlugin):

    def __init__(self, args) -> None:
        super().__init__()
        self._model = MAEImputation(args)

    @staticmethod
    def name():
        return 'mae'

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any):
        return self

    def _transform(self, X: pd.DataFrame):
        X = torch.tensor(X.values, dtype=torch.float32).to(device)
        self._model.fit(X)
        return self._model.transform(X).detach().cpu().numpy()


class MAEImputation:

    def __init__(self, args):

        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.norm_field_loss = args.norm_field_loss
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = args.warmup_epochs
        self.model = None
        self.norm_parameters = None
        self.note = args.note

    
        
        # configurable parameters
        configs = get_configs(args.dataset)
        self.embed_dim = configs['embed_dim'] if args.embed_dim is None else args.embed_dim
        self.depth = configs['depth'] if args.depth is None else args.depth
        self.decoder_depth = configs['decoder_depth'] if args.decoder_depth is None else args.decoder_depth
        self.num_heads = configs['num_heads'] if args.num_heads is None else args.num_heads
        self.mlp_ratio = configs['mlp_ratio'] if args.mlp_ratio is None else args.mlp_ratio
        self.max_epochs = configs['max_epochs'] if args.max_epochs is None else args.max_epochs
        self.mask_ratio = configs['mask_ratio'] if args.mask_ratio is None else args.mask_ratio
        self.encode_func = configs['encode_func'] if args.encode_func is None else args.encode_func
        
        
    def fit(self, X_raw: torch.Tensor):
        
        X = X_raw.clone()
        
        # Parameters
        no = len(X)
        dim = len(X[0, :])
        
        X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        
        for i in range(dim):
            min_val[i] = np.nanmin(X[:, i])
            max_val[i] = np.nanmax(X[:, i])
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] -  min_val[i] + eps)            
            
        self.norm_parameters = {"min": min_val, "max": max_val}

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(device)

        X = torch.nan_to_num(X)
        X = X.to(device)
        
        self.model = model_mae.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim, 
            depth=self.depth, 
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim, 
            decoder_depth=self.decoder_depth, 
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )

        # if self.improve and os.path.exists(self.path):     
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.model.to(device)
        #     return self

        self.model.to(device)

        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        eff_batch_size = self.batch_size * self.accum_iter 
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
        # param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
        # optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        # if self.resume and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.lr *= 0.5

        cka = {} 
        checkpoint = torch.load('/data/ting/remasker2/data/letter.pt')
        ox = checkpoint['ox'][:64]
        nox = len(ox)
        oxm = torch.ones_like(ox)
        oxm = oxm.float().to(device, non_blocking=True)
        ox = ox.unsqueeze(dim=1)
        ox = ox.to(device, non_blocking=True)
        
        self.model.train()

        miss_rates = [0.1, 0.3, 0.5, 0.7]
        for miss_rate in miss_rates:
            cka[miss_rate] = []

        for epoch in range(self.max_epochs):

            optimizer.zero_grad()
            total_loss = 0

            iter = 0
            for iter, (samples, masks) in enumerate(dataloader):

                # we use a per iteration (instead of per epoch) lr scheduler
                if iter % self.accum_iter == 0:
                    adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr, self.max_epochs, self.warmup_epochs)

                samples = samples.unsqueeze(dim=1)
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                # print(samples, masks)

                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = loss.item()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss /= self.accum_iter
                loss_scaler(loss, optimizer, parameters=self.model.parameters(), update_grad=(iter + 1) % self.accum_iter == 0)
                
                if (iter + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()
                
            total_loss = (total_loss/(iter+1))**0.5        
            
            if (epoch + 1) % 10 == 0:
                # check CKA 
                # print('epoch', epoch + 1, total_loss)
                with torch.no_grad():
                    ox_mask, _, _, _ = self.model.forward_encoder(ox, oxm, mask_ratio=0)
                    ox_mask = ox_mask[:, 1:, :].reshape((nox, -1))
                    
                    print('epoch', epoch + 1)
                    for miss_rate in miss_rates:
    
                        cka_vals = []
                        for _ in range(5):
                            mx_mask, _, _, _ = self.model.forward_encoder(ox, oxm, mask_ratio=miss_rate)
                            mx_mask = mx_mask[:, 1:, :].reshape((nox, -1))
                            cka_val = kernel_CKA(ox_mask.cpu(), mx_mask.cpu())
                            cka_vals.append(cka_val)
                        
                        print(miss_rate, cka_vals)
                        cka[miss_rate].append(cka_vals)

                
            # if total_loss < best_loss:
            #     best_loss = total_loss
            #     torch.save(self.model.state_dict(), self.path)
            # if (epoch + 1) % 50 == 0:
            #    print((epoch+1), '/', self.max_epochs, ':', total_loss)

        with open('/data/ting/remasker2/output/letter-cka' + '.json', 'w') as f:
            f.write(json.dumps(cka, indent=4))
            f.close()

        # torch.save(self.model.state_dict(), self.path)
        return self

    def transform(self, X_raw: torch.Tensor):
        
        X = X_raw.clone()

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]
        
        no, dim = X.shape
        X = X.cpu()

        # MinMaxScaler normalization
        for i in range(dim):
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)

        X = torch.from_numpy(X).to(device)
        M = M.to(device)
        
        self.model.eval()

        # Imputed data        
        with torch.no_grad():
          for i in range(no):
            sample = torch.reshape(X[i], (1, 1, -1))
            mask = torch.reshape(M[i], (1, -1))
            _, pred, _, _ = self.model(sample, mask)
            pred = pred.squeeze(dim=2)
            if i == 0:
                imputed_data = pred
            else:
                imputed_data = torch.cat((imputed_data, pred), 0) 
            
        # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()
        # print('imputed', imputed_data, M)
        # print('imputed', M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data)
        return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data


    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Imputes the provided dataset using the GAIN strategy.
        Args:
            X: np.ndarray
                A dataset with missing values.
        Returns:
            Xhat: The imputed dataset.
        """
        return self.fit(X).transform(X)



if __name__ == '__main__':

    from hyperimpute.plugins.imputers import Imputers
    from configs import get_args_parser
 

    # randomize every time
    # print(time.time())
    # random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    # torch.cuda.manual_seed(int(time.time()))
    np.random.seed(int(time.time()))
    # torch.backends.cudnn.deterministic = False

    args = get_args_parser().parse_args()
    X, y = get_dataset(args.dataset, args.path)

    # datasets = ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 'credit', 'raisin', 'bike', 'obesity', 'california', 'diabetes']
    # methods=['hyperimpute', 'miwae', 'EM', 'gain', 'ice', 'mean', 'median', 'mice', 'miracle', 'missforest', 'most_frequent', 'sinkhorn', 'softimpute']

    imputers = Imputers()
    imputers.add(MAEPlugin.name(), MAEPlugin)
    imputer = imputers.get('mae', args)

    # imputer.fit_transform(X)
    results = {}
    # for dataset in datasets:    
    
    results[args.dataset] = compare_models(
        name="exp1",
        evaluated_model=imputer,
        X_raw=X,
        ref_methods=[],
        scenarios=["MAR"],
        miss_pct=[0.3],
        n_iter=1,
        n_jobs=1
    )
    #  print(args.dataset)    
    # with open('/data/ting/mdi/exp1.json', 'w') as f:
    #     f.write(json.dumps(results, indent=4))
    #     f.close()
