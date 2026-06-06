from model.ot_imp import *


class KPI(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        sigma=[0.1,0.3,0.5,1,3,5],
        DEVICE = torch.device('cuda'),p_miss=0.3,loss='mae',stop=5
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE = DEVICE
        self.sigma = sigma
        self.p_miss = p_miss
        self.loss = loss
        self.lr = lr
        self.stop = stop

    def fit_transform(self, X: pd.DataFrame,GT=0, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        GT=GT[:,:X.shape[1]]
        train_X, valid_mask=valid_ampute(X, self.p_miss)
        # print(valid_mask);exit()
        train_mask = np.isnan(train_X) # train_mask is the mask of the training data, excluding the observations in the validation and test sets.
        if self.initializer is not None:
            imps = self.initializer.fit_transform(X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(train_mask.shape, device=DEVICE).double() + imps)[train_mask] # replace the missing values in the training data with the initial imputations.
        train_X = torch.tensor(train_X).to(DEVICE)
        train_X = train_X.clone()
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e
            
        if self.initializer is None:
            imps = (self.noise * torch.randn(train_mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        train_mask = torch.tensor(train_mask).to(DEVICE)
        valid_mask = torch.tensor(valid_mask).to(DEVICE)
        imps.requires_grad = True
        self.kernel_weight = nn.Parameter(torch.randn((len(self.sigma), 1, 1), requires_grad=True).to(DEVICE))
        self.ss_weight = nn.Parameter(torch.randn((len(self.sigma), 1, 1), requires_grad=True).to(DEVICE))
        self.bias1 = nn.Parameter(torch.randn((len(self.sigma), 1, 1), requires_grad=True).to(DEVICE))
        self.bias2 = nn.Parameter(torch.randn((len(self.sigma), 1, 1), requires_grad=True).to(DEVICE))

        optimizer = self.opt([imps,self.kernel_weight,self.ss_weight,self.bias1,self.bias2], lr=self.lr)
        tick = 0
        min_val = 999999999999
        epoch = 0
        for i in range(self.n_epochs):
            epoch += 1
            X_filled = train_X.detach().clone()
            X_filled[train_mask.bool()] = imps
            loss = 0
            loss_list = []
            loss_list2 = []
            for _ in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()
               
                    mask_t = train_mask[idx2, d].bool()
                    mask_v=valid_mask[idx2,d].bool()

                    """
                    The next two lines together compute the kernel matrix \( K^{\Delta}_{X^t X^s} \) in Equation (5)
                    The kernel matrix is computed using the `kernel_compute()` function.
                    """
                    kernel_fused_ts = torch.stack([kernel_compute(Xt, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma]) # Computing a stack of $E$ kernel matrices corresponding to different sigma hyperparameters.
                    kernel_fused_ts = ((kernel_fused_ts) * F.softmax(self.kernel_weight, dim=0)).sum(0) # Computing the ensembled kernel \( K^{\Delta}_{X^t X^s} \) with normalized weights.

                    """
                    The next two lines together compute the kernel matrix \( K^{\Delta}_{X^s X^s} \) in Equation (5)
                    The kernel matrix is computed using the `kernel_compute()` function.
                    """
                    kernel_fused_ss=torch.stack([kernel_compute(Xs, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma]) # Computing a stack of $E$ kernel matrices corresponding to different sigma hyperparameters.
                    kernel_fused_ss=((kernel_fused_ss)*F.softmax(self.ss_weight,dim=0)).sum(0) # # Computing the ensembled kernel \( K^{\Delta}_{X^s X^s} \) with normalized weights.

                    
                    """
                    The next line computes the model prediction term as K^{\Delta}_{X^t X^s} (K^{\Delta}_{X^s X^s} + λI)^{-1} Y_s in Equation (6)
                    """
                    pred = kernel_fused_ts @ torch.inverse(kernel_fused_ss + self.labda * torch.eye(Xt.shape[0]).to(DEVICE)) @ ys
                    
                    
                    """
                    The next line computes the MSE loss in Equation (6)
                    """
                    loss = (yt - pred) **2
                    
                    loss_list.append(loss[~mask_t])
                    loss_list2.append(loss[~mask_v].detach())
            
            loss = torch.concat(loss_list, axis=0).mean()
            vali_loss = torch.concat(loss_list2, axis=0).mean().item()
            
            if torch.isnan(loss).any() or torch.isinf(loss).any(): # If the loss is nan or inf, break the loop.
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            vali_loss = MAE(X_filled.detach().cpu().numpy(),GT,valid_mask.detach().cpu().numpy()) # Compute the validation loss.
            if GT is not None:
                test_loss = MAE(X_filled.detach().cpu().numpy(),GT,mask.detach().cpu().numpy()) # Compute the test loss.
                # vali_loss=loss.item()
                if vali_loss < min_val:
                    tick=0
                    min_val = vali_loss
                    res = X_filled.detach().cpu().numpy()
                else:
                    tick+=1
                if tick==self.stop:
                    break
                print(self.stop)
                print('tick: ',tick,f" {self.loss} :",'        test:',test_loss,'      min_val:',min_val)
            else:
                res=X_filled.detach().cpu().numpy()

        return res

