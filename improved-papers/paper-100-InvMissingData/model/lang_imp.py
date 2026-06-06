import numpy as np
import torch
from loguru import logger
# from ot_utils import nanmean, MAE, RMSE
import ot
import logging
from utils.utils import MAE, RMSE, nanmean
from torch.utils.data import DataLoader

from score import score_tuple
from score.utils_score import parzen_window_pdf
from score.score_trainer import Trainer
from score.xscore import vammped_parzen_score
from score.SteinVGD import xRBF
import time


"""
"""

class NeuralLangImputer(object):
    def __init__(self, initializer=None, entropy_reg=10.0, eps=0.01, lr=1.0e-1,
                 opt=torch.optim.Adam, niter=50, kernel_func=xRBF,
                 mlp_hidden=[256, 256], score_net_epoch=2000, score_net_lr=1.0e-3, score_loss_type="mask_dsm",
                 log_pdf=parzen_window_pdf, bandwidth=10.0, sampling_step=500, log_path="./neuralGFImpute",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 batchsize=128, n_pairs=1, noise=0.1, scaling=.9):
        super(NeuralLangImputer, self).__init__()
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.scaling = scaling
        self.sampling_step = sampling_step
        self.device = device
        self.mlp_hidden = mlp_hidden
        self.score_net_epoch = score_net_epoch
        self.score_loss_type = score_loss_type
        self.score_net_lr = score_net_lr
        self.log_path = log_path
        self.initializer = initializer
        self.entropy_reg = entropy_reg

        # if os.path.exists(os.path.join("./", self.log_path)):
        #     shutil.rmtree(os.path.join("./", self.log_path))
        # self.writer = score_tuple.logWriter(os.path.join("./", self.log_path))
        self.bandwidth = bandwidth
        self.score_func = vammped_parzen_score


        # kernel func and concerning grad
        self.kernel_func = kernel_func(self.bandwidth)
        self.grad_val_kernel = self.kernel_func

        # log pdf func and concerning grad
        self.log_pdf = log_pdf


        # self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
        # self.data_step = SVGDScoreFunction(score_func=self.log_pdf, kernel_func=rbf_kernel(1.0))


    def _sum_kernel(self, X, Y):
        """
        wrap the kernel function to obtain grad_and_values
        :return: scalar, kernel value
        """
        K_XY = self.kernel_func(X, Y)
        return torch.sum(K_XY), K_XY

    def svgd_sampling(self, data, bandwidth, data_number, score_func, grad_optim, iter_steps, mask_matrix):
        """
        svgd sampling function
        :param data:
        :param data_number:
        :param score_func:
        :param grad_optim:
        :param iter_steps:
        :param mask_matrix:
        :return:
        """
        for _ in range(iter_steps):
            with torch.no_grad():
                eval_score = score_func(data)
                eval_grad_k, eval_value_k = self.grad_val_kernel(data, data)

            # svgd gradient
            grad_tensor = -1.0 * (torch.matmul(eval_value_k, eval_score) - self.entropy_reg * eval_grad_k) / data_number
            # grad_tensor = -1.0 * eval_score
            if torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
                ### Catch numerical errors/overflows (should not happen)
                logging.info("Nan or inf loss")
                break
            # mask the corresponding values
            grad_optim.zero_grad()
            data.grad = grad_tensor
            grad_optim.step()
        return data

    def train_score_net(self, train_dataloader, outer_loop, score_trainer, grad_mask):
        """
        score network training function
        :param train_dataloader:
        :param outer_loop:
        :param score_trainer:
        :return:
        """
        for e in range(self.score_net_epoch):
            total_loss = 0.0
            for _, data in enumerate(train_dataloader):
                loss = score_tuple.train_step_ab(data, score_trainer, grad_mask)
                total_loss = total_loss + loss.item()


    def fit_transform(self, X, verbose=True, report_interval=10, X_true=None, OTLIM=5000):
        X, X_true = torch.tensor(X), torch.tensor(X_true)
        X = X.clone()
        n, d = X.shape

        # define the score network structure and corresponding trainer
        self.mlp_model = score_tuple.ToyMLP(input_dim=d, units=self.mlp_hidden).to(self.device)
        self.score_net = score_tuple.Energy(net=self.mlp_model).to(self.device)




        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2 ** e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")


        mask = torch.isnan(X).double()

        if self.initializer is not None:
            imps = self.initializer.fit_transform(X)
            imps = torch.tensor(imps).double().to(self.device)
            imps = (self.noise * torch.randn(mask.shape, device=self.device).double() + imps)[mask]
        else:
            imps = (self.noise * torch.randn(mask.shape).double() + (1) * nanmean(X, 0))[mask.bool()]
        grad_mask = ~mask.bool()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        X_filled = X_filled.to(self.device)
        X_filled.requires_grad_()
        grad_mask = grad_mask.to(self.device)

        optimizer = self.opt([X_filled], lr=self.lr)

        if verbose:
            logging.info(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)
            result_list = []

        for i in range(self.niter):
            logger.info(f"进来内圈循环了！")

            # trian the score network
            score_trainer = Trainer(model=self.score_net, loss_type=self.score_loss_type)
            train_start_time = time.time()
            train_dataset = score_tuple.MyDataset(data=X_filled)

            train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=n)
            self.train_score_net(train_dataloader, i, score_trainer, grad_mask)
            train_end_time = time.time()

            model_start_time = time.time()
            # fill the dataset with SVGD
            X_filled = self.svgd_sampling(data=X_filled, data_number=n, score_func=self.score_net.functorch_score,
                                          bandwidth=self.bandwidth,
                                          grad_optim=optimizer,
                                          # grad_optim=self.opt([X_filled], lr=self.lr),
                                          iter_steps=self.sampling_step, mask_matrix=grad_mask)
            model_end_time = time.time()


            if X_true is not None:
                maes[i] = MAE(X_filled.detach().cpu().numpy(), X_true.detach().cpu().numpy(), mask.detach().cpu().numpy()).item()
                rmses[i] = RMSE(X_filled.detach().cpu().numpy(), X_true.detach().cpu().numpy(), mask.detach().cpu().numpy()).item()

                if n <= OTLIM:
                    M = mask.sum(1) > 0
                    nimp = M.sum().item()
                    dist = ((X_filled.detach().cpu().numpy()[M][:, None] - X_true.detach().cpu().numpy()[M]) ** 2).sum(2) / 2.
                    wass = ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dist)
                else:
                    wass = 0.0

                logger.info(f'We are reporting testing results: Iteration {i + 1}:\t Loss: na\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')
            if verbose and ((i + 1) % report_interval == 0):

                if X_true is not None:
                    # self.writer.record(loss_item={"impute/RMSE": rmses[-1], "impute/MAE": maes[-1]}, step=i)
                    logging.info(f'Iteration {i + 1}:\t Loss: na\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')
                    result_dict = {"hidden": str(self.mlp_hidden), "entropy_reg": self.entropy_reg,
                                   "bandwidth": self.bandwidth,
                                   "score_epoch": self.score_net_epoch,  "interval": i,
                                   "rmse": rmses[i], "mae": maes[i], "wass": wass,
                                   "train_time": train_end_time - train_start_time,
                                   "imp_time": model_end_time - model_start_time}
                    result_list.append(result_dict)
                else:
                    logging.info(f'Iteration {i + 1}:\t Loss: na')

        # X_filled = X.detach().clone()
        # X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, result_list
        else:
            return X_filled


class GradFlowImputer(object):
    def __init__(self, eps=0.01, lr=01.0e-3, opt=torch.optim.Adam, niter=50,
                 kernel_func=xRBF,
                 log_pdf=parzen_window_pdf,
                 bandwidth=10.0,
                 sampling_step=500,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 batchsize=128, n_pairs=1, noise=0.1, scaling=.9):
        super(GradFlowImputer, self).__init__()
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.eps = eps
        self.scaling = scaling
        self.sampling_step = sampling_step
        self.device = device


        # kernel func and concerning grad
        self.kernel_func = kernel_func(0.5 * 2)
        self.grad_val_kernel = self.kernel_func

        # log pdf func and concerning grad
        self.log_pdf = log_pdf
        self.bandwidth = bandwidth
        self.score_func = vammped_parzen_score

        # self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
        # self.data_step = SVGDScoreFunction(score_func=self.log_pdf, kernel_func=rbf_kernel(1.0))


    def _sum_kernel(self, X, Y):
        """
        wrap the kernel function to obtain grad_and_values
        :return: scalar, kernel value
        """
        K_XY = self.kernel_func(X, Y)
        return torch.sum(K_XY), K_XY

    def svgd_sampling(self, data, bandwidth, data_number, score_func, grad_optim, iter_steps, mask_matrix):
        """
        svgd sampling function
        :param data:
        :param data_number:
        :param score_func:
        :param grad_optim:
        :param iter_steps:
        :param mask_matrix:
        :return:
        """
        for _ in range(iter_steps):
            with torch.no_grad():
                eval_score = score_func(data, data, bandwidth)
                eval_grad_k, eval_value_k = self.grad_val_kernel(data, data)

            # svgd gradient
            grad_tensor = -1.0 * (torch.matmul(eval_value_k, eval_score) - 1 * eval_grad_k) / data_number
            # grad_tensor = -1.0 * eval_score
            if torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
                ### Catch numerical errors/overflows (should not happen)
                logging.info("Nan or inf loss")
                break
            # mask the corresponding values
            grad_optim.zero_grad()
            data.grad = torch.masked_fill(input=grad_tensor, mask=mask_matrix, value=0.0)
            grad_optim.step()
        return data


    def fit_transform(self, X, verbose=True, report_interval=10, X_true=None):

        X = X.clone()
        n, d = X.shape
        # self.hcp = ances_hcp_loss(k=6, d=d, use_functorch=False)
        # self.sk = ances_sw_dist(seed=42, n_projections=40, )
        # self.sk = SamplesLoss("sinkhorn", p=2, blur=self.eps, scaling=self.scaling, backend="tensorized")

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2 ** e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(X).double()
        imps = (self.noise * torch.randn(mask.shape).double() + (1 ) * nanmean(X, 0))[mask.bool()]
        grad_mask = ~mask.bool()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        X_filled = X_filled.to(self.device)
        X_filled.requires_grad = True
        grad_mask = grad_mask.to(self.device)

        optimizer = self.opt([X_filled], lr=self.lr)

        if verbose:
            logging.info(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):
            # the filled matrix for x
            X_filled = self.svgd_sampling(data=X_filled, data_number=n, score_func=self.score_func,
                                          bandwidth=self.bandwidth,
                                          grad_optim=optimizer, iter_steps=self.sampling_step, mask_matrix=grad_mask)

            if X_true is not None:
                maes[i] = MAE(X_filled, X_true.to(self.device), mask.to(self.device)).item()
                rmses[i] = RMSE(X_filled, X_true.to(self.device), mask.to(self.device)).item()
                logger.info(f'We are reporting testing results: Iteration {i + 1}:\t Loss: na\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')
            if verbose and ((i + 1) % report_interval == 0):
                if X_true is not None:
                    logging.info(f'Iteration {i + 1}:\t Loss: na\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')
                else:
                    logging.info(f'Iteration {i + 1}:\t Loss: na')

        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled
