import torch
from torch import nn
import time
import numpy as np
import os
import os.path as osp
from componenets.metrics import metrics
from methods.teacher import (STGCNChebGraphConv, MultiLayerPerceptron)
from methods.student import RSTIB_MLP
from torch_geometric.utils import dense_to_sparse
import utils.util as util
import pandas as pd
import torch_geometric
import math
from tqdm import tqdm
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import copy
from scipy.fftpack import dct, idct
from Yaml2Params import args, logger
from scipy.special import gamma

def load_SE(num_node, d_model):
    pe = torch.zeros(num_node, d_model)
    position = torch.arange(0, num_node, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe

def Beta_Function(x, alpha, beta):
    """Beta function"""
    return gamma(alpha + beta) / gamma(alpha) / gamma(beta) * x ** (alpha - 1) * (1 - x) ** (beta - 1)

def record_metric(data_record_dict, data_list, key_list):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        data_record_dict[key] = data
    return data_record_dict
def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)
def build_sp_tensor(adj_weight):
    coo_adj = coo_matrix(adj_weight)
    sp_adj = convert_sp_mat_to_sp_tensor(coo_adj)
    return sp_adj

def MAE_torch(pred, true, mask_value=0):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def build_anneal_beta(beta_init = 0, beta_end = 0.3, max_epoch = 200):
    beta_init = beta_init
    init_length = int(max_epoch / 4)
    anneal_length = int(max_epoch / 4)
    beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
    beta_inter = beta_inter / 4 * (beta_init - beta_end) + beta_end
    beta_list = np.concatenate([np.ones(init_length) * beta_init, beta_inter, 
                                    np.ones(max_epoch - init_length - anneal_length + 1) * beta_end])
    
    return beta_list

def dct_smooth(x, threshold):
    # 
    assert len(x.shape) == 1,  'only one time series is needed'
    x = np.array(x.cpu())
    y = dct(x, norm='ortho') # DCT coefficients

    # Set a threshold and filter out the noise
    y_filtered = copy.deepcopy(y)
    y_filtered[np.abs(y) < threshold] = 0 # set small coefficients to zero

    # Apply IDCT to get the smoothed time series
    x_filtered = idct(y_filtered, norm='ortho') # smoothed signal
    x_filtered = torch.from_numpy(x_filtered)

    return x_filtered
def dwa(L_old, L_new, T=2):
    '''
    L_old: list.
    '''
    L_old = torch.tensor(L_old, dtype=torch.float32)
    L_new = torch.tensor(L_new, dtype=torch.float32)
    N = len(L_new) # task number
    r =  L_old / L_new
    
    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()

def t_model_select(**kwargs): 
    
    if args.t_model == 'stgcn':
        return STGCNChebGraphConv(Ks = 3, Kt = 3, blocks = [[args.in_dim], [64, 16, 64], [64, 16, 64], [128, 128], [args.out_dim]], T = args.lag, n_vertex = args.num_nodes, act_func = "glu", graph_conv_type = "cheb_graph_conv", gso = kwargs.get('supports'), bias = True, droprate = 0.5)
    else: 
        raise ValueError
    
ToD_dict = {'PEMS4': 288}
                    
class trainer():
    def __init__(self, scaler, sp_adj = None, sp_adj_w = None, **kwargs):
        self.scaler = scaler
        self.sp_adj = sp_adj
        self.sp_adj_w = sp_adj_w
        SE = load_SE(args.num_nodes, 64)
        SE = SE.to(args.device)
        
        if args.model == 'rstib_mlp': # student model
            if args.name.startswith('student'):
                self.t_model = t_model_select(**kwargs)
                # self.t_model.load(args.model_path)
                self.t_model.load_state_dict(torch.load(args.model_path, map_location=args.device))
                self.t_model = self.t_model.to(args.device)
            if args.if_in_out:
                assert np.array_equal(np.unique(sp_adj), [0, 1]), "adj is not binary, it has {}".format(np.unique(sp_adj))
                in_out_degree = np.count_nonzero(sp_adj, axis=0)
                self.model = RSTIB_MLP(args=args, in_out_degree = in_out_degree)
            else:
                self.model = RSTIB_MLP(args=args, timeofday = ToD_dict[args.dataset], dayofweek = 7)
        elif args.model == 'stgcn': # teacher model
            self.model = STGCNChebGraphConv(Ks = 3, Kt = 3, blocks = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]], T = args.lag, n_vertex = args.num_nodes, act_func = "glu", graph_conv_type = "cheb_graph_conv", gso = kwargs.get('supports'), bias = True, droprate = 0.5)
        else:
            raise ValueError('Model :{} error in define model'.format(args.model))
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f'Total params: # {total_params}')
        
        if args.testonly:
            self.model.load_state_dict(torch.load(args.mdir+args.name+'.pkl', map_location='cuda:1'))
            print("The training model was successfully loaded.")
            self.model = self.model.to(args.device)
        else:
            self.model = self.model.to(args.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f'Total params: # {total_params}')
        self.optimizer, self.lr_scheduler = self.get_optim()
        self.criterion = self.get_criterion()
        if args.model in ['rstib_mlp']: 
            self.criterion_s = nn.SmoothL1Loss()

        # early stop
        self.patience = args.patience 
        self.trigger = 0
        self.last_loss = 100000
        self.last_mape_loss = 100000
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        print(args.name)
        if args.name.startswith('student'): 
            logger.info('*'*20 + 'now student model' + '*'*20)
            if args.lamb_anneal: 
                self.lamb_list = build_anneal_beta(args.lamb_init, args.lamb, args.max_epoch)
            else: 
                self.lamb_list = np.ones(args.max_epoch + 1) * args.lamb
        elif args.name.startswith('teacher'): 
            logger.info('*'*20 + 'now teacher model' + '*'*20)
        else: 
            raise ValueError('The config name {} is not for Teacher nor Student'.format(args.name))
        if args.model in ['rstib_mlp']: 
            if args.info_anneal: 
                self.info_beta = build_anneal_beta(args.info_init, args.info_beta, args.max_epoch)
            else: 
                self.info_beta = np.ones(args.max_epoch + 1) * args.info_beta
        
    def init_st_graph(self, sp_adj, sp_adj_w, tem_adj):
        self.edge_idx_spat, self.edge_wg_spat = dense_to_sparse(torch.from_numpy(sp_adj_w))
        tem_adj = np.ones((args.lag, args.lag))
        self.edge_idx_temp, self.edge_wg_temp = dense_to_sparse(torch.from_numpy(tem_adj))
        self.edge_idx_spat = self.edge_idx_spat.to(args.device)
        self.edge_wg_spat = self.edge_wg_spat.to(args.device)
        self.edge_idx_temp = self.edge_idx_temp.to(args.device)
        self.edge_wg_temp = self.edge_wg_temp.to(args.device)


    def decorate_batch(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(args.device)
            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(args.device)
                elif isinstance(value, dict) or isinstance(value, list):
                    batch[key] = self.decorate_batch(value)
                # retain other value types in the batch dict
            return batch
        elif isinstance(batch, list):
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(args.device))
                elif isinstance(value, dict) or isinstance(value, list):
                    new_batch.append(self.decorate_batch(value))
                else:
                    # retain other value types in the batch list
                    new_batch.append(value)
            return new_batch
        elif isinstance(batch, torch_geometric.data.batch.DataBatch):
            return batch.to(args.device)
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))

    def train_t(self, epoch, trnloader, tra_val_metric, **kwargs):
        tra_loss = []
        pre_loss = []

        self.model.train()
                
        for batch_idx, batch in tqdm(enumerate(trnloader)):
            self.optimizer.zero_grad()
            batch = self.decorate_batch(batch)
            if args.is_sample is False:
                X, Y, TE = batch
                g = None
            else:
                inputs, g, _ = batch
                X, Y, TE = inputs

            # model forward
            if args.model in ['stgcn']:
                if g is None:
                    output, output_emb = self.model(history_data = X)
                else: 
                    output, output_emb = self.model(history_data = X, g = g)
            else: 
                raise ValueError('Model :{} error, in Model forward'.format(args.model))
            
            # cal loss
            if args.model in ['stgcn']:
                output  = self.scaler.inverse_transform(output)
                Y = self.scaler.inverse_transform(Y)
                main_loss = self.criterion(output, Y)
                loss = main_loss
                pre_loss.append(main_loss.item())
                if torch.isnan(loss): 
                    cur_state = copy.deepcopy(self.model.state_dict())
                    cur_epoch = epoch
                    if not os.path.exists(args.mdir):
                        os.makedirs(args.mdir)
                    torch.save(cur_state,args.mdir+args.name+'_{}'.format(cur_epoch)+'.pkl')
                    raise ValueError('loss explosion')
                # loss backward
                loss.backward()
                # add max grad clipping
                if args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                tra_loss.append(loss.item())
                pre_lr = self.optimizer.param_groups[0]['lr']
            else: 
                raise ValueError('Model :{} error, in Cal Loss'.format(args.model))
            
        self.lr_scheduler.step()
        tra_loss = np.mean(tra_loss)
        pre_loss = np.mean(pre_loss)
        # show loss
        if args.model in ['stgcn']:
            tra_val_metric = record_metric(tra_val_metric, [epoch, tra_loss, pre_loss, pre_lr], ['epoch', 'train loss', 'predict loss', 'pre lr'])
        else: 
            raise ValueError('Model :{} error, in Display Loss'.format(args.model))
        return tra_val_metric
    
    def train_s(self, epoch, trnloader, tra_val_metric, **kwargs):

        tra_losses = []
        pre_losses = []
        kd_losses = []
        tea_losses = []
        avg_losses = []
        ib_losses = []

        self.model.train()
        self.t_model.eval()
        
        for batch_idx, batch in tqdm(enumerate(trnloader)):
            self.optimizer.zero_grad()
            batch = self.decorate_batch(batch)
            X, Y, TE = batch
            # model forward 
            if args.model == 'rstib_mlp':
                if args.load_t_out is False:
                    if args.t_model in ['stgcn']:
                        output_t, output_emb_t = self.t_model(history_data = X)
                    output_t, output_emb_t = output_t.detach(), output_emb_t.detach()
                else: 
                    output_t, output_emb_t = self.t_model_outs[batch_idx], self.t_model_emb[batch_idx]

                # pass student
                t_emb = TE[:, :args.lag, :] # B, T, 2
                inp_te = t_emb[:, :, 0]
                t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                X = torch.cat([X, t_emb], dim = -1)
                output_s, output_emb_s, (mu, std), noise_y, (mu_y, std_y), (mu_x, std_x) = self.model(history_data = X, n_smaple = 1, te = inp_te)
                output_s_avg, _, _, _, _, _ = self.model(history_data = X, n_smaple = 12, te = inp_te) 
            else: 
                raise ValueError('Model :{} error, in Model forward'.format(args.model))
            
            # cal loss
            if args.model in ['rstib_mlp']:
                output_t  = self.scaler.inverse_transform(output_t)
                output_t = output_t + noise_y
                output_s  = self.scaler.inverse_transform(output_s)
                output_s_avg = self.scaler.inverse_transform(output_s_avg)
                Y = self.scaler.inverse_transform(Y)
                Y = Y + noise_y
                difficulty = torch.abs(output_t-Y)
                difficulty = difficulty.mean(dim=1).squeeze(-1)
                difficulty = F.softmax(difficulty, dim=1)
                difficulty_stu = 2 - difficulty.unsqueeze(1).repeat(1, Y.size(1), 1).unsqueeze(-1)

                stu_loss_pre = torch.abs(output_s - Y)
                stu_loss = difficulty_stu*stu_loss_pre
                total_loss = stu_loss.sum()
                num_elements = stu_loss.numel()
                stu_loss = total_loss / num_elements

                avg_loss = self.criterion(output_s_avg, Y)
                kd_loss = self.criterion(output_s, output_t)
                tea_loss = self.criterion(output_t, Y)
                KL_weight_loss = 1 + difficulty.unsqueeze(1).repeat(1, std.size(1), 1).unsqueeze(-1)
                KL_weight_loss_y = 1 + difficulty.unsqueeze(1).repeat(1, std_y.size(1), 1).unsqueeze(-1)
                KL_weight_loss_x = 1 + difficulty.unsqueeze(1).repeat(1, std_x.size(1), 1).unsqueeze(-1)       
                
                loss_reg = stu_loss
                
                temp = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2))
                temp = KL_weight_loss*temp
                loss_info = temp.sum(1).mean().div(math.log(2))

                temp = -0.5*(1+2*std_y.log()-mu_y.pow(2)-std_y.pow(2))
                temp = KL_weight_loss_y*temp
                loss_info_y = temp.sum(1).mean().div(math.log(2))

                temp = -0.5*(1+2*std_x.log()-mu_x.pow(2)-std_x.pow(2))
                temp = KL_weight_loss_x*temp
                loss_info_x = temp.sum(1).mean().div(math.log(2))
                
                loss = loss_reg.div(math.log(2)) + 2 * self.info_beta[epoch] * loss_info + 2 * self.info_beta[epoch] * loss_info_y + 2 * self.info_beta[epoch] * loss_info_x

                if torch.isnan(loss): 
                    cur_state = copy.deepcopy(self.model.state_dict())
                    cur_epoch = epoch
                    if not os.path.exists(args.mdir):
                        os.makedirs(args.mdir)
                    torch.save(cur_state,args.mdir+args.name+'_{}'.format(cur_epoch)+'.pkl')
                    raise ValueError('loss explosion')
                avg_losses.append(avg_loss.item())
                ib_losses.append(loss_info.item())
            else: 
                raise ValueError('Model :{} error, in Cal Loss'.format(args.model)) 
            
            kd_losses.append(kd_loss.item())
            pre_losses.append(stu_loss.item())
            tea_losses.append(tea_loss.item())
            
            loss.backward()
            
            # add max grad clipping
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            if args.ema:
                self.model_ema.update(self.model.state_dict())
            tra_losses.append(loss.item())
            pre_lr = self.optimizer.param_groups[0]['lr']
        self.lr_scheduler.step()
        tra_losses = np.mean(tra_losses)
        kd_losses = np.mean(kd_losses)
        pre_losses = np.mean(pre_losses)
        tea_losses = np.mean(tea_losses)
        
        # show loss
        avg_losses = np.mean(avg_losses)
        ib_losses = np.mean(ib_losses)
        tra_val_metric = record_metric(tra_val_metric, 
                                       [epoch, tra_losses, pre_losses, kd_losses, tea_losses, self.lamb_list[epoch], pre_lr, avg_losses, self.info_beta[epoch], ib_losses], 
                                       ['epoch', 'train loss', 'predict loss', 'kd loss', 'tea loss', 'lamb', 'pre lr', 'avg loss', 'IB beta', 'ib loss'])

        return tra_val_metric

    
    def validation(self,  epoch, valloader, tra_val_metric, **kwargs):
        val_loss = []
        trues = []
        preds = []
        avg_preds = []

        with torch.no_grad():
            self.model.eval()

            for batch_idx, batch in tqdm(enumerate(valloader)):
                batch = self.decorate_batch(batch)
                if args.is_sample is False:
                    X, Y, TE = batch
                    g = None
                else:
                    inputs, g, rows = batch
                    X, Y, TE = inputs
                # model forward
                if args.model in ['stgcn']:
                    if g is None:
                        output, output_emb = self.model(history_data = X)
                    else: 
                        output, output_emb = self.model(history_data = X, g = g)
                elif args.model == 'rstib_mlp': 
                    # pass student
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    inp_te = t_emb[:, :, 0] 
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output, _, _, _, _, _ = self.model(history_data = X, n_smaple = 1, te = inp_te)  
                    output_avg, _, _, _, _, _ = self.model(history_data = X, n_smaple = 12, te = inp_te)
                else: 
                    raise ValueError('Model :{} error, in Model forward'.format(args.model))

                output = self.scaler.inverse_transform(output)
                Y = self.scaler.inverse_transform(Y)
                if g is None:
                    preds.append(output.detach().cpu().numpy())
                    trues.append(Y.detach().cpu().numpy())
                else: 
                    cent_n_id = g['cent_n_id']
                    res_n_id = g['res_n_id']
                    # Note: we only evaluate predictions on those initial nodes (per random walk)
                    # to avoid duplicated computations
                    output = output.transpose(2, 1)
                    Y = Y.transpose(2, 1)
                    Y = Y[:, res_n_id]
                    output = output[:, res_n_id]
                    cent_n_id = cent_n_id[res_n_id]
                    forecast_length = Y.shape[2]

                    index_ptr = torch.cartesian_prod(
                        torch.arange(rows.size(0)),
                        torch.arange(cent_n_id.size(0)),
                        torch.arange(forecast_length)
                    )

                    label = pd.DataFrame({
                        'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                        'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                        'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                        'val': Y.flatten().data.cpu().numpy()
                    })

                    pred = pd.DataFrame({
                        'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                        'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                        'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                        'val': output.flatten().data.cpu().numpy()
                    })
                    pred = pred.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()
                    label = label.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()

                    trues.append(label)
                    preds.append(pred)

                # cal loss
                if args.model in ['stgcn']:
                    loss = self.criterion(output, Y)
                elif args.model in ['rstib_mlp']: 
                    loss = self.criterion(output, Y)
                    output_avg = self.scaler.inverse_transform(output_avg)
                    avg_preds.append(output_avg.detach().cpu().numpy())
                else: 
                    raise ValueError('Model :{} error, in Cal Loss'.format(args.model))
                val_loss.append(loss.item())              

        val_loss = np.mean(val_loss)
        if args.is_sample is False:
            trues, preds = np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)
        else: 
            trues, preds = self.eval_epoch_end(trues, preds)
        mae, rmse, mape, smape, corr = metrics(preds, trues, args.mae_thresh, args.mape_thresh)
        tra_val_metric = record_metric(tra_val_metric, [val_loss, mae, rmse, mape*100, smape*100, corr], ['val loss', 'mae', 'rmse', 'mape(%)', 'smape(%)', 'corr']) 

        if args.model in ['stgcn']:
            pass
        elif args.model in ['rstib_mlp']: 
            avg_preds = np.concatenate(avg_preds, axis=0)
            mae_avg, rmse_avg, mape_avg, smape_avg, corr_avg = metrics(avg_preds, trues, args.mae_thresh, args.mape_thresh)
            tra_val_metric = record_metric(tra_val_metric, [ mae_avg, rmse_avg, mape_avg*100, smape_avg*100, corr_avg], ['mae_avg', 'rmse_avg', 'mape_avg(%)', 'smape_avg(%)', 'corr_avg'])
        else: 
            raise ValueError('Model :{} error, in Cal Loss'.format(args.model))
        stopFlg = self.earlyStop( epoch, mae, mae)

        return tra_val_metric, stopFlg
    def eval_epoch_end(self, trues, preds):
        pred = pd.concat(preds, axis=0)
        label = pd.concat(trues, axis=0)
        pred = pred.groupby(['row_idx', 'node_idx','forecast_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()
        # atten_context = [x['atten'] for x in outputs]
        row_sum = len(pred['row_idx'].unique())
        node_sum = len(pred['node_idx'].unique())
        forecast_sum = len(pred['forecast_idx'].unique())    
        pred = pred.values.reshape(row_sum, node_sum, forecast_sum, 1)
        label = label.values.reshape(row_sum, node_sum, forecast_sum, 1)
        return pred, label

    def test(self, tstloader, ):
        self.model.load_state_dict(torch.load(args.mdir+args.name+'.pkl'), False)

        trues = []
        preds = []
        avg_preds = []

        with torch.no_grad():
            self.model.eval()
            t1 = time.time()

            for idx, batch in tqdm(enumerate(tstloader)):
                batch = self.decorate_batch(batch)
                if args.is_sample is False:
                    X, Y, TE = batch
                    g = None
                else:
                    inputs, g, rows = batch
                    X, Y, TE = inputs
                if args.model in ['stgcn']:
                    if g is None:
                        output, output_emb = self.model(history_data = X)
                    else: 
                        output, output_emb = self.model(history_data = X, g = g)
                elif args.model == 'rstib_mlp': # mlp_ib
                    # pass student
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    inp_te = t_emb[:, :, 0]
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output, _, _, _, _, _ = self.model(history_data = X, n_smaple = 1, te = inp_te)  
                    output_avg, _, _, _, _, _ = self.model(history_data = X, n_smaple = 50, te = inp_te)
                else: 
                    raise ValueError('Model :{} error, in Model forward'.format(args.model))
 
                Y = self.scaler.inverse_transform(Y)
                output = self.scaler.inverse_transform(output)

                if g is None:
                    preds.append(output.detach().cpu().numpy())
                    trues.append(Y.detach().cpu().numpy())
                else: 
                    cent_n_id = g['cent_n_id']
                    res_n_id = g['res_n_id']
                    # Note: we only evaluate predictions on those initial nodes (per random walk)
                    # to avoid duplicated computations
                    output = output.transpose(2, 1)
                    Y = Y.transpose(2, 1)
                    Y = Y[:, res_n_id]
                    output = output[:, res_n_id]
                    cent_n_id = cent_n_id[res_n_id]
                    forecast_length = Y.shape[2]

                    index_ptr = torch.cartesian_prod(
                        torch.arange(rows.size(0)),
                        torch.arange(cent_n_id.size(0)),
                        torch.arange(forecast_length)
                    )

                    label = pd.DataFrame({
                        'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                        'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                        'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                        'val': Y.flatten().data.cpu().numpy()
                    })

                    pred = pd.DataFrame({
                        'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                        'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                        'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                        'val': output.flatten().data.cpu().numpy()
                    })
                    pred = pred.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()
                    label = label.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()

                    trues.append(label)
                    preds.append(pred)
                if args.model in ['stgcn']:
                    pass
                elif args.model in ['rstib_mlp']:
                    # LEAP: multi-run stochastic ensemble (3 independent 50-sample runs averaged)
                    output_avg_inv = self.scaler.inverse_transform(output_avg)
                    for _ in range(2):
                        extra_avg, _, _, _, _, _ = self.model(history_data = X, n_smaple = 50, te = inp_te)
                        extra_avg_inv = self.scaler.inverse_transform(extra_avg)
                        output_avg_inv = output_avg_inv + extra_avg_inv
                    output_avg_inv = output_avg_inv / 3.0
                    avg_preds.append(output_avg_inv.detach().cpu().numpy())
                else: 
                    raise ValueError('Model :{} error, in Cal Loss'.format(args.model))
        t2 = time.time()
        logger.info('cost: {}'.format(t2 - t1))
        if args.is_sample is False:
            trues, preds = np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)
        else: 
            trues, preds = self.eval_epoch_end(trues, preds)
        for t in range(trues.shape[1]):
            mae, rmse, mape, smape, corr = metrics(preds[:, t, ...], trues[:, t, ...], args.mae_thresh, args.mape_thresh)
            log = "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
                t + 1, mae, rmse, mape * 100, smape * 100, corr)
            logger.info(log)
        mae, rmse, mape, smape, corr = metrics(preds, trues, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, Best Epoch: {}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
            self.best_epoch, mae, rmse, mape * 100, smape * 100, corr))
        if args.model in ['stgcn']:
            pass
        elif args.model in ['rstib_mlp']: 
            avg_preds = np.concatenate(avg_preds, axis=0)
            for t in range(trues.shape[1]):
                mae_avg, rmse_avg, mape_avg, smape_avg, corr_avg = metrics(avg_preds[:, t, ...], trues[:, t, ...], args.mae_thresh, args.mape_thresh)
                log = "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
                    t + 1, mae_avg, rmse_avg, mape_avg * 100, smape_avg * 100, corr_avg)
                logger.info(log)
            mae_avg, rmse_avg, mape_avg, smape_avg, corr_avg = metrics(avg_preds, trues, args.mae_thresh, args.mape_thresh)
            logger.info("Average Horizon, Best Epoch: {}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
                self.best_epoch, mae_avg, rmse_avg, mape_avg * 100, smape_avg * 100, corr))
        else: 
            raise ValueError('Model :{} error, in Cal Loss'.format(args.model))
        
    def get_optim(self, ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay = args.weight_decay, betas=(0.9, 0.999))
        steps = args.steps
        
        lr_decay_ratio = args.lr_decay_ratio
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                                gamma=lr_decay_ratio)
        return optimizer, lr_scheduler

    def get_criterion(self, ):
        if args.criterion == 'MSE':
            return nn.MSELoss()
        elif args.criterion == 'Smooth':
            return nn.SmoothL1Loss()
        elif args.criterion == 'MAE':
            return MAE_torch
    

    def earlyStop(self, epoch, current_loss, mape_loss):
        if epoch >= args.start_epoch: # default 100
            if current_loss >= self.last_loss or epoch == args.max_epoch:
        # if epoch >= 0:
        #     if epoch == 1:
                if current_loss < self.last_loss:
                    self.trigger = 0
                    self.last_loss = current_loss
                    self.last_mape_loss = mape_loss
                    self.best_epoch = epoch
                    self.best_state = copy.deepcopy(self.model.state_dict())
                else:
                    self.trigger += 1
                if self.trigger >= self.patience or epoch == args.max_epoch:   
                    print('Early Stopping! The best epoch is ' + str(self.best_epoch))
                    if not os.path.exists(args.mdir):
                        os.makedirs(args.mdir)

                    torch.save(self.best_state,args.mdir+args.name+'.pkl')
                    logger.info(args.mdir+args.name+'.pkl')
                    return True
            else:
                self.trigger = 0
                self.last_loss = current_loss
                self.last_mape_loss = mape_loss
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                return False

    def test_para(self, tstloader):
        self.model.load_state_dict(torch.load(args.mdir+args.name+'.pkl'), False)

        trues = []
        preds = []
        avg_preds = []

        with torch.no_grad():
            self.model.eval()
            t1 = time.time()
            

            for idx, batch in tqdm(enumerate(tstloader)):
                batch = self.decorate_batch(batch)
                if args.is_sample is False:
                    X, Y, TE = batch
                    g = None
                else:
                    inputs, g, rows = batch
                    X, Y, TE = inputs
                if args.model in ['stgcn']:
                    output, output_emb = self.model(history_data = X, g = g)
                elif args.model == 'rstib_mlp': # mlp_ib
                    # pass student
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    inp_te = t_emb[:, :, 0]
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                else: 
                    raise ValueError('Model :{} error, in Model forward'.format(args.model))
                if idx == 0:
                    break