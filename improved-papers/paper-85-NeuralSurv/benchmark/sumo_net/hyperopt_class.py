from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL,rand
from benchmark.sumo_net.nets import *
import torch
import os
import pickle
import numpy as np
from benchmark.sumo_net.pycox_local.pycox.evaluation import EvalSurv
import pandas as pd
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from benchmark.sumo_net.pycox_local.pycox.models import *
from benchmark.sumo_net.pycox_local.pycox.models.cox_time import MLPVanillaCoxTime,MixedInputMLPCoxTime
import torchtuples as tt
import time
# from utils.deephit_transformation_fix import *


def square(x):
    return x**2

def linear(x):
    return x

def ibs_calc(S,mask_1,mask_2,gst,gt):

    weights_2 = 1./gt
    weights_2[weights_2==float("Inf")]=0.0
    weights_1 = 1./gst
    weights_1[weights_1==float("Inf")]=0.0
    return ((S**2)*mask_1)*weights_1 + ((1.-S)**2 * mask_2)*weights_2

def ibll_calc(S,mask_1,mask_2,gst,gt):
    weights_2 = 1./gt
    weights_2[weights_2==float("Inf")]=0.0
    weights_1 = 1./gst
    weights_1[weights_1==float("Inf")]=0.0
    return ((torch.log(1-S+1e-6))*mask_1)*weights_1 + ((torch.log(S+1e-6)) * mask_2)*weights_2

def simpsons_composite(S,step_size,n):
    idx_odd = torch.arange(1,n-1,2)
    idx_even = torch.arange(2,n,2)
    S[idx_odd]=S[idx_odd]*4
    S[idx_even]=S[idx_even]*2
    return torch.sum(S)*step_size/3.

class fifo_list():
    def __init__(self,n):
        self.fifo_list = [0]*n
        self.n=n
    def insert(self,el):
        self.fifo_list.pop(0)
        self.fifo_list.append(el)
    def __len__(self):
        return len(self.fifo_list)

    def get_sum(self):
        return sum(self.fifo_list)

class hyperopt_training():
    def __init__(self,job_param,hyper_param_space,custom_dataloader=None):
        self.d_out = job_param['d_out']
        self.dataset_string = job_param['dataset_string']
        self.seed = job_param['seed']
        self.total_epochs = job_param['total_epochs']
        self.device = job_param['device']
        self.patience = job_param['patience']
        self.hyperits = job_param['hyperits']
        self.selection_criteria = job_param['selection_criteria']
        self.grid_size  = job_param['grid_size']
        self.validation_interval = job_param['validation_interval']
        self.test_grid_size  = job_param['test_grid_size']
        self.objective = job_param['objective']
        self.net_type = job_param['net_type']
        self.fold_idx = job_param['fold_idx']
        self.savedir = job_param['savedir']
        self.chunks = job_param['chunks']
        self.max_series_accumulation = job_param['max_series_accumulation']
        self.validate_train = job_param['validate_train']
        self.dataloader = job_param['dataloader']
        self.global_hyperit = 0
        self.best = np.inf
        self.debug = False
        # torch.cuda.set_device(self.device)
        self.custom_dataloader = custom_dataloader
        self.save_path = f'{self.savedir}/{self.dataset_string}_seed={self.seed}_fold_idx={self.fold_idx}_objective={self.objective}_{self.net_type}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.hyperopt_params = ['bounding_op', 'transformation', 'depth_x', 'width_x','depth_t', 'width_t', 'depth', 'width', 'bs', 'lr','direct_dif','dropout','eps','weight_decay']
        self.deephit_params= ['alpha','sigma','num_dur']
        self.get_hyperparameterspace(hyper_param_space)

    def calc_eval_objective(self,S,f,S_extended,durations,events,time_grid):
        val_likelihood = self.train_objective(S,f)
        eval_obj = EvalSurv(surv=S_extended,durations=durations,events=events,censor_surv='km') #Add index and pass as DF
        conc = eval_obj.concordance_td()
        ibs = eval_obj.integrated_brier_score(time_grid)
        inll = eval_obj.integrated_nbll(time_grid)
        return val_likelihood,conc,ibs,inll

    def benchmark_eval(self,y,events,X,wrapper):
        surv = wrapper.predict_surv_df(X)
        t_grid_np = np.linspace(y.min(), y.max(), surv.index.shape[0])
        surv = surv.set_index(t_grid_np)
        ev = EvalSurv(surv=surv, durations=y, events=events, censor_surv='km')
        conc = ev.concordance_td()
        ibs = ev.integrated_brier_score(t_grid_np)
        inll = ev.integrated_nbll(t_grid_np)
        return conc,ibs,inll

    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

        if self.net_type == 'deephit_benchmark':
            for string in self.deephit_params:
                self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def __call__(self,parameters_in):
        print(f"----------------new hyperopt iteration {self.global_hyperit}------------------")
        print(parameters_in)
        x_c = self.dataloader.dataset.X.shape[1]
        self.cycle_length = self.dataloader.__len__()//self.validation_interval+1
        print('cycle_length',self.cycle_length)
        net_init_params = {
            'd_in_x' : x_c,
            'cat_size_list': self.dataloader.dataset.unique_cat_cols,
            'd_in_y' : self.dataloader.dataset.y.shape[1],
            'd_out' : self.d_out,
            'bounding_op':parameters_in['bounding_op'],
            'transformation':parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']]*parameters_in['depth_x'],
            'layers_t': [parameters_in['width_t']]*parameters_in['depth_t'],
            'layers': [parameters_in['width']]*parameters_in['depth'],
            'direct_dif':parameters_in['direct_dif'],
            'objective':self.objective,
            'dropout':parameters_in['dropout'],
            'eps':parameters_in['eps']
        }
        cat_cols_nr = len(self.dataloader.dataset.unique_cat_cols)
        self.train_objective = get_objective(self.objective)
        if self.net_type=='weibull_net':
            self.model = weibull_net(**net_init_params).to(self.device)
        elif self.net_type=='lognormal_net':
            self.model = lognormal_net(**net_init_params).to(self.device)
        elif self.net_type=='survival_net_basic':
            self.model = survival_net_basic(**net_init_params).to(self.device)
        elif self.net_type=='cox_time_benchmark':
            if cat_cols_nr==0:
                self.model = MLPVanillaCoxTime(in_features=net_init_params['d_in_x'],
                                                  num_nodes=net_init_params['layers'],
                                                  batch_norm=False,
                                                  dropout=net_init_params['dropout'],
                                                  )  # Actual net to be used

            else:
                self.model = MixedInputMLPCoxTime(in_features=net_init_params['d_in_x'],
                                        num_nodes=net_init_params['layers'],
                                        batch_norm=False,
                                        dropout=net_init_params['dropout'],
                                                  num_embeddings=self.dataloader.dataset.unique_cat_cols,
                                                  embedding_dims=[el // 2 + 1 for el in
                                                                  self.dataloader.dataset.unique_cat_cols]
                                       ) #Actual net to be used
            self.wrapper = CoxTime(self.model,tt.optim.Adam)

        elif self.net_type=='deepsurv_benchmark':
            if cat_cols_nr ==0:
                self.model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                    num_nodes=net_init_params['layers'],
                                    batch_norm=False,
                                    dropout=net_init_params['dropout'],
                                    out_features=1) #Actual net to be used
            else:
                self.model = tt.practical.MixedInputMLP(
                    in_features=net_init_params['d_in_x'],
                    num_nodes=net_init_params['layers'],
                    batch_norm=False,
                    dropout=net_init_params['dropout'],
                    out_features=1,
                    num_embeddings=self.dataloader.dataset.unique_cat_cols,
                    embedding_dims=[el // 2 + 1 for el in
                                    self.dataloader.dataset.unique_cat_cols]
                )
            self.wrapper = CoxPH(self.model,tt.optim.Adam)

        elif self.net_type=='deephit_benchmark':
            print('num_dur', parameters_in['num_dur'])
            labtrans = LabTransDiscreteTime(parameters_in['num_dur'])

        elif self.net_type=='cox_CC_benchmark':
            if cat_cols_nr ==0:
                self.model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                    num_nodes=net_init_params['layers'],
                                    batch_norm=False,
                                    dropout=net_init_params['dropout'],
                                    out_features=1) #Actual net to be used
            else:
                self.model = tt.practical.MixedInputMLP(
                    in_features=net_init_params['d_in_x'],
                    num_nodes=net_init_params['layers'],
                    batch_norm=False,
                    dropout=net_init_params['dropout'],
                    out_features=1,
                    num_embeddings=self.dataloader.dataset.unique_cat_cols,
                    embedding_dims=[el // 2 + 1 for el in
                                    self.dataloader.dataset.unique_cat_cols]
                )
            self.wrapper = CoxCC(self.model,tt.optim.Adam)

        elif self.net_type=='cox_linear_benchmark':
            if cat_cols_nr ==0:
                self.model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                    num_nodes=[],
                                    batch_norm=False,
                                    dropout=net_init_params['dropout'],
                                    out_features=1) #Actual net to be used
            else:
                self.model = tt.practical.MixedInputMLP(
                    in_features=net_init_params['d_in_x'],
                    num_nodes=[],
                    batch_norm=False,
                    dropout=net_init_params['dropout'],
                    out_features=1,
                    num_embeddings=self.dataloader.dataset.unique_cat_cols,
                    embedding_dims=[el // 2 + 1 for el in
                                    self.dataloader.dataset.unique_cat_cols]
                )
            self.wrapper = CoxPH(self.model,tt.optim.Adam)

        if self.net_type in ['cox_time_benchmark','deepsurv_benchmark','cox_CC_benchmark','cox_linear_benchmark','deephit_benchmark']:
            y_train = (self.dataloader.dataset.train_y.squeeze().numpy(),self.dataloader.dataset.train_delta.squeeze().numpy())
            y_val = (self.dataloader.dataset.val_y.squeeze().numpy(),self.dataloader.dataset.val_delta.squeeze().numpy())
            if self.net_type=='deephit_benchmark':
                val_data_eval = tt.tuplefy(self.dataloader.dataset.val_X.numpy(), y_val)
                y_train,bool_fix_train = labtrans.fit_transform(y_train[0],y_train[1])
                val_dur,val_event,bool_fixer_minus_1 = labtrans.transform(y_val[0],y_val[1])
                y_val = (val_dur,val_event)

                if cat_cols_nr==0:
                    self.model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                                         num_nodes=net_init_params['layers'],
                                                         batch_norm=False,
                                                         dropout=net_init_params['dropout'],
                                                         out_features=labtrans.out_features)  # Actual net to be used
                    x_val = self.dataloader.dataset.val_X.numpy()[bool_fixer_minus_1]
                    x_test = self.dataloader.dataset.test_X.numpy()
                    val_data = tt.tuplefy(x_val, y_val)
                    X_in = self.dataloader.dataset.train_X.numpy()[bool_fix_train]

                else:
                    self.model = tt.practical.MixedInputMLP(in_features=net_init_params['d_in_x'],
                                                         num_nodes=net_init_params['layers'],
                                                         batch_norm=False,
                                                         dropout=net_init_params['dropout'],
                                                         out_features=labtrans.out_features,
                                                            num_embeddings=self.dataloader.dataset.unique_cat_cols,
                                                            embedding_dims=[el // 2 + 1 for el in
                                                                            self.dataloader.dataset.unique_cat_cols]
                                                            )  # Actual net to be used
                    X_tmp,X_cat_tmp = self.dataloader.dataset.val_X.numpy()[bool_fixer_minus_1],self.dataloader.dataset.val_cat_X.numpy()[bool_fixer_minus_1]
                    # X_tmp,X_cat_tmp = self.dataloader.dataset.val_X.numpy()[bool_fixer_minus_1],self.dataloader.dataset.val_cat_X.numpy()[bool_fixer_minus_1]
                    x_val = tt.tuplefy(X_tmp,X_cat_tmp)
                    x_test = tt.tuplefy((self.dataloader.dataset.test_X.numpy(),self.dataloader.dataset.test_cat_X.numpy()))
                    val_data = tt.tuplefy(x_val, y_val)
                    X_in = tt.tuplefy(
                        (self.dataloader.dataset.train_X.numpy()[bool_fix_train], self.dataloader.dataset.train_cat_X.numpy()[bool_fix_train]))

                self.wrapper = DeepHitSingle(self.model, tt.optim.Adam, alpha=parameters_in['alpha'],
                                             sigma=parameters_in['sigma'], duration_index=labtrans.cuts)
            else:
                if cat_cols_nr==0:
                    val_data = tt.tuplefy(self.dataloader.dataset.val_X.numpy(), y_val)
                    x_val =self.dataloader.dataset.val_X.numpy()
                    x_test = self.dataloader.dataset.test_X.numpy()
                    X_in = self.dataloader.dataset.train_X.numpy()

                else:
                    x_val =tt.tuplefy((self.dataloader.dataset.val_X.numpy(),self.dataloader.dataset.val_cat_X.numpy()))
                    x_test = tt.tuplefy((self.dataloader.dataset.test_X.numpy(),self.dataloader.dataset.test_cat_X.numpy()))
                    val_data = tt.tuplefy(x_val, y_val)
                    X_in = tt.tuplefy(
                        (self.dataloader.dataset.train_X.numpy(), self.dataloader.dataset.train_cat_X.numpy()))

            verbose = True
            self.wrapper.optimizer.set_lr(parameters_in['lr'])
            callbacks = [tt.callbacks.EarlyStopping()]
            print(self.model)
            log = self.wrapper.fit(input=X_in,
                              target=y_train, epochs=self.total_epochs,callbacks= callbacks, verbose=verbose,batch_size=parameters_in['bs'],
                            val_data=val_data)

            if self.net_type!='deephit_benchmark':
                base_haz = self.wrapper.compute_baseline_hazards()
                val_durations = self.dataloader.dataset.invert_duration(self.dataloader.dataset.val_y.numpy()).squeeze()
                val_events = self.dataloader.dataset.val_delta.float().squeeze().numpy()
            else:
                val_events = self.dataloader.dataset.val_delta.float().squeeze().numpy()[bool_fixer_minus_1]
                val_durations = self.dataloader.dataset.invert_duration(self.dataloader.dataset.val_y.numpy()).squeeze()[bool_fixer_minus_1]
            val_conc, val_ibs, val_inll =self.benchmark_eval(y=val_durations,events=val_events,
                                                             wrapper=self.wrapper,X=x_val)
            test_durations = self.dataloader.dataset.invert_duration(self.dataloader.dataset.test_y.numpy()).squeeze()
            test_conc, test_ibs, test_inll =self.benchmark_eval(y=test_durations,events=self.dataloader.dataset.test_delta.float().squeeze().numpy(),
                                                                wrapper=self.wrapper,X=x_test)
            val_likelihood_list = [1e99,1e99]
            test_likelihood_list = [1e99,1e99]
            if self.net_type !='deephit_benchmark':
                val_loss = callbacks[0].cur_best
            else:
                val_loss = None
            results = self.parse_results(
                                        val_likelihood_list,
                                         val_conc,
                                         val_ibs,
                                         val_inll,
                                        test_likelihood_list,
                                         test_conc,
                                         test_ibs,
                                         test_inll,
                                        val_loss_cox=val_loss)
            if results['test_loss']<self.best:
                self.best = results['test_loss']
                print(self.best)
                self.dump_model()

        else:
            self.dump_model()
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=parameters_in['lr'],weight_decay=parameters_in['weight_decay'])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience=self.patience//4,min_lr=1e-3,factor=0.9)
            results = self.full_loop()
            del self.optimizer
        self.global_hyperit+=1
        results['net_init_params'] = net_init_params
        torch.cuda.empty_cache()
        return results

    def complexity_test(self,grid_size=100):

        self.dataloader.dataset.set(mode='test')
        self.model = self.model.train()
        if self.net_type!='survival_net_basic':
            start = time.time()
            if self.net_type!='deephit_benchmark':
                base_haz = self.wrapper.compute_baseline_hazards()
            test_durations = self.dataloader.dataset.invert_duration(self.dataloader.dataset.test_y.numpy()).squeeze()
            cat_cols_nr = len(self.dataloader.dataset.unique_cat_cols)
            if cat_cols_nr==0:
                x_test = self.dataloader.dataset.test_X.numpy()
            else:
                x_test = tt.tuplefy((self.dataloader.dataset.test_X.numpy(), self.dataloader.dataset.test_cat_X.numpy()))

            surv = self.wrapper.predict_surv_df(x_test)
            t_grid_np = np.linspace(test_durations.min(), test_durations.max(), surv.index.shape[0])
            surv = surv.set_index(t_grid_np)
            end = time.time()

        else:
            start = time.time()
            S_series_container = []
            durations = []
            events = []
            chunks = self.dataloader.batch_size // 50 + 1
            t_grid_np = np.linspace(self.dataloader.dataset.min_duration, self.dataloader.dataset.max_duration,
                                    grid_size)
            time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
            for i, (X, x_cat, y, delta) in enumerate(tqdm(self.dataloader)):
                X = X.to(self.device)
                y = y.to(self.device)
                delta = delta.to(self.device)
                mask = delta == 1
                if not isinstance(x_cat, list):
                    x_cat = x_cat.to(self.device)
                if not isinstance(x_cat, list):
                    for chk, chk_cat in zip(torch.chunk(X, chunks), torch.chunk(x_cat, chunks)):
                        input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                        X_repeat = chk.repeat_interleave(grid_size, 0)
                        x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
                        S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                        S_serie = S_serie.detach()
                        S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                else:
                    x_cat_repeat = []
                    for chk in torch.chunk(X, chunks):
                        input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                        X_repeat = chk.repeat_interleave(grid_size, 0)
                        S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                        S_serie = S_serie.detach()
                        S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                durations.append(y.cpu().numpy())
                events.append(delta.cpu().numpy())
            S_series_container = pd.DataFrame(torch.cat(S_series_container, 1).numpy())
            t_grid_np = self.dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
            S_series_container = S_series_container.set_index(t_grid_np)
            end = time.time()
        timing = end-start
        return timing

    def do_metrics(self,training_loss,likelihood,reg_loss,i):
        val_likelihood, conc, ibs, ibll, _ = self.validation_score()
        if self.debug:
            test_likelihood, test_conc, test_ibs, test_inll = self.test_score()
            self.writer.add_scalar('Loss/train', training_loss, i)
            self.writer.add_scalar('Loss/val', val_likelihood[0], i)
            self.writer.add_scalar('Loss/test', test_likelihood[0], i)
            self.writer.add_scalar('conc/val', conc, i)
            self.writer.add_scalar('conc/test', test_conc, i)
            self.writer.add_scalar('ibs/val', ibs, i)
            self.writer.add_scalar('ibs/test', test_ibs, i)
            self.writer.add_scalar('inll/val', ibll, i)
            self.writer.add_scalar('inll/test', test_inll, i)
            print(
                f'test_likelihood: {test_likelihood[0]} test_likelihood: {test_likelihood[1]} test_conc: {test_conc} test_ibs: {test_ibs}  test_inll: {test_inll}')
            self.debug_list.append(test_ibs)
        if self.selection_criteria in ['train','likelihood']:
            criteria = val_likelihood[0]  # minimize #
        elif self.selection_criteria == 'concordance':
            criteria = -conc  # maximize
        elif self.selection_criteria == 'ibs':
            criteria = ibs  # minimize
        elif self.selection_criteria == 'inll':
            criteria = ibll  # "minimize"
        print(f'total_loss: {training_loss} likelihood: {likelihood} reg_loss: {reg_loss}')
        if self.validate_train:
            tr_likelihood, tr_conc, tr_ibs, tr_ibll, _ = self.train_score()
            print(f'tr_likelihood: {tr_likelihood[0]} tr_likelihood: {tr_likelihood[1]} tr_conc: {tr_conc} tr_ibs: {tr_ibs}  tr_ibll: {tr_ibll}')
        print(f'criteria score: {criteria} val likelihood: {val_likelihood[0]} val likelihood: {val_likelihood[1]} val conc:{conc} val ibs: {ibs} val inll {ibll}')
        return criteria


    def eval_func(self,i,training_loss,likelihood,reg_loss):
        if i % self.cycle_length == 0:
            criteria = self.do_metrics(training_loss,likelihood,reg_loss,i)
            self.scheduler.step(criteria)
            if criteria < self.best:
                self.best = criteria
                print('new best val score: ', self.best)
                print('Dumping model')
                self.dump_model()
                self.counter = 0
            else:
                self.counter += 1
            if self.counter>self.patience:
                return True

    def training_loop(self,epoch):
        self.dataloader.dataset.set(mode='train')
        total_loss_train=0.
        tot_likelihood=0.
        tot_reg_loss=0.
        self.model = self.model.train()
        for i,(X,x_cat,y,delta) in enumerate(tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            delta = delta.to(self.device)
            mask = delta==1
            X_f = X[mask, :]
            y_f = y[mask, :]
            if not isinstance(x_cat,list): #F
                x_cat = x_cat.to(self.device)
                x_cat_f = x_cat[mask,:]
            else:
                x_cat_f = []
            S = self.model.forward_cum(X, y, mask, x_cat)
            if S.numel() == 0:
                S=S.detach()
            f = self.model(X_f, y_f, x_cat_f)
            if f.numel() == 0:
                f=f.detach()
            # print(torch.isnan(S).sum())
            # print(torch.isnan(f).sum())
            total_loss =self.train_objective(S,f)
            self.optimizer.zero_grad()
            total_loss.backward()
            # for par in self.model.parameters():
            #     print(par.grad)

            self.optimizer.step()
            total_loss_train+=total_loss.detach()
            tot_likelihood+=total_loss.detach()
            if self.eval_func(i,total_loss_train/(i+1),tot_likelihood/(i+1),tot_reg_loss/(i+1)):
                return True
        return False

    def eval_loop(self,grid_size,chunks=50,max_series_accumulation=50000):
        self.model.eval()
        S_series_container = []
        S_log = []
        f_log = []
        durations = []
        events = []
        chunks = self.dataloader.batch_size//chunks+1
        t_grid_np = np.linspace(self.dataloader.dataset.min_duration, self.dataloader.dataset.max_duration,
                                grid_size)
        time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
        for i, (X, x_cat, y, delta) in enumerate(tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            delta = delta.to(self.device)
            mask = delta == 1
            X_f = X[mask, :]
            y_f = y[mask, :]
            if not isinstance(x_cat, list):
                x_cat = x_cat.to(self.device)
                x_cat_f = x_cat[mask, :]
            else:
                x_cat_f = []
            if torch.sum(mask).item() != mask.shape[0]:
                S = self.model.forward_cum(X, y, mask, x_cat)
                S = S.detach()
                S_log.append(S)
            if torch.sum(mask).item() != 0:
                X_f = X_f.to(self.device)
                y_f = y_f.to(self.device)
                f = self.model(X_f, y_f, x_cat_f)
                f = f.detach()
                f_log.append(f)
            if i*X.shape[0]<max_series_accumulation:
                if not isinstance(x_cat, list):
                    for chk,chk_cat in zip(torch.chunk(X, chunks),torch.chunk(x_cat, chunks)):
                        input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                        X_repeat = chk.repeat_interleave(grid_size, 0)
                        x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
                        S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                        S_serie = S_serie.detach()
                        S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                else:
                    x_cat_repeat = []
                    for chk in torch.chunk(X, chunks):
                        input_time = time_grid.repeat((chk.shape[0], 1)).to(self.device)
                        X_repeat = chk.repeat_interleave(grid_size, 0)
                        S_serie = self.model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                        S_serie = S_serie.detach()
                        S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
                durations.append(y.cpu().numpy())
                events.append(delta.cpu().numpy())
        non_normalized_durations = np.concatenate(durations)
        durations = self.dataloader.dataset.invert_duration(non_normalized_durations).squeeze()
        events = np.concatenate(events).squeeze()
        if S_log:
            S_log = torch.cat(S_log)
        else:
            S_log = torch.empty(0, 1, dtype=torch.float32).to(self.device)

        if f_log:
            f_log = torch.cat(f_log)
        else:
            f_log = torch.empty(0, 1, dtype=torch.float32).to(self.device)
        S_series_container = pd.DataFrame(torch.cat(S_series_container,1).numpy())
        t_grid_np = self.dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
        S_series_container=S_series_container.set_index(t_grid_np)
        val_likelihood,conc,ibs,inll = self.calc_eval_objective(S_log, f_log,S_series_container,durations=durations,events=events,time_grid=t_grid_np)
        self.model.train()
        return [val_likelihood.item(),val_likelihood.item()],conc,ibs,inll, {"survival": S_series_container, "times": t_grid_np}

    def train_score(self):
        self.dataloader.dataset.set(mode='train')
        return self.eval_loop(self.grid_size,chunks=self.chunks,max_series_accumulation=self.max_series_accumulation)
    def validation_score(self):
        self.dataloader.dataset.set(mode='val')
        return self.eval_loop(self.grid_size,chunks=self.chunks,max_series_accumulation=self.max_series_accumulation)

    def test_score(self):
        self.dataloader.dataset.set(mode='test')
        return self.eval_loop(self.grid_size)
    
    def dump_model(self):
        torch.save(self.model.state_dict(), self.save_path + f'best_model_{self.global_hyperit}.pt')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_path + f'best_model_{self.global_hyperit}.pt'))

    def full_loop(self):
        self.counter = 0
        if self.debug:
            self.writer =SummaryWriter()
            self.debug_list = []
        for i in range(self.total_epochs):
            if self.training_loop(i):
                break
        if self.debug:
            print(f'best test ibs {min(self.debug_list)}')
        self.load_model()
        val_likelihood,val_conc,val_ibs,val_inll, _ = self.validation_score()
        test_likelihood,test_conc,test_ibs,test_inll, test_surv = self.test_score()



        return self.parse_results(val_likelihood,val_conc,val_ibs,val_inll,
                                  test_likelihood,test_conc,test_ibs,test_inll, test_surv)

    def parse_results(self, val_likelihood,val_conc,val_ibs,val_inll,
                      test_likelihood, test_conc, test_ibs, test_inll,test_surv,val_loss_cox=None):
        if self.selection_criteria == 'train':
            if val_loss_cox is None:
                criteria = val_likelihood[0]
                criteria_test = test_likelihood[0]
            else:
                criteria = val_loss_cox
                criteria_test = val_loss_cox
        if self.selection_criteria == 'likelihood':
            criteria = val_likelihood[0]
            criteria_test = test_likelihood[0]
        elif self.selection_criteria == 'concordance':
            criteria = -val_conc
            criteria_test = -test_conc
        elif self.selection_criteria == 'ibs':
            criteria = val_ibs
            criteria_test = test_ibs
        elif self.selection_criteria == 'inll':
            criteria = val_inll
            criteria_test = test_inll

        return {'loss': criteria,
                'status': STATUS_OK,
                'test_loss': criteria_test,
                'test_loglikelihood_1':test_likelihood[0],
                'test_loglikelihood_2':test_likelihood[1],
                 'test_conc':test_conc,
                 'test_ibs':test_ibs,
                 'test_inll':test_inll,
                'val_loglikelihood_1': val_likelihood[0],
                'val_loglikelihood_2': val_likelihood[1],
                'val_conc': val_conc,
                'val_ibs': val_ibs,
                'val_inll': val_inll,
                "test_surv": test_surv
                }


    def run(self):
        if os.path.exists(self.save_path + 'hyperopt_database.p'):
            return
        trials = Trials()
        best = fmin(fn=self,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.hyperits,
                    trials=trials,
                    verbose=True)
        print(space_eval(self.hyperparameter_space, best))
        pickle.dump(trials,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))


    def post_process(self):
        trials = pickle.load(open(self.save_path + 'hyperopt_database.p',
                         "rb"))

        if self.selection_criteria in ['train','likelihood']:
            reverse = False
        elif self.selection_criteria == 'concordance':
            reverse = False
        elif self.selection_criteria in ['ibs','ibs_likelihood']:
            reverse = False
        elif self.selection_criteria == 'inll':
            reverse = False

        best_trial = sorted(trials.results, key=lambda x: x['test_loss'], reverse=reverse)[0] #low to high
        data = [best_trial['test_loglikelihood_1'],best_trial['test_loglikelihood_2'],best_trial['test_conc'],best_trial['test_ibs'],best_trial['test_inll'],
                best_trial['val_loglikelihood_1'],best_trial['val_loglikelihood_2'],best_trial['val_conc'],best_trial['val_ibs'],best_trial['val_inll']]
        df = pd.DataFrame([data],columns=['test_loglikelihood_1','test_loglikelihood_2','test_conc','test_ibs','test_inll',
                                          'val_loglikelihood_1','val_loglikelihood_2','val_conc','val_ibs','val_inll'])
        print(df)
        df.to_csv(self.save_path+'best_results.csv',index_label=False)
        return best_trial['test_surv']



















