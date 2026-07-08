import os
import glob
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import json

from copy import deepcopy

from utils_train import update_ema

from tqdm import tqdm

BAR = "=============="
def print_with_bar(log_msg):
    log_msg = BAR + log_msg + BAR
    if "End" in log_msg:
         log_msg += "\n"
    print(log_msg)

class Trainer:
    def __init__(
            self, diffusion, train_iter, dataset, test_dataset,  metrics, logger, 
            lr, weight_decay,
            steps, batch_size, check_val_every,
            sample_batch_size, model_save_path, result_save_path,
            num_samples_to_generate=None,
            lr_scheduler='reduce_lr_on_plateau',
            reduce_lr_patience=100, factor=0.9, 
            ema_decay=0.997,
            closs_weight_schedule = "fixed",
            c_lambda = 1.0,
            d_lambda = 1.0,
            device=torch.device('cuda:1'),
            ckpt_path = None,
            y_only=False,
            **kwargs
    ):
        self.y_only = y_only
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_num_schedule = deepcopy(self.diffusion.num_schedule)
        for param in self.ema_num_schedule.parameters():
            param.detach_()
        self.ema_cat_schedule = deepcopy(self.diffusion.cat_schedule)
        for param in self.ema_cat_schedule.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_decay = ema_decay
        self.lr_scheduler = lr_scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=reduce_lr_patience, verbose=True)
        self.closs_weight_schedule = closs_weight_schedule
        self.c_lambda = c_lambda
        self.d_lambda = d_lambda

        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.num_samples_to_generate = num_samples_to_generate
        self.metrics = metrics
        self.logger = logger
        self.check_val_every = check_val_every
        
        self.device = device
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.ckpt_path = ckpt_path
        if self.ckpt_path is not None:
            state_dicts = torch.load(self.ckpt_path, map_location=self.device)
            self.diffusion._denoise_fn.load_state_dict(state_dicts['denoise_fn'])
            self.diffusion.num_schedule.load_state_dict(state_dicts['num_schedule'])
            self.diffusion.cat_schedule.load_state_dict(state_dicts['cat_schedule'])   
            print(f"Weights are loaded from {self.ckpt_path}")     
        
        self.curr_epoch = int(os.path.basename(self.ckpt_path).split('_')[-1].split('.')[0]) if self.ckpt_path is not None else 0

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, closs_weight, dloss_weight):
        x = x.to(self.device)
        
        self.diffusion.train()

        self.optimizer.zero_grad()

        dloss, closs = self.diffusion.mixed_loss(x)

        loss = dloss_weight * dloss + closs_weight * closs
        loss.backward()
        self.optimizer.step()

        return dloss, closs
    
    def compute_loss(self):      # eval loss is not weighted
        curr_dloss = 0.0
        curr_closs = 0.0
        curr_count = 0
        data_iter = self.train_iter
        for batch in data_iter:
            x = batch.float().to(self.device)
            self.diffusion.eval()
            with torch.no_grad():
                batch_dloss, batch_closs = self.diffusion.mixed_loss(x)
            curr_dloss += batch_dloss.item() * len(x)
            curr_closs += batch_closs.item() * len(x)
            curr_count += len(x)
        mloss = np.around(curr_dloss / curr_count, 4)
        gloss = np.around(curr_closs / curr_count, 4)
        return mloss, gloss
    
    def run_loop(self):
        patience = 0
        closs_weight, dloss_weight = self.c_lambda, self.d_lambda
        best_loss = np.inf
        best_ema_loss = np.inf
        best_val_loss = np.inf
        start_time = time.time()
        print_with_bar(f"Starting Trainin Loop, total number of epoch = {self.steps}")
        # Set up wandb's step metric
        self.logger.define_metric("epoch")
        self.logger.define_metric("*", step_metric="epoch")
        
        start_epoch = self.curr_epoch
        if start_epoch > 0:
            print_with_bar(f"Resuming training from epoch {start_epoch}, with validation check every {self.check_val_every} epoches")
        for epoch in range (start_epoch, self.steps):
            self.curr_epoch = epoch+1
            # Set up pbar
            pbar = tqdm(self.train_iter, total=len(self.train_iter))
            pbar.set_description(f"Epoch {epoch+1}/{self.steps}")
            
            # Compute the loss weights
            if self.closs_weight_schedule == "fixed":
                pass
            elif self.closs_weight_schedule == "anneal":
                frac_done = epoch / self.steps
                closs_weight = self.c_lambda * (1 - frac_done)
            else:
                raise NotImplementedError(f"The continuous loss weight schedule {self.closs_weight_schedule} is not implemneted")

            # Training Step
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            curr_lr = self.optimizer.param_groups[0]['lr']
            for batch in pbar:
                x = batch.float().to(self.device)
                batch_dloss, batch_closs = self._run_step(x, closs_weight, dloss_weight)
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
                pbar.set_postfix({
                    "lr": curr_lr,
                    "DLoss": np.around(curr_dloss/curr_count, 4),
                    "CLoss": np.around(curr_closs/curr_count, 4),
                    "TotalLoss": np.around((curr_dloss + curr_closs)/curr_count, 4),
                    "closs_weight": closs_weight,
                    "dloss_weight": dloss_weight,
                })
                
            # Log training Loss
            log_dict = {}
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss + gloss
            if np.isnan(gloss):
                    print('Finding Nan in gaussian loss')
                    break
            loss_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "closs_weight": closs_weight,
                "dloss_weight": dloss_weight,
                "loss/c_loss": gloss,
                "loss/d_loss": mloss,
                "loss/total_loss": total_loss
            }
            log_dict.update(loss_dict)
            
            # Log the learned noise schedules for numerical dimensions
            if self.dataset.d_numerical > 0:    # numerical data is not empty
                num_noise_dict = {}
                if self.diffusion.num_schedule.rho().dim() == 0:   # non-learnable num schedule
                    num_noise_dict = {"num_noise/rho": self.diffusion.num_schedule.rho().item()}
                else:
                    num_noise_dict = {f"num_noise/rho_col_{i}": value.item() for i, value in enumerate(self.diffusion.num_schedule.rho())}
                log_dict.update(num_noise_dict)            

            # Log the learned noise schedules for categlrical dimensions
            if len(self.dataset.categories) > 0:    # categorical data is not empty
                cat_noise_dict = {}
                if self.diffusion.cat_schedule.k().dim() == 0:   # non-learnable cat schedule
                    cat_noise_dict = {"cat_noise/k": self.diffusion.cat_schedule.k().item()}
                else:
                    cat_noise_dict = {f"cat_noise/k_col_{i}": value.item() for i, value in enumerate(self.diffusion.cat_schedule.k())}
                log_dict.update(cat_noise_dict)
            
            # Adjust learning rate
            if self.lr_scheduler == 'reduce_lr_on_plateau':
                self.scheduler.step(total_loss)
            elif  self.lr_scheduler == 'anneal':
                self._anneal_lr(epoch)
            elif self.lr_scheduler == 'fixed':
                pass
            else:
                raise NotImplementedError(f"LR scheduler with name '{self.lr_scheduler}' is not implemented")
            
            # Update EMA models
            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters(), rate=self.ema_decay)
            update_ema(self.ema_num_schedule.parameters(), self.diffusion.num_schedule.parameters(), rate=self.ema_decay)
            update_ema(self.ema_cat_schedule.parameters(), self.diffusion.cat_schedule.parameters(), rate=self.ema_decay)

            # Save ckpt base on the best training loss
            if total_loss < best_loss and self.curr_epoch > 4000:
                best_loss = total_loss
                to_remove = glob.glob(os.path.join(self.model_save_path, f"best_model_*"))
                if to_remove:
                    os.remove(to_remove[0])
                state_dicts = {
                    'denoise_fn': self.diffusion._denoise_fn.state_dict(), 
                    'num_schedule':self.diffusion.num_schedule.state_dict(), 
                    'cat_schedule': self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f'best_model_{np.round(total_loss,4)}_{epoch+1}.pt'))
                patience = 0
            else:
                patience += 1   # increment patience if best loss is not surpassed
            
            # Compute and log EMA model loss
            curr_model, curr_num_schedule, curr_cat_schedule = self.to_ema_model()
            ema_mloss, ema_gloss = self.compute_loss()
            self.to_model(curr_model, curr_num_schedule, curr_cat_schedule)
            ema_total_loss = ema_mloss + ema_gloss
            ema_loss_dict = {
                "ema_loss/c_loss": ema_gloss,
                "ema_loss/d_loss": ema_mloss,
                "ema_loss/total_loss": ema_total_loss
            }
            
            # Save the best ema ckpt
            if ema_total_loss < best_ema_loss and self.curr_epoch > 4000:
                best_ema_loss = ema_total_loss
                to_remove = glob.glob(os.path.join(self.model_save_path, f"best_ema_model_*"))
                if to_remove:
                    os.remove(to_remove[0])
                state_dicts = {
                    'denoise_fn': self.ema_model.state_dict(), 
                    'num_schedule':self.ema_num_schedule.state_dict(), 
                    'cat_schedule': self.ema_cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f'best_ema_model_{np.round(ema_total_loss,4)}_{epoch+1}.pt'))
            
            # Evaluate Sample Quality
            if (epoch+1) % self.check_val_every == 0:
                state_dicts = {
                    'denoise_fn': self.diffusion._denoise_fn.state_dict(), 
                    'num_schedule':self.diffusion.num_schedule.state_dict(), 
                    'cat_schedule': self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f'model_{epoch+1}.pt'))
                
                print_with_bar(f"Routine Generation Evaluation every {self.check_val_every}, currently at epoch #{epoch+1}, wiht total_loss={total_loss}.")
                out_metrics, _, _ = self.evaluate_generation(save_metric_details=True, plot_density=True)
                log_dict.update(out_metrics)
                print(f"Eval Resutls of the Non-EMA model:\n {out_metrics}")

                # Evaluate the EMA model
                torch.save(self.ema_model.state_dict(), os.path.join(self.model_save_path, f'ema_model_{epoch+1}.pt'))
                ema_out_metrics, _, _ = self.evaluate_generation(ema=True, save_metric_details=True, plot_density=True)
                log_dict.update({
                    "ema": ema_out_metrics,
                })
                print(f"Eval Resutls of the EMA model:\n {ema_out_metrics}")
            
            # Submit logs
            self.logger.log(log_dict)

        end_time = time.time()
        print_with_bar(f"Ending Trainnig Loop, totoal training time = {end_time - start_time}")
        self.logger.log({
            'training_time': end_time - start_time
        })
        
    def report_test(self, num_runs):
        save_dir = self.result_save_path
        
        shape_ = []
        trend_ = []
        mle_ = []
        c2st_ = []
        for i in range(num_runs):
            print_with_bar(f"GENERAL Evaluation Run {i}")
            out_metrics, extras, syn_df = self.evaluate_generation()
            print(f"Results of Run {i} are: \n{out_metrics}")
            shape_.append(out_metrics["density/Shape"])
            trend_.append(out_metrics["density/Trend"])
            mle_.append(out_metrics["mle"])
            c2st_.append(out_metrics["c2st"])
            # Save samples for quality evaluation
            save_path = os.path.join(save_dir, "all_samples")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            syn_df.to_csv(os.path.join(save_path, f"samples_{i}.csv"), index=False)

        shape_ = np.array(shape_)
        trend_ = np.array(trend_)
        mle_ = np.array(mle_)
        c2st_ = np.array(c2st_)
        
        shape_error = (1 - shape_)*100
        trend_error = (1 - trend_)*100
        c2st_percent = c2st_ * 100
        
        all_results = pd.DataFrame({
            "shape": shape_error,
            "trend": trend_error,
            "mle": mle_,
            "c2st": c2st_percent,
        })
        avg = all_results.mean(axis=0).round(3)
        std = all_results.std(axis=0).round(3)
        avg_std = pd.concat([avg, std], axis=1, ignore_index=True)
        avg_std.columns = ["avg", "std"]
        avg_std.index = [
            "shape", 
            "trend", 
            "mle", 
            "c2st", 
        ]
        
        # Savings
        all_results.to_csv(f"{save_dir}/all_results.csv", index=True)
        avg_std.to_csv(f"{save_dir}/avg_std.csv", index=True)
        print_with_bar(f"The AVG over {num_runs} runs are: \n{avg_std}")
        
    def report_test_dcr(self, num_runs):
        save_dir = self.result_save_path
        
        dcr_ = []
        dcr_real_ = []
        dcr_test_ = []
        for i in range(num_runs):
            print_with_bar(f"DCR Evaluation Run {i}")
            out_metrics, extras, syn_df = self.evaluate_generation()
            print(f"Results of Run {i} are: \n{out_metrics}")
            dcr_.append(out_metrics["dcr"])
            dcr_real_.append(extras["dcr_real"])
            dcr_test_.append(extras["dcr_test"])
            save_path = os.path.join(save_dir, "all_samples")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            syn_df.to_csv(os.path.join(save_path, f"samples_{i}.csv"), index=False)

        dcr_ = np.array(dcr_)
        
        dcr_percent = dcr_ * 100
        
        all_results = pd.DataFrame({
            "dcr": dcr_percent,
        })
        avg = all_results.mean(axis=0).round(3)
        std = all_results.std(axis=0).round(3)
        avg_std = pd.concat([avg, std], axis=1, ignore_index=True)
        avg_std.columns = ["avg", "std"]
        avg_std.index = [
            "dcr", 
        ]
        
        # Savings
        all_results.to_csv(f"{save_dir}/all_results.csv", index=True)
        avg_std.to_csv(f"{save_dir}/avg_std.csv", index=True)
        dcr_real = np.concatenate(dcr_real_, axis=0)
        dcr_test = np.concatenate(dcr_test_, axis=0)
        dcr_df = pd.DataFrame({
            "dcr_real": dcr_real,
            "dcr_test": dcr_test
        })
        dcr_df.to_csv(f"{save_dir}/dcr.csv", index=False)
        
        print_with_bar(f"The AVG over {num_runs} runs are: \n{avg_std}")
        
    def test(self):    
        out_metrics, _, _ = self.evaluate_generation(save_metric_details=True, plot_density=True)
        print_with_bar(f"Results of the test are: \n{out_metrics}")
        self.logger.log(out_metrics)
        print(out_metrics)

    def evaluate_generation(self, save_metric_details=False, plot_density=False, ema=False):
        self.diffusion.eval()
        
        # Sample a synthetic table
        num_samples = self.num_samples_to_generate if self.num_samples_to_generate else self.metrics.real_data_size # By default, num_samples_to_generate is not specified. In this case, we generate the same number of samples as the real dataset. This approach is consistently used across all experiments in the paper.
        syn_df = self.sample_synthetic(num_samples, ema=ema)
        
        # Save the sample
        save_path = os.path.join(self.result_save_path, str(self.curr_epoch), "ema" if ema else "")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, "samples.csv")
        syn_df.to_csv(path, index=False)
        print(
            f"Samples are saved at {path}"
        )
        
        # Compute evaluation metrics on the sample
        syn_df_loaded = pd.read_csv(os.path.join(save_path, "samples.csv")) # In the original tabsyn code, syn_data is implicitly casted into float.64 when it gets loaded with pd.read_csv in the evaluation script. If we don't cast, the density evluation for some columns (especially those with tailed and peaked distribution) will collapse.
        out_metrics, extras = self.metrics.evaluate(syn_df_loaded)
        
        # Save metrics and metric details
        path = os.path.join(save_path, "all_results.json")
        with open(path, "w") as json_file:
            json.dump(out_metrics, json_file, indent=4, separators=(", ", ": "))        # always locally save the output metrics
        if save_metric_details:
            for name, extra in extras.items():
                if isinstance(extra, pd.DataFrame):
                    extra.to_csv(os.path.join(save_path, f"{name}.csv"))
                elif isinstance(extra, dict):
                    with open(os.path.join(save_path, f"{name}.json"), "w") as json_file:
                        json.dump(extra, json_file, indent=4, separators=(", ", ": "))
                else:
                    raise NotImplementedError(f"Extra file generated during evaluations has type {type(extra)}, and code to save this type of file is not implemented")
        
        # Plot density figures
        if plot_density:
            img = self.metrics.plot_density(syn_df_loaded)
            path = os.path.join(save_path, "density_plots.png")
            img.save(path)
            print(
                f"The density plots are saved at {path}"
            )
        return out_metrics, extras, syn_df
        

    def sample_synthetic(self, num_samples, keep_nan_samples=True, ema=False):
        if ema:
            curr_model, curr_num_schedule, curr_cat_schedule = self.to_ema_model()
        info = self.metrics.info
        
        print_with_bar(f"Starting Sampling, total samples to generate = {num_samples}")
        start_time = time.time()
        
        syn_data = self.diffusion.sample_all(num_samples, self.sample_batch_size, keep_nan_samples=keep_nan_samples)
        print(f"Shape of the generated sample = {syn_data.shape}")
        
        if keep_nan_samples:
            num_all_zero_row = (syn_data.sum(dim=1) == 0).sum()
            if num_all_zero_row:
                print(f"The generated samples contain {num_all_zero_row} Nan instances!!!")
                self.logger.log({
                    'num_Nan_sample': num_all_zero_row
                })

        # Recover tables
        num_inverse = self.dataset.num_inverse
        int_inverse = self.dataset.int_inverse
        cat_inverse = self.dataset.cat_inverse
        
        if self.y_only:
            if info['task_type'] == 'binclass':
                syn_data = cat_inverse(syn_data)
            else:
                syn_data = num_inverse(syn_data)
            syn_df = pd.DataFrame()
            syn_df[info['column_names'][info['target_col_idx'][0]]] = syn_data[:, 0]
        else:
            syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, int_inverse, cat_inverse) 
            syn_df = recover_data(syn_num, syn_cat, syn_target, info)
            

            idx_name_mapping = info['idx_name_mapping']
            idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

            syn_df.rename(columns = idx_name_mapping, inplace=True)
        
        end_time = time.time()
        print_with_bar(f"Ending Sampling, totoal sampling time = {end_time - start_time}")
        
        if ema:
            self.to_model(curr_model, curr_num_schedule, curr_cat_schedule)

        return syn_df
    
    def to_ema_model(self):
        curr_model = self.diffusion._denoise_fn
        curr_num_schedule = self.diffusion.num_schedule
        curr_cat_schedule = self.diffusion.cat_schedule
        self.diffusion._denoise_fn = self.ema_model  # temporarily install the ema parameters into the model
        self.diffusion.num_schedule = self.ema_num_schedule
        self.diffusion.cat_schedule = self.ema_cat_schedule
        
        return curr_model, curr_num_schedule, curr_cat_schedule

    def to_model(self, curr_model, curr_num_schedule, curr_cat_schedule):
        self.diffusion._denoise_fn = curr_model      # give back the parameters
        self.diffusion.num_schedule = curr_num_schedule
        self.diffusion.cat_schedule = curr_cat_schedule
        
    def test_impute(self, trail_start, trial_size, resample_rounds, impute_condition, imputed_sample_save_dir, w_num, w_cat):
        self.diffusion.eval()
        
        info = self.metrics.info
        task_type = info['task_type']
        d_numerical, categories = self.dataset.d_numerical, self.dataset.categories
        num_mask_idx, cat_mask_idx = [], []
        X_train = self.dataset.X
        X_train = X_train
        x_num_train, x_cat_train = X_train[:,:d_numerical], X_train[:,d_numerical:]
        
        if task_type == 'binclass':    # for cat cols, push the masked col to [MASK]
            cat_mask_idx += [0]
        else:      # for num cols, set the masked col to the col mean
            num_mask_idx += [0]
            avg = x_num_train[:, num_mask_idx].mean(0).to(self.device)
        
        with torch.no_grad():
            
            for trial in range(trail_start, trail_start+trial_size):
                print_with_bar(f"Imputing trial {trial}")
                X_test = self.test_dataset.X
                X_test = deepcopy(X_test).to(self.device)
                x_num_test, x_cat_test = X_test[:, :d_numerical], X_test[:, d_numerical:].long()
                
                # Apply mask to x_0
                if num_mask_idx:
                    x_num_test[:, num_mask_idx] = avg
                if cat_mask_idx:
                    x_cat_test[:, cat_mask_idx] = torch.tensor(categories, dtype=x_cat_test.dtype, device=x_cat_test.device)[cat_mask_idx]
                
                # Sample imputed tables
                syn_data = self.diffusion.sample_impute(x_num_test, x_cat_test, num_mask_idx, cat_mask_idx, resample_rounds, impute_condition, w_num, w_cat)
                print(f"Shape of the imputed sample = {syn_data.shape}")

                # Recover tables
                num_inverse = self.dataset.num_inverse
                int_inverse = self.dataset.int_inverse
                cat_inverse = self.dataset.cat_inverse
                
                if torch.any((syn_data[:, d_numerical+1:]).max(dim=0).values > (x_cat_train[:,1:]).max(dim=0).values):     # if the test set contains categories not presented in the train set, we can not do cat_inverse. So we implement a patch that set those columns to the same as the train set
                    print("Test set contains extra categories, and so does imputed syn data. We cannot do cat_inverse. So we set the cat columns as the same as the train set")
                    syn_data[:, d_numerical+1:] = x_cat_train[:syn_data.shape[0],1:]
                    
                
                syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, int_inverse, cat_inverse) 
                syn_df = recover_data(syn_num, syn_cat, syn_target, info)

                idx_name_mapping = info['idx_name_mapping']
                idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

                syn_df.rename(columns = idx_name_mapping, inplace=True)
                
                # Save imputed samples
                os.makedirs(imputed_sample_save_dir) if not os.path.exists(imputed_sample_save_dir) else None
                print(f"Imputed samples are saved to {imputed_sample_save_dir}/{trial}.csv")
                syn_df.to_csv(f'{imputed_sample_save_dir}/{trial}.csv', index = False)
        
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, int_inverse, cat_inverse):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_num = int_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)


    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df