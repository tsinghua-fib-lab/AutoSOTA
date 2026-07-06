import os
import wandb
import torch
import logging
import time
import evaluation
import csv
import torch.nn.functional as F
import random
import numpy as np

from tqdm import tqdm
from torch import optim
from utils import callbacks
from collections import defaultdict
from model.loss_func import *
from data.dataset import *
from typing import Dict, List, Optional, Tuple

from utils.utils import xavier_normal_initialization, normal_initialization


class BaseModel(nn.Module):

    def __init__(self, config : Dict, dataset_list : List[BaseDataset]) -> None:
        super().__init__()
        self.config = config
        self.ckpt_path = None
        self.logger = logging.getLogger('CDR')
        self.dataset_list = dataset_list
        self.device = config['train']['device']
        self.fuid = 'user_id'
        self.fiid = 'item_id'
        self.domain_name_list = dataset_list[0].domain_name_list
        self.training_time = 0
        self.inference_time = 0
        self.dataset_class = config['data']['dataset_class']
        
        # Paths for the sample analysis log files
        self.analysis_log_path = None
        self.validation_analysis_log_path = None
        self.test_analysis_log_path = None

        # register mode-relevant parameters
        self.embed_dim = config['model']['embed_dim']
        self.max_seq_len = config['data']['max_seq_len']
        self.num_users = dataset_list[0].num_users
        self.num_items = dataset_list[0].num_items
        self.item_embedding = nn.Embedding(self.num_items, self.embed_dim, padding_idx=0)

    def _init_model(self, train_data):
        self.apply(normal_initialization)
        self = self.to(self.device)
        self.optimizer = self._get_optimizers()
        self.loss_fn = self._get_loss_func()

    def _neg_sampling(self, batch):
        user_seq = batch['in_' + self.fiid]
        num_neg = self.config.get('neg_num')
        weight = torch.ones(user_seq.shape[0], self.num_items, device=self.device)
        
        _idx = torch.arange(user_seq.size(0), device=self.device).view(-1, 1).expand_as(user_seq)
        weight.scatter_(1, user_seq, 0.0)

        weight[:, 0] = 0

        if len(batch[self.fiid].shape) == 2:
            L = batch[self.fiid].shape[1]
            neg_idx = torch.multinomial(weight, L * num_neg, replacement=True)
            neg_idx = neg_idx.view(batch[self.fiid].shape[0], L, num_neg)
        else:
            neg_idx = torch.multinomial(weight, num_neg, replacement=True)
            
        return neg_idx


    def _get_dataset_class(config):
        if config['data']['dataset_class'] == 'condense':
            return CondenseDataset
        elif config['data']['dataset_class'] == 'general':
            return SeparateDataset
        elif config['data']['dataset_class'] == 'selection':
            return SelectionDataset
        elif config['data']['dataset_class'] == 'split':
            return SplitDataset
        elif config['data']['dataset_class'] == 'cluster':
            return ClusterDataset
        elif config['data']['dataset_class'] == 'pattern':
            return PatternDataset
        elif config['data']['dataset_class'] == 'curriculum':
            return CurriculumDataset
        else:
            raise NotImplementedError

    def _get_optimizers(self):
        opt_name = self.config['train']['optimizer']
        lr = self.config['train']['learning_rate']
        weight_decay = self.config['train']['weight_decay']
        params = self.parameters()

        if opt_name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=lr)
        else:
            optimizer = optim.Adam(params, lr=lr)

        return optimizer

    def _get_loss_func(self):
        loss_fn_name = self.config.get('loss_fn_name')
        if loss_fn_name == 'bce':
            return BinaryCrossEntropyLoss()
        elif loss_fn_name == 'bpr':
            return BPRLoss()
        elif loss_fn_name == 'sampled_softmax':
            return SampledSoftmaxLoss()
        else:
            raise NotImplementedError(f"Loss function '{loss_fn_name}' is not implemented.")

    def forward(self, batch):
        raise NotImplementedError

    def fit(self):
        self.callback = callbacks.EarlyStopping(self, 'ndcg@20', self.config['data']['dataset'], patience=self.config['train']['early_stop_patience'])
        self.analyzer = callbacks.Analyzer(self)
        self.logger.info('save_dir:' + self.callback.save_dir)
        self._init_model(self.dataset_list[0])
        self.logger.info(self)

        base_log_path = '/saved/sasrec_toy_5090/data/temp.csv' 
        self.analysis_log_path = base_log_path
        
        # Derive validation and test log paths from the base path
        base, ext = os.path.splitext(base_log_path)
        self.validation_analysis_log_path = f"{base}_validation{ext}"
        self.test_analysis_log_path = f"{base}_test{ext}"

        # Define header for all log files
        header = [
            'epoch', 'phase', 'domain', 'sample_index', 'position', 'user_id', 'target_item_id',
            'is_padding', 'loss', 'similarity_score', 'neg_similarity_score',
            'confidence', 'predicted_item_id', 'max_confidence',
            'target_item_rank', 'max_score_logit'
        ]

        # Initialize all three log files
        log_paths = {
            "Training": self.analysis_log_path,
            "Validation": self.validation_analysis_log_path,
            "Test": self.test_analysis_log_path
        }

        for name, path in log_paths.items():
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                self.logger.info(f"{name} analysis log will be saved to: {path}")
            except IOError as e:
                self.logger.error(f"Failed to initialize {name} analysis log file: {e}")
                if name == "Training": self.analysis_log_path = None
                elif name == "Validation": self.validation_analysis_log_path = None
                else: self.test_analysis_log_path = None
        
        self.fit_loop()

    def fit_loop(self):
        try:
            nepoch = 0
            self.train_start()
            for e in range(self.config['train']['epochs']):
                self.logged_metrics = {}
                self.logged_metrics['epoch'] = nepoch

                # training procedure
                tik_train = time.time()
                self.train()
                training_output_list = self.training_epoch(nepoch)
                tok_train = time.time()
                self.training_time += tok_train - tik_train

                # validation procedure
                tik_valid = time.time()
                self.eval()
                if nepoch % 1 == 0:
                    for domain in self.domain_name_list:
                        val_dataset = self.dataset_list[1]
                        val_dataset.set_eval_domain(domain)
                        self.set_eval_domain(domain)
                        val_loader = val_dataset.get_loader()
                        validation_output_list = self.validation_epoch(nepoch, val_loader)
                        # The call to validation_epoch_end is now safe because the function self-isolates.
                        self.validation_epoch_end(validation_output_list, domain, val_dataset) 
                    all_domain_result = defaultdict(float)
                    for k, v in self.logged_metrics.items():
                        for domain_name in self.domain_name_list:
                            if domain_name in k: # result of a single domain
                                all_domain_result[k.removeprefix(domain_name + '_')] += v
                                break
                    self.logged_metrics.update(all_domain_result)
                    wandb.log(all_domain_result)
                tok_valid = time.time()
                self.inference_time += tok_valid - tik_valid

                # This is the original, unmodified RNG protection block from your code.
                # It must be preserved exactly as it was.
                random_state = random.getstate()
                numpy_state = np.random.get_state()
                cpu_rng_state = torch.get_rng_state()
                gpu_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

                self.training_epoch_end(training_output_list, nepoch)
                
                random.setstate(random_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(cpu_rng_state)
                if torch.cuda.is_available() and gpu_rng_state is not None:
                    torch.cuda.set_rng_state(gpu_rng_state)

                # model is saved in callback when the callback return True.
                stop_training = self.callback(self, nepoch, self.logged_metrics)
                if stop_training:
                    break

                nepoch += 1

            self.training_end()
            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        except KeyboardInterrupt:
            # if catch keyboardinterrupt in training, save the best model.
            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        return
    
    def current_epoch_trainloaders(self, nepoch):
        if self.dataset_class == 'curriculum':
            curriculum_config = self.config['train'].get('curriculum', {})
            is_curriculum_enabled = curriculum_config.get('enabled', False)

            if not is_curriculum_enabled:
                batch_size = self.config['train']['batch_size']
                return self.dataset_list[0].get_loader(batch_size=batch_size, shuffle=True)

            schedule = curriculum_config.get('schedule', [])
            train_dataset = self.dataset_list[0]
            batch_size = self.config['train']['batch_size']
            
            cumulative_epochs = 0
            current_mix_ratio = None

            for phase_config in schedule:
                phase_duration = phase_config['epochs']
                
                if phase_duration == 0:
                    current_mix_ratio = phase_config['mix_ratio']
                    break

                start_epoch = cumulative_epochs
                end_epoch = cumulative_epochs + phase_duration
                
                if start_epoch <= nepoch < end_epoch:
                    current_mix_ratio = phase_config['mix_ratio']
                    break
                
                cumulative_epochs = end_epoch

            if current_mix_ratio is None and schedule:
                current_mix_ratio = schedule[-1]['mix_ratio']

            self.logger.info(f"Epoch {nepoch}: Applying curriculum with mix_ratio: {current_mix_ratio}")
            
            return train_dataset.get_loader(
                batch_size=batch_size, 
                shuffle=True, 
                mix_ratio=current_mix_ratio
            )
        else:
            return self.dataset_list[0].get_loader()

    def train_start(self):
        pass

    def training_epoch(self, nepoch):
        output_list = []

        trn_dataloaders = self.current_epoch_trainloaders(nepoch)
        trn_dataloaders = [trn_dataloaders]

        for loader_idx, loader in enumerate(trn_dataloaders):
            outputs = []
            loader = tqdm(
                loader,
                total=len(loader),
                ncols=75,
                desc=f"Training {nepoch:>5}",
                leave=False,
            )
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch['neg_item'] = self._neg_sampling(batch)
                self.optimizer.zero_grad()
                
                training_step_args = {'batch': batch, 'nepoch': nepoch}

                loss = self.training_step(**training_step_args)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                outputs.append({f"loss_{loader_idx}": loss.detach()})
            output_list.append(outputs)
        return output_list
    
    def training_step(self, batch, nepoch, reduce=True, return_query=False):

        query = self.forward(batch)
        pos_score = (query * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
        neg_score = (query.unsqueeze(-2) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
        
        pos_score[batch[self.fiid] == 0] = -torch.inf
        loss_per_sample_train = self.loss_fn(pos_score, neg_score, reduce=False)
        
        weights = torch.ones_like(loss_per_sample_train)

        loss_per_sample = loss_per_sample_train * weights

        if reduce:
            padding_mask = (batch[self.fiid] != 0)
            loss_value = loss_per_sample[padding_mask].sum() / padding_mask.sum().clamp(min=1)
        else:
            loss_value = loss_per_sample

        if return_query:
            return loss_value, query
        else:
            return loss_value
        
    def training_epoch_end(self, output_list, nepoch):
        output_list = [output_list] if not isinstance(output_list, list) else output_list
        for outputs in output_list:
            if isinstance(outputs, List):
                loss_metric = {'train_'+ k: torch.hstack([e[k] for e in outputs]).mean() for k in outputs[0]}
            elif isinstance(outputs, torch.Tensor):
                loss_metric = {'train_loss': outputs.item()}
            elif isinstance(outputs, Dict):
                loss_metric = {'train_'+k : v for k, v in outputs}
            self.logged_metrics.update(loss_metric)

        self.logger.info(self.logged_metrics)
        self.logger.info(f'training_time: {self.training_time}')
        self.logger.info(f'inference_time: {self.inference_time}')
        
        self._analyze_and_log_training_difficulty(nepoch)

    def training_end(self):
        pass

    def _analyze_and_log_training_difficulty(self, epoch: int):
        """
        Wrapper to analyze and log training data for a specific epoch.
        """
        if self.analysis_log_path is None:
            self.logger.warning("Training sample analysis logging is disabled because the log file path is not set.")
            return

        analysis_loader = self.dataset_list[0].get_loader(shuffle=False)
        self._log_intermediate_values(
            dataloader=analysis_loader,
            log_path=self.analysis_log_path,
            epoch=epoch,
            phase='train'
        )

    @torch.no_grad()
    def _log_intermediate_values(self, dataloader, log_path: str, epoch: int, phase: str, domain: str = 'N/A'):
        """
        Analyzes samples and logs details to a CSV file. This function assumes it is
        called from within an RNG-protected context if reproducibility is required.
        """
        if log_path is None:
            return

        is_training_before = self.training
        self.eval()
        
        try:
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                progress_desc = f"Analyzing {phase.capitalize()} (Epoch {epoch})"
                for batch in tqdm(dataloader, desc=progress_desc, ncols=75, leave=False):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    batch['neg_item'] = self._neg_sampling(batch)
                    
                    seq_output = self.forward(batch, need_pooling=False)
                    target_ids = batch[self.fiid]
                    B, L, D = seq_output.shape
                    
                    all_item_embeds = self.item_embedding.weight

                    if phase == 'train':
                        seq_output_flat = seq_output.view(B * L, D)
                        target_ids_flat = target_ids.view(B * L)
                        neg_ids_flat = batch['neg_item'].view(B * L, -1)[:, 0] 
                        all_scores = seq_output_flat @ all_item_embeds.T
                        pos_scores_reshaped = (seq_output * all_item_embeds[target_ids]).sum(-1)
                        neg_scores_reshaped = (seq_output.unsqueeze(-2) * all_item_embeds[batch['neg_item']]).sum(-1)
                        loss = self.loss_fn(pos_scores_reshaped, neg_scores_reshaped, reduce=False)
                        loss_flat = loss.view(B * L)
                        sample_indices = batch['index'].unsqueeze(1).expand(B, L).reshape(B * L)
                        positions = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L).reshape(B * L)
                        user_ids = batch[self.fuid].unsqueeze(1).expand(B, L).reshape(B * L)
                        is_padding = (target_ids_flat == 0)
                    else: # validation and test
                        seq_output_flat = seq_output[:, -1, :]
                        target_ids_flat = target_ids
                        neg_ids_flat = batch['neg_item'][:, 0]
                        all_scores = seq_output_flat @ all_item_embeds.T
                        pos_scores = (seq_output_flat * all_item_embeds[target_ids_flat]).sum(-1)
                        neg_scores = (seq_output_flat.unsqueeze(1) * all_item_embeds[batch['neg_item']]).sum(-1)
                        loss = self.loss_fn(pos_scores.unsqueeze(1), neg_scores.unsqueeze(1), reduce=False)
                        loss_flat = loss.squeeze()
                        sample_indices = batch['index']
                        positions = torch.full_like(sample_indices, fill_value=L - 1, device=self.device)
                        user_ids = batch[self.fuid]
                        is_padding = (target_ids_flat == 0)

                    probs = F.softmax(all_scores, dim=1)
                    confidence = probs.gather(1, target_ids_flat.unsqueeze(1)).squeeze(-1)
                    max_confidence, predicted_item_id = torch.max(probs, dim=1)
                    target_embeds = all_item_embeds[target_ids_flat]
                    similarity_score = (seq_output_flat * target_embeds).sum(-1)
                    neg_embeds = all_item_embeds[neg_ids_flat]
                    neg_similarity_score = (seq_output_flat * neg_embeds).sum(-1)
                    target_scores = all_scores.gather(1, target_ids_flat.unsqueeze(1))
                    num_better_scores = torch.sum(all_scores > target_scores, dim=1)
                    target_item_rank = num_better_scores + 1
                    max_score_logit, _ = torch.max(all_scores, dim=1)

                    iterable_columns = [
                        sample_indices.cpu().numpy(), positions.cpu().numpy(),
                        user_ids.cpu().numpy(), target_ids_flat.cpu().numpy(),
                        is_padding.cpu().numpy(), loss_flat.cpu().numpy(),
                        similarity_score.cpu().numpy(), neg_similarity_score.cpu().numpy(),
                        confidence.cpu().numpy(), predicted_item_id.cpu().numpy(),
                        max_confidence.cpu().numpy(), target_item_rank.cpu().numpy(),
                        max_score_logit.cpu().numpy()
                    ]

                    for row_columns in zip(*iterable_columns):
                        if not row_columns[4]:
                            writer.writerow([epoch, phase, domain] + list(row_columns))
        finally:
            self.train(is_training_before)

    @torch.no_grad()
    def validation_epoch(self, nepoch, dataloader):
        output_list = []
        dataloader = tqdm(
            dataloader,
            total=len(dataloader),
            ncols=75,
            desc=f"Evaluating {nepoch:>5}",
            leave=False,
        )
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.validation_step(batch)
            output_list.append(output)
        return output_list

    @torch.no_grad()
    def test_epoch(self, dataloader):
        output_list = []
        dataloader = tqdm(
            dataloader,
            total=len(dataloader),
            ncols=75,
            desc=f"Testing: ",
            leave=False,
        )
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.test_step(batch)
            output_list.append(output)
        return output_list

    def validation_epoch_end(self, outputs, domain, val_dataset):
        # Perform all original, non-random operations first.
        val_metrics = self.config['eval']['val_metrics']
        cutoff = self.config['eval']['cutoff']
        val_metric = evaluation.get_eval_metrics(val_metrics, cutoff, validation=True)
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end_metrics(outputs, val_metric)
            out = dict(zip(val_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end_metrics(outputs, val_metric)
        out = {domain + '_' + k: v for k, v in out.items()}
        self.logged_metrics.update(out)
        self.analyzer.analyze_epoch()
        
        # ========== SELF-ISOLATION BLOCK FOR NEW LOGGING (START) ==========
        # Save the state of ALL random number generators before the new task.
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        try:
            # Execute the new logging task, which is the sole source of new randomness.
            if self.validation_analysis_log_path:
                nepoch = self.logged_metrics.get('epoch', -1)
                analysis_loader = val_dataset.get_loader(shuffle=False)
                self._log_intermediate_values(
                    dataloader=analysis_loader,
                    log_path=self.validation_analysis_log_path,
                    epoch=nepoch,
                    phase='validation',
                    domain=domain
                )
            else:
                self.logger.warning("Validation sample analysis logging is disabled.")
        finally:
            # Restore all RNG states, ensuring this function has no random side effects.
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available() and gpu_rng_state is not None:
                torch.cuda.set_rng_state(gpu_rng_state)
        # ========== SELF-ISOLATION BLOCK FOR NEW LOGGING (END) ==========
        
        return out

    def test_epoch_end(self, outputs, domain, test_dataset):
        # Perform all original, non-random operations first.
        test_metrics = self.config['eval']['test_metrics']
        cutoff = self.config['eval']['cutoff']
        test_metric = evaluation.get_eval_metrics(test_metrics, cutoff, validation=False)
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end_metrics(outputs, test_metric)
            out = dict(zip(test_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end_metrics(outputs, test_metric)
        out = {domain + '_' + k: v for k, v in out.items()}
        self.logged_metrics.update(out)
        self.analyzer.analyze_epoch()

        # ========== SELF-ISOLATION BLOCK FOR NEW LOGGING (START) ==========
        # Save the state of ALL random number generators before the new task.
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        try:
            # Execute the new logging task for the test set.
            if self.test_analysis_log_path:
                analysis_loader = test_dataset.get_loader(shuffle=False)
                self._log_intermediate_values(
                    dataloader=analysis_loader,
                    log_path=self.test_analysis_log_path,
                    epoch=-1,
                    phase='test',
                    domain=domain
                )
            else:
                self.logger.warning("Test sample analysis logging is disabled.")
        finally:
            # Restore all RNG states, ensuring no side effects.
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available() and gpu_rng_state is not None:
                torch.cuda.set_rng_state(gpu_rng_state)
        # ========== SELF-ISOLATION BLOCK FOR NEW LOGGING (END) ==========

        return out
    
    def _test_epoch_end_metrics(self, outputs, metrics):
        if isinstance(outputs[0][0], List):
            metric, bs = zip(*outputs)
            metric = torch.tensor(metric)
            bs = torch.tensor(bs)
            out = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        elif isinstance(outputs[0][0], Dict):
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = (metric * bs).sum() / bs.sum()
        return out

    def validation_step(self, batch):
        eval_metric = self.config['eval']['val_metrics']
        cutoff = self.config['eval']['cutoff']
        return self._test_step(batch, eval_metric, cutoff)

    def test_step(self, batch):
        eval_metric = self.config['eval']['test_metrics']
        cutoffs = self.config['eval']['cutoff']
        return self._test_step(batch, eval_metric, cutoffs)
    
    def _test_step(self, batch, metric, cutoffs):
        rank_m = evaluation.get_rank_metrics(metric)
        topk = self.config['eval']['topk']
        bs = batch['user_id'].size(0)
        assert len(rank_m) > 0
        score, topk_items = self.topk(batch, topk, batch['user_hist'])
        label = batch[self.fiid].view(-1, 1) == topk_items
        pos_rating = batch['label'].view(-1, 1)

        # analyzer
        analyzer_rst = {f"{name}@{cutoff}": func(label, pos_rating, cutoff, mean=False) for cutoff in cutoffs for name, func in rank_m}
        self.analyzer.record_batch(batch[self.fuid], batch['user_hist'], analyzer_rst)

        return {k: v.mean() for k, v in analyzer_rst.items()}, bs
    
    def topk(self, batch, k, user_h=None):
        query = self.forward(batch)
        more = user_h.size(1) if user_h is not None else 0
        real_score = query @ self.item_embedding.weight[:self.num_items].T
        domain_mask = torch.zeros(1, self.num_items, dtype=torch.bool, device=self.device)
        masked_score : torch.Tensor = real_score.masked_fill(domain_mask, -torch.inf)
        masked_score = torch.scatter(masked_score, 1, user_h, -torch.inf)

        score, topk_items = torch.topk(masked_score, k)

        return score, topk_items

    def set_eval_domain(self, domain):
        self.eval_domain = domain

    def evaluate(self) -> Dict:
        test_data = self.dataset_list[-1]
        output = defaultdict(float)
        self.load_checkpoint(os.path.join(self.config['eval']['save_path'], self.ckpt_path))
        self.eval()

        for domain in self.domain_name_list:
            test_data.set_eval_domain(domain)
            self.set_eval_domain(domain)
            test_loader = test_data.get_loader()
            output_list = self.test_epoch(test_loader)
            
            # Pass the test_data object to test_epoch_end for analysis
            output.update(self.test_epoch_end(output_list, domain, test_data))

        all_domain_result = defaultdict(float)
        for k, v in output.items():
            for domain_name in self.domain_name_list:
                if domain_name in k: # result of a single domain
                    all_domain_result[k.removeprefix(domain_name + '_')] += v
        output.update(all_domain_result)

        self.logger.info(output)
        wandb.log(output)
        wandb.log({'training_time': self.training_time, 'inference_time': self.inference_time})
        return output
    
    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.config = ckpt['config']
        self.load_state_dict(ckpt['parameters'])

    @torch.no_grad()
    def generate_and_save_sequences_parallel(self, save_path: str, m: int, k: int, max_len: int, local_prob: float, min_len: int = 3):

        self.logger.info(f"The sequence will be generated and saved in the format of [Parallel V2, no repeated decoding] to: {save_path}")
        self.eval()

        self.logger.info("Step 1/4: Formatting the Raw Training Sequence...")
        formatted_data = []
        source_dataset = self.dataset_list[0]
        batch_size = self.config['train']['batch_size']
        dataloader = source_dataset.get_loader(batch_size=batch_size, shuffle=False)
        
        for batch in tqdm(dataloader, desc="Formatting raw data"):
            current_batch_size = batch[self.fuid].size(0)
            for i in range(current_batch_size):
                user_id_i = batch[self.fuid][i].item()
                history_tensor = batch['in_' + self.fiid][i]
                target_tensor = batch[self.fiid][i]
                history_len = batch['seqlen'][i].item()
                
                if history_len <= 0: continue
                domain_id_i = batch.get('domain_id', torch.ones_like(batch[self.fuid]))[i]
                domain_id_i = domain_id_i[0].item() if domain_id_i.dim() > 0 else domain_id_i.item()
                
                label_sequence = [1] * history_len + [0] * (self.max_seq_len - history_len)
                domain_id_sequence = [domain_id_i] * self.max_seq_len
                sample = [user_id_i, history_tensor.tolist(), target_tensor.tolist(), history_len, label_sequence, domain_id_sequence]
                formatted_data.append(sample)
        self.logger.info(f"Finished formatting {len(formatted_data)} raw sequences.")

        self.logger.info("Step 2/4: Generating new augmentation sequence in [Parallel V2]...")
        if 'fixed' not in self._internal_models:
            self.logger.error("The model 'fixed' is not available and cannot generate a sequence.。")
            return

        model = self._internal_models['fixed']
        model.eval()

        all_newly_generated = []
        gen_dataloader = source_dataset.get_loader(batch_size=batch_size, shuffle=False)

        for batch in tqdm(gen_dataloader, desc="Generate batches in parallel"):
            original_seqs_tensor = batch['in_' + self.fiid]
            original_seqs_list = [seq[seq != 0].tolist() for seq in original_seqs_tensor]
            
            B = original_seqs_tensor.size(0)
            effective_B = B * m
            
            generated_sequences = torch.zeros(effective_B, self.max_seq_len, dtype=torch.long, device=self.device)
            first_items = original_seqs_tensor[:, 0].repeat_interleave(m)
            generated_sequences[:, 0] = first_items

            current_lengths = torch.ones(effective_B, dtype=torch.long, device=self.device)
            is_active = torch.ones(effective_B, dtype=torch.bool, device=self.device)
            original_sequences_expanded = [seq for seq in original_seqs_list for _ in range(m)]

            for step in range(1, self.max_seq_len):
                if not torch.any(is_active):
                    break

                active_indices = torch.where(is_active)[0]
                
                model_input_seqs = generated_sequences[active_indices]
                model_input_lens = current_lengths[active_indices]

                fake_batch = {'in_' + self.fiid: model_input_seqs, 'seqlen': model_input_lens}
                
                query_all_steps = model.forward(fake_batch, need_pooling=False)
                prediction_embeddings = query_all_steps[torch.arange(len(active_indices)), model_input_lens - 1]
                all_scores = prediction_embeddings @ model.item_embedding.weight.T

                next_items = torch.zeros(len(active_indices), dtype=torch.long, device=self.device)
                indices_to_deactivate = []

                for i, full_batch_idx in enumerate(active_indices.tolist()):
                    original_seq = original_sequences_expanded[full_batch_idx]
                    generated_seq = generated_sequences[full_batch_idx, :current_lengths[full_batch_idx]].tolist()
                    
                    # --- Quality Verification ---
                    remaining_items = set(original_seq) - set(generated_seq)
                    if not remaining_items:
                        indices_to_deactivate.append(i)
                        continue

                    scores_i = all_scores[i]
                    rem_items_tensor = torch.LongTensor(list(remaining_items)).to(self.device)
                    ranks = torch.sum(scores_i > scores_i[rem_items_tensor].unsqueeze(1), dim=1) + 1
                    if not torch.any(ranks <= k):
                        indices_to_deactivate.append(i)
                        continue

                    # ==================== KEY MODIFICATION HERE ====================
                    # Before decoding, shield all generated items
                    masked_scores = scores_i.clone()
                    already_generated_tensor = torch.LongTensor(generated_seq).to(self.device)
                    masked_scores[already_generated_tensor] = -torch.inf
                    # ===============================================================
                    
                    # --- Decoding ---
                    if random.random() < local_prob:
                        candidate_pool = torch.LongTensor(list(set(original_seq))).to(self.device)
                    else:
                        candidate_pool = torch.arange(1, self.num_items, device=self.device)
                    
                    # Get candidate scores from masked scores
                    candidate_scores = masked_scores[candidate_pool]
                    
                    # Check if there are any optional items in the candidate pool
                    top_score_in_pool, _ = torch.topk(candidate_scores, 1)
                    if top_score_in_pool[0] == -torch.inf:
                        indices_to_deactivate.append(i)
                        continue

                    _, top_k_indices_in_pool = torch.topk(candidate_scores, 1)
                    next_items[i] = candidate_pool[top_k_indices_in_pool[0]].item()

                if indices_to_deactivate:
                    deactivate_mask = torch.zeros(len(active_indices), dtype=torch.bool, device=self.device)
                    deactivate_mask[indices_to_deactivate] = True
                    active_indices_to_update = active_indices[~deactivate_mask]
                    active_indices_to_deactivate = active_indices[deactivate_mask]
                    is_active[active_indices_to_deactivate] = False
                else:
                    active_indices_to_update = active_indices
                
                update_positions = current_lengths[active_indices_to_update]
                valid_next_items = next_items[~deactivate_mask] if indices_to_deactivate else next_items
                generated_sequences[active_indices_to_update, update_positions] = valid_next_items
                
                current_lengths[active_indices_to_update] += 1
            
            for i in range(effective_B):
                seq_len = current_lengths[i].item()
                if seq_len >= min_len:
                    all_newly_generated.append(generated_sequences[i, :seq_len].tolist())

        unique_sequences_set = {tuple(seq) for seq in all_newly_generated}
        unique_sequences = [list(seq) for seq in unique_sequences_set]
        self.logger.info(f"Successfully generated {len(unique_sequences)} new, unique, valid sequences.")

        self.logger.info("Step 3/4: Formatting the newly generated sequence...")
        next_user_id = self.num_users
        for seq in tqdm(unique_sequences, desc="Formatting a new sequence"):
            story_len = len(seq)
            if story_len < 2: continue
            history_len = story_len - 1
            history, target = seq[:-1], seq[1:]
            padded_history = history + [0] * (self.max_seq_len - history_len)
            padded_target = target + [0] * (self.max_seq_len - history_len)
            domain_id_i, label_sequence = 1, [1] * history_len + [0] * (self.max_seq_len - history_len)
            domain_id_sequence = [domain_id_i] * self.max_seq_len
            sample = [next_user_id, padded_history, padded_target, history_len, label_sequence, domain_id_sequence]
            formatted_data.append(sample)
            next_user_id += 1

        self.logger.info(f"Formatting completed. Total data volume: {len(formatted_data)} items.")

        self.logger.info("Step 4/4: Saving final dataset...")
        try:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            torch.save(formatted_data, save_path)
            self.logger.info(f"Successfully saved the merged dataset to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save augmented dataset: {e}")