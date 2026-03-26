import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from main.models.gcse import GCSE, GCSERoberta
from main.models.simcse import SimCSE, SimCSERoberta
from main.loaders.cl_loader import CLDataset
from main.analysis import Analysis
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class Trainer():

    def __init__(self, tokenizer, from_pretrained=None, base_from_pretrained=None, data_name='default', data_present_path=None, train_file=None, eval_file=None, test_file=None, max_seq_len=256, batch_size=16, batch_size_eval=64, eval_label_scale=5.0, hard_negative_weight=0, temp=0.05, alpha=0.17, beta=0.22, eval_mode='dev', task_name='GCSE'):
        self.tokenizer = tokenizer
        self.from_pretrained = from_pretrained
        self.base_from_pretrained = base_from_pretrained
        self.data_name = data_name
        self.data_present_path = data_present_path
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.task_name = task_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.eval_label_scale = eval_label_scale
        self.hard_negative_weight = hard_negative_weight
        self.temp = temp
        self.alpha = alpha
        self.beta = beta
        self.eval_mode = eval_mode

        self.dataloader_init()
        self.model_init()
        self.analysis = Analysis()

    def model_init(self):
        self.config = AutoConfig.from_pretrained(
            self.from_pretrained)
        base_from_pretrained = self.base_from_pretrained if self.base_from_pretrained is not None else self.from_pretrained
        if self.config.model_type == 'bert':
            self.model = GCSE(from_pretrained=self.from_pretrained,
                                pooler_type='cls', hard_negative_weight=self.hard_negative_weight, temp=self.temp, alpha=self.alpha, beta=self.beta)
            self.base_model = SimCSE(from_pretrained=base_from_pretrained,
                                pooler_type='cls', hard_negative_weight=self.hard_negative_weight, temp=self.temp)
        elif self.config.model_type == 'roberta':
            self.model = GCSERoberta(from_pretrained=self.from_pretrained,
                                       pooler_type='cls', hard_negative_weight=self.hard_negative_weight, temp=self.temp, alpha=self.alpha, beta=self.beta)
            self.base_model = SimCSERoberta(from_pretrained=base_from_pretrained,
                                       pooler_type='cls', hard_negative_weight=self.hard_negative_weight, temp=self.temp)

    def dataloader_init(self):
        if self.data_present_path is None:
            self.data_path = {
                'train': self.train_file,
                'dev': self.eval_file,
                'test': self.test_file
            }
        else:
            self.data_path = self.get_data_present(
                self.data_present_path)[self.data_name]
        self.train_set = CLDataset(
            self.tokenizer, self.data_path['train'], max_seq_len=self.max_seq_len, shuffle=True)
        self.eval_set = CLDataset(
            self.tokenizer, self.data_path['dev'], max_seq_len=self.max_seq_len, shuffle=False, is_eval=True)
        if 'test' in self.data_path and self.data_path['test'] is not None:
            self.test_set = CLDataset(
                self.tokenizer, self.data_path['test'], max_seq_len=self.max_seq_len, shuffle=False, is_eval=True)
        self.train_loader = DataLoader(
            self.train_set, self.batch_size)
        self.eval_loader = DataLoader(
            self.eval_set, self.batch_size_eval) if self.eval_mode == 'dev' else DataLoader(self.test_set, self.batch_size_eval)

    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present

    def model_to_device(self, gpu=[0]):
        self.num_gpus = len(gpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)
        self.base_model.cuda()
        self.base_model = torch.nn.DataParallel(self.base_model, device_ids=gpu).cuda()
        self.base_model.to(device)

    def __call__(self, resume_step=None, num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], eval_call_step=None, save_per_call=False):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, gpu=gpu, eval_call_step=eval_call_step, save_per_call=save_per_call)

    def train(self, resume_step=None, num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], eval_call_step=None, save_per_call=False):
        self.model_to_device(gpu=gpu)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        best_eval_score = 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0

            train_iter = tqdm(self.train_loader)
            self.model.train()
            self.base_model.eval()

            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])
                
                cos_sim = self.base_model(**it).logits.detach()
                outputs = self.model(**it, base_cos_sim=cos_sim)
                loss = outputs['loss']
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count)

                if (eval_call_step is None and train_step % 125 == 0) or eval_call_step(train_step):
                    X, Y, score = self.eval(train_step)
                    if score > best_eval_score:
                        best_eval_score = score
                        self.save_model('best')
                    elif save_per_call:
                        self.save_model(train_step, '_step')
                    self.analysis.save_pred_gold(
                        X, Y, uid=current_uid if self.task_name is None else self.task_name, step=train_step)
                    self.analysis.save_all_records(
                        uid=current_uid if self.task_name is None else self.task_name)
                    yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, 'current_best')
                    self.model.train()
            
            model_uid = self.save_model(epoch + 1)

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count
            })

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def save_model(self, current_step=0, prefix=''):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists(f'./save_model/{dir}'):
            os.makedirs(f'./save_model/{dir}')
        model_self = self.model.module if hasattr(
            self.model, 'module') else self.model
        
        model_self.save_pretrained(
            f'./save_model/{dir}/GCSE{prefix}_{current_step}', safe_serialization=False)
        self.tokenizer.save_pretrained(
            f'./save_model/{dir}/GCSE{prefix}_{current_step}')
        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch, gpu=[0], is_eval=False):
        if is_eval:
            self.model_to_device(gpu=gpu)

        with torch.no_grad():
            eval_count = 0
            eval_loss = 0
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            X = []
            Y = []
            eval_spearman = []

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                outputs = self.model(**it)
                loss = outputs['loss']
                logits = outputs['logits']
                loss = loss.mean()

                if self.num_gpus > 1:
                    p = []
                    per_size = logits.size(1)
                    for i in range(self.num_gpus):
                        p.append(torch.diag(logits[i * per_size : (i + 1) * per_size]) * self.temp)
                    p = torch.cat(p, dim=0)
                else:
                    p = torch.diag(logits) * self.temp
                
                eval_loss += loss.data.item()
                eval_count += 1

                gold = it['labels']
                X += p.tolist()
                Y += gold.tolist()
                tp += ((gold / self.eval_label_scale >= 0.5) & (p >= 0.5)).sum().item()
                fp += ((gold / self.eval_label_scale < 0.5) & (p >= 0.5)).sum().item()
                fn += ((gold / self.eval_label_scale >= 0.5) & (p < 0.5)).sum().item()
                tn += ((gold / self.eval_label_scale < 0.5) & (p < 0.5)).sum().item()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                r, r_mse, pearsonr, spearmanr = self.analysis.evaluationSAS(
                    X, Y)
                eval_spearman.append(spearmanr)

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count, Spearman=np.mean(eval_spearman), eval_acc=(tp + tn) / (tp + tn + fp + fn), precision=precision, recall=recall, f1=f1)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                'eval_acc': (tp + tn) / (tp + tn + fp + fn),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'spearman': np.mean(eval_spearman)
            })

        if is_eval:
            self.analysis.save_pred_gold(
                X, Y, uid='0' if self.task_name is None else self.task_name, step=0)
            print(f'F1: {f1}, Precision: {precision}, Recall: {recall}')

        return X, Y, np.mean(eval_spearman)

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX
