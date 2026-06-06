import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from main.models.sbert import SBert
from main.models.bert import Bert
from main.loaders.sts_loader import STSDataset
from main.analysis import Analysis
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class Trainer():

    def __init__(self, tokenizer, from_pretrained=None, model_type='bert', data_name='default', data_present_path=None, data_type='interactive', train_file=None, eval_file=None, test_file=None, max_seq_len=256, batch_size=16, batch_size_eval=64, hard_negative_weight=0, temp=0.05, eval_mode='dev', task_name='STS'):
        self.tokenizer = tokenizer
        self.from_pretrained = from_pretrained
        self.model_type = model_type
        self.data_name = data_name
        self.data_present_path = data_present_path
        self.data_type = data_type
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.task_name = task_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.hard_negative_weight = hard_negative_weight
        self.temp = temp
        self.eval_mode = eval_mode

        self.dataloader_init()
        self.model_init()
        self.analysis = Analysis()

    def model_init(self):
        self.config = AutoConfig.from_pretrained(
            self.from_pretrained)
        if self.model_type == 'sbert':
            self.model = SBert(from_pretrained=self.from_pretrained, max_seq_len=self.max_seq_len)
        elif self.model_type == 'bert':
            self.model = Bert(from_pretrained=self.from_pretrained)

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
        self.train_set = STSDataset(
            self.tokenizer, self.data_path['train'], max_seq_len=self.max_seq_len, data_type=self.data_type, shuffle=True)
        self.eval_set = STSDataset(
            self.tokenizer, self.data_path['dev'], max_seq_len=self.max_seq_len, data_type=self.data_type, shuffle=False)
        if 'test' in self.data_path and self.data_path['test'] is not None:
            self.test_set = STSDataset(
                self.tokenizer, self.data_path['test'], max_seq_len=self.max_seq_len, data_type=self.data_type, shuffle=False)
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)

    def __call__(self, resume_step=None, num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], eval_call_step=None):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, gpu=gpu, eval_call_step=eval_call_step)

    def train(self, resume_step=None, num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], eval_call_step=None):
        self.model_to_device(gpu=gpu)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        best_eval_score = 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0
            precision = []
            recall = []
            f1 = []
            X = []
            Y = []
            train_spearman = []

            train_iter = tqdm(self.train_loader)
            self.model.train()

            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                outputs = self.model(**it)
                loss = outputs['loss']
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                logits = outputs['logits']
                gold = it['labels']
                p = self.analysis.computed_batch_label(logits)
                X += p.tolist()
                Y += gold.tolist()
                eval_scores = self.analysis.computed_batch_f1(p, gold)
                precision.append(eval_scores['precision'])
                recall.append(eval_scores['recall'])
                f1.append(eval_scores['f1'])
                
                r, r_mse, pearsonr, spearmanr = self.analysis.evaluationSAS(
                    X, Y)
                train_spearman.append(spearmanr)

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count, Spearman=np.mean(spearmanr), precision=np.mean(precision), recall=np.mean(recall), f1=np.mean(f1))

                if (eval_call_step is None and train_step % 125 == 0) or eval_call_step(train_step):
                    X, Y, score = self.eval(train_step)
                    if score > best_eval_score:
                        best_eval_score = score
                        self.save_model('best')
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

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists(f'./save_model/{dir}'):
            os.makedirs(f'./save_model/{dir}')
        model_self = self.model.module if hasattr(
            self.model, 'module') else self.model
        
        model_self.save_pretrained(
            f'./save_model/{dir}/simcse_{current_step}')
        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch, gpu=[0], is_eval=False):
        if is_eval:
            self.model_to_device(gpu=gpu)

        with torch.no_grad():
            eval_count = 0
            eval_loss = 0
            precision = []
            recall = []
            f1 = []
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

                eval_loss += loss.data.item()
                eval_count += 1

                gold = it['labels']
                p = self.analysis.computed_batch_label(logits)
                X += p.tolist()
                Y += gold.tolist()
                eval_scores = self.analysis.computed_batch_f1(p, gold)
                precision.append(eval_scores['precision'])
                recall.append(eval_scores['recall'])
                f1.append(eval_scores['f1'])
                r, r_mse, pearsonr, spearmanr = self.analysis.evaluationSAS(
                    X, Y)
                eval_spearman.append(spearmanr)

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count, Spearman=np.mean(spearmanr), precision=np.mean(precision), recall=np.mean(recall), f1=np.mean(f1))

            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                'precision': np.mean(precision),
                'recall': np.mean(recall),
                'f1': np.mean(f1),
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
