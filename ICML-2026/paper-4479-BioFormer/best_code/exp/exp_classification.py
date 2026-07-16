from copy import deepcopy

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import os
from tqdm import tqdm
import warnings
from thop import profile
from utils.losses import mmd_loss, coral_loss, FocalLoss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len           # redefine seq_len
        self.args.enc_in = test_data.X.shape[2]             # redefine enc_in
        self.args.num_class = len(np.unique(test_data.y))   # define num_class
        if self.args.is_cross_subject:
            self.args.num_subjs = int(train_data.id.max()) + 1
            print("current subject num = ", self.args.num_subjs)
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # Focal Loss for imbalanced AD classification
        if self.args.use_focal:
            return FocalLoss(alpha=self.args.focal_alpha, gamma=self.args.focal_gamma)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss, total_ce, total_kl = [], [], []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask, subject in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                subject = subject.to(self.device)
                # label.shape (batch_size,)
                # TTA: multiple augmented forward passes
                n_views = self.args.tta_views if getattr(self.args, "use_tta", False) and getattr(self, "_tta_eval", False) else 1
                if n_views > 1:
                    # Enable augmentations for TTA by setting augmentation modules to train mode
                    for aug_mod in self.model.enc_embedding.augmentation:
                        aug_mod.train()
                
                all_outputs = []
                for _v in range(n_views):
                    if self.swa:
                        if self.args.method in ("FBD","SubjNorm","RAW","MixStyle","DSU"):
                            out = self.swa_model(batch_x, padding_mask, None, None, None)
                        elif self.args.method in ("GRL", "MMD","CORAL"):
                            out, * _ = self.swa_model(batch_x, None, None, None, None)
                        elif self.args.method == "Soft":
                            out, * _ = self.swa_model(batch_x)
                    else:
                        if self.args.method in ("FBD","SubjNorm","RAW","MixStyle","DSU"):
                            out = self.model(batch_x, padding_mask, None, None, None)
                        elif self.args.method in ("GRL", "MMD","CORAL"):
                            out, * _ = self.model(batch_x, None, None, None, None)
                        elif self.args.method == "Soft":
                            out, * _ = self.model(batch_x)
                    all_outputs.append(out)
                
                if n_views > 1:
                    outputs = torch.stack(all_outputs, dim=0).mean(dim=0)
                    # Restore eval mode for augmentation modules
                    for aug_mod in self.model.enc_embedding.augmentation:
                        aug_mod.eval()
                else:
                    outputs = all_outputs[0]

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)
                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        # preds.shape (total_samples, num_classes)
        trues = torch.cat(trues, 0)
        # trues.shape (total_samples,)
        probs = torch.nn.functional.softmax(
            preds
        )  
        # probs.shape (total_samples, num_classes)  convert logits to probabilities   
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # trues_onehot.shape (total_samples, num_classes) one-hot encoded true labels
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  
        # predictions.shape (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict
        # return avg_loss, avg_ce, avg_kl, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")

        # setting record
        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
    
        for epoch in range(self.args.train_epochs):

            total_loss = []
            self.model.train()

            pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{self.args.train_epochs}')
            for batch_x, label, padding_mask,subject in pbar: 
              
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)                    # (batch_size, seq_len, feature_dim)
                padding_mask = padding_mask.float().to(self.device)          # (batch_size, seq_len)
                label = label.to(self.device)                                # (batch_size,)
                subject = subject.to(self.device)                            # (batch_size,)
                
                if self.args.method in ("FBD","SubjNorm","RAW","MixStyle","DSU"):
                    outputs = self.model(batch_x, padding_mask, None, None, None)     # (batch_size, num_classes)
                elif self.args.method == "Soft":
                     outputs, moe_loss = self.model(batch_x, num_epoch_i=epoch, warm_up_epoch=5)
                elif self.args.method == "GRL":
                    p = float(epoch) / (50)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1 
                    outputs, subject_pred = self.model(batch_x, alpha, None, None, None)  # 
                elif self.args.method in ("MMD","CORAL"):
                    outputs, features = self.model(batch_x, padding_mask, None, None, None)
                    # subject: [B]
                    unique_subjects = torch.unique(subject)

                    if len(unique_subjects) < 2:
                        delat = torch.tensor(0.0, device=self.device)
                    else:
                        s1, s2 = unique_subjects[:2]

                        idx_s = subject == s1
                        idx_t = subject == s2

                        feat_s = features[idx_s]  # x_s
                        feat_t = features[idx_t]  # x_t 
                        delat = 1
    
                loss_cls = criterion(outputs, label.long())

                if self.args.method in ("FBD","SubjNorm","RAW","MixStyle","DSU"):
                    loss = loss_cls
                elif self.args.method == "GRL":
                    loss_subject = criterion(subject_pred, subject.long())
                    loss = loss_cls + loss_subject * 0.1 
                elif self.args.method == "MMD":
                    loss = loss_cls + 0.05 * mmd_loss(feat_s, feat_t) * delat  
                elif self.args.method == "CORAL":
                    loss = loss_cls + 0.05 * coral_loss(features, subject)* delat  
                elif self.args.method == "Soft":
                    loss = loss_cls + 0.01 * moe_loss

                total_loss.append(loss.item())   
               
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            self.swa_model.update_parameters(self.model)
            
            vali_loss,val_metrics_dict = self.vali(vali_data, vali_loader, criterion)            

            print(
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            )
            
            # trainning result save
            folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            )     
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


            file_name = "result_for_train.txt"
            f = open(os.path.join(folder_path, file_name), "a")
            #  Add-on Mode
            f.write(setting + "  \n")
            f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            )
            f.write("\n")
            f.write("\n")
            f.close()

            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        self._tta_eval = True
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)
        self._tta_eval = False
        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_for_test.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        #  Add-on Mode
        f.write(setting + "  \n")
        f.write(
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return
