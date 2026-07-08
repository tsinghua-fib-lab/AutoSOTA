import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from collections import OrderedDict


EPSILON = 1e-8
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        label_distribution = [(train_dataset.labels == x).sum() for x in range(self._total_classes)]
        #label_distribution = [train_dataset.labels.count(xx) for xx in OrderedDict.fromkeys(train_dataset.labels)]
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, label_distribution)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, label_distribution):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epoch"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler, label_distribution)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["epochs"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler, label_distribution)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, label_distribution):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        gradall = np.zeros(self._total_classes, dtype=float)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = regularized_cn(logits, targets, label_distribution)
                optimizer.zero_grad()
                loss.backward()
                # gradient reweighting
                grad_norm = []
                # accumulated gradients
                grad_norm.extend(torch.norm(self._network.fc.weight.grad, dim=1).data.cpu().numpy())

                gradall += np.array(grad_norm)
                # class-balance ratio
                grad_weight = np.min(gradall) / np.array(gradall)

                for i in range(self._total_classes):
                    self._network.fc.weight.grad[i, :] = grad_weight[i] * self._network.fc.weight.grad[i, :]
                    self._network.fc.bias.grad[i] = grad_weight[i] * self._network.fc.bias.grad[i]

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler, label_distribution):
        prog_bar = tqdm(range(self.args["epochs"]))
        gradall = np.zeros(self._total_classes, dtype = float)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss_clf = regularized_cn(logits, targets, label_distribution)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    self.args["T"],
                )
                loss = loss_clf + self.args["lambda"]*(self._known_classes/(self._total_classes - self._known_classes)) * loss_kd
                # # Compute gradients for the entire weight tensor
                grad_kd = []
                bias_kd = []
                grad_norm_kd = []
                temp_norm = torch.autograd.grad(loss_kd, self._network.fc.weight, retain_graph=True)[0]
                grad_norm_kd.extend(torch.norm(temp_norm[:self._known_classes, :], dim=1).data.cpu().numpy())
                grad_kd.append(temp_norm[:self._known_classes, :])
                bias_kd.append(torch.autograd.grad(loss_kd, self._network.fc.bias, retain_graph=True)[0][:self._known_classes])
                grad_cn = []
                bias_cn = []
                grad_norm_cn = []
                temp_norm = torch.autograd.grad(loss_clf, self._network.fc.weight, retain_graph=True)[0]
                grad_norm_cn.extend(torch.norm(temp_norm, dim=1).data.cpu().numpy())
                grad_cn.append(temp_norm)
                bias_cn.append(torch.autograd.grad(loss_clf, self._network.fc.bias, retain_graph=True)[0])
                gradall += np.array(grad_norm_cn)
                grad_weight = []
                task_grad = []
                old_temp = np.array(gradall[:self._known_classes])
                new_temp = np.array(gradall[self._known_classes:])
                old_temp_weight = np.min(old_temp) / old_temp
                new_temp_weight = np.min(new_temp) / new_temp
                grad_weight.extend(old_temp_weight.tolist())
                grad_weight.extend(new_temp_weight.tolist())
                grad_weight = np.array(grad_weight)
                mean_old = np.mean(old_temp)
                mean_new = np.mean(new_temp)
                rs_old = min(1., (mean_new / mean_old))
                rs_new = min(1., (mean_new / mean_old) * np.exp(-1*self.args["gammar_r"]*(self._known_classes/self._total_classes)))
                grad_weight[:self._known_classes] *= rs_old
                grad_weight[self._known_classes:] *= rs_new
                grad_both = []
                for k in range(self._total_classes):
                    grad_both.append(grad_weight[k] * grad_norm_cn[k])
                ratio_kd = np.linalg.norm(grad_both) / np.linalg.norm(grad_norm_kd)
                # start back prop
                optimizer.zero_grad()
                loss.backward()
                
                for i in range(self._known_classes):
                    self._network.fc.weight.grad[i, :] = grad_weight[i] * grad_cn[0][i, :] + ratio_kd * grad_kd[0][i, :]
                    self._network.fc.bias.grad[i] = grad_weight[i] * bias_cn[0][i] + ratio_kd * bias_kd[0][i]
                for i in range(self._known_classes, self._total_classes):
                    self._network.fc.weight.grad[i, :] = grad_weight[i] * grad_cn[0][i, :]
                    self._network.fc.bias.grad[i] = grad_weight[i] * bias_cn[0][i]

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]



def regularized_cn(outputs, targets, label_distribution):
    prior_regularization = label_distribution / np.sum(label_distribution)
    prior_regularization = torch.FloatTensor(prior_regularization).cuda()
    prior_regularization = torch.log(prior_regularization)
    loss = F.cross_entropy(outputs+prior_regularization, targets)
    return loss



    
