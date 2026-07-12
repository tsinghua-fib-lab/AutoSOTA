import time
import os
import copy
from unittest import result
import torch
from torch import tensor
from torch.nn import parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# from opacus import PrivacyEngine
# from models.losses.sign_loss import SignLoss
from models.alexnet import AlexNet
from experiments.utils import chunks, vec_mul_ten, insta_criterion
from experiments.defense_instahide import defense_insta
import torch.nn as nn
import time
import random


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TesterPrivate(object):
    def __init__(self, model, device, verbose=True):
        self.model = model
        self.device = device
        self.verbose = verbose

    def test_signature(self, kwargs, ind):
        self.model.eval()
        avg_private = 0
        count_private = 0

        with torch.no_grad():
            if kwargs != None:
                if isinstance(self.model, AlexNet):
                    for m in kwargs:
                        if kwargs[m]['flag'] == True:
                            b = kwargs[m]['b']
                            M = kwargs[m]['M']

                            M = M.to(self.device)
                            if ind == 0 or ind == 1:
                                signbit = self.model.features[int(m)].scale.view([1, -1]).mm(M).sign().to(self.device)
                                # signbit = self.model.features[int(m)].scale.view([1, -1]).sign().mm(M).sign().to(self.device)
                            if ind == 2 or ind == 3:
                                w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)
                                signbit = w.view([1, -1]).mm(M).sign().to(self.device)
                            # print(signbit)

                            privatebit = b
                            privatebit = privatebit.sign().to(self.device)

                            # print(privatebit)

                            detection = (signbit == privatebit).float().mean().item()
                            avg_private += detection
                            count_private += 1

                else:
                    for sublayer in kwargs["layer4"]:
                        for module in kwargs["layer4"][sublayer]:
                            if kwargs["layer4"][sublayer][module]['flag'] == True:
                                b = kwargs["layer4"][sublayer][module]['b']
                                M = kwargs["layer4"][sublayer][module]['M']
                                M = M.to(self.device)
                                privatebit = b
                                privatebit = privatebit.sign().to(self.device)

                                if module == 'convbnrelu_1':
                                    scale = self.model.layer4[int(sublayer)].convbnrelu_1.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbnrelu_1.conv.weight,
                                                        dim=0)
                                if module == 'convbn_2':
                                    scale = self.model.layer4[int(sublayer)].convbn_2.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbn_2.conv.weight, dim=0)

                                if ind == 0 or ind == 1:
                                    signbit = scale.view([1, -1]).mm(M).sign().to(self.device)
                                    # signbit = scale.view([1, -1]).sign().mm(M).sign().to(self.device)

                                if ind == 2 or ind == 3:
                                    signbit = conv_w.view([1, -1]).mm(M).sign().to(self.device)
                                # print(signbit)
                                # print(privatebit)
                                detection = (signbit == privatebit).float().mean().item()
                                avg_private += detection
                                count_private += 1

        if kwargs == None:
            avg_private = None
        if count_private != 0:
            avg_private /= count_private

        return avg_private


class TrainerPrivate(object):
    def __init__(self, model, train_set, device, dp, sigma, num_classes, balancedmethod, defense=None, klam=3,
                 up_bound=0.65, mix_alpha=0.01):
        self.model = model
        self.device = device
        self.tester = TesterPrivate(model, device)
        self.dp = dp
        self.sigma = sigma
        self.defense = defense
        self.klam = klam
        self.up_bound = up_bound
        self.mix_alpha = mix_alpha
        self.balancedmethod = balancedmethod

        self.num_classes = num_classes
        self.train_loader = train_set
        self.batch_size = 100

    def mixup_data(self, x, y, alpha):

        use_cuda = True
        # print('alpha:',alpha)

        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        # print('lam:',lam)
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        # print('index:',index)
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _local_update_noback(self, dataloader, local_ep, lr, optim_choice, sampling_proportion):

        if optim_choice == "sgd":

            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr,
                                       momentum=0.9,
                                       weight_decay=0.0005)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(),
                                         lr,
                                         weight_decay=0.0005)

        epoch_loss = []
        cos_scores = []
        train_ldr = dataloader

        for epoch in range(local_ep):

            loss_meter = 0
            loss_meter_a = 0
            loss_meter_v = 0
            acc_meter = 0
            sample_grads = []
            total = 0
            correct = 0
            iteration = 0
            # for batch_idx, (x, y) in enumerate(train_ldr):
            for step, (index, spec, image, label) in enumerate(train_ldr):
                sample_batch_grads = []
                # print("batch_idx:{}\n x:{} \n y:{}\n".format(batch_idx,x,y))
                spec = spec.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)


                self.optimizer.zero_grad()
                a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                weight_size = self.model.fusion_module.fc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(
                    self.model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + self.model.fusion_module.fc_out.bias / 2)

                out_a = (torch.mm(a, torch.transpose(
                    self.model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + self.model.fusion_module.fc_out.bias / 2)

                loss = F.cross_entropy(out, label)
                loss_v = F.cross_entropy(out_v, label)
                loss_a = F.cross_entropy(out_a, label)
                if self.balancedmethod == 'sanloss':
                    loss_v.backward(retain_graph=True)  # 第一步：视觉单模态损失反向
                    loss_a.backward(retain_graph=True)  # 第二步：音频单模态损失反向
                loss.backward()  # 第三步：融合损失反向（释放计算图）
                if self.balancedmethod == 'OGM':
                    softmax = nn.Softmax(dim=1)
                    relu = nn.ReLU(inplace=True)
                    tanh = nn.Tanh()

                    score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
                    score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
                    ratio_v = score_v / score_a
                    ratio_a = 1 / ratio_v
                    if ratio_v > 1:
                        coeff_v = 1 - tanh(relu(ratio_v))
                        coeff_a = 1
                    else:
                        coeff_a = 1 - tanh(relu(ratio_a))
                        coeff_v = 1
                    for name, parms in self.model.named_parameters():
                        layer = str(name).split('.')[0]
                        if parms.grad is None:
                            continue
                        if 'audio' in layer and len(parms.grad.size()) == 4:
                            parms.grad *= coeff_a
                        if 'visual' in layer and len(parms.grad.size()) == 4:
                            parms.grad *= coeff_v
                self.optimizer.step()
                loss_meter += loss.item()
                loss_meter_a += loss_a.item()
                loss_meter_v += loss_v.item()

            # sampling num = batch_size * sampling_iteration = 100 * 25 = 2500
            iteration += 1
            # print("iteration:",iteration)
            if iteration == int(sampling_proportion * len(train_ldr)):
                break

            loss_meter /= len(train_ldr)
            loss_meter_a /= len(train_ldr)
            loss_meter_v /= len(train_ldr)
            # swanlab.log({
            #     "loss_meter": loss_meter,
            #     "loss_meter_a": loss_meter_a,
            #     "loss_meter_v": loss_meter_v,
            #
            # })
            acc_meter /= len(dataloader)
            epoch_loss.append(loss_meter)


        if self.dp:
            print('DP setting ......')
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss)

    def test(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        loss_meter = 0
        loss_meter_a = 0  # 音频模态loss
        loss_meter_v = 0  # 视觉模态loss
        acc_meter = 0
        acc_meter_a = 0  # 音频模态准确率
        acc_meter_v = 0  # 视觉模态准确率
        runcount = 0
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for step, (index, spec, image, label) in enumerate(dataloader):
                spec = spec.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)
                a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                weight_size = self.model.fusion_module.fc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(
                    self.model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + self.model.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(
                    self.model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + self.model.fusion_module.fc_out.bias / 2)

                # 融合模型的loss和acc
                loss_meter += F.cross_entropy(out, label, reduction='sum').item()
                pred = softmax(out)
                pred = pred.max(1, keepdim=True)[1]
                acc_meter += pred.eq(label.view_as(pred)).sum().item()

                # 音频模态的loss和acc
                loss_meter_a += F.cross_entropy(out_a, label, reduction='sum').item()
                pred_a = softmax(out_a)
                pred_a = pred_a.max(1, keepdim=True)[1]
                acc_meter_a += pred_a.eq(label.view_as(pred_a)).sum().item()

                # 视觉模态的loss和acc
                loss_meter_v += F.cross_entropy(out_v, label, reduction='sum').item()
                pred_v = softmax(out_v)
                pred_v = pred_v.max(1, keepdim=True)[1]
                acc_meter_v += pred_v.eq(label.view_as(pred_v)).sum().item()

                runcount += image.size(0)

        loss_meter /= runcount
        loss_meter_a /= runcount
        loss_meter_v /= runcount
        acc_meter /= runcount
        acc_meter_a /= runcount
        acc_meter_v /= runcount

        return loss_meter, acc_meter, loss_meter_a, acc_meter_a, loss_meter_v, acc_meter_v




