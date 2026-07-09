import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.toolkit import tensor2numpy, accuracy_domain

from model.net import PromptNet

class HybridEnergyDistancePromptTrainer(object):

    def __init__(self, args):
        
        self._network = PromptNet(args)
        
        self.args = args
        self._cur_task = -1 # 当前域
        self._known_classes = 0 # 已发现的类（训练了）
        self._total_classes = 0 # 总类数量，和_known_classes一般相差一个increment
        self.query_type = args["query_type"]#距离计算方式
        self._device = args['device'][0]#gpu设备
        self._multiple_gpus = args['device']
        
        self.batch_size=args["batch_size"]# 数据集批次大小
        
        self.init_epoch = args["init_epoch"]#第一轮训练的轮数
        self.init_lr = args["init_lr"]#第一轮训练的学习率
        self.init_weight_decay = args["init_weight_decay"]#第一轮训练的权重衰减（正则化 防止过拟合）
        self.epochs = args["epochs"] # 训练总次数
        self.lrate = args["lrate"] # 学习率
        
        self.weight_decay = args["weight_decay"] # 权重衰减
        self.num_workers = args["num_workers"] # 线程数量
        self.knn_K=args['knn_K']#各个域Knn数量
        
        self.topk = 1  # 预测排序在前topk
        self.class_num = self._network.class_num # 每个域的类数量=increment
        self.all_keys = []#各个域聚类点的特征集合
        
        self.energy_T=args["energy_T"]#温度
        self.energy_tau=args["energy_tau"]#能量权重缩放比例
        self.distance_tau=args["distance_tau"]#距离权重缩放比例
        self.reg_loss_lamda=args["reg_loss_lamda"]#正则损失比例
        self.energy_midline=args["energy_midline"]#能量中轴线
        self.energy_border=args["energy_border"]#能量边界线
        self.dataset=args["dataset"]#数据集类型
        

    def after_task(self):
        self._known_classes = self._total_classes#更新当前已训练的类数量
        self._network.update_fc()
        
    def begin_incremental(self, data_manager):
        
        self._cur_task += 1 
        # 不同域的相同类也先按不同类``
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
    
    def incremental_train(self, data_manager):
        self.data_manager=data_manager
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # 测试数据集
        if self.dataset == "core50":
            
            test_dataset = data_manager.get_dataset(np.arange(0, self.class_num), source='test', mode='test')
        else:
            test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
       
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus) 
        self.forzen_params()
        
        self.vit_clustering(self.train_loader,self._cur_task)
        self._train(self.train_loader, self.test_loader)
         
        # 训练后的模型参数更新到最新，便于下一轮
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def forzen_params(self):
        self._network.to(self._device)
        
        paramGradTrue=["textual_prompt","visual_prompt"]
       
        for name, param in self._network.named_parameters():
            param.requires_grad_(False) 
            for item in paramGradTrue:
                if item in name:
                    param.requires_grad_(True)
                
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
    
    def getDistance(self, batch_vectors,mean_vector,dim):
        if self.query_type == "l1":
                #L1范式距离：d(x, y) = |x1 - y1| + |x2 - y2| + ... + |xn - yn|
            distances= torch.sum(torch.abs(batch_vectors-mean_vector.to(batch_vectors.device)),dim=dim) #输出为【batchSize】  
        elif self.query_type == "l2":
            #L2范式距离：d(x, y) = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2)
            distances= torch.sqrt(torch.sum((batch_vectors-mean_vector.to(batch_vectors.device))**2, dim=dim))
        return distances 
    
    def vit_clustering(self, dataloader,cur_task):
        features = []
        this_inputs=[]
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)

            features.append(feature)
            this_inputs.extend(inputs)
        features = torch.cat(features, 0).cpu().detach().numpy()
        knn_K=self.knn_K[cur_task]
        clustering = KMeans(n_clusters=knn_K, random_state=0).fit(features)
        centers=clustering.cluster_centers_
        self.all_keys.append(torch.tensor(centers).to(feature.device))
    
    def _train(self, train_loader, test_loader):
       
        if self._cur_task==0: # 第一轮训练
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.lrate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)


    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch)) 
        train_loader.dataset.newDomainsLogits(self._cur_task)
        train_loader.dataset.newDomainsWeights(self._cur_task)
        test_loader.dataset.newDomainsLogits(self._cur_task)
        test_loader.dataset.newDomainsWeights(self._cur_task)
        
        for _, epoch in enumerate(prog_bar):
            self._network.eval() 
            losses = 0.0
            correct, total = 0, 0
            
            for i, (idx, inputs, targets) in enumerate(train_loader): 
                
                idx,inputs, targets = idx.to(self._device),inputs.to(self._device), targets.to(self._device)
                
                mask = (targets >= self._known_classes).nonzero().view(-1)
                
                inputs = torch.index_select(inputs, 0, mask) 
                idx = torch.index_select(idx, 0, mask) 
                
                # 这里减去是因为真实标签是0-50
                targets = torch.index_select(targets, 0, mask) - self._known_classes
               
                logits,loss=self.getLogitLoss(train_loader,idx,inputs,targets,epoch,isTest=0)
                optimizer.zero_grad()
                loss.backward() 
                
                optimizer.step()
                losses += loss.item() # 累计损失
                # 计算准确率
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            scheduler.step() 
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
           
            test_acc = self._compute_accuracy_domain(self._network, test_loader,epoch)
            # 进度条内容显示格式
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        
        logging.info(info)
        
           
    def getDistanceWeight(self,inputs):
        with torch.no_grad():
            if isinstance(self._network, nn.DataParallel):
                feature = self._network.module.extract_vector(inputs)
            else:
                feature = self._network.extract_vector(inputs)
                
            domain_min_distances = []
            for task_centers in self.all_keys:
                tmpcentersbatch = []
                    
                for center in task_centers: 
                    tmpcentersbatch.append(self.getDistance(feature, center, dim=1))
                    
                domain_min_distances.append(torch.vstack(tmpcentersbatch).min(0)[0])
            
            domain_min_distances=torch.cat(domain_min_distances).view(len(domain_min_distances), -1)#2,128
            weight=torch.exp((torch.min(domain_min_distances,dim=0)[0]-domain_min_distances)/self.distance_tau)
        return weight.T
    
    
    def getLogitLoss(self,loader,idx,inputs,targets,epoch,isTest=0):
        
        
        if self._cur_task!=0:
            if epoch==0:
                
                old_energy_weights=[]
                old_logits=[]
                
                distance_Weights=self.getDistanceWeight(inputs)
                loader.dataset.setCurWeight(idx,distance_Weights)
                for taskId in range(self._cur_task):
                    with torch.no_grad():
                        oldLogits=self._network.getOldLogits(inputs,taskId)
                        loader.dataset.setDomainsLogits(taskId,idx,oldLogits)
                        old_logits.append(oldLogits)
                        old_logsumexp = torch.logsumexp(oldLogits / self.energy_T, dim=1, keepdim=False)#
                        old_free_energy = -1.0 * self.energy_T * old_logsumexp
                    
                        old_energy_weights.append(old_free_energy)
                        loader.dataset.setDomainsWeights(taskId,idx,old_free_energy)
                
            else:
                old_energy_weights=[]
                old_logits=[]
                
                distance_Weights=loader.dataset.getCurWeight(idx)
                for taskId in range(self._cur_task):
                    old_logits.append(loader.dataset.getDomainsLogits(taskId,idx))
                    old_energy_weights.append(loader.dataset.getDomainsWeights(taskId,idx).to(self._device))
            if isTest!=0:#测试
                with torch.no_grad():
                    logits,_=self.calLogits(inputs,targets,old_energy_weights,distance_Weights,old_logits)
                    return logits,None
                
            else:
    
                logits=self._network(inputs)
                output_logsumexp = torch.logsumexp(logits / self.energy_T, dim=1, keepdim=False)#
                cur_free_energy = -1.0 * self.energy_T * output_logsumexp
                
        else:#第一域，不用计算权重
            
            if isTest!=0:#测试
                with torch.no_grad():
                    logits=self._network(inputs)
                    return logits,None
            else:
                logits=self._network(inputs)
                output_logsumexp = torch.logsumexp(logits / self.energy_T, dim=1, keepdim=False)#
                cur_free_energy = -1.0 * self.energy_T * output_logsumexp
        classify_loss = F.cross_entropy(logits, targets)
        
        mid_align_loss,bord_align_loss=0,0
        bord_energy_diff=cur_free_energy - self.energy_border
        bord_energy_diff_max0=bord_energy_diff[bord_energy_diff>0] 
        
        if bord_energy_diff_max0.numel() != 0:
            bord_align_loss =self.reg_loss_lamda* (bord_energy_diff_max0).mean()
        mid_align_loss=self.reg_loss_lamda* (cur_free_energy.mean() - self.energy_midline).abs()
        
        loss=classify_loss+mid_align_loss+bord_align_loss
        return logits,loss
        
    def calLogits(self,inputs,targets,old_energy_weights,distance_Weights,old_logits):
        curLogits = self._network(inputs)
        output_logsumexp = torch.logsumexp(curLogits / self.energy_T, dim=1, keepdim=False)#
        cur_free_energy = -1.0 * self.energy_T * output_logsumexp
        
        old_energy_weights.append(cur_free_energy)
        energy_Weights=(torch.stack(old_energy_weights)).T
        min_score=(torch.min(energy_Weights,dim=1)[0]).unsqueeze(1)
        energy_Weights=torch.exp((min_score-energy_Weights)/self.energy_tau)
        
       
        weights=torch.mul(energy_Weights, distance_Weights)#e相乘，代表指数相加
        
        weights_sum=torch.sum(weights,dim=1).unsqueeze(1)
        domain_weights=torch.div(weights,weights_sum)
        logits=0
        for taskId in range(self._cur_task):
            old_weight=domain_weights[:,taskId].unsqueeze(1)
            oldLogits=old_logits[taskId]
            logits+=oldLogits*old_weight
        cur_weights=domain_weights[:,self._cur_task].unsqueeze(1)
        logits+=curLogits*cur_weights
        return logits,cur_free_energy
      
    
    def eval_task(self):
        if self.dataset == "core50":
            test_dataset = self.data_manager.get_dataset(np.arange(0, self.class_num), source='test', mode='test')
        else:
            test_dataset = self.data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader.dataset.newDomainsLogits(self._cur_task)
        test_loader.dataset.newDomainsWeights(self._cur_task)
        y_pred, y_true = self._eval_cnn(test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        
        return cnn_accy
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain(y_pred.T[0], y_true, self._known_classes, class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
       
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (idx, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            logits,_=self.getLogitLoss(loader,idx,inputs,targets,epoch=0,isTest=2)
                    
            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader,epoch):
        
        model.eval()
        correct, total = 0, 0
        for i, (idx, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            logits,_=self.getLogitLoss(loader,idx,inputs,targets,epoch,isTest=1)
                   
            predicts = torch.max(logits, dim=1)[1]
            #不同域相同类算预测成功，这里默认选好了域
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)
        if not isinstance(correct, int):
            correct=tensor2numpy(correct)
        return np.around(correct * 100 / total, decimals=2)
