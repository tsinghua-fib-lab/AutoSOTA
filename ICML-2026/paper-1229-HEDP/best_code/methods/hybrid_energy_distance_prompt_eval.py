import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.toolkit import tensor2numpy, accuracy_domain_total


class HybridEnergyDistancePromptEval(object):

    def __init__(self, args):
        self._device = args['device'][0]#gpu设备
        model = torch.load(args['model_path'],map_location=self._device)
        self._network=model._network.to(self._device)
        self.all_keys=model.all_keys
        self.args = args
        self._cur_task = -1 # 当前域
        self._known_classes = 0 # 已发现的类（训练了）
        self._total_classes = 0 # 总类数量，和_known_classes一般相差一个increment
        self.query_type = args["query_type"]#距离计算方式
        
        self._multiple_gpus = args['device']
        
        self.batch_size=args["batch_size"]# 数据集批次大小
        
        self.num_workers = args["num_workers"] # 线程数量
        
        self.topk = 1  # 预测排序在前topk
        self.class_num = self._network.class_num # 每个域的类数量=increment
        
        self.energy_T=args["energy_T"]#温度
        self.energy_tau=args["energy_tau"]#能量权重缩放比例
        self.distance_tau=args["distance_tau"]#距离权重缩放比例
        self.dataset=args["dataset"]#数据集类型
        self.test_task_name=args['task_name']
        self.test_task_num=len(self.test_task_name)
        self.train_task_num=args['train_task_num']
        

    def eval_task_last(self,data_manager):
        self.data_manager=data_manager
         
        self._cur_task=self.train_task_num-1
        self._network.to(self._device)
        self._network.numtask=self._cur_task
        if self.dataset == "core50":
            test_dataset = self.data_manager.get_dataset(np.arange(0, self.class_num), source='test', mode='test')
        else:
            self._total_classes=self.test_task_num*self.class_num
            test_dataset = self.data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
        y_pred, y_true = self._eval_cnn(test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info('CNN:{}'.format(cnn_accy['grouped']))



    def getDistance(self, batch_vectors,mean_vector,dim):
        if self.query_type == "l1":
                #L1范式距离：d(x, y) = |x1 - y1| + |x2 - y2| + ... + |xn - yn|
            distances= torch.sum(torch.abs(batch_vectors-mean_vector.to(batch_vectors.device)),dim=dim) #输出为【batchSize】  
        elif self.query_type == "l2":
            #L2范式距离：d(x, y) = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2)
            distances= torch.sqrt(torch.sum((batch_vectors-mean_vector.to(batch_vectors.device))**2, dim=dim))
        return distances 


    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain_total(y_pred.T[0], y_true, class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
       
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (idx, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            logits,_=self.getLogitLoss(loader,idx,inputs,targets)
                    
            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def getLogitLoss(self,loader,idx,inputs,targets):
       
              
        old_energy_weights=[]
        old_logits=[]

        distance_Weights=self.getDistanceWeight(inputs)[:,:self._cur_task+1]#128,2
        
        for taskId in range(self._cur_task+1):
            with torch.no_grad():
                oldLogits=self._network.getOldLogits(inputs,taskId)
                old_logits.append(oldLogits)
                old_logsumexp = torch.logsumexp(oldLogits / self.energy_T, dim=1, keepdim=False)#
                old_free_energy = -1.0 * self.energy_T * old_logsumexp
                old_energy_weights.append(old_free_energy)
                
        energy_Weights=(torch.stack(old_energy_weights)).T #128,2
        min_score=(torch.min(energy_Weights,dim=1)[0]).unsqueeze(1)
        energy_Weights=min_score-energy_Weights#128,2
        
        with torch.no_grad():
            logits=self.calLogits(inputs,targets,energy_Weights,distance_Weights,old_logits)
            return logits,None
                
    def getDistanceWeight(self,inputs):
        with torch.no_grad():
            if isinstance(self._network, nn.DataParallel):
                feature = self._network.module.extract_vector(inputs)
            else:
                feature = self._network.extract_vector(inputs)
                
            domain_min_distances = []
            allkeys=self.all_keys[:(self._cur_task+1)]
            for task_centers in allkeys:
                tmpcentersbatch = []
                for center in task_centers: 
                    tmpcentersbatch.append(self.getDistance(feature, center, dim=1))
                    
                domain_min_distances.append(torch.vstack(tmpcentersbatch).min(0)[0])
            
            domain_min_distances=torch.cat(domain_min_distances).view(len(domain_min_distances), -1)#2,128
            weight=torch.min(domain_min_distances,dim=0)[0]-domain_min_distances
            
        return weight.T

    def calLogits(self,inputs,targets,energy_Weights,distance_Weights,old_logits):

        energy_Weights=torch.exp(energy_Weights/self.energy_tau)#128,2
        distance_Weights=torch.exp(distance_Weights/self.distance_tau)#128,2
        
        weights=torch.mul(energy_Weights, distance_Weights)
        
        weights_sum=torch.sum(weights,dim=1).unsqueeze(1)
        domain_weights=torch.div(weights,weights_sum)
        logits=0
        for taskId in range(self._cur_task+1):
            old_weight=domain_weights[:,taskId].unsqueeze(1)
            oldLogits=old_logits[taskId]
            logits+=oldLogits*old_weight
        return logits
     
