import math
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Metric, R2Score
from typing import Tuple
import itertools
from scipy.special import rel_entr
from logdiff.utils import true_generated_image_grid_save, save_images_in_folder
from collections import defaultdict
from collections import Counter


class WeightedAverage(Metric):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.count = 0
    def update(self, value, count):
        self.total += value*count
        self.count += count
    def compute(self):
        return self.total/self.count
    def reset(self):
        self.total = 0
        self.count = 0

class CS(Metric):
    def __init__(self,classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.classifier.eval()
        self.accuracy_fn = nn.ModuleList([WeightedAverage() for _ in range(len(classifier.num_classes_per_label))])
        self.total_acc = WeightedAverage()
        self.count = 0

    @torch.no_grad()
    def update(self, generated_images, queries, y_null, guidance_scale):
        logits = self.classifier(generated_images)
        pred_vals = torch.stack([torch.argmax(logits[i],dim=1) for i in range(len(logits))],dim=1)
        pred_vals = pred_vals.unsqueeze(dim=1).repeat(1,queries.size(1),1)
        y_null = y_null.unsqueeze(dim=1).repeat(1,queries.size(1),1)
        equal = torch.logical_not(torch.logical_xor(pred_vals == queries,(torch.tensor(guidance_scale)>0).to(pred_vals.device).unsqueeze(dim=0).unsqueeze(dim=2)))
        equal = torch.where(queries == y_null, torch.tensor(True), equal) #dont consider the null token
        equal = equal.all(dim=1)
        self.total_acc.update(equal.all(dim=1).float().mean(),generated_images.size(0))
        self.count += generated_images.size(0)
        equal = equal.float().mean(dim=0)
        for i,acc in enumerate(self.accuracy_fn):
            acc.update(equal[i],generated_images.size(0))
        return 

    def compute(self):
        return_dict= {"total_acc":self.total_acc.compute()}
        for i,acc in enumerate(self.accuracy_fn):
            return_dict[f"accuracy_{i}"] = acc.compute()
        return return_dict
    
    def reset(self):
        self.total_acc.reset()
        self.count = 0
        for acc in self.accuracy_fn:
            acc.reset()
  

class R2(Metric):
    def __init__(self,sample_size:Tuple[int]):
        super().__init__()
        self.r2 = R2Score(multioutput='variance_weighted') #,num_regressors=math.prod(sample_size)
    def update(self,generated_images,true_images):
        self.r2.update(generated_images.reshape(generated_images.size(0),-1),true_images.reshape(true_images.size(0),-1))
    def compute(self):
        return {"r2":self.r2.compute()}
    def reset(self):
        self.r2.reset()

                                                            
class JSD(Metric):
    def __init__(self,num_classes_per_label:int):
        super().__init__()
        """
        hyperparameters mentioned in the paper
        JSD(c_1,c_2) between labels c_1,c_2 [ for now we only support pairwise JSD]
        num_of_timesteps = 5
        start,end = 300,600 [ refer to elucidating design space paper]
        """
        self.c_1_index,self.c_2_index = 0,1
        self.c_1_num_classes,self.c_2_num_classes = num_classes_per_label[self.c_1_index],num_classes_per_label[self.c_2_index]
        self.num_of_timesteps = 5
        self.time_limit = (300,600)
        self.jsd = WeightedAverage()
        self.count = 0
        self.guidance_accuracy = nn.ModuleList([WeightedAverage() for _ in range(2)])
    
    @torch.no_grad()
    def update(self,batch, model,scheduler):
        count = batch["X"].size(0)
        jsd,accuracy = self.guidance_evaluator(batch,model,scheduler)
        for acc,acc_module in zip(accuracy,self.guidance_accuracy):
            acc_module.update(acc,count)
        self.count += count
        self.jsd.update(jsd,count)

    def compute(self):
        return_metrics = {"jsd":self.jsd.compute()}
        for i,acc in enumerate(self.guidance_accuracy):
            return_metrics[f"guidance_accuracy_{i}"] = acc.compute()
        return return_metrics

    def reset(self):
        self.jsd.reset()
        for acc in self.guidance_accuracy:
            acc.reset()
        self.count = 0
    
    @torch.no_grad()
    def guidance_evaluator(self,batch,model,scheduler):
        """
        refactor guidance evaluator
        """

        device = self.device
        start,end = self.time_limit
        c_1,c_2 = self.c_1_num_classes,self.c_2_num_classes
        c_1_index,c_2_index = self.c_1_index,self.c_2_index

        x_og,y_og,y_null = batch["X"].to(device), batch["label"].to(device), batch["label_null"].to(device)
        all_possible_joint_labels = torch.tensor(list(itertools.product(range( self.c_1_num_classes),range(self.c_2_num_classes))),device=device,dtype=torch.long)
        
        y_all  = y_null[:1].repeat(c_1*c_2+c_1+c_2,1).to(dtype=all_possible_joint_labels.dtype,device=device)
        y_all[:c_1*c_2,c_1_index] = all_possible_joint_labels[:,0]
        y_all[:c_1*c_2,c_2_index] = all_possible_joint_labels[:,1]
        y_all[c_1*c_2:c_1*c_2+c_1,c_1_index] = torch.arange(c_1,device=device)
        y_all[c_1*c_2+c_1:,c_2_index] =torch.arange(c_2,device=device)

        y_true = y_all.repeat(self.num_of_timesteps,1)
        
        
        og_timesteps = torch.linspace(start, end, self.num_of_timesteps,device=device).long()
        timesteps = og_timesteps.repeat_interleave(y_all.size(0))

        accuracy = [0,0]
        js_divergence = []
        x_og, y_og = batch["X"].to(device), batch["label"].to(device)
        for i in range(x_og.size(0)):
            x0 = x_og[i:i+1]
            y = y_og[i:i+1]
            
            noise = torch.randn_like(x0)
            xt = scheduler.add_noise(x0, noise, og_timesteps)
            xt = xt.repeat_interleave(y_all.size(0),dim=0)
            with torch.no_grad():
                noise_pred = model(xt, timesteps,y_true)
            n_chunks = [n.mean(dim=(1,2,3)) for n in F.mse_loss(noise_pred, noise, reduction='none').chunk(self.num_of_timesteps,dim=0)]
            l = sum([(n - n.mean())/n.std() for n in n_chunks]).view(y_all.size(0),-1).mean(dim=1).detach().cpu()
            joint =F.softmax(-1*l[:c_1*c_2],dim=0).numpy()
            indp_0 = F.softmax( -1*l[c_1*c_2:c_1*c_2+c_1],dim=0)
            indp_1 = F.softmax(-1*l[c_1*c_2+c_1:],dim=0)
            mutual = ((indp_0.reshape(-1,1) @ indp_1.reshape(1,-1))).numpy()
            js_divergence_m = 0.5*(sum(rel_entr(mutual.reshape(-1),joint.reshape(-1)))+ sum(rel_entr(joint.reshape(-1),mutual.reshape(-1))))
            js_divergence.append(js_divergence_m)
            y_est = [np.argmax(indp_0) == y[0,c_1_index].cpu().numpy(), np.argmax(indp_1)== y[0,c_2_index].cpu().numpy()]
            accuracy = [a+b for a,b in zip(accuracy,y_est)]
        return sum(js_divergence)/x_og.size(0),[x*1.0/len(x_og) for x in accuracy]

class Quality:
    def __init__(self,save_dir,save_style='grid'):
        self.save_dir = save_dir
        self.counter = 0
        self.save_style = save_style
        os.makedirs(save_dir, exist_ok=True)
    def update(self,generated_images,true_images):
        if self.save_style == 'grid':
            true_generated_image_grid_save(true_images, generated_images, f"{self.save_dir}/grid_{self.counter}.png")
            self.counter += 1
        else:
            save_images_in_folder(true_images, y, path=self.save_dir, title=f'generated_samples',counter=self.counter)
            self.counter += generated_images.size(0)



class GroupAccuracy(Metric):
    def __init__(self,labels=2,confounders=2):
        super().__init__()
        groups = labels*confounders
        self.total_per_group = [0]*groups
        self.count_per_group = [0]*groups
    def update(self,logits,y,g):
        is_correct = (torch.argmax(logits,dim=1) == y).float()
        for group in range(4):
            y_mask = (y == group//2)
            g_mask = (g == group%2)
            y_g_mask = y_mask & g_mask
            self.total_per_group[group] += is_correct[y_g_mask].sum().item()
            self.count_per_group[group] += y_g_mask.sum().item()
    
    def compute(self):
        group_accuracy = [self.total_per_group[i]/self.count_per_group[i] for i in range(4) if self.count_per_group[i] > 0]
        return {
            'avg_group_accuracy': sum(group_accuracy)/len(group_accuracy), 'worst_group_accuracy': min(group_accuracy),
            'per_group_accuracy': [ f"{i//2,i%2}:{self.total_per_group[i]/self.count_per_group[i]}" for i in range(4)]
        }
    def reset(self):
        self.total_per_group = [0]*4
        self.count_per_group = [0]*4
        
    

class DROMetrics(Metric):
    def __init__(self,labels=2,confounders=2):
        super().__init__()
        self.test_accuracy = WeightedAverage()
        self.group_accuracy = GroupAccuracy()

    def update(self,logits,y,g):
        #compute accuracy
        is_correct = (torch.argmax(logits,dim=1) == y).float()
        self.test_accuracy.update(is_correct.mean().item(),y.size(0))
        self.group_accuracy.update(logits,y,g)

    def compute(self):
        return {
            'test_accuracy': self.test_accuracy.compute(),
            **self.group_accuracy.compute()
        }
    def reset(self):
        self.test_accuracy.reset()
        self.group_accuracy.reset()


class Diversity(Metric):
    def __init__(self,classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.classifier.eval()
        self.uncontrolled_diversity = defaultdict(Counter)

    @torch.no_grad()
    def update(self,generated_images,queries,y_null):
        logits = self.classifier(generated_images)
        pred_vals = torch.stack([torch.argmax(logits[i],dim=1) for i in range(len(logits))],dim=1)
        uncond_pred_vals = torch.where(queries== y_null, pred_vals, y_null)
        for p,q in zip(uncond_pred_vals,queries):
            self.uncontrolled_diversity[tuple(q.cpu().numpy())][tuple(p.cpu().numpy())] += 1
    
    def compute(self):
        entropies = []
        for k,v in self.uncontrolled_diversity.items():
            total = sum(v.values())
            entropy = -sum([(v/total)*math.log(v/total,2) for v in v.values()])
            entropies.append(entropy)
        
        return {
            'diversity': sum(entropies)/len(entropies)
        }
    def reset(self):
        self.uncontrolled_diversity = defaultdict(Counter)





