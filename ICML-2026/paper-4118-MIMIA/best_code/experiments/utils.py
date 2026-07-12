import torch 
import numpy as np
from models.alexnet import AlexNet
import torch.nn.functional as F

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def label_to_onehot(target, num_classes):
    '''Returns one-hot embeddings of scaler labels'''
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def vec_mul_ten(vec, tensor):
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res

def instahide_data( x, y, klam, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    '''Returns mixed inputs, lists of targets, and lambdas'''
    lams = np.random.normal(0, 1, size=(x.size()[0], klam))
    for i in range(x.size()[0]):
        lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
        if klam > 1:
            while lams[i].max() > 0.85 or lams[i].max() < 0.5: #0.65: #conf["upper"]:     # upper bounds a single lambda
                lams[i] = np.random.normal(0, 1, size=(1, klam))
                lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))

    lams = torch.from_numpy(lams).float().to(device)

    mixed_x = vec_mul_ten(lams[:, 0], x)
    ys = [y]

    for i in range(1, klam):
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        mixed_x += vec_mul_ten(lams[:, i], x[index, :])
        ys.append(y[index])

        sign = torch.randint(2, size=list(x.shape), device=device) * 2.0 - 1
        mixed_x *= sign.float().to(device)
    
    return mixed_x, ys, lams

def mixup_data(klam,x, y,num_class,use_cuda=True):
    '''Returns mixed inputs, lists of targets, and lambdas'''
    lams = np.random.normal(0, 1, size=(x.shape[0], klam))
    for i in range(x.shape[0]):
        lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
        if klam > 1:
            while lams[i].max() > 0.65: # args.upper:     # upper bounds a single lambda
                lams[i] = np.random.normal(0, 1, size=(1, klam))
                lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))

    lams = torch.from_numpy(lams).float()

    mixed_x = vec_mul_ten(lams[:, 0], x)
    ys = [y]

    for i in range(1, klam):
        batch_size = x.shape[0]
        index = torch.randperm(batch_size)
        mixed_x += vec_mul_ten(lams[:, i], x[index, :])
        ys.append(y[index])

    sign = torch.randint(2, size=list(x.shape)) * 2.0 - 1
    mixed_x *= sign.float()
        
    return mixed_x, ys, lams

def insta_criterion(pred, ys, lam_batch, klam, num_classes):
    '''Returns mixup loss'''
    ys_onehot = [label_to_onehot(y, num_classes) for y in ys]
    mixy = vec_mul_ten(lam_batch[:, 0], ys_onehot[0])
    # print('lam_batch:',lam_batch[:,0])

    for i in range(1, klam):
        # print(i)
        mixy += vec_mul_ten(lam_batch[:, i], ys_onehot[i])

    l = cross_entropy_for_onehot(pred, mixy)
    # print("mixed label:",mixy[0])
    # print("pred:",pred[0])
    return l

def generate_sample(klam,dataset,num_class):
    np.random.shuffle(dataset)
    inputs=dataset[:1000,:-1]
    targets=dataset[:1000,-1]
        
    #print(inputs,targets)
        
    mix_inputs, mix_targets, lams = mixup_data(klam,inputs, targets,num_class)
        
    return (mix_inputs, mix_targets, lams)

def quant(ts,bits):
    ts_min=ts.min()
    num_of_bins=2**bits
    ts_q=(ts-ts_min)
    delta=ts_q.max()/num_of_bins
    ts_q=(ts_q/delta).int().float()
    ts_q=ts_q*delta+ts_min
    return ts_q
