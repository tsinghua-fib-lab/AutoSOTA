from sklearn.mixture import GaussianMixture
from torch.distributions import MultivariateNormal
import torch
import numpy as np
import torch.nn.functional as F
import utils
import pipeline, layers
from tqdm import tqdm
import copy
import pandas as pd
from collections import defaultdict
import os

def cosine_similarity_dims(weight):
    weight = torch.flatten(weight, start_dim=1)
    num_channels = weight.shape[0]
    count = 0
    pools = []
    for i in range(num_channels):
        pools.append(weight[i])
        if len(pools) == 1:
            count += 1
        else:
            score = 0
            for j in range(len(pools) - 1):
                cos = 1 - F.cosine_similarity(pools[j].flatten(), weight[i].flatten(), dim=-1)
                score += cos
            score = score / (len(pools) - 1)
            count += score
    return count / (num_channels - 1)


def load_conv_weight(path, name):
    train_func = pipeline.TrainEnergyBlock()
    args = utils.TempArgs()
    args.save_path = path
    args.config = '{}{}_config.yaml'.format(path, name)
    args = utils.set_config2args(args)
    layer1 = train_func.make_model(args)
    ckpt = torch.load("{}{}.pth".format(path, name))
    layer1.load_state_dict(ckpt['state_dict'])
    weight = layer1.conv.weight.data
    bn_mean = layer1.bn.running_mean.data
    bn_var = layer1.bn.running_var.data
    weight = weight - bn_mean.reshape(1, -1, 1, 1)
    weight = weight / torch.sqrt(bn_var.reshape(1, -1, 1, 1) + layer1.bn.eps)
    weight_var = torch.var(weight, dim=[1, 2, 3]).cpu()
    idx = torch.argsort(weight_var, descending=True)
    weight = weight[idx].cpu()
    return weight

class GaussianMixtureModel:
    def __init__(self, num_components, mean, cov, sample_size) -> None:
        self.num_components = num_components
        self.components = []
        self.components_entropy = []
        self.ind_components = []
        self.ind_components_entropy = []
        self.samples = []
        self.ind_samples = []
        self.sample_size = sample_size
        self.mean = mean
        self.cov = cov
        self.total_sample = num_components * sample_size
        mask = torch.eye(self.mean.size(-1))
        for i in range(num_components):
            self.components.append(MultivariateNormal(mean[i], scale_tril=torch.linalg.cholesky(cov[i])))
            self.components_entropy.append(self.components[i].entropy())
            self.samples.append(self.components[i].sample([self.sample_size]))
            self.ind_components.append(MultivariateNormal(mean[i], scale_tril=torch.linalg.cholesky(cov[i] * mask)))
            self.ind_samples.append(self.ind_components[i].sample([self.sample_size]))
            self.ind_components_entropy.append(self.ind_components[i].entropy())
            
        self.components_entropy = torch.tensor(self.components_entropy)
        self.ind_components_entropy = torch.tensor(self.ind_components_entropy)
        self.samples = torch.cat(self.samples, dim=0)
        self.ind_samples = torch.cat(self.ind_samples, dim=0)
        
        self.gm = GaussianMixture(num_components)
        self.gm.weights_ = np.ones(num_components) / num_components
        self.gm.means_ = mean.numpy()
        self.gm.covariances_ = cov.numpy()
        self.gm.precisions_cholesky_ = torch.linalg.cholesky(torch.linalg.inv(cov)).numpy()
        
        self.ind_gm = GaussianMixture(num_components)
        self.ind_gm.weights_ = np.ones(num_components) / num_components
        self.ind_gm.means_ = mean.numpy()
        self.ind_gm.covariances_ = (cov * mask).numpy()
        self.ind_gm.precisions_cholesky_ = torch.linalg.cholesky(torch.linalg.inv(cov * mask)).numpy()
        
    def compute_probability(self, samples, component):
        log_prob = component.log_prob(samples)
        prob = torch.exp(log_prob)
        return prob
    
    def sampling_entropy(self, samples, dist):
        entropy = - torch.mean(dist.log_prob(samples))
        return entropy
    
    def compute_entropy(self, log_prob, dim=None):
        if dim is None:
            entropy = - torch.mean(log_prob)
        else:
            entropy = - torch.mean(log_prob, dim=dim)
        return entropy
    
    def gaussian_margin_prob(self, samples, mean, cov):
        # dim(samples) = (num_samples, dims)
        assert cov.dim() == 2 and mean.dim() == 1
        std = torch.sqrt(torch.diag(cov))
        prob = 1 / (np.sqrt(2 * np.pi) * std) * torch.exp(- (samples - mean) ** 2 / 2 / std / std)
        return prob
    
    def unconditional_response_entropy(self):
        samples = self.samples.reshape(self.total_sample, -1)
        entropy = - self.gm.score(samples)
        return entropy
    
    def unconditional_ind_response_entropy(self):
        samples = self.ind_samples.reshape(self.total_sample, -1)
        entropy = - self.ind_gm.score(samples)
        return entropy
    
    def conditional_response_entropy(self):
        entropy = torch.mean(self.components_entropy)
        return entropy
    
    def presudo_ind_uncond_entropy(self):
        samples = self.ind_samples.reshape(self.total_sample, -1)
        entropy = - self.ind_gm.score(samples)
        return entropy
    
    def unconditional_single_cell_entropy(self):
        samples = self.samples.reshape(self.total_sample, -1)
        x = []
        for i in range(self.num_components):
            u = self.mean[i]
            cov = self.cov[i]
            assert cov.dim() == 2 and u.dim() == 1
            var = torch.diag(cov)
            assert var.size() == u.size()
            x.append((- torch.pow(samples - u, 2) / var / 2 - torch.log(torch.sqrt(2 * np.pi * var) * self.num_components)).unsqueeze(0))
        x = torch.cat(x, dim=0)
        assert x.size() == (self.num_components, self.total_sample, self.mean.size(-1))
        entropy = torch.logsumexp(x, dim=0)
        entropy = torch.sum(-torch.mean(entropy, dim=0))
        return entropy
    
    def condition_single_cell_entropy(self):
        entropy = torch.mean(self.ind_components_entropy)
        return entropy
    
    def global_mutual_information(self):
        uncondtional_entropy = self.unconditional_response_entropy()
        mi = uncondtional_entropy - self.conditional_response_entropy()
        return mi
    
    def linear_term_information(self):
        entropy = self.unconditional_single_cell_entropy() - self.condition_single_cell_entropy()
        return entropy
    
    def signal_similarity_information(self):
        entropy = self.presudo_ind_uncond_entropy() - self.unconditional_single_cell_entropy()
        return entropy

def information_breakdown(num_conponents, output, sample_size):
    mean, cov = output
    cov = cov + cov * torch.eye(cov.size(-1)) + torch.permute(cov, dims=(0, 2, 1)) * (torch.ones(cov.size(-1), cov.size(-1)) - torch.eye(cov.size(-1)))
    cov = cov / 2 # prevent floating error by reinforcing cov symmetric
    prob_model = GaussianMixtureModel(num_conponents, mean, cov, sample_size)
    mi = prob_model.global_mutual_information()
    linear_info = prob_model.linear_term_information()
    sig_sim = prob_model.signal_similarity_information()
    corr_info = mi - linear_info - sig_sim
    return mi.item(), linear_info.item(), sig_sim.item(), corr_info.item()


def train_one_epoch(model, train_loader, critertion, optimizer, local_rank=0):
    model.train()
    train_loss = 0
    train_acc = 0
    for i, (data, label) in enumerate(train_loader):
        data, label = data.cuda(local_rank), label.cuda(local_rank)
        optimizer.zero_grad()
        output = model(data)
        loss = critertion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        train_acc += (pred == label).sum().item() / len(label)
    return train_loss / len(train_loader), train_acc / len(train_loader)

@torch.no_grad()
def validate_one_epoch(model, test_loader, critertion, local_rank=0):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.cuda(local_rank), label.cuda(local_rank)
            output = model(data)
            loss = critertion(output, label)
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            test_acc += (pred == label).sum().item() / len(label)
    return test_loss / len(test_loader), test_acc / len(test_loader)


def layer_wise_classification(path, save_name, local_rank=0):
    args = utils.TempArgs()
    args.config = '{}{}_config.yaml'.format(path, save_name)
    args = utils.set_config2args(args)
    train_func = pipeline.GreedyTrainPipeline()
    model = train_func.make_model(args)
    train_loader, test_loader = train_func.prepare_dataloader(args)
    ckpt = torch.load("{}{}.pth".format(path, save_name))
    model.load_state_dict(ckpt['state_dict'])
    auxililary_config = args.AUXILIARY_CONFIG
    trained_blocks = torch.nn.ModuleList()
    trained_models = []
    for i in range(2):
        trained_blocks = copy.deepcopy(model.energy_blocks[:i+1])
        with torch.no_grad():
            x: torch.Tensor = test_loader.dataset[0][0]
            while x.dim() < 4:
                x = x.unsqueeze(0)
            for m in trained_blocks:
                m.eval()
                m.local_grad = False
                x = m(x)
            for m in trained_blocks:
                m.local_grad = True
        input_dims = torch.numel(x)
        trained_blocks[-1].is_last = True
        fc1 = torch.nn.Linear(input_dims, 10)
        classifier = torch.nn.Sequential(*[torch.nn.Flatten(),torch.nn.Dropout(0.5), fc1])
        auxililary_model = layers.AuxiliaryEnergyModel(trained_blocks, classifier, **auxililary_config)
        temp_name = "{}_internal_conv={}".format(save_name, i)
        if os.path.exists("{}{}_best.pth".format(path, temp_name)):
            ckpt = torch.load("{}{}_best.pth".format(path, temp_name), map_location='cpu')
            auxililary_model.next_model.load_state_dict(ckpt['state_dict'])
            best_acc = ckpt['best_acc']
        else:
            if args.use_cuda:
                auxililary_model = auxililary_model.cuda(local_rank)
            params_group = train_func.specify_params_group(auxililary_model)
            optimizer = utils.make_optimizer(params_group, args)
            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, 'phase2_epoch', 40) - args.start_epoch)
            criterion = train_func.prepare_criterion(args)
            best_acc = -1
            for epoch in tqdm(range(args.phase2_epoch)):
                train_loss, train_acc = train_one_epoch(auxililary_model, train_loader, criterion, optimizer, local_rank)
                test_loss, test_acc = validate_one_epoch(auxililary_model, test_loader, criterion, local_rank)
                lr_schedule.step()
                if best_acc < test_acc:
                    best_acc = test_acc
                    torch.save({
                    'epoch': epoch,
                    'state_dict': auxililary_model.next_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc
                }, "{}{}_best.pth".format(path, temp_name))
            
            auxililary_model = auxililary_model.cpu()
            ckpt = torch.load("{}{}_best.pth".format(path, temp_name), map_location='cpu')
            auxililary_model.next_model.load_state_dict(ckpt['state_dict'])
        print('Using internal conv: {}, best acc is: {}'.format(i, best_acc))
        trained_models.append(auxililary_model)
    return trained_models


@torch.no_grad()
def fetch_model_outputs_only(path, save_name, local_rank=0):
    args = utils.TempArgs()
    args.config = '{}{}_config.yaml'.format(path, save_name)
    args = utils.set_config2args(args)
    train_func = pipeline.GreedyTrainPipeline()
    model = train_func.make_model(args)
    _, test_loader = train_func.prepare_dataloader(args)
    ckpt = torch.load("{}{}.pth".format(path, save_name))
    model.load_state_dict(ckpt['state_dict'])
    if args.use_cuda:
        model = model.cuda(local_rank)
    model.eval()
    model_outputs = []
    label_list = []
    for i, (data, label) in enumerate(test_loader):
            data, label = data.cuda(local_rank), label.cuda(local_rank)
            output = model(data)
            model_outputs.append(output.cpu())
            label_list.append(label.cpu())
    model_outputs = torch.cat(model_outputs, dim=0)
    label_list = torch.cat(label_list, dim=0)
    return model_outputs, label_list, model.cpu()


@torch.no_grad()
def fetch_outputs_with_model(path, save_name, model, local_rank=0):
    args = utils.TempArgs()
    args.config = '{}{}_config.yaml'.format(path, save_name)
    args = utils.set_config2args(args)
    train_func = pipeline.GreedyTrainPipeline()
    _, test_loader = train_func.prepare_dataloader(args)
    if args.use_cuda:
        model = model.cuda(local_rank)
    model.eval()
    model_outputs = []
    label_list = []
    for i, (data, label) in enumerate(test_loader):
            data, label = data.cuda(local_rank), label.cuda(local_rank)
            output = model(data)
            model_outputs.append(output.cpu())
            label_list.append(label.cpu())
    model_outputs = torch.cat(model_outputs, dim=0)
    label_list = torch.cat(label_list, dim=0)
    return model_outputs, label_list

def information_theorectical_analysis(path, save_name, local_rank=0):
    trained_model = layer_wise_classification(path, save_name, local_rank=local_rank)
    internal_layer_outputs = []
    for model in trained_model:
        outputs = fetch_outputs_with_model(path, save_name, model)
        internal_layer_outputs.append(outputs)
    final_outputs, label, _ = fetch_model_outputs_only(path, save_name)
    internal_layer_outputs.append((final_outputs, label))
    acc_list = []
    information = defaultdict(list)
    for i, (outputs, label) in enumerate(internal_layer_outputs):
        print(outputs.shape)
        pred = torch.argmax(outputs, dim=-1)
        acc = torch.sum(pred == label) / torch.numel(label)
        acc_list.append(acc)
        print(acc, outputs.shape, label.shape)
        idx = torch.argsort(label, descending=True)
        outputs = outputs[idx].reshape(10, -1, 10)
        outputs = torch.permute(outputs, (0, 2, 1))
        mean = torch.mean(outputs, dim=-1)
        cov = torch.einsum('bmi, bni -> b m n', outputs, outputs) / (outputs.size(-1)-1)
        mi, linear_info, sig_sim, corr_info = information_breakdown(num_conponents=10, output=(mean, cov), sample_size=100000)
        information['information'].extend([mi, linear_info + sig_sim, corr_info])
        information['layer'].extend(['conv{}'.format(i+1)] * 3)
        information['component'].extend(['tot', 'lin', 'cor'])
        print(mi, linear_info + sig_sim, corr_info)
    df = pd.DataFrame(information)
    return df