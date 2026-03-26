import utils
from pipeline import TrainEnergyBlock, TrainProcessCollections, GreedyTrainPipeline, general_train_pipeline, greedy_training_pipeline, inference_pipeline
import prepared_configs
import copy
import torch

def main(args):
    # The proposed method is trained on CIFAR10, CIFAR100 and MNIST datasets
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=30, local_grad=True)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'

    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar10_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=90, local_grad=True)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(1, 3):
            net_config['conv_config'][j]['projecting_dim'] = 200 - j * 50
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar100_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True,projecting_dim=30, local_grad=True)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_mnist_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)

def diversity_only_training(args):
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(consistency_mode=None, consistency_factor=0.0, diversity_factor=1.0,sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=30, local_grad=True)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar10_diversity_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(consistency_mode=None, consistency_factor=0.0, diversity_factor=1.0,sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=90, local_grad=True)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(1, 3):
            net_config['conv_config'][j]['projecting_dim'] = 200 - j * 50
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar100_diversity_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(consistency_mode=None, consistency_factor=0.0, diversity_factor=1.0, sampling_len=20, inference_mode='sampling', projecting=True,projecting_dim=30, local_grad=True)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_mnist_diversity_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    

def supervised_sampling_training(args):
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(consistency_mode='supervised',diversity_factor=0.5, consistency_factor=0.5,sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=30, local_grad=True)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'sup_sampling_cifar10_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(consistency_mode='supervised',diversity_factor=0.5, consistency_factor=0.5,sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=90, local_grad=True)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(1, 3):
            net_config['conv_config'][j]['projecting_dim'] = 200 - j * 50
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'sup_sampling_cifar100_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(consistency_mode='supervised',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True,projecting_dim=30, local_grad=True)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'sup_sampling_mnist_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
def supervised_training(args):
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(consistency_mode='supervised',diversity_factor=0.5, consistency_factor=0.5, sampling_len=-1, inference_mode='sampling', projecting=True, projecting_dim=30, local_grad=True)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'sup_cifar10_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(consistency_mode='supervised',diversity_factor=0.5, consistency_factor=0.5, sampling_len=-1, inference_mode='sampling', projecting=True, projecting_dim=90, local_grad=True)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(1, 3):
            net_config['conv_config'][j]['projecting_dim'] = 200 - j * 50
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'sup_cifar100_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(consistency_mode='supervised',diversity_factor=0.5, consistency_factor=0.5, sampling_len=-1, inference_mode='sampling', projecting=True,projecting_dim=30, local_grad=True)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'sup_mnist_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    

def random_projecting_training(args):
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=30, local_grad=True)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = torch.randint(10, 60, (1,)).item()
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar10_randproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5,sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=90, local_grad=True)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(1, 3):
            net_config['conv_config'][j]['projecting_dim'] = torch.randint(100, 300, (1,)).item()
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar100_randproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True,projecting_dim=30, local_grad=True)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = torch.randint(10, 60, (1,)).item()
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_mnist_randproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)

def fixed_projecting_training(args):
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5,sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=30, local_grad=True)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar10_fixproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=90, local_grad=True)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        for j in range(1, 3):
            net_config['conv_config'][j]['projecting_dim'] = 100
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar100_fixproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=True,projecting_dim=30, local_grad=True)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        for j in range(3):
            net_config['conv_config'][j]['projecting_dim'] = 10
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_mnist_fixproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)


def no_projecting_training(args): 
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5,sampling_len=20, inference_mode='sampling', projecting=False, projecting_dim=30, local_grad=True)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'

    for i in range(trials):
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar10_noproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5,sampling_len=20, inference_mode='sampling', projecting=False, projecting_dim=90, local_grad=True)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_cifar100_noproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(consistency_mode='feature',diversity_factor=0.5, consistency_factor=0.5, sampling_len=20, inference_mode='sampling', projecting=False,projecting_dim=30, local_grad=True)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'unsup_mnist_noproject_{}'.format(i)
        greedy_training_pipeline(args, train_func=GreedyTrainPipeline)

def bp_model_training(args):
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    
    trials = getattr(args, 'trials', 1)
    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(local_grad=False)
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        args.MODEL = net_config
        args.save_name = 'bp_cifar10_{}'.format(i)
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        #general_train_pipeline(args, train_func=TrainProcessCollections)
        save_path = getattr(args, 'save_path', './checkpoint/nips2025/')
        #inference_pipeline(save_path, args.save_name, inference_mode='usual', local_rank=0)
        
    args.dataset = 'CIFAR100'
    net_config = prepared_configs.default_network_config(local_grad=False)
    net_config['num_classes'] = 100
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    for i in range(trials):
        args.MODEL = net_config
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'bp_cifar100_{}'.format(i)
        general_train_pipeline(args, train_func=TrainProcessCollections)
        save_path = getattr(args, 'save_path', './checkpoint/nips2025/')
        inference_pipeline(save_path, args.save_name, inference_mode='usual', local_rank=0)
    
    args.dataset = 'MNIST'
    net_config = prepared_configs.default_network_config(local_grad=False)
    net_config['conv_config'][0]['in_channels'] = 1
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config['linear_dims'] = 1536 * 3 * 3
    
    for i in range(trials):
        args.MODEL = net_config
        args.MODEL['conv_config'][-1]['is_last'] = True
        args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
        args.save_name = 'bp_mnist_{}'.format(i)
        general_train_pipeline(args, train_func=TrainProcessCollections)
        save_path = getattr(args, 'save_path', './checkpoint/nips2025/')
        inference_pipeline(save_path, args.save_name, inference_mode='usual', local_rank=0)
    


def energy_block_training(args):
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.LR_SCHEDULER = {'name':"CosineAnnealingLR", 'args': {'T_max':  args.epochs, 'eta_min': 0.0001}}
    args.dataset = 'CIFAR10'
    
    net_config = prepared_configs.default_network_config('feature', sampling_len=20, inference_mode='sampling', projecting=True, projecting_dim=30, local_grad=True)
    if args.dataset == 'MNIST':
        net_config['conv_config'][0]['in_channels'] = 1
    net_config = net_config['conv_config'][0]
    net_config['projecting'] = True
    net_config['projecting_dim'] = 30
    for i in range(11):
        net_config['consistency_factor'] = 0.1 * i
        net_config['diversity_factor'] = 0.1 * (10 - i)
        args.save_name = '{}_unsup_dl={:.1f}_cl={:.1f}'.format(args.dataset, net_config['diversity_factor'], net_config['consistency_factor'])
        args.MODEL = net_config
        greedy_training_pipeline(args, TrainEnergyBlock)
        
if __name__ == '__main__':
    config = utils.deploy_config()
    #main(config)
    #energy_block_training(config)
    #bp_model_training(config)
    supervised_sampling_training(config)