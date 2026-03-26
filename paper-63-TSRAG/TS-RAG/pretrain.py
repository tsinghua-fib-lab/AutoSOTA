import os
import time
import wandb
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.moment import MOMENTPipelineWithRetrieval
from dataset import CustomPretrainDataset, Retriever_for_pretrain
from models.ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval
    
warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='ChronosBoltRetrieve')

parser.add_argument('--model_id', type=str, default='ChronosBoltRetrieve_Pretrain')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--retrieve_lookback_length', type=int, default=64)
parser.add_argument('--retrieval_database_path', type=str, default='../database/pretrain/retrieval_database_512.parquet')

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')

# model
parser.add_argument('--model', type=str, default='ChronosBoltRetrieve')
parser.add_argument('--freeze_chronos_bolt', action='store_true', help="freeze the params of chronos-bolt.")
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base/')
parser.add_argument('--context_length', type=int, default=512)
parser.add_argument('--prediction_length', type=int, default=64)

# pretrain
parser.add_argument('--data_path', type=str, default='../datasets/pretrain/50m-with-retrieval_512', help='pretrain data path')
parser.add_argument('--train_steps', type=int, default=200_000)
parser.add_argument('--evaluation_steps', type=int, default=10_000)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--shuffle_buffer_length', type=int, default=100_000)
parser.add_argument('--grad_clip_value', type=float, default=1.0)

# gpu
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--gpu_loc', type=int, default=0, help='main gpu location')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)


args = parser.parse_args()

# init wandb project
wandb.init(project=f'{args.model}_Pretrain', name=args.model_id)
wandb.config.update(args)


device = 'cuda:'+str(args.gpu_loc)

time_now = time.time()

## load model, optimizer
config = AutoConfig.from_pretrained(args.pretrained_model_path)
if args.model == 'ChronosBolt':
    model = ChronosBoltModelForForecasting.from_pretrained(args.pretrained_model_path, config=config)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
elif args.model == 'ChronosBoltRetrieve':
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(args.pretrained_model_path, config=config, augment=args.augment_mode)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer])
    if 'gate' in args.augment_mode:
        model.init_extra_weights([model.gate_layer, model.gate_linear1, model.gate_linear2])
elif args.model == 'MOMENTRetrieve':
    MOMENT_MODEL_PATH = "AutonLab/MOMENT-1-large"
    model = MOMENTPipelineWithRetrieval.from_pretrained(MOMENT_MODEL_PATH,
                                           model_kwargs={
                                               'task_name': 'forecasting',
                                               'forecast_horizon': 64,
                                           })
    model.init()
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer, model.project_before_fusion, model.project_after_fusion])
    criterion = nn.MSELoss().to(device)
else:
    print('model error')
    exit()
print(f'{args.model} model loaded')

model.to(device)
if args.use_multi_gpu:
    args.devices = [int(i) for i in args.devices.split(',')]
    model = nn.DataParallel(model, device_ids=args.devices)
    
params = model.parameters()

if args.optimizer == 'adam':
    model_optim = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    model_optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

# freeze params
if args.freeze_chronos_bolt:
    layers_to_unfreeze = ['gate_layer', 'encode_mlp', 'mha', 'ffn']
    if args.augment_mode == 'moe3':
        if args.model == 'ChronosBoltRetrieve':
            layers_to_unfreeze.append('output_patch_embedding')
        elif args.model == 'MOMENTRetrieve':
            # import pdb; pdb.set_trace()
            layers_to_unfreeze.append('head')
    elif args.augment_mode == 'gate':
        layers_to_unfreeze.append('gate_linear1')
        layers_to_unfreeze.append('gate_linear2')

    for param in model.parameters():
        param.requires_grad = False
    # unfreeze the specified layers
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in layers_to_unfreeze)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

# retrieval already done, do not need to load the embedding model
embedding_model = None

# load retriever
retriever = Retriever_for_pretrain(
    retrieval_database_path=args.retrieval_database_path,
    dimension=768,
    embedding_model=embedding_model,
)
retriever.build_index()

## load data
dataset = CustomPretrainDataset(
    args.data_path, 
    retriever=retriever, 
    mode='training',
    drop_prob=args.drop_prob,
    context_length=args.context_length,
    prediction_length=args.prediction_length,
    retrieve_lookback_length=args.retrieve_lookback_length,
    top_k=args.top_k,
).shuffle(shuffle_buffer_length=args.shuffle_buffer_length)

train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


## train
is_first = True 
iter_count = 0
train_loss = []
for i, batch in tqdm(enumerate(train_loader)):
    if i >= args.train_steps:
        print('training finished')
        break
    if is_first : 
        print(args.data_path, batch['x'].shape, batch['y'].shape, batch['distances'].shape, batch['indices'].shape)
        is_first=False
        
    iter_count += 1
    model_optim.zero_grad()
    retrieved_seqs = torch.tensor(retriever.whole_seq[batch['indices']])
    
    if not args.use_multi_gpu:
        batch['x'] = batch['x'].float().to(device)
        batch['y'] = batch['y'].float().to(device)
        batch['distances'] = batch['distances'].float().to(device)
        retrieved_seqs = retrieved_seqs.float().to(device)
    if args.model == 'ChronosBoltRetrieve':
        outputs = model(context = batch['x'].float(), 
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(), 
                        distances = batch['distances'].float())                  # ChronosBoltOutput
    elif args.model == 'MOMENTRetrieve':
        outputs = model(x_enc=batch['x'].float().unsqueeze(1), retrieved_seq=retrieved_seqs.float())
        outputs = outputs.forecast.squeeze(1)                                                     
        loss = criterion(outputs, batch['y'].float())
    else:
        print('model error')
    if args.model == 'MOMENTRetrieve':
        pass
    else:
        loss = outputs.loss
    loss = loss.mean()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.log({
            'loss': loss.item(),
            'lr': model_optim.param_groups[0]['lr']
            })

    train_loss.append(loss.item())

    if (i + 1) % args.evaluation_steps == 0:
        print("\titers: {0} | loss: {1:.7f}".format(i + 1, sum(train_loss) / len(train_loss)))
        train_loss = []
        speed = (time.time() - time_now) / iter_count
        print('\tspeed: {:.4f}s/iter'.format(speed))
        iter_count = 0
        time_now = time.time()
        # save model and optimizer
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            save_path = os.path.join(args.checkpoints, args.model_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path,f'model_steps{i}.pth'))
            torch.save(model_optim.state_dict(), os.path.join(save_path, f'optim_steps{i}.pth'))

        # adjust learning rate
        scheduler.step()
        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

    loss.backward()
    clip_grad_norm_(model.parameters(), args.grad_clip_value)
    model_optim.step()
                

