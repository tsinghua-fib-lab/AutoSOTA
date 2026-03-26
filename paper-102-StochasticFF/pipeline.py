import utils
import layers
import datasets
import torch
from enum import Enum
import time
import shutil
import math
import numpy as np
import copy
import os
from tqdm import tqdm

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        info = '\t'.join(entries)
        print(info)
        return info

    def display_summary(self):
        entries = [self.prefix + " *"]
        entries += [meter.summary() for meter in self.meters]
        info = ' '.join(entries)
        print(info)
        return info


class TrainProcessCollections:
    def __init__(self, *args, **kwargs):
        self.dump_path = './checkpoints/'
        self.data_dir = './data/'
        self.task_type = 'classify'
    
    def set_random_seed(self, seed):
        utils.seed_everything(seed)
    
    def make_model(self, args):
        model_args = args.MODEL
        model = layers.EnergyCNN(**model_args)
        return model
    
    def prepare_dataloader(self, args):
        if args.dataset == 'CIFAR10':
             trainloader, testloader = datasets.make_cifar10_dataloader(args)
        
        elif args.dataset == 'MNIST':
            trainloader, testloader = datasets.classic_mnist_loader(args.data_dir, args.bs, args.bs, num_workers=args.workers,)
        
        elif args.dataset == 'CIFAR100':
            trainloader, testloader = datasets.make_cifar100_dataloader(args)
        else:
            raise ValueError('dataset not supported')
        return trainloader, testloader
    
    def prepare_optimizer_scheduler(self, params_group, args):
        optimizer, scheduler = utils.prepare_optimizer_scheduler(params_group, args)
        return optimizer, scheduler
    
    def prepare_criterion(self, args):
        criterion = utils.make_criterion(args.CRITERION)
        return criterion
    
    def specify_params_group(self, model):
        return filter(lambda p: p.requires_grad, model.parameters())
    
    def data2device(self, data, target, args):
        if args.use_cuda:
            data = data.cuda(args.local_rank, non_blocking=True)
            if isinstance(target, torch.Tensor):
                target = target.cuda(args.local_rank, non_blocking=True)
        return data, target
    
    def metric_init(self, data_loader, epoch, prefix='Epoch: [{}]'):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, top1],
            prefix=prefix.format(epoch))
        return batch_time, data_time, losses, top1, progress
    
    def compute_model_output(self, model, inputs, args=None):
        output = model(inputs)
        return output
    
    
    def compute_loss(self, output, target, criterion, model=None, args=None, inputs=None):
        loss = criterion(output, target)
        return loss
    
    def model_processing_before_train(self, model, args):
        model.train()
        return model
    
    def model_processing_before_eval(self, model, args):
        model.eval()
        return model
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, args,):
        batch_time, data_time, losses, top1, progress = self.metric_init(train_loader, epoch)
        
        # switch to train mode
        model = self.model_processing_before_train(model, args)
        
        end = time.time()
        num_updates = epoch * len(train_loader)
        
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            optimizer.zero_grad()
            images, target = self.data2device(images, target, args)
            data_time.update(time.time() - end)
            output = self.compute_model_output(model, images, args)
            loss = self.compute_loss(output=output, target=target, criterion=criterion, model=model, args=args, inputs=images)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            num_updates += 1
            
            acc1, = self.accuracy(output, target, topk=(1, ))
            top1.update(acc1.item(), images.size(0))
            losses.update(loss.item(), images.size(0))
            end = time.time()
            if i % args.print_freq == 0 and args.local_rank == 0:
                info = progress.display(i) + '\n'
                utils.writing_log(args.log_path, info)
        if args.local_rank == 0 and getattr(args, 'log_path', None) is not None:
            info = 'Training result: ' + progress.display_summary() + '\n'
            utils.writing_log(args.log_path, info)
    
    def validate(self, val_loader, model, criterion, args, epoch=0):
        batch_time, data_time, losses, top1, progress = self.metric_init(val_loader, epoch, 'Test [Epoch:{}]: ')

        model = self.model_processing_before_eval(model, args)
    
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images, target = self.data2device(images, target, args)
                output = self.compute_model_output(model, images, args)
                loss = self.compute_loss(output=output, target=target, criterion=criterion, model=model, args=args, inputs=images)
                acc1, = self.accuracy(output, target, topk=(1, ))
                top1.update(acc1.item(), images.size(0))
                losses.update(loss.item(), images.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
            if args.local_rank == 0 and getattr(args, 'log_path', None) is not None:
                info = 'Validation result: ' + progress.display_summary() + '\n'
                utils.writing_log(args.log_path, info)
        return top1.avg
    
    def save_checkpoint(self, state, is_best, save_path, save_name='checkpoint'):
        torch.save(state, save_path + save_name + '.pth')
        if is_best:
            shutil.copyfile(save_path + save_name + '.pth', save_path + save_name + '_best_model.pth')
            
    def run_training(self, args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, best_acc1, save_path, local_rank=0):
        best_epoch = args.start_epoch
        for epoch in range(args.start_epoch, args.epochs):
            self.train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
            acc1 = self.validate(val_loader, model, criterion, args, epoch=epoch)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch
            if lr_scheduler is not None:
                lr_scheduler.step()
                
            if local_rank == 0:
                save_state = {
                    'epoch': epoch + 1,
                    'arch': args.save_name,
                    'best_acc1': best_acc1,
                    'best_epoch': best_epoch,
                    'state_dict':  model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': lr_scheduler.state_dict() if lr_scheduler is not None else None
                }
                self.save_checkpoint(save_state, is_best, save_path, save_name=args.save_name)
        if local_rank == 0:
            info = '-*- Summary: after {} epochs training, the model hit {}% top1 acc at epoch [{}]\n'.format(
                args.epochs - args.start_epoch, best_acc1, best_epoch)
            utils.writing_log(args.log_path, info)


class GreedyTrainPipeline(TrainProcessCollections):
    def compute_learning_rate(self, args, epoch):
        if args.lr_schedule == 'cosine':
            lr = args.lr * 0.5 * (1 + math.cos(np.pi * epoch / args.epochs))
        elif args.lr_schedule == 'step':
            lr = args.lr * (0.1 ** (epoch // getattr(args, 'step_size', 10)))
        else:
            lr = args.lr
        return lr

    def energy_train_one_epoch(self, train_loader, model: layers.AuxiliaryEnergyModel, epoch, args, prefix='{}, Epoch: [{}]', layer_name='conv1'):
        lr = self.compute_learning_rate(args, epoch)
        args.OPTIMIZER['args']['lr'] = lr
        params_group = self.specify_params_group(model)
        optimizer = utils.make_optimizer(params_group, args)
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix=prefix.format(layer_name, epoch))
        
        model.train()
        
        end = time.time()
        num_updates = epoch * len(train_loader)
        for i, (images, target) in enumerate(train_loader):
            images, target = self.data2device(images, target, args)
            data_time.update(time.time() - end)
            # Compute output
            optimizer.zero_grad()
            _, loss = model(images, target)
            
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            num_updates += 1
            losses.update(loss.item(), images.size(0))
            end = time.time()
            if i % args.print_freq == 0 and args.local_rank == 0:
                info = progress.display(i) + '\n'
                utils.writing_log(args.log_path, info)
        
        if args.local_rank == 0 and getattr(args, 'log_path', None) is not None:
            info = '{} Training result: '.format(layer_name) + progress.display_summary() + '\n'
            utils.writing_log(args.log_path, info)
    
    def run_training(self, args, model, train_loader, val_loader, criterion, best_acc1, save_path, local_rank=0):
        best_epoch = args.start_epoch
        auxililary_config = args.AUXILIARY_CONFIG
        for epoch in range(args.start_epoch, args.epochs):
            energy_conv = torch.nn.ModuleList()
            for i, conv in enumerate(model.energy_blocks):
                auxililary_model = layers.AuxiliaryEnergyModel(energy_conv, copy.deepcopy(conv), **auxililary_config)
                if args.use_cuda:
                    auxililary_model = auxililary_model.cuda(local_rank)
                self.energy_train_one_epoch(train_loader, auxililary_model, epoch, args, prefix='{}, Epoch: [{}]', layer_name='conv{}'.format(i + 1))
                auxililary_model = auxililary_model.cpu()
                model.energy_blocks[i].load_state_dict(auxililary_model.next_model.state_dict())
                energy_conv.append(copy.deepcopy(auxililary_model.next_model))
            
        auxililary_model = layers.AuxiliaryEnergyModel(energy_conv, copy.deepcopy(model.classifier), **auxililary_config)
        if args.use_cuda:
            auxililary_model = auxililary_model.cuda(local_rank)
        lr = self.compute_learning_rate(args, epoch)
        args.OPTIMIZER['args']['lr'] = lr
        params_group = self.specify_params_group(auxililary_model)
        optimizer = utils.make_optimizer(params_group, args)
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, 'phase2_epoch', 40))
        
        is_best = False
        for phase2_epoch in range(getattr(args, 'phase2_epoch', 60)):
            self.train_one_epoch(train_loader, auxililary_model, criterion, optimizer, phase2_epoch, args)
            acc1  = self.validate(val_loader, auxililary_model, criterion, epoch=phase2_epoch, args=args)
            if acc1 > best_acc1:
                best_acc1 = acc1
                best_epoch = phase2_epoch
            lr_schedule.step()
                
        auxililary_model = auxililary_model.cpu()
        model.classifier.load_state_dict(auxililary_model.next_model.state_dict())
        if local_rank == 0:
            save_state = {
                'epoch': epoch + 1,
                'arch': args.save_name,
                'best_acc1': best_acc1,
                'best_epoch': best_epoch,
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }
            self.save_checkpoint(save_state, is_best, save_path, save_name=args.save_name)
            
        if local_rank == 0:
            info = '-*- Summary: after {} epochs training, the model hit {}% top1 acc at epoch [{}]\n'.format(
                args.epochs - args.start_epoch, best_acc1, best_epoch)
            utils.writing_log(args.log_path, info)
            
class TrainEnergyBlock(GreedyTrainPipeline):
    def make_model(self, args):
        model_args = args.MODEL
        model = layers.EnergyConv2dEnsemble(**model_args)
        return model
    
    def run_training(self, args, model, train_loader, val_loader, criterion,  best_acc1, save_path, local_rank=0):
            best_epoch = args.start_epoch
            model = model.cuda(local_rank)
            for epoch in range(args.start_epoch, args.epochs):
                # Train the energy blocks in a greedy manner
                self.energy_train_one_epoch(train_loader, model, epoch, args, prefix='{}, Epoch: [{}]', layer_name='conv{}'.format(1))
                save_state = {
                    'epoch': epoch + 1,
                    'arch': args.save_name,
                    'best_acc1': best_acc1,
                    'best_epoch': best_epoch,
                    'state_dict':  model.state_dict(),
                }
                self.save_checkpoint(save_state, False, save_path, save_name=args.save_name)

def general_train_pipeline(args, train_func=TrainProcessCollections, **kwargs):
    train_func = train_func(args, **kwargs)
    local_rank = 0
    if args.seed is not None:
        train_func.set_random_seed(args.seed)
    best_acc1 = - 1
    if args.use_cuda:
        torch.cuda.set_device(local_rank)
        utils.cudnn.benchmark = True
    
    save_path = getattr(args, 'dump_path', train_func.dump_path) + args.dir + '/'
    args.log_path = save_path + args.save_name + '_log.txt'
    args.save_path = save_path
    if local_rank == 0:
        utils.make_dir(save_path)
        utils.record_hyper_parameter(save_path, args.save_name, **args.__dict__)
    
    model = train_func.make_model(args)
    if args.use_cuda:
        model.cuda(local_rank)
    
    train_loader, val_loader = train_func.prepare_dataloader(args)
    criterion = train_func.prepare_criterion(args)
    if args.use_cuda:
        criterion = criterion.cuda(local_rank)
    params_group = train_func.specify_params_group(model)
    optimizer, lr_scheduler = train_func.prepare_optimizer_scheduler(params_group, args)
    
    if lr_scheduler is not None and args.start_epoch > 0:
        lr_scheduler.step(args.start_epoch)
    
    train_func.run_training(args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, best_acc1, save_path, local_rank)
    

def greedy_training_pipeline(args, train_func=GreedyTrainPipeline, **kwargs):
    train_func = train_func(args, **kwargs)
    local_rank = 0
    if args.seed is not None:
        train_func.set_random_seed(args.seed)
    best_acc1 = - np.inf
    if args.use_cuda:
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
    save_path = getattr(args, 'dump_path', train_func.dump_path) + args.dir + '/'
    args.log_path = save_path + args.save_name + '_log.txt'
    args.save_path = save_path
    utils.make_dir(save_path)
    utils.record_hyper_parameter(save_path, args.save_name, **args.__dict__)
    
    model = train_func.make_model(args)
    train_loader, val_loader = train_func.prepare_dataloader(args)
    criterion = train_func.prepare_criterion(args)
    if args.use_cuda:
        criterion = criterion.cuda(local_rank)
    
    train_func.run_training(args, model, train_loader, val_loader, criterion, best_acc1, save_path, local_rank)



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

def inference_pipeline(path, save_name, inference_mode='usual', local_rank=0):
    args = utils.TempArgs()
    args.config = '{}{}_config.yaml'.format(path, save_name)
    temp_name = "{}_inference={}".format(save_name, inference_mode)
    args = utils.set_config2args(args)
    train_func = GreedyTrainPipeline()
    model = train_func.make_model(args)
    train_loader, test_loader = train_func.prepare_dataloader(args)
    ckpt = torch.load("{}{}.pth".format(path, save_name))
    model.load_state_dict(ckpt['state_dict'])
    auxililary_config = args.AUXILIARY_CONFIG
    auxililary_model = layers.AuxiliaryEnergyModel(model.energy_blocks, model.classifier, **auxililary_config)
    criterion = train_func.prepare_criterion(args)
    if args.use_cuda:
        criterion = criterion.cuda(local_rank)
    if inference_mode == 'usual':
        for i in range(len(auxililary_model.trained_model)):
            auxililary_model.trained_model[i].local_grad = False
        auxililary_model.trained_model[-1].is_last = False
    elif inference_mode == 'average':
        auxililary_model.trained_model[-1].inference_mode = 'average'
    else:
        raise ValueError("Inference mode should be usual or average")
    if args.use_cuda:
            auxililary_model = auxililary_model.cuda(local_rank)
    params_group = train_func.specify_params_group(auxililary_model)
    optimizer = utils.make_optimizer(params_group, args)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, 'phase2_epoch', 60) - args.start_epoch)
    best_acc = -1
    if os.path.exists("{}{}_best.pth".format(path, temp_name)):
        checkpoint = torch.load("{}{}_best.pth".format(path, temp_name))
        auxililary_model.next_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
    else:
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
    print(temp_name, 'Best acc:', best_acc)
    return best_acc