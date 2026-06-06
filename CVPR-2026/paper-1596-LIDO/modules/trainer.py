import os
import sys
import time
import pickle
import datetime
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cuda as cudnn
import torch.nn.functional as F

from utils.avgmeter import AverageMeter
from utils.ioueval import iouEval
from utils.lovasz_loss import Lovasz_loss
from utils.losses import OWLoss, VoxelContrastiveLoss, ObjectosphereLoss
# from utils.focal_loss import FocalLoss

from torch.utils.tensorboard import SummaryWriter

# from utils.sync_batchnorm.batchnorm import convert_model
from utils.scheduler import WarmupCosine, WarmupCosineLR

from network.minkunet import MinkUNet

# from dataloader.kitti import Parser

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, checkpoint=None, pretrained=False, fp16=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.checkpoint = checkpoint
        self.pretrained = pretrained
        self.fp16 = fp16

        # get data
        if self.ARCH['dataset']['pc_dataset_type'] == 'SemanticKITTI':
            from dataloader.kitti import Parser
        # elif self.ARCH['dataset']['pc_dataset_type'] == 'PandaSet':
        #     from dataloader.pandaset.parser import Parser
        elif self.ARCH['dataset']['pc_dataset_type'] == 'SemanticPOSS':
            from dataloader.poss import Parser
        elif self.ARCH['dataset']['pc_dataset_type'] == 'nuScenes':
            from dataloader.nuscenes import Parser
        # elif self.ARCH['dataset']['pc_dataset_type'] == 'ScribbleKITTI':
        #     from dataloader.scribblekitti.parser import Parser
        else:
            raise ValueError(f"Dataset type {self.ARCH['dataset']['pc_dataset_type']} not supported")
        
        # SemanticKITTI dataset
        self.parser = Parser(root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=None,
            labels=self.DATA["labels"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            voxel_size=self.ARCH["model_params"]["voxel_size"],
            batch_size=self.ARCH["train"]["batch_size"],
            workers=self.ARCH["train"]["workers"],
            gt=True,
            aug=True,
            shuffle_train=True)

        # weights for loss and bias
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)   # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        # set loss weights for small data setup or development testing
        if self.parser.get_train_size() < 300:
            self.loss_w = torch.ones_like(self.loss_w)
            self.loss_w[self.ARCH['dataset']['ignore_label']] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            self.model = MinkUNet(in_dim=self.ARCH['model_params']['input_dims'],
                                  num_classes=self.parser.get_n_classes(),
                                  layer_num=self.ARCH['model_params']['layer_num'],
                                  cr=self.ARCH['model_params']['cr'])

        # print details of the model
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name}, Parameters: {param.numel()}")

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of parameters {num_params/1000000} M')

        # GPU
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        print(f'Training in device: {self.device}')

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            # self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

        # Losses
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ARCH['dataset']['ignore_label'], weight=self.loss_w).to(self.device)
        self.lovasz = Lovasz_loss(ignore=self.ARCH['dataset']['ignore_label']).to(self.device)
        self.owloss = OWLoss(self.parser.get_n_classes()).to(self.device) if self.ARCH['train']['mav_loss'] else None
        self.contloss = VoxelContrastiveLoss(self.parser.get_n_classes()).to(self.device) if self.ARCH['train']['cont_loss'] else None
        self.objloss = ObjectosphereLoss().to(self.device) if self.ARCH['train']['obj_loss'] else None
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()
            self.lovasz = nn.DataParallel(self.lovasz).cuda()
            # self.owloss = nn.DataParallel(self.owloss).cuda()
            #self.focal = nn.DataParallel(self.focal).cuda()

        # self.optimizer = torch.optim.AdamW(self.model.parameters(),
        #                                    lr=self.ARCH['train']['learning_rate'],
        #                                    weight_decay=self.ARCH['train']['weight_decay'],
        #                                    eps=1e-8)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.ARCH['train']['learning_rate'],
                                         momentum=self.ARCH['train']['momentum'],
                                         weight_decay=self.ARCH['train']['weight_decay'],
                                         nesterov=self.ARCH['train']['nesterov'])

        # Scheduler
        if self.ARCH['train']['scheduler']['name'] == 'OneCycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=self.ARCH['train']['scheduler']['max_lr'],
                                                             steps_per_epoch=len(self.parser.get_train_set()),
                                                             epochs=self.ARCH['train']['epochs'],
                                                             pct_start=self.ARCH['train']['scheduler']['pct_start'])
        elif self.ARCH['train']['scheduler']['name'] == 'WarmupCosine':
            self.scheduler = WarmupCosineLR(self.optimizer,
                                            lr=self.ARCH['train']['learning_rate'],
                                            warmup_steps=5 * self.parser.get_train_size(), # TODO set warmup epochs in config file
                                            momentum=0.9,
                                            max_steps=(self.ARCH['train']['epochs'] - 5) * self.parser.get_train_size())
            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            #                                                    WarmupCosine(warmup_end= 4 * self.parser.get_train_size(),
            #                                                                 max_iter= self.ARCH['train']['epochs'] * self.parser.get_train_size(),
            #                                                                 factor_min= 0.00001 / 0.002, #self.ARCH['train']['learning_rate'] / self.ARCH['train']['scheduler']['max_lr'],
            #                                                                 ),)
        else:
            # setup a simple scheduler
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=1)

        print(f'Optimizer: {self.optimizer}')
        print(f'Scheduler: {self.scheduler}')

        # grad scaler
        self.scaler = torch.amp.GradScaler(enabled=self.fp16)

        # start epoch
        self.start_epoch = 0

        # Checkpoint model config
        if self.checkpoint is not None:
            self.load_checkpoint(self.checkpoint)

        # Pretrained model
        if self.pretrained:
            pretrained_path = './minkunet-kitti-pretrained.pt' # change this if needed
            ckpt = torch.load(pretrained_path, map_location=self.device_name)
            try:
                self.model.load_state_dict(ckpt, strict=True)
            except:
                self.model.load_state_dict(ckpt['state_dict'], strict=True)
            print(f"[INFO] Pretrained model loaded from {pretrained_path}")

        # tensorboard
        self.writer_train = SummaryWriter(log_dir=self.logdir + "/tensorboard/train/", flush_secs=30)
        self.writer_val = SummaryWriter(log_dir=self.logdir + "/tensorboard/val/", flush_secs=30)

    def train(self):
        # accuracy and IoU stuff
        best_train_iou = 0.0
        best_val_iou = 0.0

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(), self.device, self.ignore_class)

        # train for n epochs
        for epoch in range(self.start_epoch, self.ARCH['train']['epochs']):
            print(f'EPOCH {epoch+1}/{self.ARCH["train"]["epochs"]}')
            # train for 1 epoch
            acc, iou, means, vars, loss = self.train_epoch(train_loader=self.parser.get_train_set(),
                                              model=self.model,
                                              criterion=self.criterion,
                                              optimizer=self.optimizer,
                                              epoch=epoch,
                                              evaluator=self.evaluator,
                                              scheduler=self.scheduler)

            print('Train | acc: {:.2%} | mIoU: {:.2%} | loss: {:.5}'.format(acc, iou, loss))

            # update best iou and save checkpoint
            if iou > best_train_iou:
                print('Best mIoU in training set so far!')
                best_train_iou = iou
                # TODO save checkpoint
                #torch.save(self.model.state_dict(), f"{ARCH['model_architecture']}-model.pt")

            if epoch % self.ARCH['train']['report_epoch'] == 0:
                # evaluate on validation set
                val_acc, val_iou, val_loss = self.validate(val_loader=self.parser.get_valid_set(),
                                               model=self.model,
                                               criterion=self.criterion,
                                               epoch=epoch,
                                               evaluator=self.evaluator)

                print('Validation | acc: {:.2%} | mIoU: {:.2%} | loss: {:.5}'.format(val_acc, val_iou, val_loss))

                if val_iou > best_val_iou:
                    print('Best mIoU in validation so far, model saved!')
                    best_val_iou = val_iou
                    # TODO save the weights
                    #torch.save(self.model_single.state_dict(), os.path.join(self.logdir, f"{self.ARCH['model_params']['model_architecture']}-best.pt"))
                    self.save_checkpoint(epoch=epoch, best_miou=best_val_iou)

        # save mavs to a pickle
        with open(os.path.join(self.logdir, "mavs.pickle"), "wb") as h1:
            pickle.dump(means, h1, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.logdir, "vars.pickle"), "wb") as h2:
            pickle.dump(vars, h2, protocol=pickle.HIGHEST_PROTOCOL)

        # final model
        # torch.save(self.model_single.state_dict(), os.path.join(self.logdir, f"{self.ARCH['model_params']['model_architecture']}-last.pt"))

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        if self.gpu:
            torch.cuda.empty_cache()

        model.train()

        means = None
        vars = None

        # get mavs
        mavs = None
        if epoch and self.owloss is not None:
            mavs = self.owloss.read()

        for i, data_dict in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            lidar = data_dict['lidar'].to(self.device)
            points = data_dict['points'].to(self.device)
            voxel_labels = data_dict['targets'].F.to(self.device).long()
            point_labels = data_dict['targets_mapped'].to(self.device)
            # ref_index = data_dict['ref_index']
            # origin_len = data_dict['origin_len']
            # map = data_dict['map']
            # num_voxel = data_dict['num_voxel']
            invs = data_dict['inverse_map'].to(self.device)

            sem_labels = voxel_labels.clone()
            sem_labels[sem_labels == 255] = 0
            point_labels.F[point_labels.F == 255] = 0

            with torch.amp.autocast(device_type=self.device_name, enabled=self.fp16):
                output, cout = model(lidar)

                # compute loss
                ce_loss = criterion(output, sem_labels)
                lovasz_loss = self.lovasz(F.softmax(output, dim=1), sem_labels)
                mav_loss = self.owloss(output, sem_labels, is_train=True) if self.owloss is not None else torch.tensor(0.0).to(self.device)
                cont_loss = self.contloss(mavs, cout, sem_labels, epoch) if self.contloss is not None else torch.tensor(0.0).to(self.device)
                obj_loss = self.objloss(cout, voxel_labels) if self.objloss is not None else torch.tensor(0.0).to(self.device)
                # print(f'MAV loss: {mav_loss}, Cont loss: {cont_loss.item()} | Obj loss: {obj_loss.item()}')
                loss = ce_loss + 1.5 * lovasz_loss + 0.1 * mav_loss + 0.5 * obj_loss + 0.5 * cont_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            with torch.no_grad():
                evaluator.reset()
                voxel_argmax = output.argmax(dim=1)
                B = int(invs.C[:, 0].max() + 1)
                for b in range(B):
                    pts_mask = lidar.C[:, 0] == b
                    lab_mask = point_labels.C[:, 0] == b
                    inv = invs.F[invs.C[:, 0] == b]
                    pc = voxel_argmax[pts_mask][inv]
                    labels = point_labels.F[lab_mask].long()
                    evaluator.addBatch(pc, labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoUMissingClass()
            losses.update(loss.item(), B)
            acc.update(accuracy.item(), B)
            iou.update(jaccard.item(), B)

            scheduler.step()

            # tensorboard
            if i % 10 == 0 or i == len(train_loader) - 1:
                header = "Train"
                step = epoch * len(train_loader) + i
                self.writer_train.add_scalar(header + '/loss', losses.avg, step)
                self.writer_train.add_scalar(header + '/accuracy', acc.avg, step)
                self.writer_train.add_scalar(header + '/mIoU', iou.avg, step)
                self.writer_train.add_scalar(header + "/lr", self.optimizer.param_groups[0]["lr"], step)
                self.writer_train.add_scalar(header + "/ce_loss", ce_loss.item(), step)
                self.writer_train.add_scalar(header + "/lovasz_loss", lovasz_loss.item(), step)
                self.writer_train.add_scalar(header + "/mav_loss", mav_loss.item(), step)
                self.writer_train.add_scalar(header + "/cont_loss", cont_loss.item(), step)
                self.writer_train.add_scalar(header + "/obj_loss", obj_loss.item(), step)

        if self.owloss is not None:
            means, vars = self.owloss.update()

        return acc.avg, iou.avg, means, vars, losses.avg

    def validate(self, val_loader, model, criterion, epoch, evaluator):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        if self.gpu:
            torch.cuda.empty_cache()

        model.eval()
        evaluator.reset()

        # get mavs
        mavs = None
        if epoch and self.owloss is not None:
            mavs = self.owloss.read()

        with torch.no_grad():
            for i, data_dict in tqdm(enumerate(val_loader), total=len(val_loader)):

                lidar = data_dict['lidar'].to(self.device)
                points = data_dict['points'].to(self.device)
                voxel_labels = data_dict['targets'].F.to(self.device).long()
                point_labels = data_dict['targets_mapped'].to(self.device)
                # ref_index = data_dict['ref_index']
                # origin_len = data_dict['origin_len']
                # map = data_dict['map']
                # num_voxel = data_dict['num_voxel']
                invs = data_dict['inverse_map'].to(self.device)

                sem_labels = voxel_labels.clone()
                sem_labels[sem_labels == 255] = 0
                point_labels.F[point_labels.F == 255] = 0

                with torch.amp.autocast(device_type=self.device_name, enabled=False): # fp16 in validation may cause overflow issues (nan/inf output values)
                    output, cout = model(lidar)

                    # compute loss
                    ce_loss = criterion(output, sem_labels)
                    lovasz_loss = self.lovasz(F.softmax(output, dim=1), sem_labels)
                    mav_loss = self.owloss(output, sem_labels, is_train=False) if self.owloss is not None else torch.tensor(0.0).to(self.device)
                    cont_loss = self.contloss(mavs, cout, sem_labels, epoch) if self.contloss is not None else torch.tensor(0.0).to(self.device)
                    obj_loss = self.objloss(cout, voxel_labels) if self.objloss is not None else torch.tensor(0.0).to(self.device)
                    # print(f'MAV loss: {mav_loss}, Cont loss: {cont_loss.item()} | Obj loss: {obj_loss.item()}')
                    loss = ce_loss + 1.5 * lovasz_loss + 0.1 * mav_loss + 0.5 * obj_loss + 0.5 * cont_loss

                voxel_argmax = output.argmax(dim=1)
                B = int(invs.C[:, 0].max() + 1)
                for b in range(B):
                    pts_mask = lidar.C[:, 0] == b
                    lab_mask = point_labels.C[:, 0] == b
                    inv = invs.F[invs.C[:, 0] == b]
                    pc = voxel_argmax[pts_mask][inv]
                    labels = point_labels.F[lab_mask]
                    evaluator.addBatch(pc, labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoUMissingClass()

                losses.update(loss.item(), B)
                acc.update(accuracy.item(), B)
                iou.update(jaccard.item(), B)
            
            # tensorboard
            header = "Validation"
            step = epoch
            self.writer_val.add_scalar(header + '/loss', losses.avg, step)
            self.writer_val.add_scalar(header + '/accuracy', acc.avg, step)
            self.writer_val.add_scalar(header + '/mIoU', iou.avg, step)
            self.writer_val.add_scalar(header + "/ce_loss", ce_loss.item(), step)
            self.writer_val.add_scalar(header + "/lovasz_loss", lovasz_loss.item(), step)
            self.writer_val.add_scalar(header + "/mav_loss", mav_loss.item(), step)
            self.writer_val.add_scalar(header + "/cont_loss", cont_loss.item(), step)
            self.writer_val.add_scalar(header + "/obj_loss", obj_loss.item(), step)

        return acc.avg, iou.avg, losses.avg
    
    def save_checkpoint(self, epoch, best_miou=None):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict() if not self.multi_gpu else self.model_single.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.fp16 else None,
            'best_miou': best_miou if best_miou is not None else 0.0
        }
        filename = os.path.join(self.logdir, f"checkpoint-best.pt")
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No checkpoint found at '{filename}'")
        
        ckpt = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if ckpt.get('optimizer') is None:
            warnings.warn("Optimizer state not available")
        else:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') is None:
            warnings.warn("Scheduler state not available")
        else:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if self.fp16:
            if ckpt.get('scaler') is None:
                warnings.warn("GradScaler state not available")
            else:
                self.scaler.load_state_dict(ckpt['scaler'])
        if ckpt.get('epoch') is not None:
            self.start_epoch = ckpt['epoch'] + 1
        if ckpt.get('best_miou') is not None:
            self.best_miou = ckpt['best_miou']
        print(f"[INFO] Checkpoint loaded from '{filename}' at epoch {self.start_epoch} with best mIoU {self.best_miou*100:2.2f} %")